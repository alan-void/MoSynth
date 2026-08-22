using System;
using System.IO;
using AnimationTools;
using Python.Runtime;
using Unity.Mathematics;
using UnityEngine;

namespace MotionField
{
/// <summary>
/// Drives the character from a neural motion field evaluated in Python.
///
/// Each frame the field is asked for the action with the best long-term value given where the
/// player wants to go. That look-ahead is what a value function buys: a one-step-greedy policy only
/// asks which action points closest to the goal on the very next frame, so it oscillates rather
/// than committing to a turn that takes a few steps to set up. Without a trained value function the
/// stage falls back to exactly that greedy policy.
/// </summary>
[Serializable]
public class MotionFieldStage : MoSynthStage, IDisposable
{
    private bool _isInitialized;
    private bool _initializationFailed;

    private dynamic _actionPredictor;
    private dynamic _motionField;
    private dynamic _skeleton;
    private dynamic _currentX;
    private dynamic _currentV;
    private dynamic _currentContacts;

    private Transform _rootTransform;
    private MotionSynthesisComponent _owner;

    /// <summary>
    /// Sink for the Python side's own diagnostics.
    /// </summary>
    /// <remarks>
    /// Python logs to stdout, which PythonNET does not forward anywhere Unity shows, so anything it
    /// says about a rejected artefact would otherwise vanish. Static so the delegate stays rooted
    /// for as long as Python might hold it.
    /// </remarks>
    private static readonly Action<string> PythonLog = message => Debug.Log(message);

    [Tooltip("Animation database and motion field hyperparameters. Train the value function from " +
             "this asset's inspector.")]
    [SerializeField]
    public MotionFieldConfig config;

    [Tooltip("Database state the character starts from.")] [SerializeField] [Min(0)]
    public int startStateIndex = 700;

    [Tooltip("Re-execute the Python modules on start so .py edits apply without restarting Unity.")] [SerializeField]
    public bool reloadPythonModules = true;

    public enum Policy
    {
        /// <summary>Trained value function when available, greedy otherwise.</summary>
        Optimal,

        /// <summary>Force the one-step-greedy policy, ignoring any trained value function.</summary>
        Greedy,

        /// <summary>Debug: play the database back frame by frame.</summary>
        Playback,

        /// <summary>Debug: snap to the successor of the nearest database state.</summary>
        NearestNeighbour
    }

    [SerializeField] public Policy policy = Policy.Optimal;

    /// <summary>Last desired heading in world space, for gizmos and debugging.</summary>
    public Vector3 DesiredWorldDirection { get; private set; } = Vector3.forward;

    /// <summary>Last goal heading handed to Python, in radians, in the character's frame.</summary>
    public float Theta { get; private set; }

    [Header("Debug")]
    [Tooltip("Publish the UMAP embedding and the per-frame neighbourhood for MotionFieldVisualizer. " +
             "Costs one extra Python call per frame; turn it off when not debugging.")]
    [SerializeField]
    public bool collectDebugData = true;

    // Bulk debug data is exposed through methods rather than properties on purpose. These arrays
    // run to thousands of entries, and anything that walks the component's properties by
    // reflection -- a debug inspector, a scene serializer, an editor bridge -- will stall or dump
    // megabytes if they look like cheap getters.
    private Vector3[] _embedding;
    private int[] _embeddingEdges;
    private float[] _stateSpeeds;

    /// <summary>True once a matching UMAP embedding has been loaded.</summary>
    public bool HasEmbedding => _embedding != null;

    /// <summary>UMAP projection of the database, one point per state. Null when unavailable.</summary>
    public Vector3[] GetEmbedding() => _embedding;

    /// <summary>Flat (from, to) state index pairs for consecutive frames within a clip.</summary>
    public int[] GetEmbeddingEdges() => _embeddingEdges;

    /// <summary>Root speed per state, m/s. Used to colour the point cloud.</summary>
    public float[] GetStateSpeeds() => _stateSpeeds;

    /// <summary>Database states nearest the live pose on the last step.</summary>
    public int[] LastNeighbors { get; private set; } = Array.Empty<int>();

    /// <summary>Similarity weights matching <see cref="LastNeighbors"/>; sums to 1.</summary>
    public float[] LastNeighborWeights { get; private set; } = Array.Empty<float>();

    /// <summary>
    /// Index into <see cref="LastNeighbors"/> of the neighbour the chosen action emphasised, which
    /// is also the state the tug pulled toward. -1 before the first step, and in the playback and
    /// nearest-neighbour policies, which make no choice.
    /// </summary>
    public int LastChosenSlot { get; private set; } = -1;

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        if (_isInitialized || _initializationFailed) return;

        _owner = motionSynthesisComponent;

        if (config == null)
        {
            Debug.LogError("[MotionField] MotionFieldStage has no MotionFieldConfig assigned.");
            _initializationFailed = true;
            return;
        }

        _rootTransform = motionSynthesisComponent.SkeletonTransforms is { Length: > 0 }
            ? motionSynthesisComponent.SkeletonTransforms[0]
            : motionSynthesisComponent.transform;

        try
        {
            PythonRuntime.EnsureInitialized(config.pythonDllPath, config.pythonVenvPath);

            using (Py.GIL())
            {
                // Once, up front, rather than per import: every module pulled in below then comes
                // from the same generation of the source, sharing one copy of each shared class.
                if (reloadPythonModules) PythonRuntime.InvalidateProjectModules();

                _actionPredictor = PythonRuntime.Import("action_predictor");
                dynamic motionFieldModule = PythonRuntime.Import("MotionField");

                string dataPath = config.GetAssetPath();

                var animData = _actionPredictor.load_animations(dataPath, config.name);
                _skeleton = animData[0];
                dynamic poseX = animData[1];
                dynamic poseV = animData[2];
                dynamic poseY = animData[3];
                dynamic poseContacts = animData[4];
                dynamic frameTime = animData[5];

                using PyDict boneWeights = MotionFieldBoneWeights.ToPython(config);

                _motionField = motionFieldModule.MotionField(
                    poseX, poseV, poseY, _skeleton, frameTime,
                    device: config.DeviceName,
                    pos_weight: config.posWeight,
                    vel_weight: config.velWeight,
                    k_neighbors: config.kNeighbors,
                    tug_ratio: config.tugRatio,
                    knn_chunk: config.knnChunk,
                    bone_weights: boneWeights,
                    locomotion_factor: config.locomotionFactor,
                    locomotion_speed_threshold: config.locomotionSpeedThreshold);

                // A stale or absent value function degrades to greedy control rather than throwing.
                // Failing hard here would surface as an opaque managed exception in a player build.
                string valuePath = config.GetValueFunctionPath();
                if (!File.Exists(valuePath))
                {
                    Debug.LogWarning(
                        $"[MotionField] No trained value function at '{valuePath}'. Using the " +
                        "one-step-greedy policy. Press Train Motion Field on the config to fix this.");
                }
                // Refusing a stale one is not caution: the value function is indexed by database
                // row, so a mismatched pair reads plausible numbers off the wrong poses and the
                // character merely moves badly rather than visibly failing.
                else if (!config.hasTrained)
                {
                    Debug.LogWarning(
                        $"[MotionField] '{config.name}' has changed since the value function was " +
                        "trained, so the stage is running the one-step-greedy policy instead. " +
                        "Press Train Motion Field on the config.");
                }
                else
                {
                    _motionField.load_value_function(valuePath, PythonLog);
                }

                int stateCount = (int)poseX.shape[0];
                int index = Mathf.Clamp(startStateIndex, 0, Mathf.Max(0, stateCount - 1));
                _currentX = poseX[index].copy();
                _currentV = poseV[index].copy();
                _currentContacts = poseContacts[index];

                if (collectDebugData)
                {
                    LoadEmbedding(stateCount);
                }
            }

            _isInitialized = true;
        }
        catch (Exception e)
        {
            _initializationFailed = true;
            Debug.LogError("[MotionField] Failed to initialize the motion field. " +
                           "Check the Python DLL and venv paths on the MotionFieldConfig.");
            Debug.LogException(e);
        }
    }

    /// <summary>
    /// Pull the UMAP projection across for the visualizer. Must be called with the GIL held.
    ///
    /// The embedding is optional debug data, so every failure path here leaves <see cref="Embedding"/>
    /// null and lets synthesis carry on -- the visualizer simply draws nothing.
    /// </summary>
    private void LoadEmbedding(int stateCount)
    {
        string embeddingPath = config.GetEmbeddingPath();
        if (!File.Exists(embeddingPath))
        {
            Debug.LogWarning(
                $"[MotionField] No UMAP embedding at '{embeddingPath}'. MotionFieldVisualizer will " +
                "draw nothing. Press Compute UMAP Embedding on the config to generate it.");
            return;
        }

        // Init has already invalidated the module cache if it was going to, and re-invalidating
        // here would hand this module a second copy of MotionField's classes.
        dynamic embeddingModule = PythonRuntime.Import("motion_field_embedding");

        // The state count is the only thing checked over there. Nothing hashes the database, so an
        // embedding fitted on a different database of the same length is drawn as if current --
        // recompute it whenever the pose database is regenerated.
        dynamic arrays = embeddingModule.load_embedding_arrays(embeddingPath, stateCount, PythonLog);

        var flat = (float[])arrays[0];
        int[] edges = (int[])arrays[1];
        var speeds = (float[])arrays[2];
        int embeddedStates = (int)arrays[3];

        if (embeddedStates == 0 || flat.Length < embeddedStates * 3)
        {
            Debug.LogWarning(
                $"[MotionField] The embedding at '{embeddingPath}' was rejected, so the visualizer " +
                "will draw nothing. See the reason logged just above; recompute it from the config " +
                "if the pose database changed.");
            return;
        }

        var points = new Vector3[embeddedStates];
        for (int i = 0; i < embeddedStates; i++)
        {
            points[i] = new Vector3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
        }

        _embedding = points;
        _embeddingEdges = edges;
        _stateSpeeds = speeds;
    }

    /// <summary>
    /// Capture the neighbourhood the policy just used. Must be called with the GIL held.
    ///
    /// Taken from Python rather than recomputed in C# so the highlight shows the decision that was
    /// actually made; a second k-NN here could disagree with the one that produced the pose.
    /// </summary>
    private void CaptureDebugArrays()
    {
        dynamic debug = _motionField.get_debug_arrays();
        LastNeighbors = (int[])debug[0];
        LastNeighborWeights = (float[])debug[1];
        LastChosenSlot = (int)debug[2];
    }

    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        if (!_isInitialized) return true;

        try
        {
            using (Py.GIL())
            {
                dynamic nextPose = StepPolicy(deltaTime);
                _currentX = nextPose[0];
                _currentV = nextPose[1];

                if (collectDebugData)
                {
                    CaptureDebugArrays();
                }

                var poseArrays = _actionPredictor.get_pose_arrays(
                    _skeleton, _currentX, _currentV, _currentContacts);

                float[] posArray = (float[])poseArrays[0];
                float[] quatArray = (float[])poseArrays[1];
                float[] lvArray = (float[])poseArrays[2];
                float[] lavArray = (float[])poseArrays[3];

                pose.SetBool(_owner.LeftFootContactHandle, (bool)poseArrays[4]);
                pose.SetBool(_owner.RightFootContactHandle, (bool)poseArrays[5]);

                var positions = pose.Positions;
                var rotations = pose.Rotations;
                var velocities = pose.Velocities;
                var angularVelocities = pose.AngularVelocities;

                int numJoints = positions.Length;

                for (int i = 0; i < numJoints; i++)
                {
                    positions[i] = new float3(
                        posArray[i * 3], posArray[i * 3 + 1], posArray[i * 3 + 2]);

                    rotations[i] = new quaternion(
                        quatArray[i * 4], quatArray[i * 4 + 1], quatArray[i * 4 + 2], quatArray[i * 4 + 3]);

                    velocities[i] = new float3(
                        lvArray[i * 3], lvArray[i * 3 + 1], lvArray[i * 3 + 2]);

                    angularVelocities[i] = new float3(
                        lavArray[i * 3], lavArray[i * 3 + 1], lavArray[i * 3 + 2]);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            _isInitialized = false; // stop spamming the console every frame
        }

        return true;
    }

    /// <summary>Run one step of the selected policy. Must be called with the GIL held.</summary>
    private dynamic StepPolicy(float deltaTime) => policy switch
    {
        // Falls back to greedy on its own when no value function is loaded.
        Policy.Optimal => _motionField.optimal_action(
            theta: Theta, current_x: _currentX, current_v: _currentV, delta_time: deltaTime),

        Policy.Greedy => _motionField.greedy_action(
            theta: Theta, current_x: _currentX, current_v: _currentV, delta_time: deltaTime),

        Policy.Playback => _motionField.get_next_pose(
            current_x: _currentX, current_v: _currentV, delta_time: deltaTime),

        Policy.NearestNeighbour => _motionField.get_next_pose_from_field(
            current_x: _currentX, current_v: _currentV, delta_time: deltaTime),

        _ => throw new NotImplementedException($"Unknown policy {policy}."),
    };

    /// <summary>
    /// Convert the desired world heading into the goal angle the value function was trained on.
    ///
    /// Theta is the character's own heading expressed in the goal frame, which is what makes it
    /// frame-agnostic: Python accumulates its root pose independently of Unity's transform, and the
    /// only thing that couples them is the per-step yaw. Measuring the goal relative to the
    /// character's current facing means the two accumulators never have to be synchronised.
    ///
    /// The negation matches the training convention, where reward is -|theta + delta_yaw|: the
    /// action that cancels theta is the one that turns the character onto the goal.
    ///
    /// Callers are expected to push a fresh direction every frame because Theta is measured
    /// against the root's current facing; a near-zero vector keeps the previous heading and only
    /// refreshes Theta.
    /// </summary>
    public void SetDesiredDirection(Vector3 desired)
    {
        desired.y = 0f;

        if (desired.sqrMagnitude > 1e-6f)
        {
            DesiredWorldDirection = desired.normalized;
        }

        // Root-local +Z is the character's facing: PoseExtractor builds the simulation bone as
        // LookRotation(hips forward projected on the ground).
        Vector3 forward = _rootTransform != null ? _rootTransform.forward : Vector3.forward;
        forward.y = 0f;
        if (forward.sqrMagnitude < 1e-6f) forward = Vector3.forward;

        Theta = -Vector3.SignedAngle(forward, DesiredWorldDirection, Vector3.up) * Mathf.Deg2Rad;
    }

    public void Dispose()
    {
        if (!_isInitialized) return;

        using (Py.GIL())
        {
            _actionPredictor = null;
            _motionField = null;
            _skeleton = null;
            _currentX = null;
            _currentV = null;
            _currentContacts = null;
        }

        _isInitialized = false;
    }

    public override void OnDestroy()
    {
        Dispose();
    }
}
}