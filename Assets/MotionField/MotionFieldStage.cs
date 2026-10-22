using System;
using System.Collections.Generic;
using System.IO;
using AnimationTools;
using Python.Runtime;
using Unity.Mathematics;
using UnityEngine;

namespace MotionField
{
/// <summary>
/// Drives the character from a neural motion field evaluated in Python, choosing each frame the
/// action with the best long-term value toward the requested heading and speed.
/// </summary>
/// <remarks>
/// The <see cref="Policy.Optimal"/> policy needs the value function trained under its seed and
/// refuses to start without one; the one-step-greedy policy needs none, but oscillates rather than
/// committing to multi-step turns. See openwiki/motion-field/motion-field-stage.md.
/// </remarks>
[Serializable]
public class MotionFieldStage : MoSynthStage, IDisposable, IContactBoneSource, ISeededStage,
    ISynthesisInputSource
{
    private bool _isInitialized;
    private bool _initializationFailed;

    private dynamic _actionPredictor;
    private dynamic _motionField;
    private dynamic _skeleton;
    private dynamic _currentX;
    private dynamic _currentV;

    // The character frame this stage steers against: the component's own Transform.
    private Transform _characterTransform;
    private MotionSynthesisComponent _owner;

    // Static so the delegate stays rooted for as long as Python holds it; PythonNET does not
    // forward Python's stdout to Unity on its own.
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
        /// <summary>
        /// The trained value function for the stage's seed. Missing, Init fails; trained on a config
        /// that has changed since, the stage runs greedy instead.
        /// </summary>
        Optimal,

        /// <summary>Force the one-step-greedy policy, ignoring any trained value function.</summary>
        Greedy,

        /// <summary>Debug: play the database back frame by frame.</summary>
        Playback,

        /// <summary>Debug: snap to the successor of the nearest database state.</summary>
        NearestNeighbour
    }

    [SerializeField] public Policy policy = Policy.Optimal;

    [Tooltip("Seconds ahead the greedy policy judges heading at, by extrapolating each action's turn " +
             "rate. One frame of turning is too small to outweigh the travel and speed terms, so at 0 " +
             "the greedy policy strafes or backpedals toward the goal instead of turning to face it. " +
             "Also used by Optimal when it falls back to greedy; never needs a retrain.")]
    [SerializeField] [Min(0f)]
    public float greedyHorizon = 0.5f;

    /// <inheritdoc />
    public int? CheckpointSeed { get; set; }

    /// <summary>Only the optimal policy reads the value function.</summary>
    public bool LoadsCheckpoint => policy == Policy.Optimal;

    public IReadOnlyList<string> ContactBoneNames => config == null ? null : config.ContactBoneNames;

    /// <summary>Last desired heading in world space, for gizmos and debugging.</summary>
    public Vector3 DesiredWorldDirection { get; private set; } = Vector3.forward;

    /// <summary>Last goal heading handed to Python, in radians, in the character's frame.</summary>
    public float Theta { get; private set; }

    /// <summary>Last ground speed handed to Python, in m/s.</summary>
    public float DesiredSpeed { get; private set; } = 1f;

    [Header("Debug")]
    [Tooltip("Publish the UMAP embedding and the per-frame neighbourhood for MotionFieldVisualizer. " +
             "Costs one extra Python call per frame; turn it off when not debugging.")]
    [SerializeField]
    public bool collectDebugData = true;

    // Exposed through methods rather than properties: these arrays run to thousands of entries, and
    // tools that walk properties by reflection would stall on them.
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

    /// <summary>Drawn length of the current facing, in metres; the task's arrow is drawn as one second of travel.</summary>
    private const float FacingDrawLength = 0.6f;

    /// <summary>The shortest a requested heading is drawn, so a standing request still shows which way.</summary>
    private const float MinimumHeadingDrawLength = 0.25f;

    private SynthesisInputs _consumedInputs;
    private string _policyName;

    /// <summary>The task last handed to the policy and the travel of the state it chose, for overlays.</summary>
    public SynthesisInputs ConsumedInputs => _consumedInputs;

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

        _characterTransform = motionSynthesisComponent.transform;
        _policyName = policy.ToString();
        _consumedInputs = new SynthesisInputs("motion field task", 3, 6)
        {
            RequestLabel = "goal heading",
            AnswerLabel = "chosen travel",
            StateLabel = "facing"
        };

        try
        {
            PythonRuntime.EnsureInitialized();

            using (Py.GIL())
            {
                // Once, up front, so every import below comes from one generation of the source.
                if (reloadPythonModules) PythonRuntime.InvalidateProjectModules();

                _actionPredictor = PythonRuntime.Import("motion_field.action_predictor");
                dynamic motionFieldModule = PythonRuntime.Import("motion_field.field");

                var dataPath = config.GetAssetPath();

                var animData = _actionPredictor.load_animations(dataPath, config.name);
                _skeleton = animData[0];
                dynamic poseX = animData[1];
                dynamic poseV = animData[2];
                dynamic poseY = animData[3];
                dynamic poseContacts = animData[4];
                dynamic nextContacts = animData[5];
                dynamic frameTime = animData[6];

                string[] names = PythonRuntime.ToStringArrayOrNull(animData[7].contact_bone_names);
                if (names != null)
                {
                    if (!ContactBoneSources.TryMatchCheckpoint(names.Length, names,
                            _owner.ContactBoneNames, out var contactError))
                    {
                        Debug.LogError($"[MotionField] '{config.name}' database does not fit this character: " +
                                       $"{contactError}. Regenerate the pose database.");
                        _initializationFailed = true;
                        return;
                    }
                }

                using var boneWeights = MotionFieldBoneWeights.ToPython(config);

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
                    locomotion_speed_threshold: config.locomotionSpeedThreshold,
                    travel_factor: config.travelFactor,
                    speed_factor: config.speedFactor,
                    greedy_horizon: greedyHorizon,
                    pose_contacts: poseContacts,
                    next_contacts: nextContacts);

                var valuePath = config.GetValueFunctionPath(CheckpointSeed ?? config.seed);
                if (!File.Exists(valuePath))
                {
                    // Running greedy instead would report a seeded benchmark run under the wrong policy.
                    if (LoadsCheckpoint)
                    {
                        Debug.LogError(MissingValueFunctionMessage(valuePath));
                        _initializationFailed = true;
                        return;
                    }
                }
                // The value function is indexed by database row, so a stale one reads plausible
                // numbers off the wrong poses rather than failing visibly.
                else if (!config.hasTrained)
                {
                    Debug.LogWarning(
                        $"[MotionField] '{config.name}' has changed since the value function was " +
                        "trained, so the stage is running the one-step-greedy policy instead. " +
                        "Press Train Motion Field on the config.");
                }
                // Python has already logged why it refused the file, e.g. a heading-only value
                // function from before speed was a task dimension.
                else if (!(bool)_motionField.load_value_function(valuePath, PythonLog) && LoadsCheckpoint)
                {
                    Debug.LogError($"[MotionField] The {policy} policy could not load the value function at " +
                                   $"'{valuePath}'. Press Train Motion Field on '{config.name}' to rebuild it.");
                    _initializationFailed = true;
                    return;
                }

                var stateCount = (int)poseX.shape[0];
                var index = Mathf.Clamp(startStateIndex, 0, Mathf.Max(0, stateCount - 1));
                _currentX = poseX[index].copy();
                _currentV = poseV[index].copy();

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
                           "Check Project Settings > MoSynth > Python.");
            Debug.LogException(e);
        }
    }

    private string MissingValueFunctionMessage(string valuePath)
    {
        var message = $"[MotionField] The {policy} policy needs a trained value function at " +
                      $"'{valuePath}', and there is none. Train Motion Field for that seed, or run " +
                      "the Greedy policy.";
        var unseeded = config.GetUnseededValueFunctionPath();
        if (File.Exists(unseeded))
            message += $" '{unseeded}' predates per-seed value functions and is not read; retrain " +
                       "rather than rename it, since it was trained without a seed.";
        return message;
    }

    /// <summary>
    /// Pull the UMAP projection across for the visualizer. Must be called with the GIL held.
    /// Every failure path leaves <see cref="GetEmbedding"/> null and lets synthesis carry on.
    /// </summary>
    private void LoadEmbedding(int stateCount)
    {
        var embeddingPath = config.GetEmbeddingPath();
        if (!File.Exists(embeddingPath))
        {
            Debug.LogWarning(
                $"[MotionField] No UMAP embedding at '{embeddingPath}'. MotionFieldVisualizer will " +
                "draw nothing. Press Compute UMAP Embedding on the config to generate it.");
            return;
        }

        // Not re-invalidated: that would hand this module a second copy of MotionField's classes.
        dynamic embeddingModule = PythonRuntime.Import("motion_field.embedding");

        // Only the state count is checked, so an embedding from a different database of the same
        // length is drawn as if current. Recompute it after regenerating the pose database.
        dynamic arrays = embeddingModule.load_embedding_arrays(embeddingPath, stateCount, PythonLog);

        var flat = (float[])arrays[0];
        var edges = (int[])arrays[1];
        var speeds = (float[])arrays[2];
        var embeddedStates = (int)arrays[3];

        if (embeddedStates == 0 || flat.Length < embeddedStates * 3)
        {
            Debug.LogWarning(
                $"[MotionField] The embedding at '{embeddingPath}' was rejected, so the visualizer " +
                "will draw nothing. See the reason logged just above; recompute it from the config " +
                "if the pose database changed.");
            return;
        }

        var points = new Vector3[embeddedStates];
        for (var i = 0; i < embeddedStates; i++)
        {
            points[i] = new Vector3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
        }

        _embedding = points;
        _embeddingEdges = edges;
        _stateSpeeds = speeds;
    }

    /// <summary>
    /// Capture the neighbourhood the policy just used, from Python rather than a second k-NN that
    /// could disagree with it. Must be called with the GIL held.
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
                dynamic contacts = nextPose[2];

                if (collectDebugData)
                {
                    CaptureDebugArrays();
                }

                var poseArrays = _actionPredictor.get_pose_arrays(
                    _skeleton, _currentX, _currentV, contacts);

                var posArray = (float[])poseArrays[0];
                var quatArray = (float[])poseArrays[1];
                var lvArray = (float[])poseArrays[2];
                var lavArray = (float[])poseArrays[3];

                dynamic contactFlags = poseArrays[4];
                var contactHandles = _owner.ContactHandles;
                for (var slot = 0; slot < contactHandles.Count; slot++)
                {
                    pose.SetBool(contactHandles[slot], (bool)contactFlags[slot]);
                }

                var positions = pose.Positions;
                var rotations = pose.Rotations;
                var velocities = pose.Velocities;
                var angularVelocities = pose.AngularVelocities;

                var numJoints = positions.Length;

                for (var i = 0; i < numJoints; i++)
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

            RecordConsumedInputs(pose, deltaTime);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            // Stop rather than log the same failure every frame.
            _isInitialized = false;
        }

        return true;
    }

    /// <summary>
    /// Copies the task this tick handed the policy into <see cref="ConsumedInputs"/>: the goal heading
    /// as the policy received it, relative to the facing, and the travel of the state it chose.
    /// </summary>
    private void RecordConsumedInputs(PoseBuffer pose, float deltaTime)
    {
        var inputs = _consumedInputs;
        inputs.ClearMarkers();
        inputs.ClearReadouts();

        float3 root = _characterTransform.position;
        quaternion facingRotation = _characterTransform.rotation;
        var facing = math.normalizesafe(math.mul(facingRotation, math.forward()) * new float3(1f, 0f, 1f),
            math.forward());
        inputs.AddArrow(SynthesisInputRole.State, root, facing * FacingDrawLength);

        var taskDriven = policy == Policy.Optimal || policy == Policy.Greedy;
        if (taskDriven)
        {
            // Theta is the negated signed angle from facing to goal, so the goal is the facing turned by -Theta.
            var heading = math.mul(quaternion.AxisAngle(math.up(), -Theta), facing);
            inputs.AddArrow(SynthesisInputRole.Request, root,
                heading * math.max(DesiredSpeed, MinimumHeadingDrawLength));
        }

        SimulationFrame.ComputeVelocity(pose, _owner.SkeletonData, _owner.SimulationFrame, deltaTime,
            out var frameVelocity, out _);
        var travel = math.mul(facingRotation, frameVelocity);
        inputs.AddArrow(SynthesisInputRole.Answer, root, travel);

        inputs.AddReadout("policy", _policyName);
        if (taskDriven)
        {
            inputs.AddReadout("theta", math.degrees(Theta), "0.0", " deg");
            inputs.AddReadout("desired speed", DesiredSpeed, "0.00", " m/s");
        }

        inputs.AddReadout("chosen speed", math.length(travel), "0.00", " m/s");
        if (collectDebugData && LastChosenSlot >= 0 && LastChosenSlot < LastNeighbors.Length)
        {
            inputs.AddReadout("chosen state", LastNeighbors[LastChosenSlot], "0");
            inputs.AddReadout("its weight", LastNeighborWeights[LastChosenSlot], "0.00");
        }
    }

    /// <summary>Run one step of the selected policy. Must be called with the GIL held.</summary>
    private dynamic StepPolicy(float deltaTime) => policy switch
    {
        // Falls back to greedy on its own when no value function is loaded.
        Policy.Optimal => _motionField.optimal_action(
            theta: Theta, desired_speed: DesiredSpeed, current_x: _currentX, current_v: _currentV,
            delta_time: deltaTime),

        Policy.Greedy => _motionField.greedy_action(
            theta: Theta, desired_speed: DesiredSpeed, current_x: _currentX, current_v: _currentV,
            delta_time: deltaTime),

        Policy.Playback => _motionField.get_next_pose(
            current_x: _currentX, current_v: _currentV, delta_time: deltaTime),

        Policy.NearestNeighbour => _motionField.get_next_pose_from_field(
            current_x: _currentX, current_v: _currentV, delta_time: deltaTime),

        _ => throw new NotImplementedException($"Unknown policy {policy}."),
    };

    /// <summary>
    /// Convert the desired world heading into the goal angle the value function was trained on.
    /// Call every frame, since Theta is measured against the current facing; a near-zero vector
    /// keeps the previous heading and only refreshes Theta.
    /// </summary>
    /// <remarks>
    /// Measuring against the character's facing means Python's root accumulator never has to be
    /// synchronised with Unity's transform. The negation matches the training reward
    /// -|theta + delta_yaw|.
    /// </remarks>
    public void SetDesiredDirection(Vector3 desired)
    {
        desired.y = 0f;

        if (desired.sqrMagnitude > 1e-6f)
        {
            DesiredWorldDirection = desired.normalized;
        }

        // The character Transform's +Z is the facing: MotionSynthesisComponent keeps it aligned
        // with the yaw-only simulation frame the pose derives.
        var forward = _characterTransform != null ? _characterTransform.forward : Vector3.forward;
        forward.y = 0f;
        if (forward.sqrMagnitude < 1e-6f) forward = Vector3.forward;

        Theta = -Vector3.SignedAngle(forward, DesiredWorldDirection, Vector3.up) * Mathf.Deg2Rad;
    }

    /// <summary>
    /// The ground speed the policy should move at, in m/s. Values outside the speed range the value
    /// function was trained over are answered as its nearest end.
    /// </summary>
    public void SetDesiredSpeed(float speed)
    {
        DesiredSpeed = Mathf.Max(0f, speed);
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
        }

        _isInitialized = false;
    }

    public override void OnDestroy()
    {
        Dispose();
    }
}
}