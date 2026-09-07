using System;
using AnimationTools;
using Python.Runtime;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace Pfnn
{
/// <summary>
/// Synthesises motion with a trained phase-functioned network (Holden et al. 2017), running the
/// model in Python behind PythonNET.
/// </summary>
/// <remarks>
/// Each tick the stage assembles what the network was trained to see — a trajectory window either
/// side of now, the joints as they stand, the gait phase — reads back the next pose, and writes it
/// through <see cref="CharacterSpacePose.Apply"/>. The character travels because the predicted root
/// delta is written into bone 0's velocity channels, which
/// <see cref="MotionSynthesisComponent"/> reads back to advance the transform.
/// <para>
/// The stage owns the state — phase, joints, trajectory history — rather than leaving it in Python.
/// A motion field's state is an index into a database and is cheap to keep across the boundary; a
/// PFNN's state is a pose that has to be written into a <see cref="PoseBuffer"/> anyway.
/// </para>
/// <para>
/// See the wiki's PFNN section for the vector layouts and the training-time definitions these have
/// to agree with.
/// </para>
/// </remarks>
[Serializable]
public class PfnnStage : MoSynthStage, IDisposable
{
    [SerializeField]
    [Tooltip("The trained config. Its checkpoint decides which bones this stage writes.")]
    public PfnnConfig config;

    [SerializeField]
    [Tooltip("Re-read the Python modules on Init, so edits apply without restarting the editor. " +
             "Turn off for builds.")]
    public bool reloadPythonModules = true;

    [SerializeField]
    [Tooltip("Gait phase to start on, as a fraction of a cycle. Which foot leads out of a " +
             "standing start.")]
    [Range(0f, 1f)]
    public float startPhase;

    [SerializeField]
    [Tooltip("Where inference runs. A single-sample forward pass is small, so CPU usually beats " +
             "the per-call transfer to a GPU.")]
    public PfnnConfig.ComputeDevice inferenceDevice = PfnnConfig.ComputeDevice.Cpu;

    // Static so the delegate stays rooted for as long as Python holds it; PythonNET does not
    // forward Python's stdout to Unity on its own.
    private static readonly Action<string> PythonLog = message => Debug.Log(message);

    private const float Tau = 2f * math.PI;

    private bool _isInitialized;
    private bool _initializationFailed;

    private dynamic _policy;
    private MotionSynthesisComponent _owner;
    private Transform _characterTransform;
    private SkeletonData _skeletonData;
    private SimulationFrameDef _frameDef;

    /// <summary>Whatever is steering the character, when it is a PFNN input; null while nothing is.</summary>
    public PfnnControlInput ControlInput => _owner?.ControlInput as PfnnControlInput;

    /// <summary>Current gait phase in radians, for gizmos and diagnostics.</summary>
    public float Phase { get; private set; }

    /// <summary>Samples in the trajectory window, once one has been assembled.</summary>
    public int WindowSampleCount => _windowFilled ? _windowOffsets.Length : 0;

    /// <summary>
    /// One sample of the window last handed to the network, taken back out of the character frame
    /// it was packed into.
    /// </summary>
    /// <remarks>
    /// Reading the packed floats rather than re-asking the control input is the point: this is what
    /// the model was queried with, so a packing or frame error shows up as a visibly wrong drawing
    /// instead of hiding behind a picture that redraws the intent correctly.
    /// </remarks>
    /// <param name="framesAhead">The sample's offset from now, negative for the history half.</param>
    public void GetWindowSample(int index, out int framesAhead, out float2 world, out float2 direction)
    {
        framesAhead = _windowOffsets[index];
        world = PfnnTrajectory.FromFrame(
            new float2(_windowPositions[index * 2], _windowPositions[index * 2 + 1]),
            _windowOrigin, _windowFrameYaw);
        direction = PfnnTrajectory.DirectionFromFrame(
            new float2(_windowDirections[index * 2], _windowDirections[index * 2 + 1]),
            _windowFrameYaw);
    }

    // Packing, read off the checkpoint at Init ---------------------------------------------------
    private int[] _boneIndices;      // checkpoint slot -> skeleton bone
    private int[] _slotOfBone;       // skeleton bone -> checkpoint slot, or -1
    private int[] _windowOffsets;
    private float _databaseFrameTime = 1f / 30f;

    // Per-tick working arrays, all in the character frame ----------------------------------------
    private NativeArray<float3> _positions;
    private NativeArray<quaternion> _rotations;
    private NativeArray<float3> _velocities;
    private NativeArray<float3> _angularVelocities;
    private NativeArray<quaternion> _previousRotations;
    private bool _hasPreviousPose;

    // What crosses the boundary -----------------------------------------------------------------
    private float[] _windowPositions;
    private float[] _windowDirections;
    private float[] _jointPositions;
    private float[] _jointVelocities;
    private readonly float[] _contacts = new float[2];

    private PfnnTrajectory _trajectory;

    // The frame the window above was packed into, kept so a gizmo can take it back out again.
    private float2 _windowOrigin;
    private float _windowFrameYaw;
    private bool _windowFilled;

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        if (_isInitialized || _initializationFailed) return;

        _owner = motionSynthesisComponent;

        if (config == null)
        {
            Debug.LogError("[PFNN] PfnnStage has no config assigned.");
            _initializationFailed = true;
            return;
        }

        _characterTransform = motionSynthesisComponent.transform;
        _skeletonData = motionSynthesisComponent.SkeletonData;
        _frameDef = motionSynthesisComponent.SimulationFrame;

        try
        {
            PythonRuntime.EnsureInitialized();

            using (Py.GIL())
            {
                // Once, up front, so every import below comes from one generation of the source.
                if (reloadPythonModules) PythonRuntime.InvalidateProjectModules();

                dynamic runtime = PythonRuntime.Import("pfnn_runtime");
                _policy = runtime.PfnnPolicy(config.GetCheckpointPath(), PythonLog,
                    DeviceName(inferenceDevice));

                if (!TryBindCheckpoint()) return;
                Debug.Log($"[PFNN] {config.name}: {(string)_policy.describe()}");
            }

            AllocateBuffers();
            SeedFromCurrentPose();
            WarnIfTickRateDisagrees();

            _isInitialized = true;
        }
        catch (Exception e)
        {
            _initializationFailed = true;
            Debug.LogError($"[PFNN] failed to start. Check Project Settings > MoSynth > Python, " +
                           $"and that '{config.GetCheckpointPath()}' exists — press Train on the " +
                           "config if it does not.");
            Debug.LogException(e);
        }
    }

    /// <summary>
    /// Maps the checkpoint's bones onto this rig, which is the check that stops a model being fed
    /// something other than what it was trained on. Must be called with the GIL held.
    /// </summary>
    private bool TryBindCheckpoint()
    {
        var checkpointBones = (string[])_policy.bone_names();

        if (!PfnnBoneSelection.TryResolve(checkpointBones, _owner.Skeleton,
                out _boneIndices, out var error))
        {
            _initializationFailed = true;
            Debug.LogError($"[PFNN] '{config.name}' checkpoint does not fit this character: {error}. " +
                           "Regenerate the database and retrain if the rig changed.");
            return false;
        }

        _slotOfBone = new int[_skeletonData.BoneCount];
        for (var i = 0; i < _slotOfBone.Length; i++) _slotOfBone[i] = -1;
        for (var slot = 0; slot < _boneIndices.Length; slot++) _slotOfBone[_boneIndices[slot]] = slot;

        _windowOffsets = (int[])_policy.window_offsets();
        _databaseFrameTime = (float)_policy.frame_time();
        return true;
    }

    private void AllocateBuffers()
    {
        var boneCount = _skeletonData.BoneCount;
        _positions = new NativeArray<float3>(boneCount, Allocator.Domain);
        _rotations = new NativeArray<quaternion>(boneCount, Allocator.Domain);
        _velocities = new NativeArray<float3>(boneCount, Allocator.Domain);
        _angularVelocities = new NativeArray<float3>(boneCount, Allocator.Domain);
        _previousRotations = new NativeArray<quaternion>(boneCount, Allocator.Domain);

        _windowPositions = new float[_windowOffsets.Length * 2];
        _windowDirections = new float[_windowOffsets.Length * 2];
        _jointPositions = new float[_boneIndices.Length * 3];
        _jointVelocities = new float[_boneIndices.Length * 3];

        var reach = 0;
        foreach (var offset in _windowOffsets) reach = math.max(reach, -offset);
        _trajectory = new PfnnTrajectory(reach);
    }

    /// <summary>
    /// Take the first input from the rig as it stands, so the character starts from its own pose
    /// rather than from an arbitrary database frame.
    /// </summary>
    private void SeedFromCurrentPose()
    {
        CharacterSpacePose.Extract(_owner.CurrentPose, _skeletonData, _frameDef, _databaseFrameTime,
            _positions, _rotations, _velocities, _angularVelocities);

        for (var slot = 0; slot < _boneIndices.Length; slot++)
        {
            var bone = _boneIndices[slot];
            WriteFloat3(_jointPositions, slot, _positions[bone]);
            WriteFloat3(_jointVelocities, slot, _velocities[bone]);
        }

        _previousRotations.CopyFrom(_rotations);
        _hasPreviousPose = false;

        _contacts[0] = _owner.CurrentPose.GetBool(_owner.LeftFootContactHandle) ? 1f : 0f;
        _contacts[1] = _owner.CurrentPose.GetBool(_owner.RightFootContactHandle) ? 1f : 0f;

        Phase = startPhase * Tau;
        _trajectory.Seed(CurrentGroundPosition(), CurrentYaw());
    }

    /// <summary>
    /// The network steps one database frame per synthesis tick, so the two rates have to agree or
    /// the animation plays at the wrong speed even though the travel distance stays right.
    /// </summary>
    private void WarnIfTickRateDisagrees()
    {
        var tick = 1f / math.max(1f, _owner.synthesisFrameRate);
        if (math.abs(tick - _databaseFrameTime) < 1e-3f) return;

        Debug.LogWarning(
            $"[PFNN] synthesisFrameRate is {_owner.synthesisFrameRate} Hz but '{config.name}' was " +
            $"trained at {1f / _databaseFrameTime:0.#} Hz. The character will travel at the right " +
            "speed but its limbs will move at the wrong one. Match the two.");
    }

    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        if (!_isInitialized) return true;

        try
        {
            _trajectory.Push(CurrentGroundPosition(), CurrentYaw());
            FillTrajectoryWindow();

            float rootHeight, dx, dz, dyaw, phaseDelta;
            bool leftContact, rightContact;
            float[] rotations6d, jointVelocities;

            using (Py.GIL())
            {
                dynamic result = _policy.step(_windowPositions, _windowDirections,
                    _jointPositions, _jointVelocities, _contacts, Phase);

                rotations6d = (float[])result[0];
                jointVelocities = (float[])result[1];
                rootHeight = (float)result[2];
                dx = (float)result[3];
                dz = (float)result[4];
                dyaw = (float)result[5];
                phaseDelta = (float)result[6];
                leftContact = (bool)result[7];
                rightContact = (bool)result[8];
            }

            BuildCharacterSpacePose(rotations6d, jointVelocities, rootHeight);

            // Rates rather than per-tick deltas, because that is what the pose channels mean and
            // what MotionSynthesisComponent integrates over its own timestep.
            var frameVelocity = new float3(dx, 0f, dz) / _databaseFrameTime;
            var frameYawRate = dyaw / _databaseFrameTime;

            // The pose is written in a frame at the origin: the component re-anchors bone 0 against
            // the frame the pose implies and accumulates the travel onto its own Transform, so an
            // absolute frame here would be applied twice.
            CharacterSpacePose.Apply(pose, _skeletonData, float3.zero, quaternion.identity,
                frameVelocity, frameYawRate, _positions, _rotations,
                _velocities, _angularVelocities);

            pose.SetBool(_owner.LeftFootContactHandle, leftContact);
            pose.SetBool(_owner.RightFootContactHandle, rightContact);

            _contacts[0] = leftContact ? 1f : 0f;
            _contacts[1] = rightContact ? 1f : 0f;

            // Gait phase only ever advances: GaitPhase builds it as a monotone unwrapped angle, so
            // every training target was non-negative and a negative prediction is the network
            // extrapolating outside what it was shown. Letting one through would run the cycle
            // backwards, which no amount of later frames recovers from.
            Phase = Repeat(Phase + math.max(0f, phaseDelta), Tau);
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
    /// The window the network expects: history behind the character, the control input's wishes
    /// ahead of it, both in the character's own frame.
    /// </summary>
    private void FillTrajectoryWindow()
    {
        var origin = CurrentGroundPosition();
        var frameYaw = CurrentYaw();
        _windowOrigin = origin;
        _windowFrameYaw = frameYaw;
        _windowFilled = true;

        for (var i = 0; i < _windowOffsets.Length; i++)
        {
            var offset = _windowOffsets[i];
            float2 world;
            float yaw;

            if (offset <= 0)
            {
                _trajectory.Sample(-offset, out world, out yaw);
            }
            else if (ControlInput == null ||
                     !ControlInput.TryGetFutureSample(offset, out world, out var direction))
            {
                // Nothing steering: the character is being asked to stand where it is.
                world = origin;
                yaw = frameYaw;
            }
            else
            {
                yaw = PfnnTrajectory.YawOf(direction);
            }

            var local = PfnnTrajectory.ToFrame(world, origin, frameYaw);
            var localDirection = PfnnTrajectory.DirectionInFrame(yaw, frameYaw);

            _windowPositions[i * 2] = local.x;
            _windowPositions[i * 2 + 1] = local.y;
            _windowDirections[i * 2] = localDirection.x;
            _windowDirections[i * 2 + 1] = localDirection.y;
        }
    }

    /// <summary>
    /// Turns the network's output into a full character-frame pose, and keeps the part of it that
    /// becomes the next tick's input.
    /// </summary>
    /// <remarks>
    /// One forward pass over the skeleton. A bone the model does not predict takes its rest local
    /// rotation against its parent, which is well defined because the predicted set is closed under
    /// parent — a bone is always excluded together with its subtree.
    /// </remarks>
    private void BuildCharacterSpacePose(float[] rotations6d, float[] jointVelocities,
        float rootHeight)
    {
        _previousRotations.CopyFrom(_rotations);

        for (var bone = 0; bone < _skeletonData.BoneCount; bone++)
        {
            var slot = _slotOfBone[bone];
            var parent = _skeletonData.ParentIndices[bone];

            _rotations[bone] = slot >= 0
                ? RotationFrom6D(rotations6d, slot)
                : math.mul(_rotations[parent], _skeletonData.RestLocalRotations[bone]);

            _positions[bone] = bone == 0
                ? new float3(0f, rootHeight, 0f)
                : _positions[parent] +
                  math.rotate(_rotations[parent], _skeletonData.RestLocalPositions[bone]);

            // The network predicts a linear rate per bone but no angular one, so the angular rate
            // is differenced from the rotations — which is the same quantity training_data stores,
            // differences of consecutive frame-local poses. On the first tick there is no previous
            // frame to difference against, only the rig's rest pose, and bone 0's share of that
            // would be integrated onto the character's Transform as a tick of root motion.
            _velocities[bone] = slot >= 0 ? ReadFloat3(jointVelocities, slot) : float3.zero;
            _angularVelocities[bone] = _hasPreviousPose
                ? AngularVelocity(_previousRotations[bone], _rotations[bone], _databaseFrameTime)
                : float3.zero;
        }

        _hasPreviousPose = true;

        for (var slot = 0; slot < _boneIndices.Length; slot++)
        {
            var bone = _boneIndices[slot];
            WriteFloat3(_jointPositions, slot, _positions[bone]);
            WriteFloat3(_jointVelocities, slot, _velocities[bone]);
        }
    }

    /// <summary>
    /// A rotation from the two-axis form the network regresses, by Gram-Schmidt.
    /// </summary>
    /// <remarks>
    /// The C# counterpart of <c>training_data.rotations_from_6d</c>. A regressed pair of columns is
    /// not orthonormal, and this is the projection that makes it a rotation again (Zhou et al.).
    /// </remarks>
    private static quaternion RotationFrom6D(float[] source, int slot)
    {
        var offset = slot * 6;
        var first = new float3(source[offset], source[offset + 1], source[offset + 2]);
        var second = new float3(source[offset + 3], source[offset + 4], source[offset + 5]);

        var column0 = math.normalize(first);
        var column1 = math.normalize(second - math.dot(column0, second) * column0);
        var column2 = math.cross(column0, column1);

        return new quaternion(new float3x3(column0, column1, column2));
    }

    /// <summary>The rotation vector carrying <paramref name="from"/> to <paramref name="to"/>, per second.</summary>
    private static float3 AngularVelocity(quaternion from, quaternion to, float deltaTime)
    {
        var delta = math.mul(to, math.inverse(from));

        // Taken the short way round, so a rotation just past half a turn does not read as almost a
        // full turn the other way.
        if (delta.value.w < 0f) delta.value = -delta.value;

        var sinHalf = math.length(delta.value.xyz);
        if (sinHalf < 1e-6f) return float3.zero;

        var angle = 2f * math.atan2(sinHalf, delta.value.w);
        return delta.value.xyz / sinHalf * (angle / deltaTime);
    }

    private float2 CurrentGroundPosition()
    {
        var position = _characterTransform.position;
        return new float2(position.x, position.z);
    }

    private float CurrentYaw()
    {
        var forward = _characterTransform.forward;
        return PfnnTrajectory.YawOf(new float2(forward.x, forward.z));
    }

    private static string DeviceName(PfnnConfig.ComputeDevice device) => device switch
    {
        PfnnConfig.ComputeDevice.Cuda => "cuda",
        PfnnConfig.ComputeDevice.Cpu => "cpu",
        _ => "auto"
    };

    private static float Repeat(float value, float length)
    {
        var wrapped = value - math.floor(value / length) * length;
        return math.clamp(wrapped, 0f, length);
    }

    private static float3 ReadFloat3(float[] source, int slot) =>
        new(source[slot * 3], source[slot * 3 + 1], source[slot * 3 + 2]);

    private static void WriteFloat3(float[] destination, int slot, float3 value)
    {
        destination[slot * 3] = value.x;
        destination[slot * 3 + 1] = value.y;
        destination[slot * 3 + 2] = value.z;
    }

    public void Dispose()
    {
        if (_isInitialized)
        {
            // The GIL is held while dropping the handle because that is what decrefs the object.
            using (Py.GIL())
            {
                _policy = null;
            }
        }

        if (_positions.IsCreated) _positions.Dispose();
        if (_rotations.IsCreated) _rotations.Dispose();
        if (_velocities.IsCreated) _velocities.Dispose();
        if (_angularVelocities.IsCreated) _angularVelocities.Dispose();
        if (_previousRotations.IsCreated) _previousRotations.Dispose();

        _isInitialized = false;
    }

    public override void OnDestroy() => Dispose();
}
}
