using System;
using AnimationTools;
using Unity.Collections;
using Unity.Mathematics;

namespace MotionMatching
{
/// <summary>
/// Smooths pose jumps by inertialization: on a jump the target pose is taken instantly, and the
/// offset from the previous output to it is decayed with a spring, so only one pose stream is needed.
/// </summary>
/// <remarks>
/// Place after whatever produces pose jumps. It keys off
/// <see cref="MotionSynthesisComponent.PoseDiscontinuity"/>, so a stage that jumps without raising
/// that flag will not be smoothed.
/// <para>
/// Bone 0 is blended in the space of its pose's own <see cref="SimulationFrame"/>, not in world
/// space, so a jump to another part of a clip does not smear the character across the world.
/// See openwiki/motion-matching/inertialization.md.
/// </para>
/// </remarks>
[Serializable]
public class Inertialization : MoSynthStage
{
    // State for the explicit-transition API (PoseTransition / *ContactTransition / Update), in which a
    // caller drives transitions itself; Apply does not use those entry points.
    #region Explicit-transition state

    [NonSerialized] public quaternion[] InertializedRotations;
    private float3[] _inertializedAngularVelocities;
    [NonSerialized] public float3 InertializedRootPosition;
    [NonSerialized] public float3 InertializedRootVelocity;

    // Index 0 holds bone 0's frame-local offset; every other entry is parent-local.
    private quaternion[] _offsetRotations;
    private float3[] _offsetAngularVelocities;
    private float3 _offsetRootPosition;

    private float3 _offsetRootVelocity;

    // Contacts
    private float3 _offsetLeftContact;
    private float3 _offsetLeftContactVelocity;
    private float3 _offsetRightContact;
    private float3 _offsetRightContactVelocity;

    #endregion

    private MotionSynthesisComponent _owner;
    private SkeletonData _skeletonData;
    private SimulationFrameDef _simulationFrame;

    public Inertialization()
    {
    }

    public Inertialization(int jointCount)
    {
        InertializedRotations = new quaternion[jointCount];
        _inertializedAngularVelocities = new float3[jointCount];
        _offsetRotations = new quaternion[jointCount];
        for (var i = 0; i < jointCount; i++) _offsetRotations[i] = quaternion.identity;
        _offsetAngularVelocities = new float3[jointCount];
    }

    /// <summary>
    /// Stores the output of this stage from the previous frame.
    /// Inertialization is applied from this pose to the input pose passed <see cref="Inertialization.Apply"/>.
    /// </summary>
    /// <remarks>
    /// If inertialization needs to be applied from the currently
    /// displayed pose, the inverted retargeting to needs to be
    /// fed to this stage.
    /// </remarks>
    private PoseBuffer _currentPose;

    private bool _isPoseInitialized = false;
    public float halfLife = 0.1f;

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        _owner = motionSynthesisComponent;
        _skeletonData = motionSynthesisComponent.SkeletonData;
        _simulationFrame = motionSynthesisComponent.SimulationFrame;
        _currentPose = PoseBuffer.Allocate(motionSynthesisComponent.PoseLayout, Allocator.Persistent);

        var jointCount = _currentPose.Rotations.Length;
        _offsetRotations = new quaternion[jointCount];
        for (var i = 0; i < jointCount; i++) _offsetRotations[i] = quaternion.identity;
        _offsetAngularVelocities = new float3[jointCount];
        _offsetRootPosition = float3.zero;
        _offsetRootVelocity = float3.zero;
    }

    public override void OnDestroy()
    {
        if (_currentPose.IsCreated) _currentPose.Dispose();
    }

    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        if (!_isPoseInitialized)
        {
            _currentPose.CopyFrom(pose);
            _isPoseInitialized = true;
            return true;
        }

        var rotations = pose.Rotations;
        var angularVelocities = pose.AngularVelocities;
        var positions = pose.Positions;
        var velocities = pose.Velocities;

        ReadRootFrameLocalState(pose, deltaTime, out var targetFrame, out var targetRoot);

        // Re-anchor the offsets only when the target pose stream jumped; between jumps the
        // offsets just decay so continuous animation passes through unfiltered (no lag).
        if (_owner.PoseDiscontinuity)
        {
            var lastOutputRotations = _currentPose.Rotations;
            var lastOutputAngularVelocities = _currentPose.AngularVelocities;
            for (var i = 1; i < rotations.Length; i++)
            {
                // _currentPose is last frame's output, so a jump during an ongoing blend
                // folds the remaining offset into the new one.
                _offsetRotations[i] = math.normalizesafe(MathExtensions.Abs(
                    math.mul(math.inverse(rotations[i]), lastOutputRotations[i])));
                _offsetAngularVelocities[i] = lastOutputAngularVelocities[i] - angularVelocities[i];
            }

            // Each side of the jump is read against the frame it derives for itself, so the two
            // are compared as two characters each standing at their own origin.
            ReadRootFrameLocalState(_currentPose, deltaTime, out _, out var lastOutputRoot);

            _offsetRotations[0] = math.normalizesafe(MathExtensions.Abs(
                math.mul(math.inverse(targetRoot.Rotation), lastOutputRoot.Rotation)));
            _offsetAngularVelocities[0] = lastOutputRoot.AngularVelocity - targetRoot.AngularVelocity;
            _offsetRootPosition = lastOutputRoot.Position - targetRoot.Position;
            _offsetRootVelocity = lastOutputRoot.Velocity - targetRoot.Velocity;
        }

        for (var i = 1; i < rotations.Length; i++)
        {
            InertializeJointUpdate(rotations[i], angularVelocities[i],
                halfLife, deltaTime,
                ref _offsetRotations[i], ref _offsetAngularVelocities[i],
                out var newRot, out var newAngularVel);
            rotations[i] = newRot;
            angularVelocities[i] = newAngularVel;
        }

        InertializeJointUpdate(targetRoot.Rotation, targetRoot.AngularVelocity,
            halfLife, deltaTime,
            ref _offsetRotations[0], ref _offsetAngularVelocities[0],
            out var newRootRotation, out var newRootAngularVelocity);
        InertializeJointUpdate(targetRoot.Position, targetRoot.Velocity,
            halfLife, deltaTime,
            ref _offsetRootPosition, ref _offsetRootVelocity,
            out var newRootPosition, out var newRootVelocity);

        // Back to world space through the target's frame, which is where the rest of the pose
        // already is. The velocity channels have to follow, because the component re-derives the
        // frame's own motion from them when it advances the character.
        positions[0] = SimulationFrame.FromFrameLocal(newRootPosition, targetFrame.Position,
            targetFrame.Rotation);
        rotations[0] = SimulationFrame.FromFrameLocal(newRootRotation, targetFrame.Rotation);
        SimulationFrame.RecomposeRootVelocity(targetFrame.Position, targetFrame.Rotation,
            targetFrame.LinearVelocity, targetFrame.YawRate, positions[0],
            newRootVelocity, newRootAngularVelocity,
            out var worldRootVelocity, out var worldRootAngularVelocity);
        velocities[0] = worldRootVelocity;
        angularVelocities[0] = worldRootAngularVelocity;

        _currentPose.CopyFrom(pose);
        return true;
    }

    /// <summary>The simulation frame a pose derives, and how fast that frame is moving.</summary>
    private struct FrameState
    {
        public float3 Position;
        public quaternion Rotation;
        public float3 LinearVelocity;
        public float YawRate;
    }

    /// <summary>Bone 0 relative to that frame, with the rates left over once the frame's own
    /// motion is taken out.</summary>
    private struct RootState
    {
        public float3 Position;
        public quaternion Rotation;
        public float3 Velocity;
        public float3 AngularVelocity;
    }

    private void ReadRootFrameLocalState(PoseBuffer pose, float deltaTime, out FrameState frame,
        out RootState root)
    {
        SimulationFrame.Compute(pose, _skeletonData, _simulationFrame,
            out var framePosition, out var frameRotation);
        SimulationFrame.ComputeVelocity(pose, _skeletonData, _simulationFrame, deltaTime,
            out var frameLinearVelocity, out var frameYawRate);

        var worldPosition = pose.Positions[0];
        SimulationFrame.DecomposeRootVelocity(framePosition, frameRotation, frameLinearVelocity,
            frameYawRate, worldPosition, pose.Velocities[0], pose.AngularVelocities[0],
            out var localVelocity, out var localAngularVelocity);

        frame = new FrameState
        {
            Position = framePosition,
            Rotation = frameRotation,
            LinearVelocity = frameLinearVelocity,
            YawRate = frameYawRate
        };
        root = new RootState
        {
            Position = SimulationFrame.ToFrameLocal(worldPosition, framePosition, frameRotation),
            Rotation = SimulationFrame.ToFrameLocal(pose.Rotations[0], frameRotation),
            Velocity = localVelocity,
            AngularVelocity = localAngularVelocity
        };
    }

    #region Explicit-transition API and shared maths

    /// <summary>
    /// Sets up a transition from one database pose to another; advance it with <see cref="Update"/>.
    /// </summary>
    public void PoseTransition(PoseSet poseSet, int sourcePoseIndex, int targetPoseIndex)
    {
        var sourcePose = poseSet.GetPoseBuffer(sourcePoseIndex);
        var targetPose = poseSet.GetPoseBuffer(targetPoseIndex);

        var sourceRotations = sourcePose.Rotations;
        var targetRotations = targetPose.Rotations;
        var sourceAngularVelocities = sourcePose.AngularVelocities;
        var targetAngularVelocities = targetPose.AngularVelocities;

        for (var i = 1; i < sourceRotations.Length; i++)
        {
            quaternion sourceJointRotation = sourceRotations[i];
            quaternion targetJointRotation = targetRotations[i];
            float3 sourceJointAngularVelocity = sourceAngularVelocities[i];
            float3 targetJointAngularVelocity = targetAngularVelocities[i];
            InertializeJointTransition(sourceJointRotation, sourceJointAngularVelocity,
                targetJointRotation, targetJointAngularVelocity,
                ref _offsetRotations[i], ref _offsetAngularVelocities[i]);
        }

        // The root blends in world space here, unlike Apply's frame-local blend.
        var sourcePositions = sourcePose.Positions;
        var targetPositions = targetPose.Positions;
        var sourceVelocities = sourcePose.Velocities;
        var targetVelocities = targetPose.Velocities;
        float3 sourceRootPosition = sourcePositions[0];
        float3 targetRootPosition = targetPositions[0];
        float3 sourceRootVelocity = sourceVelocities[0];
        float3 targetRootVelocity = targetVelocities[0];
        InertializeJointTransition(sourceRootPosition, sourceRootVelocity,
            targetRootPosition, targetRootVelocity,
            ref _offsetRootPosition, ref _offsetRootVelocity);
    }

    /// <summary>Sets up a left-contact transition; advance it with <see cref="UpdateLeftContact"/>.</summary>
    public void LeftContactTransition(float3 sourceLeftContact, float3 sourceLeftContactVelocity,
        float3 targetLeftContact, float3 targetLeftContactVelocity)
    {
        InertializeJointTransition(sourceLeftContact, sourceLeftContactVelocity,
            targetLeftContact, targetLeftContactVelocity,
            ref _offsetLeftContact, ref _offsetLeftContactVelocity);
    }

    /// <summary>Sets up a right-contact transition; advance it with <see cref="UpdateRightContact"/>.</summary>
    public void RightContactTransition(float3 sourceRightContact, float3 sourceRightContactVelocity,
        float3 targetRightContact, float3 targetRightContactVelocity)
    {
        InertializeJointTransition(sourceRightContact, sourceRightContactVelocity,
            targetRightContact, targetRightContactVelocity,
            ref _offsetRightContact, ref _offsetRightContactVelocity);
    }

    public void UpdateLeftContact(float3 targetPos, float3 targetVelocity, float halfLife, float deltaTime,
        out float3 newPos, out float3 newVel)
    {
        InertializeJointUpdate(targetPos, targetVelocity,
            halfLife, deltaTime,
            ref _offsetLeftContact, ref _offsetLeftContactVelocity,
            out newPos, out newVel);
    }

    public void UpdateRightContact(float3 targetPos, float3 targetVelocity, float halfLife, float deltaTime,
        out float3 newPos, out float3 newVel)
    {
        InertializeJointUpdate(targetPos, targetVelocity,
            halfLife, deltaTime,
            ref _offsetRightContact, ref _offsetRightContactVelocity,
            out newPos, out newVel);
    }

    /// <summary>
    /// Decays the offset set up by <see cref="PoseTransition"/> and applies it to <paramref name="targetPose"/>.
    /// </summary>
    public void Update(PoseBuffer targetPose, float halfLife, float deltaTime)
    {
        var targetRotations = targetPose.Rotations;
        var targetAngularVelocities = targetPose.AngularVelocities;

        for (var i = 1; i < targetRotations.Length; i++)
        {
            quaternion targetJointRotation = targetRotations[i];
            float3 targetAngularVelocity = targetAngularVelocities[i];
            InertializeJointUpdate(targetJointRotation, targetAngularVelocity,
                halfLife, deltaTime,
                ref _offsetRotations[i], ref _offsetAngularVelocities[i],
                out InertializedRotations[i], out _inertializedAngularVelocities[i]);
        }

        var targetPositions = targetPose.Positions;
        var targetVelocities = targetPose.Velocities;
        float3 targetRootPosition = targetPositions[0];
        float3 targetRootVelocity = targetVelocities[0];
        InertializeJointUpdate(targetRootPosition, targetRootVelocity,
            halfLife, deltaTime,
            ref _offsetRootPosition, ref _offsetRootVelocity,
            out InertializedRootPosition, out InertializedRootVelocity);
    }

    /// <summary>
    /// Computes the offset from source to target. In/out, so a transition can start in the middle of another.
    /// </summary>
    public static void InertializeJointTransition(quaternion sourceRot, float3 sourceAngularVel,
        quaternion targetRot, float3 targetAngularVel,
        ref quaternion offsetRot, ref float3 offsetAngularVel)
    {
        offsetRot = math.normalizesafe(MathExtensions.Abs(math.mul(math.inverse(targetRot),
            math.mul(sourceRot, offsetRot))));
        offsetAngularVel = (sourceAngularVel + offsetAngularVel) - targetAngularVel;
    }

    /// <summary>
    /// Computes the offset from source to target. In/out, so a transition can start in the middle of another.
    /// </summary>
    public static void InertializeJointTransition(float3 currentPos, float3 currentVel,
        float3 targetPos, float3 targetVel,
        ref float3 offset, ref float3 offsetVel)
    {
        offset = (currentPos + offset) - targetPos;
        offsetVel = (currentVel + offsetVel) - targetVel;
    }

    /// <summary>
    /// Computes the offset from source to target. In/out, so a transition can start in the middle of another.
    /// </summary>
    public static void InertializeJointTransition(float source, float sourceVel,
        float target, float targetVel,
        ref float offset, ref float offsetVel)
    {
        offset = (source + offset) - target;
        offsetVel = (sourceVel + offsetVel) - targetVel;
    }

    /// <summary>Decays the offset and applies it on top of the target.</summary>
    public static void InertializeJointUpdate(quaternion targetRot, float3 targetAngularVel,
        float halfLife, float deltaTime,
        ref quaternion offsetRot, ref float3 offsetAngularVel,
        out quaternion newRot, out float3 newAngularVel)
    {
        Spring.DecaySpringDamperImplicit(ref offsetRot, ref offsetAngularVel, halfLife, deltaTime);
        newRot = math.mul(targetRot, offsetRot);
        newAngularVel = targetAngularVel + offsetAngularVel;
    }

    /// <summary>Decays the offset and applies it on top of the target.</summary>
    public static void InertializeJointUpdate(float3 target, float3 targetVel,
        float halfLife, float deltaTime,
        ref float3 offset, ref float3 offsetVel,
        out float3 newValue, out float3 newVel)
    {
        Spring.DecaySpringDamperImplicit(ref offset, ref offsetVel, halfLife, deltaTime);
        newValue = target + offset;
        newVel = targetVel + offsetVel;
    }

    /// <summary>Decays the offset and applies it on top of the target.</summary>
    public static void InertializeJointUpdate(float target, float targetVel,
        float halfLife, float deltaTime,
        ref float offset, ref float offsetVel,
        out float newValue, out float newVel)
    {
        Spring.DecaySpringDamperImplicit(ref offset, ref offsetVel, halfLife, deltaTime);
        newValue = target + offset;
        newVel = targetVel + offsetVel;
    }

    #endregion
}
}