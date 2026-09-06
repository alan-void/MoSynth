using System;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Pulls the synthesized character onto something else that decides where it should be — a capsule
/// a controller drives, a point travelling along a path. With the half-lives at zero the character
/// is exactly on the target every tick, so it cannot lag behind it.
/// </summary>
/// <remarks>
/// The correction is written as frame velocity on bone 0, not as a Transform move, because the
/// component integrates that velocity to advance the character. Going through the same channel the
/// animation does is what lets the two be blended and rate-limited against each other; writing the
/// Transform directly would put a correction where nothing downstream could see it.
/// <para>
/// Place it after whatever produces the pose, and after <c>Inertialization</c> — the blend smooths
/// the pose stream, and running it afterwards would smooth the correction away with it.
/// </para>
/// <para>
/// The character frame is ground-projected, so this follows a target in the plane only. See
/// <c>openwiki/animation-tools/root-following.md</c>.
/// </para>
/// </remarks>
[Serializable]
public class RootFollowStage : MoSynthStage
{
    [Tooltip("What the character should be standing on. A component implementing IFrameTarget is " +
             "preferred over the Transform itself, for a target that moves without moving its object.")]
    public Transform target;

    [Tooltip("Time to close half the gap to the target's position. 0 lands on it every tick.")]
    [Min(0f)] public float positionHalfLife;

    [Tooltip("Time to close half the gap to the target's facing. 0 matches it every tick.")]
    [Min(0f)] public float rotationHalfLife;

    [Tooltip("Cap on the correction's own speed, in m/s, so it hides inside the animation's travel " +
             "instead of sliding. 0 is uncapped.")]
    [Min(0f)] public float maxCorrectionSpeed;

    [Tooltip("Cap on the facing correction, in degrees per second. 0 is uncapped.")]
    [Min(0f)] public float maxCorrectionYawRate;

    [Tooltip("Move the target onto the character at startup. Without it a target authored somewhere " +
             "else teleports the character on the first tick.")]
    public bool alignTargetOnInit = true;

    private MotionSynthesisComponent _owner;
    private IFrameTarget _frameTarget;
    private bool _isUsable;

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        _owner = motionSynthesisComponent;
        _isUsable = false;

        if (target == null)
        {
            Debug.LogWarning($"{nameof(RootFollowStage)} on \"{_owner.name}\": no target assigned; " +
                             "the character will not follow anything.", _owner);
            return;
        }

        // Following a Transform the character carries would make the character chase itself.
        if (target.IsChildOf(_owner.transform))
        {
            Debug.LogError($"{nameof(RootFollowStage)} on \"{_owner.name}\": the target \"{target.name}\" " +
                           "is under the character, so following it is circular. Disabling the stage.",
                _owner);
            return;
        }

        if (!_owner.rootPositionsMask)
        {
            Debug.LogWarning($"{nameof(RootFollowStage)} on \"{_owner.name}\": rootPositionsMask is off, " +
                             "so the character is never advanced and a correction cannot reach it.", _owner);
            return;
        }

        _frameTarget = target.GetComponent<IFrameTarget>();
        _isUsable = true;

        if (alignTargetOnInit && _frameTarget == null)
        {
            var characterTransform = _owner.transform;
            target.SetPositionAndRotation(
                new Vector3(characterTransform.position.x, target.position.y, characterTransform.position.z),
                Quaternion.Euler(0f, characterTransform.eulerAngles.y, 0f));
        }
    }

    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        if (!_isUsable || deltaTime <= 0f) return true;
        if (!TryReadTarget(out var targetPositionXZ, out var targetYaw)) return true;

        var skeletonData = _owner.SkeletonData;
        var frameDef = _owner.SimulationFrame;
        SimulationFrame.Compute(pose, skeletonData, frameDef, out var framePosition, out var frameRotation);
        SimulationFrame.ComputeVelocity(pose, skeletonData, frameDef, deltaTime,
            out var frameVelocity, out var frameYawRate);

        var characterTransform = _owner.transform;
        RootFollow.Solve(
            new float2(characterTransform.position.x, characterTransform.position.z),
            SimulationFrame.Yaw(characterTransform.rotation),
            targetPositionXZ, targetYaw,
            frameVelocity, frameYawRate,
            positionHalfLife, rotationHalfLife,
            maxCorrectionSpeed, math.radians(maxCorrectionYawRate), deltaTime,
            out var correctedVelocity, out var correctedYawRate);

        // The frame's motion is not stored, only implied by bone 0's rates, so a change to it is
        // made by taking those rates apart against the old frame and putting them back against the
        // new one. Same route Inertialization writes the blended root through.
        var positions = pose.Positions;
        var velocities = pose.Velocities;
        var angularVelocities = pose.AngularVelocities;

        SimulationFrame.DecomposeRootVelocity(framePosition, frameRotation, frameVelocity, frameYawRate,
            positions[0], velocities[0], angularVelocities[0],
            out var rootVelocityLocal, out var rootAngularVelocityLocal);
        SimulationFrame.RecomposeRootVelocity(framePosition, frameRotation, correctedVelocity,
            correctedYawRate, positions[0], rootVelocityLocal, rootAngularVelocityLocal,
            out var worldRootVelocity, out var worldRootAngularVelocity);

        velocities[0] = worldRootVelocity;
        angularVelocities[0] = worldRootAngularVelocity;
        return true;
    }

    /// <summary>Where the target is, preferring what it says about itself over where its object sits.</summary>
    private bool TryReadTarget(out float2 positionXZ, out float yaw)
    {
        if (_frameTarget != null) return _frameTarget.TryGetFrame(out positionXZ, out yaw);

        positionXZ = new float2(target.position.x, target.position.z);
        yaw = SimulationFrame.Yaw(target.rotation);
        return true;
    }
}
}
