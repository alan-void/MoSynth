using System;
using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// Which bone the character/simulation frame is read from, and which of that bone's local axes
/// points along the character's facing. A rig's bone roll is arbitrary, so the forward axis is a
/// property of the rest pose rather than a constant — see <see cref="Skeleton.RestLocalAxis"/>.
/// </summary>
[Serializable]
public struct SimulationFrameDef
{
    public int ReferenceBoneIndex;
    public float3 ForwardAxisLocal;

    /// <summary>The structural default: the skeleton root, facing along character forward.</summary>
    public static SimulationFrameDef Default(Skeleton skeleton) => new()
    {
        ReferenceBoneIndex = 0,
        ForwardAxisLocal = skeleton.RestLocalAxis(0, math.forward())
    };
}

/// <summary>
/// The character frame of a pose, derived rather than stored.
/// </summary>
/// <remarks>
/// <para>
/// Storage convention: a pose is exactly the clip's own space. Bone 0 is the rig's real root and
/// carries world position and rotation; bones 1.. carry rest offsets and parent-local rotations.
/// Velocity channels are plain finite differences of those same channels, so bone 0's velocity is
/// world-space too.
/// </para>
/// <para>
/// The simulation frame is the ground-projected, facing-aligned frame a character controller
/// works in. It is computed from a reference bone (<see cref="SimulationFrameDef"/>) on demand:
/// position is the reference bone's world position flattened onto the XZ plane, rotation is the
/// yaw that aims at the bone's flattened forward axis. It is therefore yaw-only by construction,
/// and never stored — nothing can drift out of sync with the pose it came from.
/// </para>
/// <para>
/// Idempotence: expressing a pose's bone 0 frame-locally (<see cref="ToFrameLocal(float3,float3,quaternion)"/>)
/// yields a pose whose own derived frame is the identity, which is what lets a pose be re-anchored
/// under an arbitrary world transform and re-derived without loss.
/// </para>
/// </remarks>
public static class SimulationFrame
{
    /// <summary>
    /// Ground-projected position and yaw-only rotation of the character frame this pose implies.
    /// </summary>
    public static void Compute(in PoseBuffer pose, in SkeletonData skeleton, in SimulationFrameDef def,
        out float3 framePos, out quaternion frameRot)
    {
        var referencePosition = skeleton.CharacterSpacePosition(pose, def.ReferenceBoneIndex);
        var referenceRotation = skeleton.CharacterSpaceRotation(pose, def.ReferenceBoneIndex);

        framePos = new float3(referencePosition.x, 0f, referencePosition.z);
        frameRot = quaternion.LookRotationSafe(FlattenForward(referenceRotation, def.ForwardAxisLocal), math.up());
    }

    /// <summary>
    /// Rate of change of the frame <see cref="Compute"/> derives, in the finite-step form the
    /// extraction pipeline stores: the reference bone's channels are advanced by one step of
    /// <paramref name="dt"/> and the frame is re-derived, so a pose holding one-frame finite
    /// differences reproduces the neighbouring frame's simulation frame exactly.
    /// </summary>
    /// <param name="linVelFrameLocal">Ground-plane velocity of the frame origin, in frame space.</param>
    /// <param name="yawRate">Signed rotation rate about +Y, radians per second.</param>
    public static void ComputeVelocity(in PoseBuffer pose, in SkeletonData skeleton, in SimulationFrameDef def,
        float dt, out float3 linVelFrameLocal, out float yawRate)
    {
        var referenceRotation = skeleton.CharacterSpaceRotation(pose, def.ReferenceBoneIndex);
        var forward = FlattenForward(referenceRotation, def.ForwardAxisLocal);
        var frameRot = quaternion.LookRotationSafe(forward, math.up());

        var referenceVelocity = skeleton.CharacterSpaceVelocity(pose, def.ReferenceBoneIndex);
        linVelFrameLocal = math.mul(math.inverse(frameRot),
            new float3(referenceVelocity.x, 0f, referenceVelocity.z));

        var referenceAngularVelocity = skeleton.CharacterSpaceAngularVelocity(pose, def.ReferenceBoneIndex);
        var nextRotation = math.mul(MathExtensions.QuaternionFromScaledAngleAxis(referenceAngularVelocity * dt),
            referenceRotation);
        var nextForward = FlattenForward(nextRotation, def.ForwardAxisLocal);

        yawRate = SignedYawAngle(forward, nextForward) / dt;
    }

    /// <summary>World position expressed relative to a simulation frame.</summary>
    public static float3 ToFrameLocal(float3 worldPos, float3 framePos, quaternion frameRot) =>
        math.mul(math.inverse(frameRot), worldPos - framePos);

    /// <summary>World rotation expressed relative to a simulation frame.</summary>
    public static quaternion ToFrameLocal(quaternion worldRot, quaternion frameRot) =>
        math.mul(math.inverse(frameRot), worldRot);

    /// <summary>Inverse of <see cref="ToFrameLocal(float3,float3,quaternion)"/>.</summary>
    public static float3 FromFrameLocal(float3 localPos, float3 framePos, quaternion frameRot) =>
        math.mul(frameRot, localPos) + framePos;

    /// <summary>Inverse of <see cref="ToFrameLocal(quaternion,quaternion)"/>.</summary>
    public static quaternion FromFrameLocal(quaternion localRot, quaternion frameRot) =>
        math.mul(frameRot, localRot);

    /// <summary>
    /// Splits bone 0's world velocity channels into the part the simulation frame already carries
    /// and the part left over, expressed in frame space. The frame's own motion contributes both
    /// its linear velocity and the tangential velocity its yaw imparts at bone 0's offset from the
    /// frame origin.
    /// </summary>
    public static void DecomposeRootVelocity(float3 framePos, quaternion frameRot, float3 linVelFrameLocal,
        float yawRate, float3 rootPosition, float3 rootVelocity, float3 rootAngularVelocity,
        out float3 linVelLocal, out float3 angVelLocal)
    {
        var frameLinVelWorld = math.mul(frameRot, linVelFrameLocal);
        var yawWorld = yawRate * math.up();
        var tangential = math.cross(yawWorld, rootPosition - framePos);

        var inverseFrameRot = math.inverse(frameRot);
        linVelLocal = math.mul(inverseFrameRot, rootVelocity - frameLinVelWorld - tangential);
        angVelLocal = math.mul(inverseFrameRot, rootAngularVelocity - yawWorld);
    }

    /// <summary>Exact inverse of <see cref="DecomposeRootVelocity"/>.</summary>
    public static void RecomposeRootVelocity(float3 framePos, quaternion frameRot, float3 linVelFrameLocal,
        float yawRate, float3 rootPosition, float3 linVelLocal, float3 angVelLocal,
        out float3 rootVelocity, out float3 rootAngularVelocity)
    {
        var frameLinVelWorld = math.mul(frameRot, linVelFrameLocal);
        var yawWorld = yawRate * math.up();
        var tangential = math.cross(yawWorld, rootPosition - framePos);

        rootVelocity = math.mul(frameRot, linVelLocal) + frameLinVelWorld + tangential;
        rootAngularVelocity = math.mul(frameRot, angVelLocal) + yawWorld;
    }

    /// <summary>
    /// The reference bone's forward axis taken into character space, flattened onto the ground
    /// plane and normalized. A bone aimed straight up or down leaves nothing to aim at, so the
    /// character-forward axis stands in.
    /// </summary>
    private static float3 FlattenForward(quaternion referenceRotation, float3 forwardAxisLocal)
    {
        var forward = math.mul(referenceRotation, forwardAxisLocal);
        forward.y = 0f;
        return math.normalizesafe(forward, math.forward());
    }

    /// <summary>Signed angle in radians from <paramref name="from"/> to <paramref name="to"/> about +Y.</summary>
    private static float SignedYawAngle(float3 from, float3 to) =>
        math.atan2(math.dot(math.cross(from, to), math.up()), math.dot(from, to));
}
}
