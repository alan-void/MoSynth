using System;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// A <see cref="SplineControlInput"/> whose spline carries pose keypoints: when a prediction horizon
/// lands on a <see cref="SplinePoseKeypoint"/> keypoint, the bone trajectory channels (e.g. the foot
/// keypoint channels) switch on with that keypoint's bone position as the query target. Away from
/// keypoints the channels stay off, so the stage weight-masks them and the search behaves like plain
/// spline following.
/// </summary>
/// <remarks>
/// <para>
/// The facing this reports comes from the keypoints, not from the path tangent. The database bakes
/// its Direction channels as the character's facing, and nothing downstream steers the character
/// toward the controller — <see cref="AnimationTools.MotionSynthesisComponent"/> integrates the
/// matched clip's own yaw rate — so the direction query is the only influence this input has over
/// which way the character ends up facing. Feeding it the direction of travel would fight the foot
/// constraints on any pose whose facing differs from its heading.
/// </para>
/// <para>
/// Assumes the keypoint animation uses the same rig as the motion-matching data, so a channel's bone
/// resolves against the keypoint skeleton.
/// </para>
/// </remarks>
public class SplinePoseKeypointControlInput : SplineControlInput
{
    [Tooltip("Source of the spline and the keypoint poses this input constrains toward.")]
    public SplinePoseKeypoint poseKeypoints;

    [Min(0)]
    [Tooltip("How close, in database frames of travel, a prediction horizon must land to a keypoint " +
             "for its constraint to switch on.")]
    public int activationWindowFrames = 2;

    private bool HasKeypoints => poseKeypoints != null && poseKeypoints.Keypoints.Count > 0;

    protected override void Start()
    {
        AdoptKeypointSpline();
        base.Start();

        if (HasKeypoints && SplineContainer != poseKeypoints.SplineContainer)
        {
            // Facing is looked up by spline parameter on the keypoints' own curve, so following a
            // different one would read a parameter from one path against another's shape.
            Debug.LogWarning("[SplinePoseKeypointControlInput] Following a different spline than the " +
                             "keypoints were built against; the facing and the bone targets will not " +
                             "line up with the path.", this);
        }
    }

    /// <summary>
    /// Follows the spline the keypoints were built against, unless one was assigned explicitly.
    /// Public so a component wired up at runtime can adopt it without waiting for the next
    /// <see cref="Start"/>.
    /// </summary>
    public void AdoptKeypointSpline()
    {
        if (poseKeypoints != null && SplineContainer == null)
        {
            SplineContainer = poseKeypoints.SplineContainer;
        }
    }

    /// <summary>
    /// The path's stored facing rather than the direction it travels in. Overriding here rather than
    /// at <see cref="GetTrajectoryFeature"/> means the trajectory query, the debug gizmos and
    /// <see cref="SplineControlInput.GetCurrentRotation"/> all report the same thing.
    /// </summary>
    protected override float2 SampleDirection(float t, float3 positionAtT)
    {
        if (!HasKeypoints) return base.SampleDirection(t, positionAtT);

        var facing = poseKeypoints.GetWorldFacing(t);
        var flat = new float2(facing.x, facing.z);
        // Falling back costs an extra spline evaluation, so only pay it if the facing is unusable.
        return math.lengthsq(flat) > 1e-8f ? math.normalize(flat) : base.SampleDirection(t, positionAtT);
    }

    public override bool GetBoneTrajectoryFeature(TrajectoryFeatureChannel feature, int index, Transform character,
        Span<float> output)
    {
        if (poseKeypoints == null || feature.featureType != TrajectoryFeatureChannel.Type.Position) return false;

        var tFuture = GetPredictedSplineT(feature.predictionFrames[index]);
        var tolerance = activationWindowFrames * speed / splineContainer.CalculateLength() * DatabaseDeltaTime;
        if (!poseKeypoints.TryGetKeypointNearT(tFuture, tolerance, out var keypoint)) return false;

        var boneIndex = feature.bone?.ResolveIndex(poseKeypoints.Skeleton) ?? -1;
        if (boneIndex < 0) return false;
        if (!poseKeypoints.TryGetWorldBonePosition(keypoint, boneIndex, out var world)) return false;

        // Bake-exact character space: yaw-only rotation, ground-projected origin, Y kept as absolute
        // height — InverseTransformPoint would subtract the transform's Y and apply its scale.
        var forward = math.normalizesafe(new float3(character.forward.x, 0f, character.forward.z), math.forward());
        var local = math.mul(math.inverse(quaternion.LookRotation(forward, math.up())),
            world - new float3(character.position.x, 0f, character.position.z));

        var f = 0;
        if (!feature.zeroX) output[f++] = local.x;
        if (!feature.zeroY) output[f++] = local.y;
        if (!feature.zeroZ) output[f] = local.z;
        return true;
    }

    /// <summary>The start of the path, so a character spawns on it rather than wherever this
    /// GameObject sits.</summary>
    public override float3 GetWorldInitPosition() =>
        HasKeypoints ? poseKeypoints.SplineContainer.EvaluatePosition(0f) : base.GetWorldInitPosition();

    /// <summary>The facing the path carries at its start, not the direction it sets off in.</summary>
    public override float3 GetWorldInitDirection() =>
        HasKeypoints ? poseKeypoints.GetWorldFacing(0f) : base.GetWorldInitDirection();
}
}
