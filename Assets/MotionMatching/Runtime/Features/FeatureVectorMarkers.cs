using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace MotionMatching
{
/// <summary>
/// Turns a matching feature vector, a query or a stored frame, into world-space points and arrows:
/// one per channel and horizon, placed in a given character frame.
/// </summary>
/// <remarks>
/// Positions become points. Directions and velocities become arrows hung from the position the same
/// vector gives for that bone (or for the simulation frame at that horizon), else from the live
/// joint. Shared by <c>MotionMatchingDataVisualiser</c>'s gizmos and the stages' recorded
/// inputs, so what the editor shows and what a video shows cannot drift apart.
/// </remarks>
public sealed class FeatureVectorMarkers
{
    /// <summary>Drawn length of a unit direction feature, in metres.</summary>
    public const float DirectionLength = 0.4f;

    /// <summary>A velocity is drawn as the distance it covers in this many seconds.</summary>
    public const float VelocitySeconds = 0.2f;

    private readonly MotionMatchingData _mmData;
    private readonly FeatureSet _featureSet;

    private readonly int[] _trajectoryBones;   // -1 for a simulation-frame feature
    private readonly int[] _trajectoryAnchors; // direction feature -> the position feature it hangs from, or -1
    private readonly int[] _poseBones;
    private readonly int[] _poseAnchors;       // velocity feature -> the position feature of its bone, or -1

    public FeatureVectorMarkers(MotionMatchingData mmData, FeatureSet featureSet, Skeleton skeleton)
    {
        _mmData = mmData;
        _featureSet = featureSet;

        var trajectory = mmData.trajectoryFeatures;
        _trajectoryBones = new int[trajectory.Count];
        _trajectoryAnchors = new int[trajectory.Count];
        for (var t = 0; t < trajectory.Count; t++)
        {
            _trajectoryBones[t] = trajectory[t].simulationBone ? -1 : ResolveBone(trajectory[t].bone, skeleton);
        }

        for (var t = 0; t < trajectory.Count; t++)
        {
            _trajectoryAnchors[t] = -1;
            if (trajectory[t].featureType != TrajectoryFeatureChannel.Type.Direction) continue;

            for (var other = 0; other < trajectory.Count; other++)
            {
                if (trajectory[other].featureType != TrajectoryFeatureChannel.Type.Position ||
                    _trajectoryBones[other] != _trajectoryBones[t] ||
                    featureSet.GetPredictionCount(other) != featureSet.GetPredictionCount(t)) continue;

                _trajectoryAnchors[t] = other;
                break;
            }
        }

        var pose = mmData.poseFeatures;
        _poseBones = new int[pose.Count];
        _poseAnchors = new int[pose.Count];
        for (var p = 0; p < pose.Count; p++) _poseBones[p] = ResolveBone(pose[p].bone, skeleton);

        for (var p = 0; p < pose.Count; p++)
        {
            _poseAnchors[p] = -1;
            if (pose[p].featureType != PoseFeatureChannel.Type.Velocity) continue;

            for (var other = 0; other < pose.Count; other++)
            {
                if (pose[other].featureType != PoseFeatureChannel.Type.Position ||
                    _poseBones[other] != _poseBones[p]) continue;

                _poseAnchors[p] = other;
                break;
            }
        }
    }

    /// <summary>Adds every trajectory channel of <paramref name="featureVector"/> to <paramref name="output"/>.</summary>
    /// <param name="weights">The search's per-float weights; a channel weighted zero throughout is skipped. Empty draws everything.</param>
    /// <param name="joints">The live rig, for arrows with no position of their own to hang from; may be null.</param>
    /// <returns>How many channel horizons were skipped for carrying no weight.</returns>
    public int AddTrajectory(ReadOnlySpan<float> featureVector, ReadOnlySpan<float> weights, float3 origin,
        quaternion rotation, Transform[] joints, SynthesisInputRole role, SynthesisInputs output)
    {
        var skipped = 0;
        var trajectory = _mmData.trajectoryFeatures;
        for (var t = 0; t < trajectory.Count; t++)
        {
            var feature = trajectory[t];
            var floats = _featureSet.GetTrajectoryFeatureFloatCount(t);
            for (var p = 0; p < _featureSet.GetPredictionCount(t); p++)
            {
                if (!IsWeighted(weights, _featureSet.GetTrajectoryFeatureOffset(t, p), floats))
                {
                    skipped++;
                    continue;
                }

                var local = feature.Unpack(featureVector, _featureSet, t, p);
                if (feature.featureType == TrajectoryFeatureChannel.Type.Position)
                {
                    output.AddPoint(role, origin + math.mul(rotation, local));
                    continue;
                }

                var anchor = _trajectoryAnchors[t];
                var tail = anchor >= 0 &&
                           IsWeighted(weights, _featureSet.GetTrajectoryFeatureOffset(anchor, p), floats)
                    ? origin + math.mul(rotation, trajectory[anchor].Unpack(featureVector, _featureSet, anchor, p))
                    : JointOrOrigin(joints, _trajectoryBones[t], origin);
                output.AddArrow(role, tail, math.mul(rotation, local) * DirectionLength);
            }
        }

        return skipped;
    }

    /// <summary>Adds every pose channel of <paramref name="featureVector"/> to <paramref name="output"/>.</summary>
    /// <inheritdoc cref="AddTrajectory"/>
    public int AddPose(ReadOnlySpan<float> featureVector, ReadOnlySpan<float> weights, float3 origin,
        quaternion rotation, Transform[] joints, SynthesisInputRole role, SynthesisInputs output)
    {
        var skipped = 0;
        var pose = _mmData.poseFeatures;
        for (var p = 0; p < pose.Count; p++)
        {
            if (!IsWeighted(weights, _featureSet.GetPoseFeatureOffset(p), FeatureSet.FloatsPerPoseFeature))
            {
                skipped++;
                continue;
            }

            var local = _featureSet.GetPoseFeature(featureVector, p);
            if (pose[p].featureType == PoseFeatureChannel.Type.Position)
            {
                output.AddPoint(role, origin + math.mul(rotation, local));
                continue;
            }

            var anchor = _poseAnchors[p];
            var tail = anchor >= 0 &&
                       IsWeighted(weights, _featureSet.GetPoseFeatureOffset(anchor), FeatureSet.FloatsPerPoseFeature)
                ? origin + math.mul(rotation, _featureSet.GetPoseFeature(featureVector, anchor))
                : JointOrOrigin(joints, _poseBones[p], origin);
            output.AddArrow(role, tail, math.mul(rotation, local) * VelocitySeconds);
        }

        return skipped;
    }

    private static bool IsWeighted(ReadOnlySpan<float> weights, int offset, int count)
    {
        if (weights.IsEmpty) return true;
        for (var f = 0; f < count; f++)
        {
            if (weights[offset + f] != 0f) return true;
        }

        return false;
    }

    private static float3 JointOrOrigin(Transform[] joints, int bone, float3 origin)
    {
        return joints != null && bone >= 0 && bone < joints.Length && joints[bone] != null
            ? (float3)joints[bone].position
            : origin;
    }

    private static int ResolveBone(SkeletonBone bone, Skeleton skeleton)
    {
        var index = bone?.ResolveIndex(skeleton) ?? -1;
        return index >= 0 ? index : 0;
    }
}
}
