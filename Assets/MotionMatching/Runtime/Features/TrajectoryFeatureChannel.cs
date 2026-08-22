using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Serialization;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace MotionMatching
{
/// <summary>
/// A bone (or the simulation bone) sampled a number of frames into the future, stored once per
/// entry of <see cref="predictionFrames"/>. Axes can be masked out, so one prediction is one to
/// three floats wide.
/// </summary>
/// <remarks>
/// This is both the authored definition and the layout channel. <see cref="PoseLayout"/> keys its
/// offset table on the descriptor, so rebuild the layout after editing a feature rather than
/// mutating one a built layout already holds.
/// </remarks>
[Serializable]
public sealed class TrajectoryFeatureChannel : ChannelDescriptor, IMatchingFeature
{
    public enum Type
    {
        Position,
        Direction
    }

    [FormerlySerializedAs("Name")] public string name;
    [FormerlySerializedAs("FeatureType")] public Type featureType;

    [FormerlySerializedAs("predictionFrame")] [FormerlySerializedAs("framesPrediction")] [FormerlySerializedAs("FramesPrediction")]
    public int[] predictionFrames = Array.Empty<int>(); // Number of frames in the future for each point of the trajectory

    [FormerlySerializedAs("SimulationBone")]
    public bool
        simulationBone; // Use the simulation bone (articial root added during pose extraction) instead of a bone

    [FormerlySerializedAs("Bone")]
    public SkeletonBone bone = new(); // Bone used to compute the trajectory in the feature set

    [FormerlySerializedAs("ZeroX")] public bool zeroX; // Zero the X, Y and/or Z component of the trajectory feature
    [FormerlySerializedAs("ZeroY")] public bool zeroY; // Zero the X, Y and/or Z component of the trajectory feature
    [FormerlySerializedAs("ZeroZ")] public bool zeroZ; // Zero the X, Y and/or Z component of the trajectory feature

    [FormerlySerializedAs("IsMainPositionFeature")]
    public bool
        isMainPositionFeature; // Only for position feature type. Used for visualizing gizmos of other trajectory features colocated with this position feature.

    public string Name => name;

    /// <summary>Floats one prediction occupies, once the zeroed axes are dropped.</summary>
    public int FloatsPerPrediction => 3 - (zeroX ? 1 : 0) - (zeroY ? 1 : 0) - (zeroZ ? 1 : 0);

    public int PredictionCount => predictionFrames.Length;

    public override int FloatCount => FloatsPerPrediction * PredictionCount;

    public override int SectionKey => FeatureSections.Trajectory;

    public void Extract(PoseSet poseSet, MotionMatchingData mmData, int poseIndex, int boneIndex, ChannelHandle handle,
        StateBuffer frame)
    {
        var skeleton = poseSet.Skeleton.GetSkeletonData();
        var characterPose = poseSet.GetPoseBuffer(poseIndex);

        for (var p = 0; p < predictionFrames.Length; ++p)
        {
            var futurePose = poseSet.GetPoseBuffer(poseIndex + predictionFrames[p]);
            var value = float3.zero;
            switch (featureType)
            {
                case Type.Position:
                {
                    value = FeatureSet.GetLocalJointPositionFromCharacter(skeleton, characterPose, futurePose,
                        boneIndex);
                }
                    break;
                case Type.Direction:
                {
                    value = GetDirection(skeleton, characterPose, futurePose, boneIndex, mmData);
                    if (zeroX) value.x = 0;
                    if (zeroY) value.y = 0;
                    if (zeroZ) value.z = 0;
                    value = math.normalize(value);
                }
                    break;
                default:
                    Debug.Assert(false, "Unsupported Feature Type: " + featureType);
                    break;
            }

            Pack(value, frame, handle, p);
        }
    }

    /// <summary>
    /// Rebuilds the full vector of one prediction, with the masked axes back at zero.
    /// </summary>
    public float3 Unpack(FeatureSet featureSet, int frameIndex, int trajectoryFeatureIndex, int predictionIndex)
    {
        var value = float3.zero;
        var axis = 0;
        for (var f = 0; f < FloatsPerPrediction; ++f)
        {
            axis = SkipZeroedAxes(axis);
            value[axis] = featureSet.GetTrajectoryFloat(frameIndex, trajectoryFeatureIndex, predictionIndex, f, true);
            axis += 1;
        }

        return value;
    }

    private float3 GetDirection(in SkeletonData skeleton, PoseBuffer characterPose, PoseBuffer pose, int boneIndex,
        MotionMatchingData mmData)
    {
        quaternion worldRotation;
        float3 localForward;
        if (simulationBone)
        {
            worldRotation = pose.Rotations[0];
            localForward = math.forward();
        }
        else
        {
            worldRotation = PoseBufferFK.WorldRotation(skeleton, pose, boneIndex);
            // Forward vector of the joint in its own local space, taken from the rig's rest pose.
            localForward = mmData.GetLocalForward(boneIndex);
        }

        return FeatureSet.GetLocalDirectionFromCharacter(characterPose, math.mul(worldRotation, localForward));
    }

    private void Pack(float3 value, StateBuffer frame, ChannelHandle handle, int predictionIndex)
    {
        var start = predictionIndex * FloatsPerPrediction;
        var axis = 0;
        for (var f = 0; f < FloatsPerPrediction; ++f)
        {
            axis = SkipZeroedAxes(axis);
            frame.SetFloat(handle, start + f, value[axis]);
            axis += 1;
        }
    }

    /// <summary>
    /// A masked axis is not stored, so packing and unpacking step over it. A trailing zeroed Z
    /// needs no case of its own; the shorter float count ends the loop first.
    /// </summary>
    private int SkipZeroedAxes(int axis)
    {
        if (axis == 0 && zeroX) axis += 1;
        if (axis == 1 && zeroY) axis += 1;
        return axis;
    }

    public override bool Equals(ChannelDescriptor other)
    {
        return other is TrajectoryFeatureChannel channel
               && channel.featureType == featureType
               && channel.simulationBone == simulationBone
               && channel.bone?.Name == bone?.Name
               && channel.zeroX == zeroX
               && channel.zeroY == zeroY
               && channel.zeroZ == zeroZ
               && SamePredictionFrames(channel.predictionFrames, predictionFrames);
    }

    public override int GetHashCode()
    {
        unchecked
        {
            var hash = typeof(TrajectoryFeatureChannel).GetHashCode();
            hash = hash * 31 + (int)featureType;
            hash = hash * 31 + (simulationBone ? 1 : 0);
            hash = hash * 31 + (bone?.Name != null ? bone.Name.GetHashCode() : 0);
            hash = hash * 31 + (zeroX ? 1 : 0);
            hash = hash * 31 + (zeroY ? 1 : 0);
            hash = hash * 31 + (zeroZ ? 1 : 0);
            for (var i = 0; i < predictionFrames.Length; i++)
            {
                hash = hash * 31 + predictionFrames[i];
            }

            return hash;
        }
    }

    public override int GetContentHash() => GetHashCode();

    private static bool SamePredictionFrames(int[] a, int[] b)
    {
        if (ReferenceEquals(a, b)) return true;
        if (a == null || b == null || a.Length != b.Length) return false;
        for (var i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }

        return true;
    }
}
}
