using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Serialization;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace MotionMatching
{
/// <summary>
/// A joint of the current pose, expressed in the character frame. Always three floats.
/// </summary>
/// <remarks>
/// This is both the authored definition and the layout channel. <see cref="PoseLayout"/> keys its
/// offset table on the descriptor, so rebuild the layout after editing a feature rather than
/// mutating one a built layout already holds.
/// </remarks>
[Serializable]
public sealed class PoseFeatureChannel : ChannelDescriptor, IMatchingFeature
{
    public enum Type
    {
        Position,
        Velocity
    }

    [FormerlySerializedAs("Name")] public string name;
    [FormerlySerializedAs("FeatureType")] public Type featureType;
    [FormerlySerializedAs("Bone")] public SkeletonBone bone = new();

    public string Name => name;

    public override int FloatCount => 3;

    public override int SectionKey => FeatureSections.Pose;

    public void Extract(PoseSet poseSet, MotionMatchingData mmData, int poseIndex, int boneIndex, ChannelHandle handle,
        StateBuffer frame)
    {
        var skeleton = poseSet.Skeleton.GetSkeletonData();
        var simulationFrame = poseSet.SimulationFrame;
        var characterPose = poseSet.GetPoseBuffer(poseIndex);

        var value = float3.zero;
        switch (featureType)
        {
            case Type.Position:
                value = FeatureSet.GetLocalJointPositionFromCharacter(skeleton, simulationFrame, characterPose,
                    characterPose, boneIndex);
                break;
            case Type.Velocity:
            {
                var nextPose = poseSet.GetPoseBuffer(NextPoseIndex(poseSet, poseIndex));
                var position =
                    FeatureSet.GetLocalJointPositionFromCharacter(skeleton, simulationFrame, characterPose,
                        characterPose, boneIndex);
                var nextPosition =
                    FeatureSet.GetLocalJointPositionFromCharacter(skeleton, simulationFrame, characterPose,
                        nextPose, boneIndex);
                value = (nextPosition - position) / poseSet.FrameTime;
                break;
            }
            default:
                Debug.Assert(false, "Unknown PoseFeatureChannel.Type: " + featureType);
                break;
        }

        frame.SetFloat3(handle, value);
    }

    /// <summary>
    /// The pose one frame later within the same clip. Clips are stored back to back, so the frame
    /// after a clip's last is an unrelated animation; differencing against it would report a
    /// jump-cut as joint velocity. On a clip's last frame the pose stands in for its own successor
    /// and the velocity comes out zero.
    /// </summary>
    private static int NextPoseIndex(PoseSet poseSet, int poseIndex)
    {
        var nextPoseIndex = poseIndex + 1;
        return nextPoseIndex < poseSet.GetClipContaining(poseIndex).End ? nextPoseIndex : poseIndex;
    }

    public override bool Equals(ChannelDescriptor other)
    {
        return other is PoseFeatureChannel channel
               && channel.featureType == featureType
               && channel.bone?.Name == bone?.Name;
    }

    public override int GetHashCode()
    {
        unchecked
        {
            var hash = typeof(PoseFeatureChannel).GetHashCode();
            hash = hash * 31 + (int)featureType;
            hash = hash * 31 + (bone?.Name != null ? bone.Name.GetHashCode() : 0);
            return hash;
        }
    }

    public override int GetContentHash() => GetHashCode();
}
}
