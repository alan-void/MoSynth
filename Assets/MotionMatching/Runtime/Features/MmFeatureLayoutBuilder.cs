using AnimationTools;
using UnityEngine;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace MotionMatching
{
/// <summary>
/// Section keys for a motion-matching feature vector. They sit after every built-in pose section,
/// so trajectory floats come first, then pose floats, and the boundary between them is the
/// pose-section start.
/// </summary>
public static class FeatureSections
{
    public const int Trajectory = ChannelSections.Bool + 1;
    public const int Pose = ChannelSections.Bool + 2;
}

/// <summary>
/// Turns a <see cref="MotionMatchingData"/> feature configuration into the layout its feature
/// vectors are stored in.
/// </summary>
public static class MmFeatureLayoutBuilder
{
    public static MmFeatureLayout Build(MotionMatchingData mmData, Skeleton skeleton)
    {
        var trajectoryChannels = mmData.trajectoryFeatures.ToArray();
        var poseChannels = mmData.poseFeatures.ToArray();

        var boneIndices = new int[trajectoryChannels.Length + poseChannels.Length];

        for (var i = 0; i < trajectoryChannels.Length; i++)
        {
            var feature = trajectoryChannels[i];
            // -1 rather than a bone: a simulation-frame feature is derived from the whole pose,
            // so anything that indexes a bone with it is wrong and should fail loudly.
            boneIndices[i] = feature.simulationBone ? -1 : FindJointIndexOrZero(skeleton, feature.bone, feature.name);
        }

        for (var i = 0; i < poseChannels.Length; i++)
        {
            var feature = poseChannels[i];
            boneIndices[trajectoryChannels.Length + i] = FindJointIndexOrZero(skeleton, feature.bone, feature.name);
        }

        return new MmFeatureLayout(trajectoryChannels, poseChannels, boneIndices);
    }

    private static int FindJointIndexOrZero(Skeleton skeleton, SkeletonBone bone, string featureName)
    {
        var index = bone?.ResolveIndex(skeleton) ?? -1;
        if (index >= 0) return index;

        Debug.LogWarning($"MmFeatureLayoutBuilder: feature \"{featureName}\" has no resolvable bone; falling back to the skeleton root.");
        return 0;
    }
}
}
