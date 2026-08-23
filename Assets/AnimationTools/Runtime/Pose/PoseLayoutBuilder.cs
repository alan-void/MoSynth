using System.Collections.Generic;
using AnimationTools;

namespace AnimationTools
{
/// <summary>
/// Builds the single pose layout the motion-matching runtime shares: a full pose
/// (parent-local position + rotation per bone, element index == bone index), per-bone
/// velocities/angular velocities, and two foot-contact Bool channels. <see cref="PoseSet"/> and
/// MotionSynthesisComponent both build through here over the same
/// <see cref="AnimationTools.Skeleton"/>, so <see cref="PoseLayout.Build"/>'s cache hands them the
/// same instance and frames can be copied between them (<see cref="PoseBuffer.CopyFrom"/>
/// requires equal layout hashes). The contact host bone for each side must stay a pure function
/// of the skeleton's bone names -- never of the skeleton's source -- so two independently built
/// skeletons that are structurally equal always pick the same host bones and hash the same.
/// </summary>
    public static class PoseLayoutBuilder
{
    public readonly struct ContactHandles
    {
        public readonly ChannelHandle Left;
        public readonly ChannelHandle Right;

        internal ContactHandles(ChannelHandle left, ChannelHandle right)
        {
            Left = left;
            Right = right;
        }
    }

    /// <summary>
    /// The full-pose channel set without the contact bools, for consumers that append their
    /// own extra channels before calling <see cref="PoseLayout.Build"/> themselves.
    /// </summary>
    public static List<ChannelDescriptor> BuildFullPoseChannels(Skeleton skeleton)
    {
        var boneCount = skeleton.BoneCount;
        var channels = new List<ChannelDescriptor>(boneCount * 4 + 2);

        for (var i = 0; i < boneCount; i++)
        {
            var boneId = skeleton.GetBoneId(i);
            channels.Add(new PositionChannel(boneId));
            channels.Add(new RotationChannel(boneId));
        }

        for (var i = 0; i < boneCount; i++)
        {
            channels.Add(new VelocityChannel(skeleton.GetBoneId(i)));
        }

        for (var i = 0; i < boneCount; i++)
        {
            channels.Add(new AngularVelocityChannel(skeleton.GetBoneId(i)));
        }

        return channels;
    }

    /// <summary>
    /// The authoritative motion-matching pose layout: full-pose channels plus the two
    /// foot-contact Bool channels, with their handles bound.
    /// </summary>
    public static PoseLayout Build(Skeleton skeleton, out ContactHandles contacts)
    {
        var channels = BuildFullPoseChannels(skeleton);

        var leftContactBone = BoneNameConventions.TryFindContactBone(skeleton, true, out var leftIndex)
            ? leftIndex
            : 0;
        var rightContactBone = BoneNameConventions.TryFindContactBone(skeleton, false, out var rightIndex)
            ? rightIndex
            : skeleton.BoneCount - 1;
        // The layout rejects duplicate channel identities, so a skeleton missing one side must
        // still end up with two distinct host bones.
        if (rightContactBone == leftContactBone)
        {
            rightContactBone = leftContactBone == 0 ? skeleton.BoneCount - 1 : 0;
        }

        channels.Add(new BoolChannel(skeleton.GetBoneId(leftContactBone), ChannelUsage.Contact));
        channels.Add(new BoolChannel(skeleton.GetBoneId(rightContactBone), ChannelUsage.Contact));

        var layout = PoseLayout.Build(skeleton, channels);
        contacts = new ContactHandles(
            layout.BindChannel(new BoolChannel(skeleton.GetBoneId(leftContactBone), ChannelUsage.Contact)),
            layout.BindChannel(new BoolChannel(skeleton.GetBoneId(rightContactBone), ChannelUsage.Contact)));
        return layout;
    }
}
}
