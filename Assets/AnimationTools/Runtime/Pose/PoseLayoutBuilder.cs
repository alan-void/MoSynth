using System;
using System.Collections.Generic;

namespace AnimationTools
{
/// <summary>
/// Builds the single pose layout the pipeline shares: parent-local position + rotation per bone
/// (element index == bone index), per-bone velocities and angular velocities, and one contact Bool
/// channel per contact bone.
/// </summary>
/// <remarks>
/// <see cref="PoseSet"/> and <see cref="MotionSynthesisComponent"/> both build through here from the
/// same contact-bone list, so they get the same cached layout and frames copy between them.
/// </remarks>
public static class PoseLayoutBuilder
{
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
    /// The authoritative pose layout: full-pose channels plus one contact Bool channel per entry of
    /// <paramref name="contactBoneIndices"/>, in that order, with their handles bound.
    /// </summary>
    /// <exception cref="ArgumentException">An index is out of range or listed twice.</exception>
    public static PoseLayout Build(Skeleton skeleton, IReadOnlyList<int> contactBoneIndices,
        out ContactHandles contacts)
    {
        var channels = BuildFullPoseChannels(skeleton);

        var count = contactBoneIndices?.Count ?? 0;
        var boneIndices = new int[count];
        for (var slot = 0; slot < count; slot++)
        {
            var bone = contactBoneIndices[slot];
            if (bone < 0 || bone >= skeleton.BoneCount)
                throw new ArgumentException($"Contact bone index {bone} is outside the skeleton's " +
                                            $"{skeleton.BoneCount} bones.", nameof(contactBoneIndices));
            if (Array.IndexOf(boneIndices, bone, 0, slot) >= 0)
                throw new ArgumentException($"Contact bone \"{skeleton.GetBone(bone).Name}\" is listed twice.",
                    nameof(contactBoneIndices));

            boneIndices[slot] = bone;
            channels.Add(new BoolChannel(skeleton.GetBoneId(bone), ChannelUsage.Contact));
        }

        var layout = PoseLayout.Build(skeleton, channels);

        var handles = new ChannelHandle[count];
        for (var slot = 0; slot < count; slot++)
        {
            handles[slot] = layout.BindChannel(
                new BoolChannel(skeleton.GetBoneId(boneIndices[slot]), ChannelUsage.Contact));
        }

        contacts = new ContactHandles(boneIndices, handles);
        return layout;
    }
}
}
