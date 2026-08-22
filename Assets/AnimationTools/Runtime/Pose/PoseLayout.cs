using System;
using System.Collections.Generic;

namespace AnimationTools
{
/// <summary>
/// Unmanaged, Burst-embeddable section offsets/counts for a <see cref="PoseLayout"/>.
/// Starts are in floats, counts are in elements (e.g. one Position element is 3 floats).
/// The buffer-wide width and hash live in <see cref="StateBufferLayoutData"/>, which a
/// <see cref="StateBuffer"/> carries alongside this.
/// </summary>
/// <remarks>
/// The named sections describe only the six built-in channel kinds. A layout may hold further
/// sections after them (see <see cref="ChannelSections"/>); <see cref="ExtraStart"/> and
/// <see cref="ExtraCount"/> cover that tail in floats, so nothing here assumes the named sections
/// span the whole buffer.
/// </remarks>
public struct PoseLayoutData
{
    public int PositionStart, PositionCount;
    public int RotationStart, RotationCount;
    /// <summary>4 for Quaternion, 3 for RotationVector/EulerDegrees.</summary>
    public byte RotationStride;
    public int ScaleStart, ScaleCount;
    public int VelocityStart, VelocityCount;
    public int AngularVelocityStart, AngularVelocityCount;
    /// <summary>Bools are stored as one float each (0f/1f).</summary>
    public int BoolStart, BoolCount;
    /// <summary>Floats belonging to sections beyond the six built-in kinds.</summary>
    public int ExtraStart, ExtraCount;
}

/// <summary>
/// A <see cref="StateBufferLayout"/> tied to a skeleton: the six built-in channel kinds get named
/// sections (<see cref="Data"/>) so pose consumers can address positions/rotations/etc. without
/// handles. This layout IS the pose file's self-description: a serialized pose asset is expected
/// to store enough of it to reconstruct a <see cref="PoseLayout"/> without external context.
/// Structurally-equal skeletons from different sources (e.g. a <see cref="PoseSet"/>'s skeleton and
/// a deserialized one with the same bones) share one cached layout instance, which is what lets
/// <see cref="PoseBuffer.CopyFrom"/> work between frames built over either.
/// </summary>
public sealed class PoseLayout : StateBufferLayout
{
    public const int PoseFormatVersion = 1;

    private static readonly Dictionary<int, PoseLayout> Cache = new();

    public Skeleton Skeleton { get; }
    public PoseLayoutData Data { get; }

    /// <summary>The shared representation of every Rotation channel in this layout.</summary>
    public RotationRepresentation RotationFormat { get; }

    private PoseLayout(Skeleton skeleton, IReadOnlyList<ChannelDescriptor> channels, int layoutHash,
        RotationRepresentation rotationFormat)
        : base(channels, layoutHash)
    {
        Skeleton = skeleton;
        RotationFormat = rotationFormat;

        var data = new PoseLayoutData
        {
            RotationStride = (byte)(rotationFormat == RotationRepresentation.Quaternion ? 4 : 3)
        };

        if (TryGetSectionRange(ChannelSections.Position, out var start, out _)) data.PositionStart = start;
        data.PositionCount = GetSectionElementCount(ChannelSections.Position);
        if (TryGetSectionRange(ChannelSections.Rotation, out start, out _)) data.RotationStart = start;
        data.RotationCount = GetSectionElementCount(ChannelSections.Rotation);
        if (TryGetSectionRange(ChannelSections.Scale, out start, out _)) data.ScaleStart = start;
        data.ScaleCount = GetSectionElementCount(ChannelSections.Scale);
        if (TryGetSectionRange(ChannelSections.Velocity, out start, out _)) data.VelocityStart = start;
        data.VelocityCount = GetSectionElementCount(ChannelSections.Velocity);
        if (TryGetSectionRange(ChannelSections.AngularVelocity, out start, out _)) data.AngularVelocityStart = start;
        data.AngularVelocityCount = GetSectionElementCount(ChannelSections.AngularVelocity);
        if (TryGetSectionRange(ChannelSections.Bool, out start, out _)) data.BoolStart = start;
        data.BoolCount = GetSectionElementCount(ChannelSections.Bool);

        // Built-in section keys are the lowest, so anything else forms a contiguous tail.
        var extraStart = FloatCount;
        for (var i = 0; i < Channels.Count; i++)
        {
            var key = Channels[i].SectionKey;
            if (key > ChannelSections.Bool && TryGetSectionRange(key, out start, out _) && start < extraStart)
                extraStart = start;
        }

        data.ExtraStart = extraStart;
        data.ExtraCount = FloatCount - extraStart;

        Data = data;
    }

    /// <summary>
    /// Builds (or returns the cached) layout for a channel set. Throws
    /// <see cref="ArgumentException"/> if any boneId is not present in
    /// <paramref name="skeleton"/>, if a channel identity repeats, if Rotation channels don't
    /// share one representation, or if a non-built-in channel claims a built-in section key.
    /// </summary>
    public static PoseLayout Build(Skeleton skeleton, IReadOnlyList<ChannelDescriptor> channels)
    {
        if (skeleton == null) throw new ArgumentException("skeleton must not be null.", nameof(skeleton));
        if (channels == null) throw new ArgumentException("channels must not be null.", nameof(channels));

        RotationRepresentation? rotationFormat = null;

        for (var i = 0; i < channels.Count; i++)
        {
            var channel = channels[i];
            if (channel == null) throw new ArgumentException($"Channel {i} is null.", nameof(channels));

            if (channel is BoneChannelDescriptor boneChannel && skeleton.IndexOfId(boneChannel.BoneId) < 0)
                throw new ArgumentException($"Channel {i} references bone id {boneChannel.BoneId}, which is not present in skeleton \"{skeleton.Name}\".", nameof(channels));

            if (channel.SectionKey <= ChannelSections.Bool && !IsBuiltIn(channel))
                throw new ArgumentException($"Channel {Describe(channel)} claims built-in section key {channel.SectionKey}; keys up to {ChannelSections.Bool} are reserved.", nameof(channels));

            if (channel is RotationChannel rotation)
            {
                if (rotationFormat == null) rotationFormat = rotation.Representation;
                else if (rotationFormat != rotation.Representation)
                    throw new ArgumentException($"Mixed rotation representations in one layout: {rotationFormat} and {rotation.Representation}. All Rotation channels must share one representation.", nameof(channels));
            }
        }

        var hash = ComputeHash(skeleton, channels);
        if (Cache.TryGetValue(hash, out var cached))
        {
            // A skeleton is a Transform tree, so a cached entry's rig can be destroyed out from
            // under it by a reimport or a scene unload. Such an entry reports zero bones and can
            // never match again, so evict it rather than let it block the slot forever.
            if (cached.Skeleton == null || !cached.Skeleton.IsSet)
            {
                Cache.Remove(hash);
                cached = null;
            }
            else if (AnimationTools.Skeleton.StructurallyEqual(cached.Skeleton, skeleton))
            {
                return cached;
            }
        }

        var layout = new PoseLayout(skeleton, channels, hash,
            rotationFormat ?? RotationRepresentation.Quaternion);

        // A hash collision between structurally different skeletons must not evict the cached
        // entry another skeleton is relying on; the fresh layout is simply left uncached.
        if (cached == null) Cache[hash] = layout;
        return layout;
    }

    /// <summary>
    /// One ParentLocal Position + one ParentLocal Quaternion Rotation channel per bone, in
    /// skeleton DFS order, so element index equals bone index. Optionally adds per-bone
    /// ParentLocal Velocity/AngularVelocity channels, likewise bone-index aligned. All channels
    /// use <see cref="ChannelUsage.Default"/>.
    /// </summary>
    public static PoseLayout CreateFullPose(Skeleton skeleton, bool includeVelocities, bool includeAngularVelocities)
    {
        if (skeleton == null) throw new ArgumentException("skeleton must not be null.", nameof(skeleton));

        var boneCount = skeleton.BoneCount;
        var channels = new List<ChannelDescriptor>(boneCount * 2);

        for (var i = 0; i < boneCount; i++)
        {
            var boneId = skeleton.GetBoneId(i);
            channels.Add(new PositionChannel(boneId));
            channels.Add(new RotationChannel(boneId));
        }

        if (includeVelocities)
        {
            for (var i = 0; i < boneCount; i++)
            {
                channels.Add(new VelocityChannel(skeleton.GetBoneId(i)));
            }
        }

        if (includeAngularVelocities)
        {
            for (var i = 0; i < boneCount; i++)
            {
                channels.Add(new AngularVelocityChannel(skeleton.GetBoneId(i)));
            }
        }

        return Build(skeleton, channels);
    }

    private static bool IsBuiltIn(ChannelDescriptor channel)
    {
        return channel is PositionChannel or RotationChannel or ScaleChannel or VelocityChannel
            or AngularVelocityChannel or BoolChannel;
    }

    private static int ComputeHash(Skeleton skeleton, IReadOnlyList<ChannelDescriptor> channels)
    {
        unchecked
        {
            var hash = 17;
            hash = hash * 31 + skeleton.ContentHash;

            return HashChannels(hash, channels);
        }
    }
}
}
