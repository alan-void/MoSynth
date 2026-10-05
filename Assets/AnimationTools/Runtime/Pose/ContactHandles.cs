using System;
using System.Collections.Generic;

namespace AnimationTools
{
/// <summary>
/// The contact Bool channels of a pose layout, one per slot in contact-list order, and the bone each
/// slot is hosted on. Built by <see cref="PoseLayoutBuilder.Build"/>.
/// </summary>
public sealed class ContactHandles
{
    public static readonly ContactHandles Empty = new(Array.Empty<int>(), Array.Empty<ChannelHandle>());

    private readonly int[] _boneIndices;
    private readonly ChannelHandle[] _handles;

    internal ContactHandles(int[] boneIndices, ChannelHandle[] handles)
    {
        _boneIndices = boneIndices;
        _handles = handles;
    }

    /// <summary>Number of contact slots.</summary>
    public int Count => _handles.Length;

    /// <summary>The Bool channel of slot <paramref name="slot"/>.</summary>
    public ChannelHandle this[int slot] => _handles[slot];

    /// <summary>Skeleton index of the bone slot <paramref name="slot"/> flags.</summary>
    public int GetBoneIndex(int slot) => _boneIndices[slot];

    /// <summary>Bone indices in slot order.</summary>
    public IReadOnlyList<int> BoneIndices => _boneIndices;

    /// <summary>The slot that flags <paramref name="boneIndex"/>; false when no slot does.</summary>
    public bool TryGetSlot(int boneIndex, out int slot)
    {
        slot = Array.IndexOf(_boneIndices, boneIndex);
        return slot >= 0;
    }
}
}
