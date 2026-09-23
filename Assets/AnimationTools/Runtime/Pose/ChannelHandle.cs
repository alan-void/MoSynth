namespace AnimationTools
{
/// <summary>
/// A handle to one channel of a <see cref="StateBuffer"/>, procured by presenting the channel's
/// <see cref="ChannelDescriptor"/> to <see cref="StateBufferLayout.BindChannel"/>.
/// </summary>
/// <remarks>
/// Index is an absolute offset into the buffer's data, so a handle is only valid against the layout
/// it was bound to; it carries that LayoutHash and every accessor asserts the match. See
/// openwiki/animation-tools/channel-layout-system.md.
/// </remarks>
public readonly struct ChannelHandle
{
    internal readonly int Index;
    internal readonly int Count;
    internal readonly int LayoutHash;

    internal ChannelHandle(int index, int count, int layoutHash)
    {
        Index = index;
        Count = count;
        LayoutHash = layoutHash;
    }

    /// <summary>Width of this channel in floats.</summary>
    public int FloatCount => Count;

    /// <summary>
    /// Offset of this channel from the start of a buffer, in floats. Exposed because per-dimension
    /// side tables (normalisation statistics, per-float weights) have to be indexed positionally;
    /// reading buffer data should still go through <see cref="StateBuffer"/>, which checks the
    /// handle against the buffer's layout.
    /// </summary>
    public int FloatOffset => Index;

    public bool IsValid => Index >= 0;
    public static ChannelHandle Invalid => new(-1, 0, 0);
}
}
