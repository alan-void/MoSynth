using System;
using Unity.Collections;

namespace AnimationTools
{
/// <summary>
/// A contiguous run of <see cref="FrameCount"/> frames sharing one <see cref="StateBufferLayout"/>,
/// stored back to back in a single flat float array.
/// </summary>
/// <remarks>
/// <see cref="GetFrame"/> returns a view aliasing <see cref="Data"/>; never Dispose it. Whoever
/// allocated <see cref="Data"/> disposes it: call <see cref="Dispose"/> only on sequences made by
/// <see cref="Allocate"/> or <see cref="FromArray"/>, never on one wrapping an external array.
/// See openwiki/animation-tools/channel-layout-system.md.
/// </remarks>
public class StateSequence
{
    public StateBufferLayout Layout { get; }
    public NativeArray<float> Data { get; }
    public int FrameCount { get; }

    /// <summary>
    /// Wraps an existing array; does not copy. Throws <see cref="ArgumentNullException"/> if
    /// <paramref name="layout"/> is null, or <see cref="ArgumentException"/> if
    /// <paramref name="data"/>'s length isn't a positive multiple of
    /// <see cref="StateBufferLayout.FloatCount"/>.
    /// </summary>
    public StateSequence(StateBufferLayout layout, NativeArray<float> data)
    {
        if (layout == null) throw new ArgumentNullException(nameof(layout));
        if (data.Length <= 0 || data.Length % layout.FloatCount != 0)
            throw new ArgumentException($"data.Length ({data.Length}) must be a positive multiple of layout.FloatCount ({layout.FloatCount}).", nameof(data));

        Layout = layout;
        Data = data;
        FrameCount = data.Length / layout.FloatCount;
    }

    public static StateSequence Allocate(StateBufferLayout layout, int frameCount, Allocator allocator)
    {
        return new StateSequence(layout, AllocateData(layout, frameCount, allocator));
    }

    public static StateSequence FromArray(StateBufferLayout layout, float[] frameData, Allocator allocator)
    {
        return new StateSequence(layout, CopyData(layout, frameData, allocator));
    }

    protected static NativeArray<float> AllocateData(StateBufferLayout layout, int frameCount, Allocator allocator)
    {
        if (layout == null) throw new ArgumentNullException(nameof(layout));
        if (frameCount <= 0) throw new ArgumentException("frameCount must be positive.", nameof(frameCount));

        return new NativeArray<float>(frameCount * layout.FloatCount, allocator, NativeArrayOptions.ClearMemory);
    }

    protected static NativeArray<float> CopyData(StateBufferLayout layout, float[] frameData, Allocator allocator)
    {
        if (layout == null) throw new ArgumentNullException(nameof(layout));
        if (frameData == null) throw new ArgumentNullException(nameof(frameData));

        return new NativeArray<float>(frameData, allocator);
    }

    /// <summary>
    /// View over one frame — the caller must NEVER Dispose the returned buffer (it is a
    /// <see cref="NativeArray{T}.GetSubArray"/> view; disposing it throws).
    /// </summary>
    public StateBuffer GetFrame(int frameIndex)
    {
        if (frameIndex < 0 || frameIndex >= FrameCount)
            throw new ArgumentOutOfRangeException(nameof(frameIndex), frameIndex, $"frameIndex must be in [0, {FrameCount}).");

        return new StateBuffer
        {
            Data = Data.GetSubArray(frameIndex * Layout.FloatCount, Layout.FloatCount),
            Layout = Layout.Data
        };
    }

    public void CopyTo(float[] destination)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (destination.Length != Data.Length)
            throw new ArgumentException($"destination.Length ({destination.Length}) must equal Data.Length ({Data.Length}).", nameof(destination));

        Data.CopyTo(destination);
    }

    /// <summary>
    /// Disposes <see cref="Data"/>. Only for sequences built by <see cref="Allocate"/> or
    /// <see cref="FromArray"/>; an externally owned (e.g. Domain-backed) array throws here.
    /// </summary>
    public void Dispose()
    {
        if (Data.IsCreated) Data.Dispose();
    }
}
}
