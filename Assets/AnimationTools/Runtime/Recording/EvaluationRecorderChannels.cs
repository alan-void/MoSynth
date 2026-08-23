using System;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine;

namespace AnimationTools
{
// Recorder channels that exist to be measured rather than replayed. Kept out of
// BuiltInRecorderChannels.cs because these three only make sense while something is benchmarking:
// two of them read state the synthesis component publishes solely for a harness.

/// <summary>
/// Per-stage and total wall-clock cost of one synthesis tick, plus managed bytes allocated during
/// it. Writes <c>stages.Count + 2</c> floats: one millisecond figure per stage in pipeline order,
/// then the total, then the allocation.
/// </summary>
/// <remarks>
/// Binding this channel switches <see cref="MotionSynthesisComponent.MeasureStageCost"/> on, which
/// is what makes the component take the timestamps in the first place. Nothing turns it back off —
/// the component is torn down at the end of a benchmark run anyway.
/// </remarks>
[Serializable]
public sealed class StageCostChannel : RecorderChannel
{
    [NonSerialized] private MotionSynthesisComponent _synthesizer;
    [NonSerialized] private string[] _stageNames = Array.Empty<string>();
    [NonSerialized] private ProfilerRecorder _gcRecorder;
    [NonSerialized] private double _ticksToMilliseconds;

    /// <summary>Index of the total-milliseconds float within this channel.</summary>
    public int TotalOffset => _stageNames.Length;

    /// <summary>Index of the GC-bytes float within this channel.</summary>
    public int GcBytesOffset => _stageNames.Length + 1;

    public override int FloatCount => _stageNames.Length + 2;

    public override void Bind(MotionSynthesisComponent synthesizer)
    {
        _synthesizer = synthesizer;
        _synthesizer.MeasureStageCost = true;

        var stages = synthesizer.stages;
        _stageNames = new string[stages.Count];
        for (var i = 0; i < stages.Count; i++)
        {
            _stageNames[i] = stages[i] != null ? stages[i].GetType().Name : "null";
        }

        _ticksToMilliseconds = 1000.0 / System.Diagnostics.Stopwatch.Frequency;

        // Started here rather than in Sample so the first tick already has a window to report.
        _gcRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Memory, "GC Allocated In Frame");
    }

    public override void Sample(in RecorderSampleContext context, ChannelHandle handle, StateBuffer frame)
    {
        var ticks = _synthesizer.StageApplyTicks;
        var total = 0.0;
        for (var i = 0; i < _stageNames.Length; i++)
        {
            var ms = i < ticks.Length ? ticks[i] * _ticksToMilliseconds : 0.0;
            frame.SetFloat(handle, i, (float)ms);
            total += ms;
        }

        frame.SetFloat(handle, TotalOffset, (float)total);
        frame.SetFloat(handle, GcBytesOffset, _gcRecorder.Valid ? _gcRecorder.LastValue : float.NaN);
    }

    public override void PopulateManifest(RecordingManifestChannel entry)
    {
        var components = new string[FloatCount];
        Array.Copy(_stageNames, components, _stageNames.Length);
        components[TotalOffset] = "total";
        components[GcBytesOffset] = "gcBytes";
        entry.components = components;
    }

    /// <summary>
    /// Releases the profiler recorder. Not part of the RecorderChannel contract, so the benchmark
    /// driver calls it after stopping the recorder; leaking one is harmless but noisy in the
    /// profiler's own bookkeeping.
    /// </summary>
    public void DisposeRecorder()
    {
        if (_gcRecorder.Valid) _gcRecorder.Dispose();
    }

    public override bool Equals(ChannelDescriptor other) =>
        other is StageCostChannel c && c.name == name;

    public override int GetHashCode()
    {
        unchecked { return GetType().GetHashCode() * 31 + (name?.GetHashCode() ?? 0); }
    }

    public override int GetContentHash() => GetHashCode();
}

/// <summary>Two floats, 0 or 1: whether each foot was flagged in contact on this tick.</summary>
[Serializable]
public sealed class FootContactChannel : RecorderChannel
{
    [NonSerialized] private ChannelHandle _left;
    [NonSerialized] private ChannelHandle _right;

    public override int FloatCount => 2;

    public override void Bind(MotionSynthesisComponent synthesizer)
    {
        _left = synthesizer.LeftFootContactHandle;
        _right = synthesizer.RightFootContactHandle;
    }

    public override void Sample(in RecorderSampleContext context, ChannelHandle handle, StateBuffer frame)
    {
        var pose = context.Pose;
        var hasPose = pose.IsCreated;
        frame.SetFloat(handle, 0, hasPose && _left.IsValid && pose.GetBool(_left) ? 1f : 0f);
        frame.SetFloat(handle, 1, hasPose && _right.IsValid && pose.GetBool(_right) ? 1f : 0f);
    }

    public override void PopulateManifest(RecordingManifestChannel entry)
    {
        entry.components = new[] { "left", "right" };
    }

    public override bool Equals(ChannelDescriptor other) =>
        other is FootContactChannel c && c.name == name;

    public override int GetHashCode()
    {
        unchecked { return GetType().GetHashCode() * 31 + (name?.GetHashCode() ?? 0); }
    }

    public override int GetContentHash() => GetHashCode();
}

/// <summary>
/// One float, 0 or 1: whether a stage replaced the pose discontinuously on this tick. Counting these
/// is how a method that hits its path by teleporting is told apart from one that walks there.
/// </summary>
[Serializable]
public sealed class PoseDiscontinuityChannel : RecorderChannel
{
    public override int FloatCount => 1;

    public override void Sample(in RecorderSampleContext context, ChannelHandle handle, StateBuffer frame)
    {
        frame.SetFloat(handle, 0, context.Synthesizer.PoseDiscontinuity ? 1f : 0f);
    }

    public override void PopulateManifest(RecordingManifestChannel entry)
    {
        entry.components = new[] { "discontinuity" };
    }

    public override bool Equals(ChannelDescriptor other) =>
        other is PoseDiscontinuityChannel c && c.name == name;

    public override int GetHashCode()
    {
        unchecked { return GetType().GetHashCode() * 31 + (name?.GetHashCode() ?? 0); }
    }

    public override int GetContentHash() => GetHashCode();
}

/// <summary>
/// World position of one side's foot-contact bone, resolved the same way the pose layout resolves
/// the contact flags themselves.
/// </summary>
/// <remarks>
/// Deliberately not a <see cref="BoneWorldPositionChannel"/> with a hand-picked bone: footskate is
/// the distance this bone travels while its flag says planted, so the position and the flag have to
/// name the same bone. Going through <see cref="BoneNameConventions"/>, exactly as
/// <see cref="PoseLayoutBuilder.Build"/> does, is what guarantees that rather than leaving it to
/// whoever fills in the inspector.
/// </remarks>
[Serializable]
public sealed class ContactBoneWorldPositionChannel : RecorderChannel
{
    public bool left = true;

    [NonSerialized] private int _boneIndex;
    [NonSerialized] private string _resolvedBoneName;

    public override int FloatCount => 3;

    public override void Bind(MotionSynthesisComponent synthesizer)
    {
        var skeleton = synthesizer.Skeleton;
        if (!BoneNameConventions.TryFindContactBone(skeleton, left, out _boneIndex))
        {
            Debug.LogWarning($"{nameof(ContactBoneWorldPositionChannel)} \"{name}\": no " +
                             $"{(left ? "left" : "right")} foot bone found by name; footskate will be measured on bone 0.");
            _boneIndex = 0;
        }

        _resolvedBoneName = skeleton.GetBone(_boneIndex).Name;
    }

    public override void Sample(in RecorderSampleContext context, ChannelHandle handle, StateBuffer frame)
    {
        frame.SetFloat3(handle, (float3)context.Synthesizer.SkeletonTransforms[_boneIndex].position);
    }

    public override void PopulateManifest(RecordingManifestChannel entry)
    {
        entry.bone = _resolvedBoneName;
        entry.space = "World";
        entry.components = new[] { "x", "y", "z" };
    }

    public override bool Equals(ChannelDescriptor other) =>
        other is ContactBoneWorldPositionChannel c && c.name == name && c.left == left;

    public override int GetHashCode()
    {
        unchecked
        {
            var hash = GetType().GetHashCode();
            hash = hash * 31 + (name?.GetHashCode() ?? 0);
            hash = hash * 31 + left.GetHashCode();
            return hash;
        }
    }

    public override int GetContentHash() => GetHashCode();
}
}
