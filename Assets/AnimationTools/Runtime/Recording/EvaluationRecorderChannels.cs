using System;
using Unity.Mathematics;
using Unity.Profiling;
using UnityEngine;

namespace AnimationTools
{
// Recorder channels that exist to be measured rather than replayed; they only make sense while
// something is benchmarking.

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

/// <summary>
/// One float per contact slot of the synthesizer, 0 or 1: whether that bone was flagged in contact on
/// this tick. Manifest components are the contact bone names, in slot order.
/// </summary>
[Serializable]
public sealed class FootContactChannel : RecorderChannel
{
    [NonSerialized] private ContactHandles _contacts = ContactHandles.Empty;
    [NonSerialized] private string[] _boneNames = Array.Empty<string>();

    public override int FloatCount => _contacts.Count;

    public override void Bind(MotionSynthesisComponent synthesizer)
    {
        _contacts = synthesizer.ContactHandles;
        _boneNames = new string[synthesizer.ContactBoneNames.Count];
        for (var slot = 0; slot < _boneNames.Length; slot++) _boneNames[slot] = synthesizer.ContactBoneNames[slot];
    }

    public override void Sample(in RecorderSampleContext context, ChannelHandle handle, StateBuffer frame)
    {
        var pose = context.Pose;
        var hasPose = pose.IsCreated;
        for (var slot = 0; slot < _contacts.Count; slot++)
        {
            frame.SetFloat(handle, slot, hasPose && pose.GetBool(_contacts[slot]) ? 1f : 0f);
        }
    }

    public override void PopulateManifest(RecordingManifestChannel entry)
    {
        entry.components = _boneNames;
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
/// World position of the bone behind one contact slot of the synthesizer.
/// </summary>
/// <remarks>
/// Not a hand-picked <see cref="BoneWorldPositionChannel"/>: footskate needs the position and the
/// contact flag to name the same bone, so this takes the bone from the slot.
/// </remarks>
[Serializable]
public sealed class ContactBoneWorldPositionChannel : RecorderChannel
{
    /// <summary>Contact slot whose bone this records, an index into the synthesizer's contact list.</summary>
    public int slot;

    [NonSerialized] private int _boneIndex;
    [NonSerialized] private string _resolvedBoneName;

    public override int FloatCount => 3;

    public override void Bind(MotionSynthesisComponent synthesizer)
    {
        var contacts = synthesizer.ContactHandles;
        if (slot < 0 || slot >= contacts.Count)
        {
            Debug.LogWarning($"{nameof(ContactBoneWorldPositionChannel)} \"{name}\": slot {slot} is outside the " +
                             $"synthesizer's {contacts.Count} contact slots; recording bone 0 instead.");
            _boneIndex = 0;
        }
        else
        {
            _boneIndex = contacts.GetBoneIndex(slot);
        }

        _resolvedBoneName = synthesizer.Skeleton.GetBone(_boneIndex).Name;
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
        other is ContactBoneWorldPositionChannel c && c.name == name && c.slot == slot;

    public override int GetHashCode()
    {
        unchecked
        {
            var hash = GetType().GetHashCode();
            hash = hash * 31 + (name?.GetHashCode() ?? 0);
            hash = hash * 31 + slot;
            return hash;
        }
    }

    public override int GetContentHash() => GetHashCode();
}
}
