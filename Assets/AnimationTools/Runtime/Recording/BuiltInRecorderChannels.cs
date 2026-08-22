using System;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
// The recorder channel kinds every MotionRecorder can use out of the box. Grouped in one file
// because they differ mainly in width and Sample body, the same way BuiltInChannels.cs groups
// the pose channel kinds.

/// <summary>Two floats: seconds since recording started, and this sample's delta time.</summary>
[Serializable]
public sealed class TimeChannel : RecorderChannel
{
    public override int FloatCount => 2;

    public override void Sample(in RecorderSampleContext context, ChannelHandle handle, StateBuffer frame)
    {
        frame.SetFloat(handle, 0, context.Time);
        frame.SetFloat(handle, 1, context.DeltaTime);
    }

    public override void PopulateManifest(RecordingManifestChannel entry)
    {
        entry.components = new[] { "time", "deltaTime" };
    }

    public override bool Equals(ChannelDescriptor other) =>
        other is TimeChannel c && c.name == name;

    public override int GetHashCode()
    {
        unchecked { return GetType().GetHashCode() * 31 + (name?.GetHashCode() ?? 0); }
    }

    public override int GetContentHash() => GetHashCode();
}

/// <summary>The world-space position of one bone (or the simulation bone) at sample time.</summary>
[Serializable]
public sealed class BoneWorldPositionChannel : RecorderChannel
{
    public bool simulationBone;
    public SkeletonBone bone = new();

    [NonSerialized] private int _boneIndex;
    [NonSerialized] private string _resolvedBoneName;

    public override int FloatCount => 3;

    public override void Bind(MotionSynthesisComponent synthesizer)
    {
        if (simulationBone)
        {
            _boneIndex = 0;
            _resolvedBoneName = "SimulationBone";
        }
        else
        {
            var index = bone?.ResolveIndex(synthesizer.Skeleton) ?? -1;
            if (index < 0) index = synthesizer.Skeleton.IndexOfName("Hips");
            if (index < 0)
            {
                Debug.LogWarning($"{nameof(BoneWorldPositionChannel)} \"{name}\": bone \"{bone?.Name}\" not found; defaulting to bone 0.");
                index = 0;
            }

            _boneIndex = index;
            _resolvedBoneName = synthesizer.Skeleton.GetBone(_boneIndex).Name;
        }
    }

    public override void Sample(in RecorderSampleContext context, ChannelHandle handle, StateBuffer frame)
    {
        // Sampling always happens after the pose has been applied to SkeletonTransforms, so the
        // transform's world position is this tick's answer, not last tick's.
        frame.SetFloat3(handle, (float3)context.Synthesizer.SkeletonTransforms[_boneIndex].position);
    }

    public override void PopulateManifest(RecordingManifestChannel entry)
    {
        entry.bone = _resolvedBoneName;
        entry.space = "World";
        entry.components = new[] { "x", "y", "z" };
    }

    public override bool Equals(ChannelDescriptor other) =>
        other is BoneWorldPositionChannel c && c.name == name
        && c.simulationBone == simulationBone && c.bone?.Name == bone?.Name;

    public override int GetHashCode()
    {
        unchecked
        {
            var hash = GetType().GetHashCode();
            hash = hash * 31 + (name?.GetHashCode() ?? 0);
            hash = hash * 31 + simulationBone.GetHashCode();
            hash = hash * 31 + (bone?.Name?.GetHashCode() ?? 0);
            return hash;
        }
    }

    public override int GetContentHash() => GetHashCode();
}

/// <summary>The world-space forward direction of one bone (or the simulation bone) at sample time.</summary>
[Serializable]
public sealed class BoneWorldForwardChannel : RecorderChannel
{
    public bool simulationBone;
    public SkeletonBone bone = new();

    [NonSerialized] private int _boneIndex;
    [NonSerialized] private string _resolvedBoneName;

    public override int FloatCount => 3;

    public override void Bind(MotionSynthesisComponent synthesizer)
    {
        if (simulationBone)
        {
            _boneIndex = 0;
            _resolvedBoneName = "SimulationBone";
        }
        else
        {
            var index = bone?.ResolveIndex(synthesizer.Skeleton) ?? -1;
            if (index < 0) index = synthesizer.Skeleton.IndexOfName("Hips");
            if (index < 0)
            {
                Debug.LogWarning($"{nameof(BoneWorldForwardChannel)} \"{name}\": bone \"{bone?.Name}\" not found; defaulting to bone 0.");
                index = 0;
            }

            _boneIndex = index;
            _resolvedBoneName = synthesizer.Skeleton.GetBone(_boneIndex).Name;
        }
    }

    public override void Sample(in RecorderSampleContext context, ChannelHandle handle, StateBuffer frame)
    {
        // Sampling always happens after the pose has been applied to SkeletonTransforms, so the
        // transform's world forward is this tick's answer, not last tick's.
        frame.SetFloat3(handle, (float3)context.Synthesizer.SkeletonTransforms[_boneIndex].forward);
    }

    public override void PopulateManifest(RecordingManifestChannel entry)
    {
        entry.bone = _resolvedBoneName;
        entry.space = "World";
        entry.components = new[] { "x", "y", "z" };
    }

    public override bool Equals(ChannelDescriptor other) =>
        other is BoneWorldForwardChannel c && c.name == name
        && c.simulationBone == simulationBone && c.bone?.Name == bone?.Name;

    public override int GetHashCode()
    {
        unchecked
        {
            var hash = GetType().GetHashCode();
            hash = hash * 31 + (name?.GetHashCode() ?? 0);
            hash = hash * 31 + simulationBone.GetHashCode();
            hash = hash * 31 + (bone?.Name?.GetHashCode() ?? 0);
            return hash;
        }
    }

    public override int GetContentHash() => GetHashCode();
}

/// <summary>
/// The entire synthesized pose buffer, copied verbatim into the recording each sample. Useful
/// for dumping every joint without hand-listing bone channels.
/// </summary>
[Serializable]
public sealed class FullPoseChannel : RecorderChannel
{
    [NonSerialized] private PoseLayout _poseLayout;
    [NonSerialized] private int _floatCount;

    public override int FloatCount
    {
        get
        {
            if (_poseLayout == null)
                throw new InvalidOperationException(
                    $"{nameof(FullPoseChannel)} \"{name}\" was not bound: FloatCount depends on the " +
                    "synthesizer's PoseLayout, which is only known after Bind() runs.");
            return _floatCount;
        }
    }

    public override void Bind(MotionSynthesisComponent synthesizer)
    {
        _poseLayout = synthesizer.PoseLayout;
        _floatCount = _poseLayout.FloatCount;
    }

    public override void Sample(in RecorderSampleContext context, ChannelHandle handle, StateBuffer frame)
    {
        var slice = frame.GetSlice(handle);
        var pose = context.Pose;
        if (!pose.IsCreated)
        {
            for (var i = 0; i < slice.Length; i++) slice[i] = 0f;
            return;
        }

        slice.CopyFrom(pose.Data);
    }

    public override void PopulateManifest(RecordingManifestChannel entry)
    {
        var data = _poseLayout.Data;
        entry.poseLayout = new RecordingManifestPoseLayout
        {
            rotationFormat = _poseLayout.RotationFormat.ToString(),
            positionStart = data.PositionStart,
            positionCount = data.PositionCount,
            rotationStart = data.RotationStart,
            rotationCount = data.RotationCount,
            rotationStride = data.RotationStride,
            velocityStart = data.VelocityStart,
            velocityCount = data.VelocityCount,
            angularVelocityStart = data.AngularVelocityStart,
            angularVelocityCount = data.AngularVelocityCount,
            boolStart = data.BoolStart,
            boolCount = data.BoolCount
        };

        var skeleton = _poseLayout.Skeleton;
        var boneNames = new string[skeleton.BoneCount];
        for (var i = 0; i < boneNames.Length; i++) boneNames[i] = skeleton.GetBone(i).Name;
        entry.poseLayout.boneNames = boneNames;
    }

    public override bool Equals(ChannelDescriptor other) =>
        other is FullPoseChannel c && c.name == name;

    public override int GetHashCode()
    {
        unchecked { return GetType().GetHashCode() * 31 + (name?.GetHashCode() ?? 0); }
    }

    public override int GetContentHash() => GetHashCode();
}
}
