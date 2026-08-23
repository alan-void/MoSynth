using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
// Recorder channels specific to motion matching. Generic ones -- time, bone transforms, the full
// pose -- live in AnimationTools' BuiltInRecorderChannels; these need the control input, which only
// exists in a motion matching stack.

/// <summary>
/// Finds the control input to sample from. Recorder channels are bound to a synthesizer, not to a
/// stage, so they have to go looking; the first motion matching stage that has one wins.
/// </summary>
internal static class RecorderChannelControlInputLookup
{
    internal static MotionMatchingControlInput FindControlInput(MotionSynthesisComponent synthesizer)
    {
        foreach (var stage in synthesizer.stages)
        {
            if (stage is MotionMatchingStage mmStage && mmStage.controlInput != null)
            {
                return mmStage.controlInput;
            }
        }

        return null;
    }
}

/// <summary>
/// Records where the control input <em>wanted</em> the character to be. Paired with where it actually
/// was, this is what turns a run into a path-following error measure. NaN when no control input was
/// found, so a broken setup shows up in the data rather than reading as the origin.
/// </summary>
[Serializable]
public sealed class PathTargetPositionChannel : RecorderChannel
{
    public override int FloatCount => 3;

    private MotionMatchingControlInput _controlInput;

    public override void Bind(MotionSynthesisComponent synthesizer)
    {
        _controlInput = RecorderChannelControlInputLookup.FindControlInput(synthesizer);
        if (_controlInput == null)
        {
            Debug.LogError($"PathTargetPositionChannel \"{name}\": no MotionMatchingStage with a controlInput " +
                            "was found on the synthesizer.");
        }
    }

    public override void Sample(in RecorderSampleContext context, ChannelHandle handle, StateBuffer frame)
    {
        if (_controlInput == null)
        {
            frame.SetFloat3(handle, new float3(float.NaN, float.NaN, float.NaN));
            return;
        }

        frame.SetFloat3(handle, _controlInput.GetPosition());
    }

    public override void PopulateManifest(RecordingManifestChannel entry)
    {
        entry.space = "World";
        entry.components = new[] { "x", "y", "z" };
    }

    public override bool Equals(ChannelDescriptor other) =>
        other is PathTargetPositionChannel c && c.name == name;

    public override int GetHashCode()
    {
        unchecked { return GetType().GetHashCode() * 31 + (name?.GetHashCode() ?? 0); }
    }

    public override int GetContentHash() => GetHashCode();
}

/// <summary>
/// Records the control input's intended speed, to compare against the speed actually achieved.
/// NaN when no control input was found.
/// </summary>
[Serializable]
public sealed class TargetSpeedChannel : RecorderChannel
{
    public override int FloatCount => 1;

    private MotionMatchingControlInput _controlInput;

    public override void Bind(MotionSynthesisComponent synthesizer)
    {
        _controlInput = RecorderChannelControlInputLookup.FindControlInput(synthesizer);
        if (_controlInput == null)
        {
            Debug.LogError($"TargetSpeedChannel \"{name}\": no MotionMatchingStage with a controlInput " +
                            "was found on the synthesizer.");
        }
    }

    public override void Sample(in RecorderSampleContext context, ChannelHandle handle, StateBuffer frame)
    {
        if (_controlInput == null)
        {
            frame.SetFloat(handle, 0, float.NaN);
            return;
        }

        frame.SetFloat(handle, 0, _controlInput.GetTargetSpeed());
    }

    public override bool Equals(ChannelDescriptor other) =>
        other is TargetSpeedChannel c && c.name == name;

    public override int GetHashCode()
    {
        unchecked { return GetType().GetHashCode() * 31 + (name?.GetHashCode() ?? 0); }
    }

    public override int GetContentHash() => GetHashCode();
}
}
