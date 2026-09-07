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
/// Finds the control input to sample from: the one steering the character, when it is a motion
/// matching input. Null on a character driven by something else, or by nothing yet.
/// </summary>
internal static class RecorderChannelControlInputLookup
{
    internal static MotionMatchingControlInput FindControlInput(MotionSynthesisComponent synthesizer) =>
        synthesizer.ControlInput as MotionMatchingControlInput;
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
            Debug.LogError($"PathTargetPositionChannel \"{name}\": no MotionMatchingControlInput steering the character " +
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
            Debug.LogError($"TargetSpeedChannel \"{name}\": no MotionMatchingControlInput steering the character " +
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
