using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// Keeps a matcher's <see cref="SynthesisInputs"/>: the query it last searched with and the
/// candidate it got back, both placed in the character frame the query was built in, plus the
/// costs and the frame playing.
/// </summary>
/// <remarks>
/// Shared by <see cref="MotionMatchingStage"/> and the learned matcher, which ask the same question.
/// The markers are world positions as of the search, so between searches they stay where the
/// question put them while the character walks on.
/// </remarks>
public sealed class MatchingInputRecorder
{
    private const int ReadoutCapacity = 8;

    private readonly FeatureVectorMarkers _markers;
    private readonly PoseSet _poseSet;

    /// <summary>One display name per database clip, or null when the config's clip list cannot be lined up with the database's.</summary>
    private readonly string[] _clipNames;

    private float _heldCost = float.NaN;
    private float _answerCost = float.NaN;
    private int _maskedChannels;

    public MatchingInputRecorder(MotionMatchingData mmData, FeatureSet featureSet, PoseSet poseSet, Skeleton skeleton)
    {
        _markers = new FeatureVectorMarkers(mmData, featureSet, skeleton);
        _poseSet = poseSet;

        var channels = featureSet.PoseFeatureCount;
        for (var t = 0; t < featureSet.NumberTrajectoryFeatures; t++) channels += featureSet.GetPredictionCount(t);
        Inputs = new SynthesisInputs("matching query", 2 * channels, ReadoutCapacity)
        {
            RequestLabel = "query",
            AnswerLabel = "best match",
            Searches = true
        };

        _clipNames = ClipNames(mmData, poseSet);
    }

    public SynthesisInputs Inputs { get; }

    public void BeginTick() => Inputs.BeginTick();

    /// <summary>Records one search: what was asked, what came back, and what each scores.</summary>
    /// <param name="answer">The best candidate's feature vector, whether or not it was taken.</param>
    /// <param name="character">The Transform the query was built against.</param>
    /// <param name="heldCost">The state already playing, scored against the query; <see cref="float.MaxValue"/> when unscoreable.</param>
    public void RecordSearch(ReadOnlySpan<float> query, ReadOnlySpan<float> answer, ReadOnlySpan<float> weights,
        Transform character, Transform[] joints, float heldCost, float answerCost, bool answerChanged)
    {
        Inputs.Searched = true;
        Inputs.AnswerChanged = answerChanged;
        Inputs.ClearMarkers();

        float3 origin = character.position;
        quaternion rotation = character.rotation;
        _maskedChannels =
            _markers.AddTrajectory(query, weights, origin, rotation, joints, SynthesisInputRole.Request, Inputs) +
            _markers.AddPose(query, weights, origin, rotation, joints, SynthesisInputRole.Request, Inputs);
        _markers.AddTrajectory(answer, weights, origin, rotation, joints, SynthesisInputRole.Answer, Inputs);
        _markers.AddPose(answer, weights, origin, rotation, joints, SynthesisInputRole.Answer, Inputs);

        _heldCost = Displayable(heldCost);
        _answerCost = Displayable(answerCost);
    }

    /// <summary>Refreshes the readouts at the end of a tick.</summary>
    /// <param name="frame">The database frame playing, or -1 when the state is not a database row.</param>
    public void EndTick(int frame)
    {
        Inputs.ClearReadouts();
        Inputs.AddReadout("held cost", _heldCost, "0.000");
        Inputs.AddReadout("match cost", _answerCost, "0.000");

        if (frame >= 0)
        {
            var clipIndex = _poseSet.GetAnimationClipIndex(frame);
            var clip = _poseSet.GetAnimationClip(clipIndex);
            Inputs.AddReadout("frame", frame, "0");
            if (_clipNames != null) Inputs.AddReadout("clip", _clipNames[clipIndex]);
            else Inputs.AddReadout("clip", clipIndex, "0");
            Inputs.AddReadout("clip frame", frame - clip.Start, "0");
        }

        Inputs.AddReadout("masked channels", _maskedChannels, "0");
    }

    private static float Displayable(float cost) => cost >= float.MaxValue ? float.NaN : cost;

    /// <summary>
    /// The database stores clips in the config's order, each followed by its mirror when mirroring is
    /// on. A clip that failed to bake breaks that correspondence, which the count check catches.
    /// </summary>
    private static string[] ClipNames(MotionMatchingData mmData, PoseSet poseSet)
    {
        var clips = mmData.AnimationClips;
        var perClip = mmData.MirrorClips ? 2 : 1;
        if (clips == null || poseSet.NumberClips != clips.Count * perClip) return null;

        var names = new string[poseSet.NumberClips];
        for (var i = 0; i < names.Length; i++)
        {
            var clip = clips[i / perClip];
            if (clip == null) return null;
            names[i] = i % perClip == 1 ? clip.name + " (mirrored)" : clip.name;
        }

        return names;
    }
}
}
