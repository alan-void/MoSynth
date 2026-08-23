using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

namespace AnimationTools.Editor
{
/// <summary>
/// Turns one finished run's raw recording into the three metric blocks of a
/// <see cref="BenchmarkRunResult"/>.
/// </summary>
/// <remarks>
/// Everything is computed from the recording on disk rather than from live state, so a report can
/// be rebuilt from the .bin/.json pair alone, and so the calculators stay pure and unit-testable.
/// Channels are looked up by name and treated as optional: a recording made with a narrower channel
/// set still yields whichever metrics its columns support, with the rest left NaN.
/// </remarks>
public static class BenchmarkRunEvaluator
{
    /// <summary>
    /// Fills <paramref name="result"/>'s metric blocks from the recording at
    /// <paramref name="manifestPath"/>. Throws only if the recording itself cannot be read; a
    /// missing channel degrades the affected metric to NaN instead.
    /// </summary>
    public static void Evaluate(
        string manifestPath,
        SynthesisBenchmarkConfig config,
        float targetSpeed,
        SplineContainer spline,
        BenchmarkRunResult result)
    {
        var reader = RecordingReader.Load(manifestPath);
        var frameCount = reader.FrameCount;
        result.framesTotal = frameCount;

        if (frameCount == 0)
        {
            result.error = "recording is empty";
            return;
        }

        var timeChannel = reader.GetChannel("time");
        var times = new float[frameCount];
        for (var f = 0; f < frameCount; f++) times[f] = reader.GetFloat(f, timeChannel, 0);

        var rootPositions = ReadFloat3(reader, "root", frameCount);
        var rootForwards = ReadFloat3(reader, "rootForward", frameCount);

        if (rootPositions != null && rootForwards != null && spline != null)
        {
            result.pathFollowing = PathFollowingMetricsCalculator.Evaluate(
                spline.Spline,
                (float4x4)spline.transform.localToWorldMatrix,
                rootPositions,
                rootForwards,
                times,
                targetSpeed,
                config.settleTime,
                config.speedSmoothingWindow);
        }

        result.pathFollowing.label = $"{result.method} on {result.path}";

        if (rootPositions != null)
        {
            var contacts = ReadContacts(reader, frameCount);
            result.motionQuality = MotionQualityMetricsCalculator.Evaluate(
                rootPositions,
                ReadFloat3(reader, "footL", frameCount),
                ReadFloat3(reader, "footR", frameCount),
                contacts.Left,
                contacts.Right,
                ReadFlags(reader, "discontinuity", frameCount),
                times,
                config.settleTime);
        }
        else
        {
            // Every motion-quality figure is anchored on the root's trajectory, so without it there
            // is nothing to compute rather than something to compute badly.
            Debug.LogWarning($"[Benchmark] \"{manifestPath}\" has no \"root\" channel; motion quality is unavailable.");
        }

        result.cost = EvaluateCost(reader, times, config.settleTime);
    }

    /// <summary>
    /// Reduces the cost columns over the analysis window. The settle frames are dropped here as
    /// well as from the quality metrics, because the first ticks of a run carry one-off setup —
    /// a motion field importing its Python modules, a search warming its caches — that would
    /// otherwise dominate the mean.
    /// </summary>
    private static SynthesisCostMetricsResult EvaluateCost(RecordingReader reader, float[] times, float settleTime)
    {
        if (!reader.TryGetChannel("cost", out var costChannel)) return new SynthesisCostMetricsResult();

        // The channel is laid out as one column per stage, then the total, then the allocation.
        var components = costChannel.components;
        var stageCount = math.max(0, costChannel.floatCount - 2);
        var stageNames = new string[stageCount];
        for (var s = 0; s < stageCount; s++)
        {
            stageNames[s] = components != null && s < components.Length ? components[s] : $"stage{s}";
        }

        var analysisFrames = new List<int>(times.Length);
        for (var f = 0; f < times.Length && f < reader.FrameCount; f++)
        {
            if (times[f] >= settleTime) analysisFrames.Add(f);
        }

        var stageMilliseconds = new float[stageCount][];
        for (var s = 0; s < stageCount; s++) stageMilliseconds[s] = new float[analysisFrames.Count];

        var totals = new float[analysisFrames.Count];
        var gcBytes = new float[analysisFrames.Count];

        for (var i = 0; i < analysisFrames.Count; i++)
        {
            var f = analysisFrames[i];
            for (var s = 0; s < stageCount; s++) stageMilliseconds[s][i] = reader.GetFloat(f, costChannel, s);
            totals[i] = reader.GetFloat(f, costChannel, stageCount);
            gcBytes[i] = reader.GetFloat(f, costChannel, stageCount + 1);
        }

        return SynthesisCostMetrics.Evaluate(stageNames, stageMilliseconds, totals, gcBytes);
    }

    private static float3[] ReadFloat3(RecordingReader reader, string channelName, int frameCount)
    {
        if (!reader.TryGetChannel(channelName, out var channel)) return null;

        var values = new float3[frameCount];
        for (var f = 0; f < frameCount; f++) values[f] = reader.GetFloat3(f, channel);
        return values;
    }

    private static (bool[] Left, bool[] Right) ReadContacts(RecordingReader reader, int frameCount)
    {
        if (!reader.TryGetChannel("contacts", out var channel)) return (null, null);

        var left = new bool[frameCount];
        var right = new bool[frameCount];
        for (var f = 0; f < frameCount; f++)
        {
            left[f] = reader.GetFloat(f, channel, 0) > 0.5f;
            right[f] = reader.GetFloat(f, channel, 1) > 0.5f;
        }

        return (left, right);
    }

    private static bool[] ReadFlags(RecordingReader reader, string channelName, int frameCount)
    {
        if (!reader.TryGetChannel(channelName, out var channel)) return null;

        var flags = new bool[frameCount];
        for (var f = 0; f < frameCount; f++) flags[f] = reader.GetFloat(f, channel, 0) > 0.5f;
        return flags;
    }
}
}
