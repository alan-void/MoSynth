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
/// Computed from the recording on disk, so a report can be rebuilt from the .bin/.json pair alone.
/// Channels are optional: a metric whose columns are missing is left NaN.
/// </remarks>
public static class BenchmarkRunEvaluator
{
    /// <summary>Recording channel holding one contact flag per slot.</summary>
    public const string ContactsChannel = "contacts";

    /// <summary>Recording channel holding the world position of contact slot <paramref name="slot"/>'s bone.</summary>
    public static string ContactPositionChannel(int slot) => $"contact{slot}";

    /// <summary>
    /// Fills <paramref name="result"/>'s metric blocks from the recording at
    /// <paramref name="manifestPath"/>. Throws only if the recording itself cannot be read; a
    /// missing channel degrades the affected metric to NaN instead.
    /// </summary>
    /// <param name="floor">Each contact bone's floor height, or null; without it the metrics judged
    /// against that height are NaN.</param>
    public static void Evaluate(
        string manifestPath,
        SynthesisBenchmarkConfig config,
        float targetSpeed,
        SplineContainer spline,
        ContactFloor floor,
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
            var contactPositions = ReadContactPositions(reader, frameCount);
            if (contactPositions == null)
            {
                Debug.LogWarning($"[Benchmark] \"{manifestPath}\" records no contact bone positions; footskate is unavailable.");
            }

            float[] restHeights = null, meanContactHeights = null;
            if (contactPositions != null)
                ResolveContactHeights(reader, contactPositions.Length, floor, out restHeights, out meanContactHeights);

            result.motionQuality = MotionQualityMetricsCalculator.Evaluate(
                rootPositions,
                contactPositions,
                ReadContactFlags(reader, contactPositions?.Length ?? 0, frameCount),
                ReadFlags(reader, "discontinuity", frameCount),
                times,
                config.settleTime,
                restHeights,
                meanContactHeights,
                config.groundContact);
        }
        else
        {
            // Every motion-quality figure is anchored on the root's trajectory.
            Debug.LogWarning($"[Benchmark] \"{manifestPath}\" has no \"root\" channel; motion quality is unavailable.");
        }

        result.cost = EvaluateCost(reader, times, config.settleTime);
    }

    /// <summary>
    /// Reduces the cost columns over the analysis window. Settle frames are dropped because a
    /// run's first ticks carry one-off setup (Python imports, cache warm-up).
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

    /// <summary>
    /// Each contact slot's rest and mean contact heights, found by the bone name its position channel
    /// records. Both null when there is no floor or any slot's bone has no height in it.
    /// </summary>
    private static void ResolveContactHeights(RecordingReader reader, int slotCount, ContactFloor floor,
        out float[] restHeights, out float[] meanContactHeights)
    {
        restHeights = null;
        meanContactHeights = null;
        if (floor == null) return;

        var rest = new float[slotCount];
        var mean = new float[slotCount];
        for (var slot = 0; slot < slotCount; slot++)
        {
            reader.TryGetChannel(ContactPositionChannel(slot), out var channel);
            var boneName = channel?.bone;
            if (!floor.TryGetHeights(boneName, out rest[slot], out mean[slot]))
            {
                Debug.LogWarning($"[Benchmark] The ground reference has no contact height for \"{boneName}\"; " +
                                 "footskate judged against it is unavailable.");
                return;
            }
        }

        restHeights = rest;
        meanContactHeights = mean;
    }

    private static float3[] ReadFloat3(RecordingReader reader, string channelName, int frameCount)
    {
        if (!reader.TryGetChannel(channelName, out var channel)) return null;

        var values = new float3[frameCount];
        for (var f = 0; f < frameCount; f++) values[f] = reader.GetFloat3(f, channel);
        return values;
    }

    /// <summary>
    /// Every contact slot's bone position, indexed <c>[slot][frame]</c>, read from the consecutive
    /// <see cref="ContactPositionChannel"/>s starting at slot 0. Null when there is none.
    /// </summary>
    private static float3[][] ReadContactPositions(RecordingReader reader, int frameCount)
    {
        var slotPositions = new List<float3[]>();
        while (true)
        {
            var positions = ReadFloat3(reader, ContactPositionChannel(slotPositions.Count), frameCount);
            if (positions == null) break;

            slotPositions.Add(positions);
        }

        return slotPositions.Count > 0 ? slotPositions.ToArray() : null;
    }

    /// <summary>
    /// The method's own contact flags, indexed <c>[slot][frame]</c>. Null when the recording has no
    /// contact channel or its slot count differs from <paramref name="slotCount"/>.
    /// </summary>
    private static bool[][] ReadContactFlags(RecordingReader reader, int slotCount, int frameCount)
    {
        if (slotCount == 0 || !reader.TryGetChannel(ContactsChannel, out var channel)) return null;
        if (channel.floatCount != slotCount) return null;

        var slotContacts = new bool[slotCount][];
        for (var slot = 0; slot < slotCount; slot++)
        {
            slotContacts[slot] = new bool[frameCount];
            for (var f = 0; f < frameCount; f++)
            {
                slotContacts[slot][f] = reader.GetFloat(f, channel, slot) > 0.5f;
            }
        }

        return slotContacts;
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
