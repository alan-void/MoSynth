using System.Collections.Generic;

namespace AnimationTools
{
/// <summary>
/// Pure metrics for what one synthesis tick costs. See <see cref="SynthesisCostMetricsResult"/> for
/// what each field means and the caveats around comparing it across machines.
/// </summary>
public static class SynthesisCostMetrics
{
    /// <summary>
    /// Evaluates cost metrics for one recorded run. Inputs are already filtered to the analysis
    /// window by the caller — unlike the other calculators, there's no settle-time cutoff here.
    /// </summary>
    /// <param name="stageNames">Concrete stage type name per pipeline stage, in pipeline order.</param>
    /// <param name="stageMilliseconds">Per-stage cost per tick; <c>stageMilliseconds[s][i]</c> is stage
    /// <paramref name="stageNames"/>[s]'s cost on analysis tick i. Length must match
    /// <paramref name="stageNames"/>.</param>
    /// <param name="totalMilliseconds">Total time across all stages' Apply() per tick.</param>
    /// <param name="gcBytes">Managed bytes allocated per tick, or null if the profiler recorder was
    /// unavailable for this run. Individual entries may be NaN for the same reason.</param>
    public static SynthesisCostMetricsResult Evaluate(
        string[] stageNames,
        float[][] stageMilliseconds,
        float[] totalMilliseconds,
        float[] gcBytes)
    {
        var result = new SynthesisCostMetricsResult();

        if (totalMilliseconds == null || totalMilliseconds.Length == 0)
        {
            result.ticksEvaluated = 0;
            result.applyMsMean = float.NaN;
            result.applyMsP50 = float.NaN;
            result.applyMsP95 = float.NaN;
            result.applyMsMax = float.NaN;
            result.gcBytesPerTick = float.NaN;
            return result;
        }

        result.ticksEvaluated = totalMilliseconds.Length;
        result.applyMsMean = MetricsMath.Mean(totalMilliseconds);
        result.applyMsP50 = MetricsMath.Percentile(totalMilliseconds, 0.5f);
        result.applyMsP95 = MetricsMath.Percentile(totalMilliseconds, 0.95f);
        result.applyMsMax = MetricsMath.Max(totalMilliseconds);

        result.gcBytesPerTick = MeanFinite(gcBytes);

        for (var s = 0; s < stageNames.Length; s++)
        {
            var costs = stageMilliseconds[s];
            result.stages.Add(new StageCostBreakdown
            {
                stage = stageNames[s],
                meanMs = MetricsMath.Mean(costs),
                p95Ms = MetricsMath.Percentile(costs, 0.95f),
                maxMs = MetricsMath.Max(costs),
            });
        }

        return result;
    }

    /// <summary>Mean of the finite entries in <paramref name="values"/>. NaN if there are none, or the
    /// array itself is null — the profiler recorder being unavailable for a whole run looks the same
    /// as it dropping every sample.</summary>
    private static float MeanFinite(float[] values)
    {
        if (values == null) return float.NaN;

        var sum = 0.0;
        var count = 0;
        for (var i = 0; i < values.Length; i++)
        {
            if (float.IsNaN(values[i])) continue;
            sum += values[i];
            count++;
        }

        return count > 0 ? (float)(sum / count) : float.NaN;
    }
}
}
