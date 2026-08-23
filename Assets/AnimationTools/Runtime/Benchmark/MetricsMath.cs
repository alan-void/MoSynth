using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>Small reductions shared by the benchmark metric calculators.</summary>
public static class MetricsMath
{
    /// <summary>
    /// Nearest-rank percentile of <paramref name="values"/>, which is left untouched. NaN for an
    /// empty input. Nearest-rank rather than interpolated because these samples are per-tick costs
    /// and jerks: the interesting figure is a value that actually occurred, not a blend of two.
    /// </summary>
    public static float Percentile(IReadOnlyList<float> values, float quantile)
    {
        if (values == null || values.Count == 0) return float.NaN;

        var sorted = new float[values.Count];
        for (var i = 0; i < sorted.Length; i++) sorted[i] = values[i];
        Array.Sort(sorted);

        var rank = (int)math.ceil(math.clamp(quantile, 0f, 1f) * sorted.Length) - 1;
        return sorted[math.clamp(rank, 0, sorted.Length - 1)];
    }

    /// <summary>Mean of <paramref name="values"/>, accumulated in double. NaN for an empty input.</summary>
    public static float Mean(IReadOnlyList<float> values)
    {
        if (values == null || values.Count == 0) return float.NaN;

        var sum = 0.0;
        for (var i = 0; i < values.Count; i++) sum += values[i];
        return (float)(sum / values.Count);
    }

    /// <summary>Largest of <paramref name="values"/>. NaN for an empty input.</summary>
    public static float Max(IReadOnlyList<float> values)
    {
        if (values == null || values.Count == 0) return float.NaN;

        var max = float.NegativeInfinity;
        for (var i = 0; i < values.Count; i++)
        {
            if (values[i] > max) max = values[i];
        }

        return max;
    }
}
}
