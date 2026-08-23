using System;
using System.Collections.Generic;

namespace AnimationTools
{
/// <summary>Per-stage slice of a <see cref="SynthesisCostMetricsResult"/>.</summary>
[Serializable]
public class StageCostBreakdown
{
    /// <summary>Concrete stage type name, as recorded by StageCostChannel's manifest entry.</summary>
    public string stage;

    public float meanMs;
    public float p95Ms;
    public float maxMs;
}

/// <summary>
/// Output of <see cref="SynthesisCostMetrics.Evaluate"/>: what one synthesis tick costs.
/// </summary>
/// <remarks>
/// Wall clock inside a (possibly batchmode) Editor is not player performance. It is valid for
/// comparing methods against each other on one machine in one sweep, which is the only claim the
/// benchmark makes — the run's JSON report records the machine and Unity version so the numbers are
/// not accidentally compared across either.
/// </remarks>
[Serializable]
public class SynthesisCostMetricsResult
{
    public int ticksEvaluated;

    /// <summary>Total time in all stages' Apply() for one tick, in milliseconds.</summary>
    public float applyMsMean;

    public float applyMsP50;
    public float applyMsP95;
    public float applyMsMax;

    /// <summary>Mean bytes allocated on the managed heap per tick. NaN when the recording carries no allocation column.</summary>
    public float gcBytesPerTick;

    /// <summary>One entry per pipeline stage, in pipeline order.</summary>
    public List<StageCostBreakdown> stages = new();
}
}
