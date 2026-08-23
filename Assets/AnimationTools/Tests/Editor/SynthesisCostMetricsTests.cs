using NUnit.Framework;

namespace AnimationTools.Tests
{
public class SynthesisCostMetricsTests
{
    [Test]
    public void KnownStageArrays_ProduceExpectedStatsAndOrder()
    {
        var stageNames = new[] { "StageA", "StageB" };
        var stageA = new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
        var stageB = new[] { 0.9f, 1.8f, 2.7f, 3.6f, 4.5f };
        var stageMilliseconds = new[] { stageA, stageB };
        var totalMilliseconds = new[] { 1f, 2f, 3f, 4f, 5f };
        var gcBytes = new[] { 100f, 200f, float.NaN, 300f, 400f };

        var result = SynthesisCostMetrics.Evaluate(stageNames, stageMilliseconds, totalMilliseconds, gcBytes);

        Assert.AreEqual(5, result.ticksEvaluated);
        Assert.AreEqual(3f, result.applyMsMean, 1e-4f);
        Assert.AreEqual(3f, result.applyMsP50, 1e-4f);
        Assert.AreEqual(5f, result.applyMsP95, 1e-4f);
        Assert.AreEqual(5f, result.applyMsMax, 1e-4f);
        Assert.AreEqual(250f, result.gcBytesPerTick, 1e-4f);

        Assert.AreEqual(2, result.stages.Count);

        Assert.AreEqual("StageA", result.stages[0].stage);
        Assert.AreEqual(0.3f, result.stages[0].meanMs, 1e-4f);
        Assert.AreEqual(0.5f, result.stages[0].p95Ms, 1e-4f);
        Assert.AreEqual(0.5f, result.stages[0].maxMs, 1e-4f);

        Assert.AreEqual("StageB", result.stages[1].stage);
        Assert.AreEqual(2.7f, result.stages[1].meanMs, 1e-4f);
        Assert.AreEqual(4.5f, result.stages[1].p95Ms, 1e-4f);
        Assert.AreEqual(4.5f, result.stages[1].maxMs, 1e-4f);
    }

    [Test]
    public void AllNaNGcBytes_GivesNaN()
    {
        var stageNames = new[] { "StageA" };
        var stageMilliseconds = new[] { new[] { 1f, 2f, 3f } };
        var totalMilliseconds = new[] { 1f, 2f, 3f };
        var gcBytes = new[] { float.NaN, float.NaN, float.NaN };

        var result = SynthesisCostMetrics.Evaluate(stageNames, stageMilliseconds, totalMilliseconds, gcBytes);

        Assert.IsNaN(result.gcBytesPerTick);
    }

    [Test]
    public void NullGcBytes_GivesNaN()
    {
        var stageNames = new[] { "StageA" };
        var stageMilliseconds = new[] { new[] { 1f, 2f, 3f } };
        var totalMilliseconds = new[] { 1f, 2f, 3f };

        var result = SynthesisCostMetrics.Evaluate(stageNames, stageMilliseconds, totalMilliseconds, null);

        Assert.IsNaN(result.gcBytesPerTick);
    }

    [Test]
    public void NullTotalMilliseconds_GivesZeroTicksEvaluated()
    {
        var stageNames = new[] { "StageA" };

        var result = SynthesisCostMetrics.Evaluate(stageNames, null, null, null);

        Assert.AreEqual(0, result.ticksEvaluated);
        Assert.IsNaN(result.applyMsMean);
        Assert.IsNaN(result.applyMsP50);
        Assert.IsNaN(result.applyMsP95);
        Assert.IsNaN(result.applyMsMax);
        Assert.IsNaN(result.gcBytesPerTick);
        Assert.AreEqual(0, result.stages.Count);
    }

    [Test]
    public void EmptyTotalMilliseconds_GivesZeroTicksEvaluated()
    {
        var stageNames = new[] { "StageA" };
        var stageMilliseconds = new[] { new float[0] };

        var result = SynthesisCostMetrics.Evaluate(stageNames, stageMilliseconds, new float[0], null);

        Assert.AreEqual(0, result.ticksEvaluated);
        Assert.AreEqual(0, result.stages.Count);
    }
}
}
