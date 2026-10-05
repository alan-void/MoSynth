using NUnit.Framework;

namespace AnimationTools.Tests
{
public class OpenPathRestWatchTests
{
    private const float Tick = 0.1f;

    [Test]
    public void SteadyProgressNeverRests()
    {
        var watch = new OpenPathRestWatch(restTimeout: 1f, minProgressT: 0.01f);
        for (var i = 0; i < 100; i++) watch.Sample(i * 0.005f, Tick);
        Assert.IsFalse(watch.HasRested);
    }

    [Test]
    public void StandingStillRestsAfterTheTimeout()
    {
        var watch = new OpenPathRestWatch(restTimeout: 1f, minProgressT: 0.01f);
        watch.Sample(0.5f, Tick);
        for (var i = 0; i < 9; i++) watch.Sample(0.5f, Tick);
        Assert.IsFalse(watch.HasRested);

        watch.Sample(0.5f, Tick);
        Assert.IsTrue(watch.HasRested);
    }

    [Test]
    public void PauseShorterThanTheTimeoutDoesNotRest()
    {
        var watch = new OpenPathRestWatch(restTimeout: 1f, minProgressT: 0.01f);
        watch.Sample(0.5f, Tick);
        for (var i = 0; i < 8; i++) watch.Sample(0.5f, Tick);
        watch.Sample(0.52f, Tick);
        for (var i = 0; i < 8; i++) watch.Sample(0.52f, Tick);
        Assert.IsFalse(watch.HasRested);
    }

    [Test]
    public void DriftingBackwardsCountsAsRest()
    {
        var watch = new OpenPathRestWatch(restTimeout: 1f, minProgressT: 0.01f);
        watch.Sample(0.5f, Tick);
        for (var i = 1; i <= 10; i++) watch.Sample(0.5f - i * 0.001f, Tick);
        Assert.IsTrue(watch.HasRested);
    }

    [Test]
    public void CreepingBelowTheMinimumStepCountsAsRest()
    {
        var watch = new OpenPathRestWatch(restTimeout: 1f, minProgressT: 0.01f);
        watch.Sample(0.5f, Tick);
        for (var i = 1; i <= 10; i++) watch.Sample(0.5f + i * 0.0009f, Tick);
        Assert.IsTrue(watch.HasRested);
    }
}
}
