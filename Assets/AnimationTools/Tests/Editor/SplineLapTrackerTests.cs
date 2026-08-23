using NUnit.Framework;

namespace AnimationTools.Tests
{
public class SplineLapTrackerTests
{
    private const float Tolerance = 1e-5f;

    private static SplineLapTracker Feed(bool closed, params float[] samples)
    {
        var tracker = new SplineLapTracker(closed);
        foreach (var t in samples) tracker.Sample(t);
        return tracker;
    }

    [Test]
    public void FirstSampleOnlyAnchors()
    {
        var tracker = Feed(closed: true, 0.4f);
        Assert.AreEqual(0f, tracker.Progress, Tolerance);
        Assert.AreEqual(0.4f, tracker.LatestT, Tolerance);
    }

    [Test]
    public void ClosedSplineAccumulatesForwardProgress()
    {
        var tracker = Feed(closed: true, 0f, 0.25f, 0.5f, 0.75f);
        Assert.AreEqual(0.75f, tracker.Progress, Tolerance);
    }

    [Test]
    public void ClosedSplineCountsAFullLapAcrossTheSeam()
    {
        var tracker = Feed(closed: true, 0f, 0.25f, 0.5f, 0.75f, 0f);
        Assert.AreEqual(1f, tracker.Progress, Tolerance);
        Assert.IsTrue(tracker.HasCompleted(1f));
    }

    [Test]
    public void ClosedSplineCountsMultipleLaps()
    {
        var tracker = Feed(closed: true, 0f, 0.3f, 0.6f, 0.9f, 0.2f, 0.5f, 0.8f, 0.1f);
        Assert.AreEqual(2.1f, tracker.Progress, Tolerance);
        Assert.IsTrue(tracker.HasCompleted(2f));
        Assert.IsFalse(tracker.HasCompleted(3f));
    }

    [Test]
    public void BackwardJitterSubtractsRatherThanCountingAsALap()
    {
        // A synthesized root drifts backwards a little every gait cycle; that must not read as
        // progress in the other direction around the loop.
        var tracker = Feed(closed: true, 0f, 0.10f, 0.09f, 0.20f);
        Assert.AreEqual(0.20f, tracker.Progress, Tolerance);
    }

    [Test]
    public void RunningBackwardsGivesNegativeProgressAndNeverCompletes()
    {
        var tracker = Feed(closed: true, 0.5f, 0.4f, 0.3f);
        Assert.Less(tracker.Progress, 0f);
        Assert.IsFalse(tracker.HasCompleted(1f));
    }

    [Test]
    public void OpenSplineProgressIsTheParameterItself()
    {
        var tracker = Feed(closed: false, 0f, 0.25f, 0.6f);
        Assert.AreEqual(0.6f, tracker.Progress, Tolerance);
        Assert.IsFalse(tracker.HasCompleted(1f));
    }

    [Test]
    public void OpenSplineCompletesAtTheFarEnd()
    {
        var tracker = Feed(closed: false, 0f, SplineLapTracker.OpenSplineEndT);
        Assert.IsTrue(tracker.HasCompleted(1f));
    }

    [Test]
    public void OpenSplineIgnoresTheRequestedLapCount()
    {
        // An open path cannot be lapped, so reaching the end is the only completion it has.
        var tracker = Feed(closed: false, 0f, 1f);
        Assert.IsTrue(tracker.HasCompleted(3f));
    }

    [Test]
    public void ResetDropsProgressAndReAnchors()
    {
        var tracker = Feed(closed: true, 0f, 0.4f);
        tracker.Reset();
        Assert.AreEqual(0f, tracker.Progress, Tolerance);

        // The sample after a Reset only re-anchors, so no delta is taken against the pre-Reset value.
        tracker.Sample(0.4f);
        Assert.AreEqual(0f, tracker.Progress, Tolerance);

        tracker.Sample(0.5f);
        Assert.AreEqual(0.1f, tracker.Progress, Tolerance);
    }

    [Test]
    public void WrapDeltaFoldsLargeStepsIntoTheShortWayRound()
    {
        Assert.AreEqual(0.1f, SplineLapTracker.WrapDelta(0.1f), Tolerance);
        Assert.AreEqual(-0.1f, SplineLapTracker.WrapDelta(-0.1f), Tolerance);
        Assert.AreEqual(0.1f, SplineLapTracker.WrapDelta(-0.9f), Tolerance);
        Assert.AreEqual(-0.1f, SplineLapTracker.WrapDelta(0.9f), Tolerance);
    }
}
}
