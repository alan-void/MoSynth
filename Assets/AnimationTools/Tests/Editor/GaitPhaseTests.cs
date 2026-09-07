using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;

namespace AnimationTools.Tests
{
/// <summary>
/// Turning footfalls into a gait phase.
/// </summary>
/// <remarks>
/// This is the only implementation of the rule, and the database carries its output to the training
/// set, so a mistake here does not throw — it makes a trained network wrong. The stretches with no
/// anchors in them are the interesting half, and have their own section below.
/// </remarks>
public class GaitPhaseTests
{
    private const float FrameTime = 1f / 30f;
    private const float Tolerance = 1e-4f;
    private const float StandingSpeed = 0.1f;
    private const float StandingPeriod = 1f;

    /// <summary>Radians per frame a standing stretch sweeps at, for these settings.</summary>
    private const float StandingSlope = GaitPhase.Tau * FrameTime / StandingPeriod;

    /// <summary>Radians per second the same sweep reports.</summary>
    private const float StandingRate = GaitPhase.Tau / StandingPeriod;

    private static List<GaitPhase.Footfall> Footfalls(params (int frame, GaitPhase.Foot foot)[] entries)
    {
        var footfalls = new List<GaitPhase.Footfall>();
        foreach (var (frame, foot) in entries) footfalls.Add(new GaitPhase.Footfall(frame, foot));
        return footfalls;
    }

    private static void Evaluate(List<GaitPhase.Footfall> footfalls, int frameCount,
        out float[] phase, out float[] rate)
    {
        phase = new float[frameCount];
        rate = new float[frameCount];
        GaitPhase.Evaluate(footfalls, FrameTime, phase, rate);
    }

    private static void Evaluate(List<GaitPhase.Footfall> footfalls, float[] speed,
        out float[] phase, out float[] rate)
    {
        phase = new float[speed.Length];
        rate = new float[speed.Length];
        GaitPhase.Evaluate(footfalls, FrameTime, speed, StandingSpeed, StandingPeriod, phase, rate);
    }

    /// <summary>Speeds for a character standing still except over <c>[from, to)</c>.</summary>
    private static float[] MovingBetween(int frameCount, int from, int to)
    {
        var speed = new float[frameCount];
        for (var frame = from; frame < to; frame++) speed[frame] = 1f;
        return speed;
    }

    [Test]
    public void FootfallsLandOnTheHalfCycleMarks()
    {
        // Right at 10, left at 20, right at 30: the cycle is zeroed on a right footfall, so those
        // are 0, pi and 2pi — which wraps back to 0.
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Left),
            (30, GaitPhase.Foot.Right));

        Evaluate(footfalls, 40, out var phase, out _);

        Assert.That(phase[10], Is.EqualTo(0f).Within(Tolerance));
        Assert.That(phase[20], Is.EqualTo(math.PI).Within(Tolerance));
        Assert.That(phase[30], Is.EqualTo(0f).Within(Tolerance));
    }

    [Test]
    public void AClipOpeningOnALeftFootfallIsOffsetByHalfACycle()
    {
        // Otherwise "phase 0" would mean a different pose depending on which foot a clip started on.
        var footfalls = Footfalls((10, GaitPhase.Foot.Left), (20, GaitPhase.Foot.Right));

        Evaluate(footfalls, 30, out var phase, out _);

        Assert.That(phase[10], Is.EqualTo(math.PI).Within(Tolerance));
        Assert.That(phase[20], Is.EqualTo(0f).Within(Tolerance));
    }

    [Test]
    public void PhaseIsLinearBetweenFootfalls()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Left));

        Evaluate(footfalls, 30, out var phase, out _);

        Assert.That(phase[15], Is.EqualTo(0.5f * math.PI).Within(Tolerance));
    }

    [Test]
    public void TheSameFootTwiceAdvancesAWholeCycle()
    {
        // What a missed contact looks like: the other foot fell in between and was not detected.
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (30, GaitPhase.Foot.Right));

        Evaluate(footfalls, 40, out var phase, out var rate);

        Assert.That(phase[20], Is.EqualTo(math.PI).Within(Tolerance), "half way is half a cycle");
        Assert.That(rate[20] * 20 * FrameTime, Is.EqualTo(GaitPhase.Tau).Within(Tolerance));
    }

    [Test]
    public void WrappedPhaseStaysInsideOneCycle()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Left),
            (40, GaitPhase.Foot.Right), (60, GaitPhase.Foot.Left), (80, GaitPhase.Foot.Right));

        Evaluate(footfalls, 120, out var phase, out _);

        foreach (var value in phase)
        {
            Assert.That(value, Is.GreaterThanOrEqualTo(0f));
            Assert.That(value, Is.LessThan(GaitPhase.Tau));
        }
    }

    [Test]
    public void TheRateIsTheSlopeOfTheSegment()
    {
        // Twenty frames for half a cycle, so pi over (20/30) seconds.
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (30, GaitPhase.Foot.Left));

        Evaluate(footfalls, 40, out _, out var rate);

        Assert.That(rate[15], Is.EqualTo(math.PI / (20f * FrameTime)).Within(Tolerance));
    }

    // --- Stretches with no anchors in them -------------------------------------------------------
    //
    // Standing and a missed contact look identical in the anchors alone, so how fast the character
    // was travelling is what separates them. With no speed measured at all, everything is held.

    [Test]
    public void FramesOutsideTheAnchorsReportNoCycle()
    {
        // This is what stops a standing intro being given phase it never walked. On the untrimmed
        // walk1_subject1 clip the first footfall is at frame 132, and extrapolating backwards was
        // inventing 2.87 complete cycles over 4.4 seconds of standing still.
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (30, GaitPhase.Foot.Left));

        Evaluate(footfalls, 50, out _, out var rate);

        Assert.That(rate[0], Is.EqualTo(0f), "before the first footfall");
        Assert.That(rate[9], Is.EqualTo(0f), "still before the first footfall");
        Assert.That(rate[20], Is.Not.EqualTo(0f), "between them");
        Assert.That(rate[30], Is.EqualTo(0f), "on and after the last footfall");
        Assert.That(rate[49], Is.EqualTo(0f), "after the last footfall");
    }

    [Test]
    public void PhaseIsHeldRatherThanExtrapolatedOutsideTheAnchors()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (30, GaitPhase.Foot.Left));

        Evaluate(footfalls, 50, out var phase, out _);

        Assert.That(phase[0], Is.EqualTo(phase[10]).Within(Tolerance));
        Assert.That(phase[49], Is.EqualTo(phase[30]).Within(Tolerance));
    }

    [Test]
    public void AStandingLeadInSweepsAndLandsOnTheFirstFootfall()
    {
        // The character stands, then walks from frame 10. Standing frames still teach a model a
        // stationary pose, so they sweep the cycle rather than sitting at one point of it — but the
        // sweep is wound back from the footfall, so the phase stays continuous where the walk starts.
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (30, GaitPhase.Foot.Left));

        Evaluate(footfalls, MovingBetween(50, 10, 50), out var phase, out var rate);

        Assert.That(rate[0], Is.EqualTo(StandingRate).Within(Tolerance));
        Assert.That(rate[9], Is.EqualTo(StandingRate).Within(Tolerance));
        Assert.That(phase[10], Is.EqualTo(0f).Within(Tolerance), "the anchor itself");
        Assert.That(phase[9], Is.EqualTo(GaitPhase.Tau - StandingSlope).Within(Tolerance),
            "one frame short of it, wound back at the standing rate");
    }

    [Test]
    public void AStandingLeadOutSweepsOnFromTheLastFootfall()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (30, GaitPhase.Foot.Left));

        Evaluate(footfalls, MovingBetween(50, 0, 30), out var phase, out var rate);

        Assert.That(rate[30], Is.EqualTo(StandingRate).Within(Tolerance));
        Assert.That(rate[49], Is.EqualTo(StandingRate).Within(Tolerance));
        Assert.That(phase[30], Is.EqualTo(math.PI).Within(Tolerance), "the anchor itself");
        Assert.That(phase[31], Is.EqualTo(math.PI + StandingSlope).Within(Tolerance));
    }

    [Test]
    public void AnAnchorlessStretchTheCharacterMovedThroughIsStillRefused()
    {
        // A missed contact leaves the same hole in the anchors that standing does, and it must stay
        // a missed contact: there is no evidence of a cycle, so the frames are held and dropped.
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (30, GaitPhase.Foot.Left));

        Evaluate(footfalls, MovingBetween(50, 0, 50), out var phase, out var rate);

        Assert.That(rate[0], Is.EqualTo(0f));
        Assert.That(rate[49], Is.EqualTo(0f));
        Assert.That(phase[0], Is.EqualTo(phase[10]).Within(Tolerance));
    }

    [Test]
    public void AStandingGapAdvancesWholeCyclesRatherThanOneSlowStride()
    {
        // Four seconds of standing between two footfalls is not one stride taken slowly. The gap
        // sweeps near the standing rate instead, by whole cycles, so it still lands on the foot the
        // anchor names.
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (130, GaitPhase.Foot.Left));

        Evaluate(footfalls, MovingBetween(140, 0, 10), out var phase, out var rate);

        var oneStride = math.PI / (120f * FrameTime);
        Assert.That(rate[50], Is.EqualTo(StandingRate).Within(StandingRate * 0.25f));
        Assert.That(rate[50], Is.GreaterThan(oneStride * 2f));
        Assert.That(phase[130], Is.EqualTo(math.PI).Within(Tolerance), "still the left foot");
    }

    [Test]
    public void AMovingGapKeepsTheMissedContactRule()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (130, GaitPhase.Foot.Left));

        Evaluate(footfalls, MovingBetween(140, 0, 140), out _, out var rate);

        Assert.That(rate[50], Is.EqualTo(math.PI / (120f * FrameTime)).Within(Tolerance));
    }

    [Test]
    public void AClipStoodThroughEndToEndSweepsFromZero()
    {
        // No anchor to line up with, which is what decides the standing rate is a fixed period
        // rather than one carried from a neighbouring segment: here there is no neighbour.
        Evaluate(new List<GaitPhase.Footfall>(), new float[30], out var phase, out var rate);

        Assert.That(rate[0], Is.EqualTo(StandingRate).Within(Tolerance));
        Assert.That(phase[0], Is.EqualTo(0f).Within(Tolerance));
        Assert.That(phase[1], Is.EqualTo(StandingSlope).Within(Tolerance));
    }

    [Test]
    public void AZeroStandingPeriodLeavesEverythingHeld()
    {
        // The switch that turns the rule off, for a caller that would rather drop those frames.
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (30, GaitPhase.Foot.Left));
        var phase = new float[50];
        var rate = new float[50];

        GaitPhase.Evaluate(footfalls, FrameTime, new float[50], StandingSpeed, 0f, phase, rate);

        Assert.That(rate[0], Is.EqualTo(0f));
        Assert.That(rate[49], Is.EqualTo(0f));
    }

    // --- Degenerate input -------------------------------------------------------------------------

    [Test]
    public void FewerThanTwoFootfallsIsReportedAsNoCycleRatherThanGuessed()
    {
        Evaluate(Footfalls((10, GaitPhase.Foot.Right)), 30, out var phase, out var rate);

        foreach (var value in rate) Assert.That(value, Is.EqualTo(0f));
        foreach (var value in phase) Assert.That(value, Is.EqualTo(0f));
    }

    [Test]
    public void NoFootfallsIsReportedAsNoCycle()
    {
        Evaluate(new List<GaitPhase.Footfall>(), 30, out _, out var rate);

        foreach (var value in rate) Assert.That(value, Is.EqualTo(0f));
    }

    [Test]
    public void TwoFootfallsOnOneFrameDoNotDivideByZero()
    {
        // A zero-length segment has no defined phase across it. The Python side divides by zero on
        // this input; here the repeat is dropped.
        var footfalls = Footfalls((10, GaitPhase.Foot.Left), (10, GaitPhase.Foot.Right),
            (30, GaitPhase.Foot.Left));

        Evaluate(footfalls, 40, out var phase, out var rate);

        foreach (var value in phase) Assert.That(float.IsFinite(value), Is.True);
        foreach (var value in rate) Assert.That(float.IsFinite(value), Is.True);
        Assert.That(rate[20], Is.Not.EqualTo(0f), "the surviving segment still has a cycle");
    }

    [Test]
    public void FootfallsOutsideTheFrameRangeAreIgnored()
    {
        var footfalls = Footfalls((-5, GaitPhase.Foot.Right), (10, GaitPhase.Foot.Right),
            (30, GaitPhase.Foot.Left), (900, GaitPhase.Foot.Right));

        Evaluate(footfalls, 40, out _, out var rate);

        Assert.That(rate[35], Is.EqualTo(0f), "the out-of-range anchor must not extend the cycle");
        Assert.That(rate[20], Is.Not.EqualTo(0f));
    }

    [Test]
    public void RepeatedFeetAreCounted()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Right),
            (30, GaitPhase.Foot.Left), (40, GaitPhase.Foot.Left), (50, GaitPhase.Foot.Right));

        Assert.That(GaitPhase.CountRepeatedFeet(footfalls), Is.EqualTo(2));
    }
}
}
