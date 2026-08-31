using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;

namespace AnimationTools.Tests
{
/// <summary>
/// Turning footfalls into a gait phase.
/// </summary>
/// <remarks>
/// The fixtures mirror <c>Python/tests/test_gait_phase.py</c>, because the two implementations have
/// to agree on this definition and a disagreement does not throw — it just makes a trained network
/// wrong. The one place they deliberately differ is outside the outer anchors, which has its own
/// tests here.
/// </remarks>
public class GaitPhaseTests
{
    private const float FrameTime = 1f / 30f;
    private const float Tolerance = 1e-4f;

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

    // --- The deliberate difference from the Python side ------------------------------------------

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
