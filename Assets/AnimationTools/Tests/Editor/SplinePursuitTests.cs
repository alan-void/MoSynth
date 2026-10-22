using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Splines;

namespace AnimationTools.Tests
{
public class SplinePursuitTests
{
    private const float Tolerance = 1e-2f;

    /// <summary>Ten metres along +x.</summary>
    private static Spline BuildStraightSpline()
    {
        var spline = new Spline();
        spline.Add(new BezierKnot(new float3(0f, 0f, 0f)), TangentMode.Linear);
        spline.Add(new BezierKnot(new float3(10f, 0f, 0f)), TangentMode.Linear);
        return spline;
    }

    /// <summary>A 40 m square loop: +x, then +z, then -x, then back down -z to the start.</summary>
    private static Spline BuildClosedSquareSpline()
    {
        var spline = new Spline();
        spline.Add(new BezierKnot(new float3(0f, 0f, 0f)), TangentMode.Linear);
        spline.Add(new BezierKnot(new float3(10f, 0f, 0f)), TangentMode.Linear);
        spline.Add(new BezierKnot(new float3(10f, 0f, 10f)), TangentMode.Linear);
        spline.Add(new BezierKnot(new float3(0f, 0f, 10f)), TangentMode.Linear);
        spline.Closed = true;
        return spline;
    }

    private static SplinePursuit Pursue(Spline spline, float3 character) =>
        Pursue(spline, float4x4.identity, character);

    private static SplinePursuit Pursue(Spline spline, float4x4 localToWorld, float3 character)
    {
        var pursuit = new SplinePursuit();
        Assert.IsTrue(pursuit.Update(spline, localToWorld, character));
        return pursuit;
    }

    private static void AssertNear(float3 expected, float3 actual, string message = null) =>
        Assert.That(math.distance(expected, actual), Is.LessThan(Tolerance),
            $"{message} expected {expected}, got {actual}");

    private static void AssertNear(float2 expected, float2 actual, string message = null) =>
        Assert.That(math.distance(expected, actual), Is.LessThan(Tolerance),
            $"{message} expected {expected}, got {actual}");

    [Test]
    public void ProjectsTheCharacterOntoTheNearestPoint()
    {
        var pursuit = Pursue(BuildStraightSpline(), new float3(3f, 0f, 2f));

        Assert.AreEqual(3f, pursuit.NearestDistance, Tolerance);
        AssertNear(new float3(3f, 0f, 0f), pursuit.NearestWorld);
        Assert.AreEqual(10f, pursuit.Length, Tolerance);
        Assert.IsFalse(pursuit.Closed);
    }

    [Test]
    public void ProjectsThroughTheContainerTransform()
    {
        var shifted = float4x4.Translate(new float3(5f, 0f, -4f));
        var pursuit = Pursue(BuildStraightSpline(), shifted, new float3(8f, 0f, -3f));

        Assert.AreEqual(3f, pursuit.NearestDistance, Tolerance);
        AssertNear(new float3(8f, 0f, -4f), pursuit.NearestWorld);
        AssertNear(new float3(10f, 0f, -4f), pursuit.PositionAt(5f), "sampled positions are world-space too:");
    }

    [Test]
    public void SteersTowardTheLookaheadPointPastTheProjection()
    {
        var character = new float3(3f, 0f, 2f);
        var pursuit = Pursue(BuildStraightSpline(), character);

        AssertNear(new float3(4.5f, 0f, 0f), pursuit.TargetWorld(1.5f));
        AssertNear(new float3(1.5f, 0f, -2f), pursuit.DesiredWorldDirection(character, 1.5f));
    }

    [Test]
    public void DesiredDirectionStaysOnTheGroundPlane()
    {
        var character = new float3(3f, 1.7f, 0f);
        var pursuit = Pursue(BuildStraightSpline(), character);

        Assert.AreEqual(0f, pursuit.DesiredWorldDirection(character, 1.5f).y, 1e-6f);
    }

    [Test]
    public void SamplesAreLaidFromTheProjectionAtTheTargetSpeed()
    {
        // Two metres off the path: the samples start from where the character projects, not from it.
        var pursuit = Pursue(BuildStraightSpline(), new float3(2f, 0f, 2f));

        AssertNear(new float3(3f, 0f, 0f), pursuit.PositionAt(pursuit.DistanceAt(1f, 1f)));
        AssertNear(new float3(4f, 0f, 0f), pursuit.PositionAt(pursuit.DistanceAt(1f, 2f)), "double the speed:");
        AssertNear(new float3(1.5f, 0f, 0f), pursuit.PositionAt(pursuit.DistanceAt(-0.5f, 1f)), "a past horizon:");
        AssertNear(new float2(1f, 0f), pursuit.DirectionAt(pursuit.DistanceAt(1f, 1f)));
    }

    [Test]
    public void OpenPathClampsEveryHorizonAtItsEnd()
    {
        var character = new float3(9.5f, 0f, 0f);
        var pursuit = Pursue(BuildStraightSpline(), character);

        AssertNear(new float3(10f, 0f, 0f), pursuit.TargetWorld(1.5f), "lookahead:");
        AssertNear(new float3(10f, 0f, 0f), pursuit.PositionAt(pursuit.DistanceAt(1f, 1f)), "near horizon:");
        AssertNear(new float3(10f, 0f, 0f), pursuit.PositionAt(pursuit.DistanceAt(3f, 1f)), "far horizon:");
        AssertNear(new float3(0f, 0f, 0f), pursuit.PositionAt(-5f), "before the start:");
    }

    [Test]
    public void OpenPathEndKeepsTheLastDirectionOfTravel()
    {
        var pursuit = Pursue(BuildStraightSpline(), new float3(9.5f, 0f, 0f));

        AssertNear(new float2(1f, 0f), pursuit.DirectionAt(10f));
        AssertNear(new float2(1f, 0f), pursuit.DirectionAt(25f), "past the end:");
    }

    [Test]
    public void OpenPathAsksTheCharacterToStopAtItsEnd()
    {
        var character = new float3(9.5f, 0f, 0f);
        var pursuit = Pursue(BuildStraightSpline(), character);

        Assert.AreEqual(0.5f, pursuit.RemainingDistance, Tolerance);
        Assert.AreEqual(1f / 3f, pursuit.SpeedToward(1f, 1.5f), Tolerance);

        var arrived = Pursue(BuildStraightSpline(), new float3(10f, 0f, 0f));
        Assert.AreEqual(0f, arrived.SpeedToward(1f, 1.5f), Tolerance);
        Assert.That(math.length(arrived.DesiredWorldDirection(new float3(10f, 0f, 0f), 1.5f)),
            Is.LessThan(Tolerance), "arrived, so there is no heading left to ask for");
    }

    [Test]
    public void ClosedPathWrapsBothWaysPastTheSeam()
    {
        var ahead = Pursue(BuildClosedSquareSpline(), new float3(9f, 0f, -0.5f));
        Assert.AreEqual(9f, ahead.NearestDistance, Tolerance);
        AssertNear(new float3(10f, 0f, 1f), ahead.PositionAt(ahead.DistanceAt(2f, 1f)), "round the corner:");
        AssertNear(new float2(0f, 1f), ahead.DirectionAt(ahead.DistanceAt(2f, 1f)));

        var behind = Pursue(BuildClosedSquareSpline(), new float3(1f, 0f, -0.5f));
        AssertNear(new float3(0f, 0f, 1f), behind.PositionAt(behind.DistanceAt(-2f, 1f)), "back past the seam:");
        AssertNear(new float3(0f, 0f, 0f), behind.PositionAt(behind.DistanceAt(39f, 1f)), "forward past the seam:");
    }

    [Test]
    public void ClosedPathNeverAsksToSlowDown()
    {
        var pursuit = Pursue(BuildClosedSquareSpline(), new float3(1f, 0f, 0f));

        Assert.IsTrue(float.IsPositiveInfinity(pursuit.RemainingDistance));
        Assert.AreEqual(1.25f, pursuit.SpeedToward(1.25f, 1.5f), 1e-6f);
    }

    [Test]
    public void FollowsTheCharacterAlongThePath()
    {
        var spline = BuildStraightSpline();
        var pursuit = new SplinePursuit();

        for (var x = 1f; x <= 9f; x += 0.5f)
        {
            Assert.IsTrue(pursuit.Update(spline, float4x4.identity, new float3(x, 0f, 0.3f)));
            Assert.AreEqual(x, pursuit.NearestDistance, Tolerance, $"at x = {x}");
        }
    }

    [Test]
    public void NoUsablePathReportsNoPath()
    {
        var pursuit = new SplinePursuit();
        Assert.IsFalse(pursuit.Update(null, float4x4.identity, float3.zero));
        Assert.IsFalse(pursuit.HasPath);

        var single = new Spline();
        single.Add(new BezierKnot(float3.zero));
        Assert.IsFalse(pursuit.Update(single, float4x4.identity, float3.zero));
        Assert.AreEqual(0f, pursuit.RemainingDistance);
    }

    [Test]
    public void ResetForgetsThePath()
    {
        var pursuit = Pursue(BuildStraightSpline(), new float3(3f, 0f, 0f));
        pursuit.Reset();

        Assert.IsFalse(pursuit.HasPath);
    }

    [Test]
    public void FoldedDistancesWrapAClosedPathAndClampAnOpenOne()
    {
        Assert.AreEqual(5f, SplineFold.Distance(45f, 40f, true), 1e-5f);
        Assert.AreEqual(38f, SplineFold.Distance(-2f, 40f, true), 1e-5f);
        Assert.AreEqual(40f, SplineFold.Distance(45f, 40f, false), 1e-5f);
        Assert.AreEqual(0f, SplineFold.Distance(-2f, 40f, false), 1e-5f);
        Assert.AreEqual(12f, SplineFold.Distance(12f, 40f, false), 1e-5f);
    }
}
}
