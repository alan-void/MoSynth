using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Splines;

namespace AnimationTools.Tests
{
public class SplineProjectorTests
{
    private static Spline BuildStraightSpline()
    {
        var spline = new Spline();
        spline.Add(new BezierKnot(new float3(0f, 0f, 0f)), TangentMode.Linear);
        spline.Add(new BezierKnot(new float3(10f, 0f, 0f)), TangentMode.Linear);
        return spline;
    }

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

    /// <summary>
    /// A lemniscate, matching the FigureEight benchmark path. The point of the fixture is the
    /// crossing at the origin, where the two branches come arbitrarily close together.
    /// </summary>
    private static Spline BuildFigureEightSpline(float scale = 6f, int knots = 12)
    {
        var spline = new Spline();
        for (var i = 0; i < knots; i++)
        {
            var t = 2f * math.PI * i / knots;
            var denominator = 1f + math.sin(t) * math.sin(t);
            spline.Add(
                new BezierKnot(new float3(
                    scale * math.cos(t) / denominator,
                    0f,
                    scale * math.sin(t) * math.cos(t) / denominator)),
                TangentMode.AutoSmooth);
        }

        spline.Closed = true;
        return spline;
    }

    [Test]
    public void FirstProjectionMatchesGlobalSearch()
    {
        var spline = BuildStraightSpline();
        var projector = new SplineProjector();

        Assert.IsFalse(projector.HasAnchor);
        var t = projector.Project(spline, new float3(3f, 0f, 2f));

        Assert.IsTrue(projector.HasAnchor);
        Assert.AreEqual(0.3f, t, 0.01f);
    }

    [Test]
    public void StraightSplineAgreesWithGlobalSearchAlongItsLength()
    {
        var spline = BuildStraightSpline();
        var projector = new SplineProjector();

        for (var i = 0; i <= 20; i++)
        {
            var x = i * 0.5f;
            var t = projector.Project(spline, new float3(x, 0f, 0.25f));
            SplineUtility.GetNearestPoint(spline, new float3(x, 0f, 0.25f), out float3 _, out var expected,
                SplineUtility.PickResolutionMax, 4);
            Assert.AreEqual(expected, t, 0.01f, $"sample {i} at x={x}");
        }
    }

    [Test]
    public void ClosedSplineProgressesMonotonicallyAroundThePerimeter()
    {
        var spline = BuildClosedSquareSpline();
        var projector = new SplineProjector();

        float3 Perimeter(float s)
        {
            s %= 40f;
            if (s < 10f) return new float3(s, 0f, 0f);
            if (s < 20f) return new float3(10f, 0f, s - 10f);
            if (s < 30f) return new float3(30f - s, 0f, 10f);
            return new float3(0f, 0f, 40f - s);
        }

        var previous = projector.Project(spline, Perimeter(0f));
        var wrapped = 0;

        for (var s = 1f; s <= 40f; s += 1f)
        {
            var t = projector.Project(spline, Perimeter(s));
            var delta = t - previous;
            if (delta < -0.5f) { delta += 1f; wrapped++; }

            Assert.Greater(delta, 0f, $"progress went backwards at s={s}");
            Assert.Less(delta, 0.1f, $"progress jumped at s={s}");
            previous = t;
        }

        Assert.AreEqual(1, wrapped, "should have crossed the seam exactly once");
    }

    [Test]
    public void CoarseSamplingStillTracks()
    {
        // The metrics calculator is fed recorded frames that can be seconds apart; at 5 m/s with 2 s
        // spacing that is a quarter of this perimeter per sample. The window scales with the point's
        // own displacement precisely so that this does not lag behind.
        var spline = BuildClosedSquareSpline();
        var projector = new SplineProjector();

        float3 Perimeter(float s)
        {
            s %= 40f;
            if (s < 10f) return new float3(s, 0f, 0f);
            if (s < 20f) return new float3(10f, 0f, s - 10f);
            if (s < 30f) return new float3(30f - s, 0f, 10f);
            return new float3(0f, 0f, 40f - s);
        }

        for (var s = 0f; s <= 80f; s += 10f)
        {
            var t = projector.Project(spline, Perimeter(s));
            var nearest = spline.EvaluatePosition(t);
            Assert.AreEqual(0f, math.distance(nearest, Perimeter(s)), 0.2f, $"lost the path at s={s}");
        }
    }

    [Test]
    public void FigureEightDoesNotJumpBranchesAtTheCrossing()
    {
        // The regression this class exists for. Walking one lobe through the crossing, a global
        // nearest-point search flips to the other branch as the two come together; a continuous one
        // must not.
        var spline = BuildFigureEightSpline();
        var projector = new SplineProjector();

        const int samples = 400;
        var previous = -1f;
        var wrapped = 0;

        for (var i = 0; i <= samples; i++)
        {
            var u = (float)i / samples;
            var point = spline.EvaluatePosition(u);
            var t = projector.Project(spline, point);

            Assert.AreEqual(u, t, 0.02f, $"drifted off the traced branch at u={u:0.000}");

            if (previous >= 0f)
            {
                var delta = t - previous;
                if (delta < -0.5f) { delta += 1f; wrapped++; }
                Assert.Greater(delta, -1e-3f, $"went backwards at u={u:0.000}");
                Assert.Less(delta, 0.1f, $"jumped branches at u={u:0.000}");
            }

            previous = t;
        }

        Assert.AreEqual(0, wrapped, "one pass of the spline should not wrap the seam");
    }

    [Test]
    public void FigureEightGlobalSearchDoesJumpBranches()
    {
        // Establishes that the fixture actually exercises the bug: the same walk, projected globally,
        // lands on the far branch somewhere near the crossing. Without this the test above could pass
        // on a fixture that was never ambiguous.
        var spline = BuildFigureEightSpline();
        var worst = 0f;

        for (var i = 0; i <= 400; i++)
        {
            var u = (float)i / 400;
            var point = spline.EvaluatePosition(u);
            SplineUtility.GetNearestPoint(spline, point, out float3 _, out var t);

            var delta = math.abs(t - u);
            if (delta > 0.5f) delta = 1f - delta;
            if (delta > worst) worst = delta;
        }

        Assert.Greater(worst, 0.05f,
            "the global search was expected to pick the wrong branch on this fixture");
    }

    [Test]
    public void ReacquiresAfterATeleport()
    {
        var spline = BuildClosedSquareSpline();
        var projector = new SplineProjector();

        projector.Project(spline, new float3(1f, 0f, 0f));
        var t = projector.Project(spline, new float3(2f, 0f, 10f));

        var nearest = spline.EvaluatePosition(t);
        Assert.AreEqual(0f, math.distance(nearest, new float3(2f, 0f, 10f)), 0.2f);
    }

    [Test]
    public void ResetForgetsTheAnchor()
    {
        var spline = BuildClosedSquareSpline();
        var projector = new SplineProjector();

        projector.Project(spline, new float3(9f, 0f, 0f));
        Assert.IsTrue(projector.HasAnchor);

        projector.Reset();
        Assert.IsFalse(projector.HasAnchor);

        // Re-seeded globally, so a point on the far side resolves there rather than being dragged
        // back toward the stale anchor.
        var t = projector.Project(spline, new float3(0f, 0f, 5f));
        var nearest = spline.EvaluatePosition(t);
        Assert.AreEqual(0f, math.distance(nearest, new float3(0f, 0f, 5f)), 0.2f);
    }

    [Test]
    public void DegenerateSplineReturnsZero()
    {
        var spline = new Spline();
        spline.Add(new BezierKnot(float3.zero), TangentMode.Linear);

        var projector = new SplineProjector();
        Assert.AreEqual(0f, projector.Project(spline, new float3(5f, 0f, 5f)), 1e-4f);
    }
}
}
