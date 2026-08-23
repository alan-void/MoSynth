using System;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine.Splines;

namespace AnimationTools.Tests
{
public class PathFollowingMetricsCalculatorTests
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

    /// <summary>A lemniscate, matching the FigureEight benchmark path: it crosses itself at the origin.</summary>
    private static Spline BuildFigureEightSpline()
    {
        var spline = new Spline();
        const int knots = 12;
        const float scale = 6f;
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

    private static (float[] times, float3[] positions, float3[] forwards) BuildFrames(
        int fps, float duration, Func<float, float3> position, Func<float, float3> forward)
    {
        var frameCount = (int)math.round(duration * fps) + 1;
        var times = new float[frameCount];
        var positions = new float3[frameCount];
        var forwards = new float3[frameCount];
        for (var i = 0; i < frameCount; i++)
        {
            var t = i / (float)fps;
            times[i] = t;
            positions[i] = position(t);
            forwards[i] = forward(t);
        }

        return (times, positions, forwards);
    }

    [Test]
    public void PerfectFollowing_AllErrorsNearZero()
    {
        var spline = BuildStraightSpline();
        var (times, positions, forwards) = BuildFrames(
            30, 5f,
            t => new float3(1.5f * t, 0f, 0f),
            t => new float3(1f, 0f, 0f));

        var result = PathFollowingMetricsCalculator.Evaluate(
            spline, float4x4.identity, positions, forwards, times,
            1.5f, 0f, 0.2f);

        Assert.AreEqual(0f, result.meanTrajectoryError, 0.02f);
        Assert.AreEqual(0f, result.meanHeadingErrorDeg, 1e-3f);
        Assert.AreEqual(0f, result.meanVelocityError, 0.02f);
        Assert.AreEqual(1.5f, result.meanActualSpeed, 0.02f);
    }

    [Test]
    public void LateralOffset_ErrorEqualsOffset()
    {
        var spline = BuildStraightSpline();
        var (times, positions, forwards) = BuildFrames(
            30, 5f,
            t => new float3(1.5f * t, 0f, 0.5f),
            t => new float3(1f, 0f, 0f));

        var result = PathFollowingMetricsCalculator.Evaluate(
            spline, float4x4.identity, positions, forwards, times,
            1.5f, 0f, 0.2f);

        Assert.AreEqual(0.5f, result.meanTrajectoryError, 0.02f);
    }

    [Test]
    public void SlowerThanTarget_VelocityErrorMatches()
    {
        var spline = BuildStraightSpline();
        var (times, positions, forwards) = BuildFrames(
            30, 4f,
            t => new float3(1f * t, 0f, 0f),
            t => new float3(1f, 0f, 0f));

        var result = PathFollowingMetricsCalculator.Evaluate(
            spline, float4x4.identity, positions, forwards, times,
            1.5f, 0f, 0.2f);

        Assert.AreEqual(0f, result.meanTrajectoryError, 0.02f);
        Assert.AreEqual(0.5f, result.meanVelocityError, 0.02f);
    }

    [Test]
    public void ConstantYawOffset_HeadingErrorMatches()
    {
        var spline = BuildStraightSpline();
        var offsetForward = math.normalize(new float3(1f, 0f, 1f));
        var (times, positions, forwards) = BuildFrames(
            30, 2f,
            t => new float3(1f * t, 0f, 0f),
            t => offsetForward);

        var result = PathFollowingMetricsCalculator.Evaluate(
            spline, float4x4.identity, positions, forwards, times,
            1f, 0f, 0.2f);

        Assert.AreEqual(45f, result.meanHeadingErrorDeg, 0.5f);
    }

    [Test]
    public void NaNTargetSpeed_TrajectoryStillComputed_VelocityNaN()
    {
        var spline = BuildStraightSpline();
        var (times, positions, forwards) = BuildFrames(
            30, 5f,
            t => new float3(1.5f * t, 0f, 0f),
            t => new float3(1f, 0f, 0f));

        var result = PathFollowingMetricsCalculator.Evaluate(
            spline, float4x4.identity, positions, forwards, times,
            float.NaN, 0f, 0.2f);

        Assert.IsFalse(float.IsNaN(result.meanTrajectoryError));
        Assert.IsNaN(result.meanVelocityError);
        Assert.IsFalse(float.IsNaN(result.meanActualSpeed));
    }

    [Test]
    public void ClosedSpline_NearestPointFollowsPerimeter()
    {
        var spline = BuildClosedSquareSpline();
        const float perimeter = 40f;
        const float targetSpeed = 5f;
        const float startPhase = 36f;

        float3 PerimeterPosition(float s)
        {
            s %= perimeter;
            if (s < 0f) s += perimeter;
            if (s < 10f) return new float3(s, 0f, 0f);
            if (s < 20f) return new float3(10f, 0f, s - 10f);
            if (s < 30f) return new float3(30f - s, 0f, 10f);
            return new float3(0f, 0f, 40f - s);
        }

        var (times, positions, forwards) = BuildFrames(
            30, 2f,
            t => PerimeterPosition(startPhase + targetSpeed * t),
            t => new float3(1f, 0f, 0f));

        var result = PathFollowingMetricsCalculator.Evaluate(
            spline, float4x4.identity, positions, forwards, times,
            targetSpeed, 0f, 0.2f);

        Assert.AreEqual(0f, result.meanTrajectoryError, 0.05f);
    }

    [Test]
    public void SelfIntersectingSpline_PerfectFollowingScoresNearZero()
    {
        // A run tracing the path exactly should score ~0 on both error terms. Projecting each frame
        // globally instead would match the far branch near the crossing and manufacture a large
        // heading error out of a run that never left the path.
        var spline = BuildFigureEightSpline();
        var frameCount = 361;
        var times = new float[frameCount];
        var positions = new float3[frameCount];
        var forwards = new float3[frameCount];

        for (var i = 0; i < frameCount; i++)
        {
            var u = i / (float)(frameCount - 1);
            times[i] = i / 30f;
            positions[i] = spline.EvaluatePosition(u);
            forwards[i] = math.normalize(spline.EvaluateTangent(u));
        }

        var result = PathFollowingMetricsCalculator.Evaluate(
            spline, float4x4.identity, positions, forwards, times,
            float.NaN, 0f, 0.2f);

        Assert.AreEqual(0f, result.meanTrajectoryError, 0.05f);
        Assert.AreEqual(0f, result.meanHeadingErrorDeg, 2f);
        Assert.Less(result.maxHeadingErrorDeg, 15f);
    }

    [Test]
    public void SettleTime_ExcludesEarlyFrames()
    {
        var spline = BuildStraightSpline();
        var (times, positions, forwards) = BuildFrames(
            30, 4f,
            t => t < 1f ? new float3(100f, 0f, 100f) : new float3(1.5f * (t - 1f), 0f, 0f),
            t => t < 1f ? new float3(0f, 0f, 1f) : new float3(1f, 0f, 0f));

        var result = PathFollowingMetricsCalculator.Evaluate(
            spline, float4x4.identity, positions, forwards, times,
            1.5f, 1.5f, 0.2f);

        Assert.AreEqual(0f, result.meanTrajectoryError, 0.02f);
        Assert.AreEqual(0f, result.meanHeadingErrorDeg, 1e-3f);
        Assert.AreEqual(0f, result.meanVelocityError, 0.02f);
        Assert.Less(result.framesEvaluated, times.Length);
    }

    [Test]
    public void TooFewFrames_ReturnsZeroEvaluated()
    {
        var spline = BuildStraightSpline();
        var (times, positions, forwards) = BuildFrames(
            30, 2f,
            t => new float3(1.5f * t, 0f, 0f),
            t => new float3(1f, 0f, 0f));

        var result = PathFollowingMetricsCalculator.Evaluate(
            spline, float4x4.identity, positions, forwards, times,
            1.5f, 3f, 0.2f);

        Assert.AreEqual(0, result.framesEvaluated);
        Assert.IsNaN(result.meanTrajectoryError);
        Assert.IsNaN(result.meanHeadingErrorDeg);
    }
}
}
