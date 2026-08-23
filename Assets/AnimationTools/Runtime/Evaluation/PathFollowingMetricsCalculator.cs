using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine.Splines;

namespace AnimationTools
{
/// <summary>
/// Output of <see cref="PathFollowingMetricsCalculator.Evaluate"/>: how closely a recorded run of
/// root positions/forwards tracked a spline in shape, heading and pace.
/// </summary>
[Serializable]
public class PathFollowingMetricsResult
{
    public string label;
    public int framesEvaluated;

    /// <summary>Target pace, m/s. <see cref="float.NaN"/> when the run has no speed model.</summary>
    public float targetSpeed;

    /// <summary>Mean XZ distance in meters from the root to the nearest point on the spline.</summary>
    public float meanTrajectoryError;

    /// <summary>Mean absolute angle in degrees between the root's forward and the spline tangent at the nearest point.</summary>
    public float meanHeadingErrorDeg;

    /// <summary>Largest single-frame heading error in degrees; spikes here usually mean an overshot corner.</summary>
    public float maxHeadingErrorDeg;

    /// <summary>Mean actual speed in the XZ plane, m/s.</summary>
    public float meanActualSpeed;

    /// <summary>Mean |actual - target| speed, m/s. NaN when <see cref="targetSpeed"/> is NaN.</summary>
    public float meanVelocityError;
}

/// <summary>
/// Pure metrics for how well a recorded run followed a spline path. Everything here is XZ-flattened
/// (height is ignored) and the spline container transform is assumed unscaled, same as
/// <see cref="MotionField.MotionFieldSplineControlInput"/> — scaling it would skew arc-length math.
/// </summary>
public static class PathFollowingMetricsCalculator
{
    /// <summary>
    /// Evaluates path-following quality for one run against one spline.
    ///
    /// Three metrics are computed, all XZ-flattened:
    /// - <b>Trajectory error</b>: per analysis frame, the XZ distance from the root to the nearest
    ///   point on the spline — how far off the path the character is, regardless of pace.
    /// - <b>Heading error</b>: the angle between the root's forward direction and the spline's
    ///   tangent at the nearest point, per analysis frame.
    /// - <b>Velocity error</b>: the (smoothed) actual XZ speed, finite-differenced from consecutive
    ///   analysis-frame positions, compared against the constant target speed.
    ///
    /// Frames before <paramref name="settleTime"/> are dropped so start-up transients don't skew
    /// the result. All inputs are world-space; the spline is evaluated in its own local space via
    /// <paramref name="splineLocalToWorld"/>, matching <see cref="SplineUtility.GetNearestPoint"/>'s
    /// contract.
    /// </summary>
    /// <param name="spline">The path being followed, in its own local space.</param>
    /// <param name="splineLocalToWorld">The spline container's localToWorldMatrix. Must be unscaled;
    /// the recorded world-space positions are pulled into spline-local space through its inverse.</param>
    /// <param name="rootPositions">World-space root (simulation bone) position per recorded frame.</param>
    /// <param name="rootForwards">World-space root forward direction per recorded frame.</param>
    /// <param name="times">Seconds since recording start per frame, strictly increasing.</param>
    /// <param name="targetSpeed">Desired pace in m/s, used only for velocity error; pass
    /// <see cref="float.NaN"/> when the control input has no speed model — velocity error is then NaN.</param>
    /// <param name="settleTime">Seconds at the start of the recording to exclude, giving the
    /// character time to reach the path from its spawn point.</param>
    /// <param name="speedSmoothingWindowSeconds">Width in seconds of the moving-average window
    /// applied to the finite-differenced speed before computing velocity error. Per-tick speed is
    /// dominated by synthesis jitter (frame switches, inertialization, gait cycle sway), so the raw
    /// signal would inflate |actual - target|; a window of ~0.2 s keeps gait-level speed changes
    /// visible while suppressing that noise. 0 (or anything below one frame) disables smoothing.</param>
    public static PathFollowingMetricsResult Evaluate(
        Spline spline,
        float4x4 splineLocalToWorld,
        float3[] rootPositions,
        float3[] rootForwards,
        float[] times,
        float targetSpeed,
        float settleTime,
        float speedSmoothingWindowSeconds)
    {
        var hasSpeed = !float.IsNaN(targetSpeed);

        var result = new PathFollowingMetricsResult
        {
            targetSpeed = targetSpeed,
        };

        var analysisFrames = new List<int>(times.Length);
        for (var t = 0; t < times.Length; t++)
        {
            if (times[t] >= settleTime) analysisFrames.Add(t);
        }

        if (analysisFrames.Count < 2)
        {
            result.framesEvaluated = 0;
            result.meanTrajectoryError = float.NaN;
            result.meanHeadingErrorDeg = float.NaN;
            result.maxHeadingErrorDeg = float.NaN;
            result.meanActualSpeed = float.NaN;
            result.meanVelocityError = float.NaN;
            return result;
        }

        var worldToLocal = math.inverse(splineLocalToWorld);

        // Walked forward across the analysis frames rather than re-derived per frame: on a path that
        // crosses itself the globally nearest point flips branches at the crossing, which would score
        // the character against the wrong tangent and manufacture a ~180 degree heading error.
        var projector = new SplineProjector();

        var errorSum = 0.0;

        var headingSum = 0.0;
        var headingCount = 0;
        var maxHeading = float.NegativeInfinity;

        for (var i = 0; i < analysisFrames.Count; i++)
        {
            var t = analysisFrames[i];
            var localRoot = math.transform(worldToLocal, rootPositions[t]);
            var nearestT = projector.Project(spline, localRoot);
            var nearestLocal = spline.EvaluatePosition(nearestT);

            var nearestWorld = math.transform(splineLocalToWorld, nearestLocal);
            var rootWorld = rootPositions[t];
            errorSum += math.distance(new float2(nearestWorld.x, nearestWorld.z), new float2(rootWorld.x, rootWorld.z));

            var tangentLocal = spline.EvaluateTangent(nearestT);
            if (math.lengthsq(tangentLocal) >= 1e-10f)
            {
                var tangentWorld = math.mul(splineLocalToWorld, new float4(tangentLocal, 0f)).xyz;
                var tangentXZ = new float2(tangentWorld.x, tangentWorld.z);
                var forward = rootForwards[t];
                var forwardXZ = new float2(forward.x, forward.z);
                if (math.lengthsq(tangentXZ) >= 1e-10f && math.lengthsq(forwardXZ) >= 1e-10f)
                {
                    var a = math.normalize(tangentXZ);
                    var b = math.normalize(forwardXZ);
                    // atan2 form: acos(dot) loses precision for nearly-parallel directions.
                    var cross = a.x * b.y - a.y * b.x;
                    var angleDeg = math.degrees(math.atan2(math.abs(cross), math.dot(a, b)));
                    headingSum += math.abs(angleDeg);
                    headingCount++;
                    if (angleDeg > maxHeading) maxHeading = angleDeg;
                }
            }
        }

        result.meanTrajectoryError = (float)(errorSum / analysisFrames.Count);
        result.meanHeadingErrorDeg = headingCount > 0 ? (float)(headingSum / headingCount) : float.NaN;
        result.maxHeadingErrorDeg = headingCount > 0 ? maxHeading : float.NaN;

        var rawSpeeds = new List<float>(analysisFrames.Count);
        var dtSum = 0.0;
        for (var i = 0; i < analysisFrames.Count - 1; i++)
        {
            var t0 = analysisFrames[i];
            var t1 = analysisFrames[i + 1];
            var dt = times[t1] - times[t0];
            if (dt <= 0f) continue;

            var p0 = rootPositions[t0];
            var p1 = rootPositions[t1];
            var dist = math.distance(new float2(p1.x, p1.z), new float2(p0.x, p0.z));
            rawSpeeds.Add(dist / dt);
            dtSum += dt;
        }

        float meanActualSpeed;
        float meanVelocityError;
        if (rawSpeeds.Count == 0)
        {
            meanActualSpeed = float.NaN;
            meanVelocityError = float.NaN;
        }
        else
        {
            var meanDt = (float)(dtSum / rawSpeeds.Count);
            var window = meanDt > 0f ? math.max(1, (int)math.round(speedSmoothingWindowSeconds / meanDt)) : 1;
            window = math.min(window, rawSpeeds.Count);

            var smoothed = SmoothCentered(rawSpeeds, window);

            var speedSum = 0.0;
            var velocityErrorSum = 0.0;
            for (var i = 0; i < smoothed.Length; i++)
            {
                speedSum += smoothed[i];
                if (hasSpeed) velocityErrorSum += math.abs(smoothed[i] - targetSpeed);
            }

            meanActualSpeed = (float)(speedSum / smoothed.Length);
            meanVelocityError = hasSpeed ? (float)(velocityErrorSum / smoothed.Length) : float.NaN;
        }

        result.framesEvaluated = analysisFrames.Count;
        result.meanActualSpeed = meanActualSpeed;
        result.meanVelocityError = meanVelocityError;
        return result;
    }

    /// <summary>
    /// Centered moving average: each sample becomes the mean of itself and up to window/2 neighbours
    /// on each side. Centered (rather than trailing) so the smoothed speed stays time-aligned with
    /// the frames it came from instead of lagging by half a window. Near the ends the window is
    /// truncated to what exists, so edge samples are averaged over fewer neighbours rather than
    /// padded or dropped.
    /// </summary>
    private static float[] SmoothCentered(List<float> raw, int window)
    {
        var smoothed = new float[raw.Count];
        var half = window / 2;
        for (var i = 0; i < raw.Count; i++)
        {
            var lo = math.max(0, i - half);
            var hi = math.min(raw.Count - 1, i + half);
            var sum = 0.0;
            for (var k = lo; k <= hi; k++) sum += raw[k];
            smoothed[i] = (float)(sum / (hi - lo + 1));
        }
        return smoothed;
    }
}
}
