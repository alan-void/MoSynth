using System.Collections.Generic;
using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// Pure metrics for how clean a recorded run's motion is, independent of whether it went where it
/// was asked to. See <see cref="MotionQualityMetricsResult"/> for what each field measures.
/// </summary>
public static class MotionQualityMetricsCalculator
{
    /// <summary>
    /// Evaluates motion-quality metrics for one recorded run. Frames before
    /// <paramref name="settleTime"/> are dropped, same as <see cref="PathFollowingMetricsCalculator"/>.
    /// </summary>
    /// <param name="rootPositions">World-space root (simulation bone) position per recorded frame.</param>
    /// <param name="leftFootPositions">World-space left foot position per recorded frame, or null if the
    /// foot channel wasn't recorded.</param>
    /// <param name="rightFootPositions">World-space right foot position per recorded frame, or null.</param>
    /// <param name="leftContacts">Left foot ground-contact flag per recorded frame, or null.</param>
    /// <param name="rightContacts">Right foot ground-contact flag per recorded frame, or null.</param>
    /// <param name="discontinuities">Whether a stage replaced the pose discontinuously on that frame, or
    /// null if the channel wasn't recorded.</param>
    /// <param name="times">Seconds since recording start per frame, strictly increasing.</param>
    /// <param name="settleTime">Seconds at the start of the recording to exclude, giving the character
    /// time to reach steady state.</param>
    public static MotionQualityMetricsResult Evaluate(
        float3[] rootPositions,
        float3[] leftFootPositions,
        float3[] rightFootPositions,
        bool[] leftContacts,
        bool[] rightContacts,
        bool[] discontinuities,
        float[] times,
        float settleTime)
    {
        var result = new MotionQualityMetricsResult();

        var analysisFrames = new List<int>(times.Length);
        for (var t = 0; t < times.Length; t++)
        {
            if (times[t] >= settleTime) analysisFrames.Add(t);
        }

        // Jerk needs four positions to produce a single sample; below that nothing here is meaningful.
        if (analysisFrames.Count < 4)
        {
            result.framesEvaluated = 0;
            result.footskatePerMeter = float.NaN;
            result.meanFootskateSpeed = float.NaN;
            result.contactFraction = float.NaN;
            result.rootJerkMean = float.NaN;
            result.rootJerkP95 = float.NaN;
            result.discontinuitiesPerSecond = float.NaN;
            return result;
        }

        result.framesEvaluated = analysisFrames.Count;

        var hasFeet = leftFootPositions != null && rightFootPositions != null
            && leftContacts != null && rightContacts != null;

        if (hasFeet)
        {
            EvaluateFootskate(analysisFrames, rootPositions, leftFootPositions, rightFootPositions,
                leftContacts, rightContacts, times, result);
        }
        else
        {
            result.footskatePerMeter = float.NaN;
            result.meanFootskateSpeed = float.NaN;
            result.contactFraction = float.NaN;
        }

        EvaluateRootJerk(analysisFrames, rootPositions, times, result);

        result.discontinuitiesPerSecond = EvaluateDiscontinuities(analysisFrames, discontinuities, times);

        return result;
    }

    private static void EvaluateFootskate(
        List<int> analysisFrames,
        float3[] rootPositions,
        float3[] leftFootPositions,
        float3[] rightFootPositions,
        bool[] leftContacts,
        bool[] rightContacts,
        float[] times,
        MotionQualityMetricsResult result)
    {
        var totalSlip = 0.0;
        var totalContactSeconds = 0.0;

        for (var i = 1; i < analysisFrames.Count; i++)
        {
            var t0 = analysisFrames[i - 1];
            var t1 = analysisFrames[i];
            var dt = times[t1] - times[t0];

            // Counted once per foot: a pair with both feet planted contributes dt twice, matching the
            // slip sum which is also over both feet, so the resulting ratio stays a per-foot speed.
            if (leftContacts[t0] && leftContacts[t1])
            {
                totalSlip += XZDistance(leftFootPositions[t0], leftFootPositions[t1]);
                if (dt > 0f) totalContactSeconds += dt;
            }

            if (rightContacts[t0] && rightContacts[t1])
            {
                totalSlip += XZDistance(rightFootPositions[t0], rightFootPositions[t1]);
                if (dt > 0f) totalContactSeconds += dt;
            }
        }

        var totalRootTravelXZ = 0.0;
        for (var i = 1; i < analysisFrames.Count; i++)
        {
            var t0 = analysisFrames[i - 1];
            var t1 = analysisFrames[i];
            totalRootTravelXZ += XZDistance(rootPositions[t0], rootPositions[t1]);
        }

        // No contact frame at all means the stage never wrote foot contacts, not that the feet never
        // slipped. Reporting 0 there would score a method with no contact data as having perfect
        // foot planting, which is exactly backwards.
        var hasContactData = totalContactSeconds > 0.0;

        result.footskatePerMeter = hasContactData && totalRootTravelXZ >= 1e-4
            ? (float)(totalSlip / totalRootTravelXZ)
            : float.NaN;
        result.meanFootskateSpeed = hasContactData ? (float)(totalSlip / totalContactSeconds) : float.NaN;

        var contactFrameCount = 0;
        for (var i = 0; i < analysisFrames.Count; i++)
        {
            var t = analysisFrames[i];
            if (leftContacts[t] || rightContacts[t]) contactFrameCount++;
        }

        result.contactFraction = (float)contactFrameCount / analysisFrames.Count;
    }

    private static float XZDistance(float3 a, float3 b)
    {
        return math.distance(new float2(a.x, a.z), new float2(b.x, b.z));
    }

    private static void EvaluateRootJerk(List<int> analysisFrames, float3[] rootPositions, float[] times,
        MotionQualityMetricsResult result)
    {
        var magnitudes = new List<float>(analysisFrames.Count);

        for (var i = 0; i <= analysisFrames.Count - 4; i++)
        {
            var i0 = analysisFrames[i];
            var i1 = analysisFrames[i + 1];
            var i2 = analysisFrames[i + 2];
            var i3 = analysisFrames[i + 3];

            var dt1 = times[i1] - times[i0];
            var dt2 = times[i2] - times[i1];
            var dt3 = times[i3] - times[i2];
            if (dt1 <= 0f || dt2 <= 0f || dt3 <= 0f) continue;

            var v0 = (rootPositions[i1] - rootPositions[i0]) / dt1;
            var v1 = (rootPositions[i2] - rootPositions[i1]) / dt2;
            var v2 = (rootPositions[i3] - rootPositions[i2]) / dt3;

            var a0 = (v1 - v0) / dt2;
            var a1 = (v2 - v1) / dt3;

            var jerk = (a1 - a0) / dt3;
            magnitudes.Add(math.length(jerk));
        }

        result.rootJerkMean = MetricsMath.Mean(magnitudes);
        result.rootJerkP95 = MetricsMath.Percentile(magnitudes, 0.95f);
    }

    private static float EvaluateDiscontinuities(List<int> analysisFrames, bool[] discontinuities, float[] times)
    {
        if (discontinuities == null) return float.NaN;

        var span = times[analysisFrames[^1]] - times[analysisFrames[0]];
        if (span <= 0f) return float.NaN;

        var count = 0;
        for (var i = 0; i < analysisFrames.Count; i++)
        {
            if (discontinuities[analysisFrames[i]]) count++;
        }

        return count / span;
    }
}
}
