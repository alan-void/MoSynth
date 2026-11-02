using System;
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
    /// <param name="rootPositions">World-space simulation-frame position per recorded frame.</param>
    /// <param name="contactBonePositions">World-space position of each contact bone per recorded frame,
    /// indexed <c>[slot][frame]</c>, or null if they weren't recorded.</param>
    /// <param name="contacts">The method's own ground-contact flag of each contact bone per recorded
    /// frame, indexed like <paramref name="contactBonePositions"/>, or null. Feeds only the
    /// <c>flag*</c> metrics and <see cref="MotionQualityMetricsResult.contactFlagAgreement"/>.</param>
    /// <param name="discontinuities">Whether a stage replaced the pose discontinuously on that frame, or
    /// null if the channel wasn't recorded.</param>
    /// <param name="times">Seconds since recording start per frame, strictly increasing.</param>
    /// <param name="settleTime">Seconds at the start of the recording to exclude, giving the character
    /// time to reach steady state.</param>
    /// <param name="restHeights">Each contact bone's floor (<see cref="ContactFloor.RestHeights"/>), indexed
    /// like <paramref name="contactBonePositions"/>, or null. Without it the geometric metrics and
    /// <see cref="MotionQualityMetricsResult.contactFlagAgreement"/> are NaN.</param>
    /// <param name="meanContactHeights">Each contact bone's mean database contact height
    /// (<see cref="ContactFloor.MeanContactHeights"/>), indexed the same way, or null. Without it
    /// <see cref="MotionQualityMetricsResult.flagOnsetHeightError"/> is NaN.</param>
    /// <param name="contactSettings">Thresholds for judging ground contact from position alone.</param>
    public static MotionQualityMetricsResult Evaluate(
        float3[] rootPositions,
        float3[][] contactBonePositions,
        bool[][] contacts,
        bool[] discontinuities,
        float[] times,
        float settleTime,
        float[] restHeights,
        float[] meanContactHeights,
        GroundContactSettings contactSettings)
    {
        if (contactSettings == null) throw new ArgumentNullException(nameof(contactSettings));

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
            SetGeometricFootskateUnavailable(result);
            SetFlagFootskateUnavailable(result);
            result.rootJerkMean = float.NaN;
            result.rootJerkP95 = float.NaN;
            result.rootVerticalJerkMean = float.NaN;
            result.rootVerticalJerkP95 = float.NaN;
            result.discontinuitiesPerSecond = float.NaN;
            return result;
        }

        result.framesEvaluated = analysisFrames.Count;

        if (HasPositionSeries(contactBonePositions))
        {
            var slotCount = contactBonePositions.Length;
            var hasRestHeights = restHeights != null && restHeights.Length == slotCount;
            var hasMeanContactHeights = meanContactHeights != null && meanContactHeights.Length == slotCount;

            bool[][] geometricContacts = null;
            if (hasRestHeights)
            {
                geometricContacts = EvaluateGeometricFootskate(
                    analysisFrames, rootPositions, contactBonePositions, times, restHeights, contactSettings,
                    result);
            }
            else
            {
                SetGeometricFootskateUnavailable(result);
            }

            if (HasFlagSeries(contacts, contactBonePositions.Length))
            {
                EvaluateFlagFootskate(analysisFrames, rootPositions, contactBonePositions, contacts, times, result);
                result.contactFlagAgreement = geometricContacts != null
                    ? EvaluateFlagAgreement(analysisFrames, contacts, geometricContacts)
                    : float.NaN;
                result.flagOnsetHeightError = hasMeanContactHeights
                    ? EvaluateFlagOnsetHeightError(analysisFrames, contactBonePositions, contacts, meanContactHeights)
                    : float.NaN;
            }
            else
            {
                SetFlagFootskateUnavailable(result);
            }
        }
        else
        {
            SetGeometricFootskateUnavailable(result);
            SetFlagFootskateUnavailable(result);
        }

        EvaluateRootJerk(analysisFrames, rootPositions, times, result);

        result.discontinuitiesPerSecond = EvaluateDiscontinuities(analysisFrames, discontinuities, times);

        return result;
    }

    private static void SetGeometricFootskateUnavailable(MotionQualityMetricsResult result)
    {
        result.footskatePerMeter = float.NaN;
        result.meanFootskateSpeed = float.NaN;
        result.contactFraction = float.NaN;
    }

    private static void SetFlagFootskateUnavailable(MotionQualityMetricsResult result)
    {
        result.flagFootskatePerMeter = float.NaN;
        result.flagMeanFootskateSpeed = float.NaN;
        result.flagContactFraction = float.NaN;
        result.contactFlagAgreement = float.NaN;
        result.flagOnsetHeightError = float.NaN;
    }

    /// <summary>At least one bone, and a position series for every one of them.</summary>
    private static bool HasPositionSeries(float3[][] positions)
    {
        if (positions == null || positions.Length == 0) return false;

        for (var slot = 0; slot < positions.Length; slot++)
        {
            if (positions[slot] == null) return false;
        }

        return true;
    }

    /// <summary>A flag series for each of <paramref name="boneCount"/> bones.</summary>
    private static bool HasFlagSeries(bool[][] contacts, int boneCount)
    {
        if (contacts == null || contacts.Length != boneCount) return false;

        for (var slot = 0; slot < contacts.Length; slot++)
        {
            if (contacts[slot] == null) return false;
        }

        return true;
    }

    /// <summary>
    /// The headline footskate, from positions only: each bone's horizontal slip weighted by how close
    /// it is to its database floor. Returns each bone's geometric contact, indexed
    /// <c>[slot][analysis frame]</c>.
    /// </summary>
    private static bool[][] EvaluateGeometricFootskate(
        List<int> analysisFrames,
        float3[] rootPositions,
        float3[][] contactBonePositions,
        float[] times,
        float[] restHeights,
        GroundContactSettings settings,
        MotionQualityMetricsResult result)
    {
        var boneCount = contactBonePositions.Length;
        var frameCount = analysisFrames.Count;

        var weightedSlip = 0.0;
        var weightedContactSeconds = 0.0;
        var geometricContacts = new bool[boneCount][];
        var heights = new float[frameCount];

        for (var slot = 0; slot < boneCount; slot++)
        {
            var positions = contactBonePositions[slot];

            for (var i = 0; i < frameCount; i++) heights[i] = positions[analysisFrames[i]].y - restHeights[slot];

            // Gated on height alone: gating on low speed as well would cap the very slip being measured.
            for (var i = 1; i < frameCount; i++)
            {
                var t0 = analysisFrames[i - 1];
                var t1 = analysisFrames[i];
                var dt = times[t1] - times[t0];
                var weight = ContactWeight(0.5f * (heights[i - 1] + heights[i]), settings.heightThreshold);

                weightedSlip += weight * XZDistance(positions[t0], positions[t1]);
                if (dt > 0f) weightedContactSeconds += weight * dt;
            }

            var inContact = new bool[frameCount];
            for (var i = 0; i < frameCount; i++)
            {
                inContact[i] = heights[i] < settings.heightThreshold &&
                               HorizontalSpeed(analysisFrames, positions, times, i) < settings.speedThreshold;
            }

            geometricContacts[slot] = inContact;
        }

        var totalRootTravelXZ = RootTravelXZ(analysisFrames, rootPositions);

        result.footskatePerMeter = totalRootTravelXZ >= 1e-4
            ? (float)(weightedSlip / totalRootTravelXZ)
            : float.NaN;
        result.meanFootskateSpeed = weightedContactSeconds > 0.0
            ? (float)(weightedSlip / weightedContactSeconds)
            : float.NaN;

        var contactFrameCount = 0;
        for (var i = 0; i < frameCount; i++)
        {
            if (AnyInContact(geometricContacts, i)) contactFrameCount++;
        }

        result.contactFraction = (float)contactFrameCount / frameCount;

        return geometricContacts;
    }

    /// <summary>
    /// <c>clamp(2 - 2^(h/H), 0, 1)</c>: full weight at or below the contact height, falling to zero
    /// at <paramref name="heightThreshold"/> above it.
    /// </summary>
    private static float ContactWeight(float height, float heightThreshold)
    {
        return math.saturate(2f - math.pow(2f, height / heightThreshold));
    }

    /// <summary>Central-difference XZ speed at analysis frame <paramref name="i"/>, one-sided at the ends.</summary>
    private static float HorizontalSpeed(List<int> analysisFrames, float3[] positions, float[] times, int i)
    {
        var a = analysisFrames[math.max(i - 1, 0)];
        var b = analysisFrames[math.min(i + 1, analysisFrames.Count - 1)];
        var dt = times[b] - times[a];
        return dt > 0f ? XZDistance(positions[a], positions[b]) / dt : 0f;
    }

    /// <summary>The method's self-reported footskate: slip measured only while its own flag is set.</summary>
    private static void EvaluateFlagFootskate(
        List<int> analysisFrames,
        float3[] rootPositions,
        float3[][] contactBonePositions,
        bool[][] contacts,
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

            // Counted once per bone: a pair with two bones planted contributes dt twice, matching the
            // slip sum which is also over every bone, so the resulting ratio stays a per-bone speed.
            for (var slot = 0; slot < contacts.Length; slot++)
            {
                if (!contacts[slot][t0] || !contacts[slot][t1]) continue;

                totalSlip += XZDistance(contactBonePositions[slot][t0], contactBonePositions[slot][t1]);
                if (dt > 0f) totalContactSeconds += dt;
            }
        }

        var totalRootTravelXZ = RootTravelXZ(analysisFrames, rootPositions);

        // No flagged frame at all means the stage never wrote foot contacts, not that the feet never
        // slipped, so reporting 0 would score missing data as perfect planting.
        var hasContactData = totalContactSeconds > 0.0;

        result.flagFootskatePerMeter = hasContactData && totalRootTravelXZ >= 1e-4
            ? (float)(totalSlip / totalRootTravelXZ)
            : float.NaN;
        result.flagMeanFootskateSpeed = hasContactData ? (float)(totalSlip / totalContactSeconds) : float.NaN;

        var contactFrameCount = 0;
        for (var i = 0; i < analysisFrames.Count; i++)
        {
            if (AnyInContact(contacts, analysisFrames[i])) contactFrameCount++;
        }

        result.flagContactFraction = (float)contactFrameCount / analysisFrames.Count;
    }

    /// <summary>
    /// Mean height above the bone's mean database contact height at which its flag rises, over every
    /// rise of every bone. A flag already set on the first analysis frame is not a rise. NaN when none rises.
    /// </summary>
    private static float EvaluateFlagOnsetHeightError(
        List<int> analysisFrames,
        float3[][] contactBonePositions,
        bool[][] contacts,
        float[] meanContactHeights)
    {
        var sum = 0.0;
        var rises = 0;

        for (var slot = 0; slot < contacts.Length; slot++)
        {
            for (var i = 1; i < analysisFrames.Count; i++)
            {
                var frame = analysisFrames[i];
                if (!contacts[slot][frame] || contacts[slot][analysisFrames[i - 1]]) continue;

                sum += contactBonePositions[slot][frame].y - meanContactHeights[slot];
                rises++;
            }
        }

        return rises > 0 ? (float)(sum / rises) : float.NaN;
    }

    /// <summary>
    /// Fraction of analysis frames on which a bone's flag matches its geometric contact, averaged over
    /// bones.
    /// </summary>
    private static float EvaluateFlagAgreement(List<int> analysisFrames, bool[][] contacts, bool[][] geometricContacts)
    {
        var sum = 0.0;
        for (var slot = 0; slot < contacts.Length; slot++)
        {
            var matches = 0;
            for (var i = 0; i < analysisFrames.Count; i++)
            {
                if (contacts[slot][analysisFrames[i]] == geometricContacts[slot][i]) matches++;
            }

            sum += (double)matches / analysisFrames.Count;
        }

        return (float)(sum / contacts.Length);
    }

    private static bool AnyInContact(bool[][] contacts, int frame)
    {
        for (var slot = 0; slot < contacts.Length; slot++)
        {
            if (contacts[slot][frame]) return true;
        }

        return false;
    }

    private static double RootTravelXZ(List<int> analysisFrames, float3[] rootPositions)
    {
        var total = 0.0;
        for (var i = 1; i < analysisFrames.Count; i++)
        {
            total += XZDistance(rootPositions[analysisFrames[i - 1]], rootPositions[analysisFrames[i]]);
        }

        return total;
    }

    private static float XZDistance(float3 a, float3 b)
    {
        return math.distance(new float2(a.x, a.z), new float2(b.x, b.z));
    }

    private static void EvaluateRootJerk(List<int> analysisFrames, float3[] rootPositions, float[] times,
        MotionQualityMetricsResult result)
    {
        var groundMagnitudes = new List<float>(analysisFrames.Count);
        var verticalMagnitudes = new List<float>(analysisFrames.Count);

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
            // Split by plane: a gait's normal vertical bob is tens of m/s^3 and would bury horizontal pops.
            groundMagnitudes.Add(math.length(jerk.xz));
            verticalMagnitudes.Add(math.abs(jerk.y));
        }

        result.rootJerkMean = MetricsMath.Mean(groundMagnitudes);
        result.rootJerkP95 = MetricsMath.Percentile(groundMagnitudes, 0.95f);
        result.rootVerticalJerkMean = MetricsMath.Mean(verticalMagnitudes);
        result.rootVerticalJerkP95 = MetricsMath.Percentile(verticalMagnitudes, 0.95f);
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
