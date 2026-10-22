using System.Collections.Generic;
using Unity.Mathematics;

namespace Style
{
/// <summary>
/// The arithmetic that turns a style model's output into a pose edit: weighting and composing the
/// posture and dynamics deltas, crossfading two styles, keeping the character frame's yaw, and
/// rewriting the root's travel. Pure, so it is tested without a pipeline.
/// </summary>
public static class StyleComposition
{
    /// <summary>
    /// <paramref name="rotation"/> raised to <paramref name="weight"/>: the identity at 0, the
    /// rotation itself at 1, and the same axis turned further above 1. Taken the short way round.
    /// </summary>
    public static quaternion Scale(quaternion rotation, float weight)
    {
        var value = rotation.value;
        if (value.w < 0f) value = -value;

        var sinHalf = math.length(value.xyz);
        if (sinHalf < 1e-7f) return quaternion.identity;

        var angle = 2f * math.atan2(sinHalf, value.w);
        return quaternion.AxisAngle(value.xyz / sinHalf, angle * weight);
    }

    /// <summary>
    /// One bone's delta, <c>slerp(I, P, wp) · slerp(I, D, wd)</c>, applied on the right of the
    /// bone's own rotation. The identity when both weights are zero.
    /// </summary>
    public static quaternion ComposeDelta(quaternion posture, quaternion dynamics,
        float postureWeight, float dynamicsWeight) =>
        math.mul(Scale(posture, postureWeight), Scale(dynamics, dynamicsWeight));

    /// <summary>
    /// Blend from one style's composed delta to another's, <paramref name="t"/> being the share of
    /// <paramref name="to"/>. Slerps the results rather than mixing the styles, so the blend never
    /// passes through a style that does not exist.
    /// </summary>
    public static quaternion Crossfade(quaternion from, quaternion to, float t) =>
        math.slerp(from, to, math.saturate(t));

    /// <summary>
    /// <paramref name="modified"/> turned about world up so its ground-projected forward has the
    /// heading <paramref name="original"/>'s had. Pitch and roll survive.
    /// </summary>
    /// <remarks>
    /// The character frame is derived from bone 0's flattened forward, so a delta that turned bone 0
    /// would turn the whole character. A rotation on the left about up changes the flattened heading
    /// and nothing else.
    /// </remarks>
    /// <param name="forwardAxisLocal">The bone-local forward axis the character frame reads.</param>
    public static quaternion StripYaw(quaternion original, quaternion modified, float3 forwardAxisLocal)
    {
        var from = FlattenedForward(original, forwardAxisLocal);
        var to = FlattenedForward(modified, forwardAxisLocal);
        var turned = math.atan2(math.dot(math.cross(from, to), math.up()), math.dot(from, to));
        return math.mul(quaternion.AxisAngle(math.up(), -turned), modified);
    }

    /// <summary>The ground-plane heading the character frame reads off a rotation.</summary>
    public static float3 FlattenedForward(quaternion rotation, float3 forwardAxisLocal)
    {
        var forward = math.rotate(rotation, forwardAxisLocal);
        forward.y = 0f;
        // Same fallback SimulationFrame uses for a bone aimed straight up or down.
        return math.normalizesafe(forward, math.forward());
    }

    /// <summary>
    /// Each bone's weight: that of its nearest listed ancestor-or-self, or 1 when none is listed.
    /// </summary>
    /// <param name="parentIndices">Parent per bone, -1 for the root; parents precede children.</param>
    /// <param name="listed">Weights authored per bone index.</param>
    /// <param name="weights">Filled with one weight per bone.</param>
    public static void ResolveRegionWeights(IReadOnlyList<int> parentIndices,
        IReadOnlyDictionary<int, float> listed, float[] weights)
    {
        for (var bone = 0; bone < parentIndices.Count; bone++)
        {
            if (listed != null && listed.TryGetValue(bone, out var weight))
            {
                weights[bone] = weight;
                continue;
            }

            var parent = parentIndices[bone];
            weights[bone] = parent >= 0 ? weights[parent] : 1f;
        }
    }

    /// <summary>
    /// The frame velocity under root styling, <c>v·(1 + (k−1)·w) + r·w</c>: the speed ratio and the
    /// oscillating residual, both faded in by <paramref name="weight"/>.
    /// </summary>
    public static float3 StylizeRootVelocity(float3 velocity, float speedRatio, float3 residual,
        float weight) =>
        velocity * (1f + (speedRatio - 1f) * weight) + residual * weight;

    /// <summary>The frame yaw rate under root styling, <c>ω + r_yaw·w</c>.</summary>
    public static float StylizeYawRate(float yawRate, float residualYawRate, float weight) =>
        yawRate + residualYawRate * weight;
}
}
