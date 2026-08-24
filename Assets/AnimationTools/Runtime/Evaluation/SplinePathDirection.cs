using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

namespace AnimationTools
{
/// <summary>
/// The world direction a path sets off in. Shared because every spline-following control input needs
/// it to answer where a character should spawn facing, and they live in assemblies that cannot see
/// each other.
/// </summary>
public static class SplinePathDirection
{
    /// <summary>
    /// Ground-plane direction of the spline at its start, in world space, normalized;
    /// <see cref="float3.zero"/> when the path is too degenerate to have one.
    /// </summary>
    /// <remarks>
    /// This is the direction of <em>travel</em>. A path that carries its own facing — see
    /// <c>SplinePoseKeypoint</c> — should answer from that instead.
    /// </remarks>
    public static float3 WorldStartDirection(SplineContainer container)
    {
        if (container == null) return float3.zero;

        var spline = container.Spline;
        if (spline == null || spline.Count < 2) return float3.zero;

        var splineTransform = container.transform;
        var direction = (float3)splineTransform.TransformDirection((Vector3)spline.EvaluateTangent(0f));
        direction.y = 0f;

        if (math.lengthsq(direction) <= 1e-8f)
        {
            // A knot authored with a linear tangent evaluates to a zero tangent, so a path with
            // square corners has no analytic direction at its start. A short chord along the path
            // does, and agrees with the tangent everywhere the tangent exists.
            var length = spline.GetLength();
            if (length <= 1e-3f) return float3.zero;

            var step = math.min(0.1f, length * 0.05f) / length;
            var start = (float3)splineTransform.TransformPoint((Vector3)spline.EvaluatePosition(0f));
            var ahead = (float3)splineTransform.TransformPoint((Vector3)spline.EvaluatePosition(step));
            direction = ahead - start;
            direction.y = 0f;
        }

        return math.lengthsq(direction) > 1e-8f ? math.normalize(direction) : float3.zero;
    }
}
}
