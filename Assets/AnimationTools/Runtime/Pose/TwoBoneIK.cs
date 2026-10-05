using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// Analytic two-bone IK over a three-joint chain A → B → C (for a leg: hip, knee, ankle): new global
/// rotations for A and B that put C on a target, in closed form rather than by iteration.
/// </summary>
/// <remarks>
/// Two stages. <em>Extension</em>: the cosine rule gives the interior angles that make the chain
/// exactly as long as the distance to the target, and A and B rotate by the difference about the
/// bend plane's normal, which leaves the direction A → C unchanged. <em>Aiming</em>: A rotates once
/// more to swing that direction onto the target. An unreachable target is pulled in to the limit
/// rather than failing. All inputs and outputs share one space; nothing here depends on which.
/// </remarks>
public static class TwoBoneIK
{
    /// <summary>Below this sine of the angle at A the chain counts as straight and has no plane of its own.</summary>
    private const float StraightChainSine = 1e-4f;

    /// <summary>
    /// Rotates the chain so C lands on <paramref name="target"/>, bending in the plane the chain
    /// already lies in.
    /// </summary>
    /// <param name="fallbackBendDirection">
    /// Which way B bends when the chain is straight and so defines no plane — for a leg, character
    /// forward, which picks a forward-bending knee.
    /// </param>
    public static void Solve(float3 a, float3 b, float3 c, quaternion globalRotationA, quaternion globalRotationB,
        float3 target, float3 fallbackBendDirection,
        out quaternion newGlobalRotationA, out quaternion newGlobalRotationB)
    {
        var lengthAB = math.distance(a, b);
        var lengthBC = math.distance(b, c);
        var lengthAT = ReachableDistance(math.distance(a, target), lengthAB, lengthBC);

        var directionAB = math.normalizesafe(b - a);
        var directionAC = math.normalizesafe(c - a, directionAB);
        var directionAT = math.normalizesafe(target - a, directionAC);

        var bendAxis = BendAxis(directionAC, directionAB, fallbackBendDirection);

        // Extension: rotate A and B so |AC| equals |AT|.
        var interiorAngleA = math.acos(math.clamp(math.dot(directionAC, directionAB), -1f, 1f));
        var interiorAngleB = math.acos(math.clamp(
            math.dot(math.normalizesafe(a - b), math.normalizesafe(c - b)), -1f, 1f));
        var desiredInteriorAngleA = math.acos(math.clamp(
            (lengthBC * lengthBC - lengthAB * lengthAB - lengthAT * lengthAT) / (-2f * lengthAB * lengthAT),
            -1f, 1f));
        var desiredInteriorAngleB = math.acos(math.clamp(
            (lengthAT * lengthAT - lengthAB * lengthAB - lengthBC * lengthBC) / (-2f * lengthAB * lengthBC),
            -1f, 1f));
        var extendA = quaternion.AxisAngle(bendAxis, desiredInteriorAngleA - interiorAngleA);
        var extendB = quaternion.AxisAngle(bendAxis, desiredInteriorAngleB - interiorAngleB);

        // Aiming: swing A so AC points along AT. Antiparallel directions have no cross product, and
        // the bend axis is perpendicular to AC, so it serves as the swing axis there.
        var aimAngle = math.acos(math.clamp(math.dot(directionAC, directionAT), -1f, 1f));
        var aimAxis = math.normalizesafe(math.cross(directionAC, directionAT), bendAxis);
        var aim = quaternion.AxisAngle(aimAxis, aimAngle);

        var deltaA = math.mul(aim, extendA);
        newGlobalRotationA = math.mul(deltaA, globalRotationA);
        // B is A's child, so it inherits A's change before adding its own.
        newGlobalRotationB = math.mul(deltaA, math.mul(extendB, globalRotationB));
    }

    /// <summary>
    /// Distance to the target, clamped to what the chain can reach. Held an epsilon short at both
    /// ends, since at full extension or full fold the cosine rule is degenerate.
    /// </summary>
    private static float ReachableDistance(float distance, float lengthAB, float lengthBC)
    {
        const float epsilon = 0.001f;
        return math.clamp(distance, epsilon, lengthAB + lengthBC - epsilon);
    }

    /// <summary>
    /// Normal of the plane the chain bends in, oriented so a positive rotation about it opens the
    /// angle at A.
    /// </summary>
    private static float3 BendAxis(float3 directionAC, float3 directionAB, float3 fallbackBendDirection)
    {
        var planeNormal = math.cross(directionAC, directionAB);
        if (math.length(planeNormal) > StraightChainSine) return math.normalize(planeNormal);

        var fallbackNormal = math.cross(directionAC, fallbackBendDirection);
        return math.normalizesafe(fallbackNormal, AnyPerpendicular(directionAC));
    }

    private static float3 AnyPerpendicular(float3 direction)
    {
        var helper = math.abs(direction.y) < 0.9f ? math.up() : math.right();
        return math.normalize(math.cross(direction, helper));
    }
}
}
