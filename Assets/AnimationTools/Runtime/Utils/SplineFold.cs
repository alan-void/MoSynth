using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// How a position along a path behaves past either end: a closed path wraps, an open one stops.
/// </summary>
/// <remarks>
/// Wrapping an open path would teleport the predicted trajectory back to the start near the end,
/// which the character reads as an instruction to turn around. See openwiki/animation-tools/benchmarking.md.
/// </remarks>
public static class SplineFold
{
    /// <summary>Folds a normalized (0..1) spline parameter into the path.</summary>
    public static float Normalized(float t, bool closed) =>
        closed ? math.frac(t) : math.clamp(t, 0f, 1f);

    /// <summary>
    /// Folds a distance along the path, in the same units as <paramref name="length"/>.
    /// </summary>
    /// <remarks>
    /// Folding a lookahead in metres, rather than as a parameter, keeps the fold independent of the
    /// spline's parameterisation.
    /// </remarks>
    public static float Distance(float distance, float length, bool closed) =>
        closed ? distance % length : math.min(distance, length);
}
}
