using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// How a position along a path behaves past either end: a closed path wraps, an open one stops.
/// </summary>
/// <remarks>
/// Shared because every control input that reads a trajectory off a spline has to answer this, and
/// answering it differently in two of them is not a difference anyone would author on purpose.
/// Wrapping an open path teleports the predicted trajectory back to the start once the reference
/// nears the end, which reads as an instruction to turn around and walk back — so the character
/// never reaches the far end and a benchmark run can only end by timing out.
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
    /// The distance form exists alongside <see cref="Normalized"/> because a lookahead is
    /// naturally measured in metres, and converting it to a parameter first would make the fold
    /// depend on the spline's arc-length parameterisation rather than on its real length.
    /// </remarks>
    public static float Distance(float distance, float length, bool closed) =>
        closed ? distance % length : math.min(distance, length);
}
}
