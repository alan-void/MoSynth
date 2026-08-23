using System;

namespace AnimationTools
{
/// <summary>
/// Output of <see cref="MotionQualityMetricsCalculator.Evaluate"/>: how clean the synthesized motion
/// itself is, independent of whether it went where it was asked to.
/// </summary>
/// <remarks>
/// These are the artefacts that separate two methods which score the same on path following. A
/// motion field that snaps between database neighbours and a motion matching search that jumps to a
/// distant frame both hit their trajectory, and both look wrong — footskate and root jerk are what
/// that wrongness measures.
/// </remarks>
[Serializable]
public class MotionQualityMetricsResult
{
    public int framesEvaluated;

    /// <summary>
    /// Meters a foot slides while flagged as planted, per meter the root travelled. Scale-free, so
    /// it compares across methods running at different speeds. 0 is a perfectly planted foot.
    /// </summary>
    public float footskatePerMeter;

    /// <summary>Mean slip speed in m/s, averaged over contact frames only. NaN when nothing was ever in contact.</summary>
    public float meanFootskateSpeed;

    /// <summary>Fraction of evaluated frames with at least one foot flagged in contact. A sanity check on the metric above.</summary>
    public float contactFraction;

    /// <summary>Mean magnitude of the simulation bone's third positional derivative, m/s^3.</summary>
    public float rootJerkMean;

    /// <summary>95th percentile root jerk, m/s^3 — where the single-frame pops show up rather than the average sway.</summary>
    public float rootJerkP95;

    /// <summary>
    /// Ticks per second on which a stage replaced the pose discontinuously (a motion matching search
    /// jumping frames, a field snapping to a new neighbour).
    /// </summary>
    public float discontinuitiesPerSecond;
}
}
