using System;

namespace AnimationTools
{
/// <summary>
/// Output of <see cref="MotionQualityMetricsCalculator.Evaluate"/>: how clean the synthesized motion
/// itself is, independent of whether it went where it was asked to.
/// </summary>
/// <remarks>
/// These separate methods that score the same on path following: a search that jumps frames can hit
/// its trajectory and still look wrong.
/// </remarks>
[Serializable]
public class MotionQualityMetricsResult
{
    public int framesEvaluated;

    /// <summary>
    /// Meters the contact bones slide while flagged as planted, summed over bones, per meter the root
    /// travelled. Scale-free, so it compares across methods running at different speeds. 0 is
    /// perfect planting.
    /// </summary>
    public float footskatePerMeter;

    /// <summary>Mean slip speed in m/s, averaged over contact frames only. NaN when nothing was ever in contact.</summary>
    public float meanFootskateSpeed;

    /// <summary>Fraction of evaluated frames with at least one contact bone flagged in contact. A sanity check on the metric above.</summary>
    public float contactFraction;

    /// <summary>Mean magnitude of the simulation frame's third positional derivative, m/s^3.</summary>
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
