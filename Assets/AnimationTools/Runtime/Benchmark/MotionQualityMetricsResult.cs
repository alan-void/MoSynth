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
    /// Meters the contact bones slide horizontally, each step weighted by how near the bone is to its
    /// database rest height (<see cref="ContactFloor.RestHeights"/>, <see cref="GroundContactSettings"/>), summed
    /// over bones, per meter the root travelled. Judged from positions alone, so every method is scored
    /// by the same rule. 0 is perfect planting; NaN without a contact floor.
    /// </summary>
    public float footskatePerMeter;

    /// <summary>The same weighted slip per weighted contact-second, m/s. NaN when no bone ever came near its database rest height.</summary>
    public float meanFootskateSpeed;

    /// <summary>
    /// Fraction of evaluated frames with at least one contact bone near its database rest height and
    /// slow enough to be in geometric contact.
    /// </summary>
    public float contactFraction;

    /// <summary>
    /// <see cref="footskatePerMeter"/> measured only while the method's own contact flags are set. A
    /// method flagging fewer contacts scores lower here, and a contact-locking stage pins exactly the
    /// flagged frames, so this reports the method's own view, not a comparable figure.
    /// </summary>
    public float flagFootskatePerMeter;

    /// <summary>Flag-gated slip speed in m/s, averaged over flagged frames only. NaN when nothing was ever flagged.</summary>
    public float flagMeanFootskateSpeed;

    /// <summary>
    /// Mean height in meters at which a contact flag rises, above the bone's mean database contact height
    /// (<see cref="ContactFloor.MeanContactHeights"/>). Positive means the flag rises higher than the
    /// database's average planted height, so a contact lock plants the bone in the air.
    /// </summary>
    public float flagOnsetHeightError;

    /// <summary>Fraction of evaluated frames with at least one contact bone flagged in contact.</summary>
    public float flagContactFraction;

    /// <summary>Fraction of frames on which a bone's contact flag matches its geometric contact, averaged over bones.</summary>
    public float contactFlagAgreement;

    /// <summary>
    /// Mean magnitude of the simulation frame's third positional derivative in the ground plane (XZ),
    /// m/s^3. Vertical motion is kept out so a gait's normal bob doesn't count as jerk.
    /// </summary>
    public float rootJerkMean;

    /// <summary>95th percentile ground-plane root jerk, m/s^3 — where the single-frame pops show up rather than the average sway.</summary>
    public float rootJerkP95;

    /// <summary>Mean magnitude of the root's vertical (Y) jerk, m/s^3. Includes the gait's bob, so compare like gaits only.</summary>
    public float rootVerticalJerkMean;

    /// <summary>95th percentile vertical root jerk, m/s^3, where a vertical pop shows up.</summary>
    public float rootVerticalJerkP95;

    /// <summary>
    /// Ticks per second on which a stage replaced the pose discontinuously (a motion matching search
    /// jumping frames, a field snapping to a new neighbour).
    /// </summary>
    public float discontinuitiesPerSecond;
}
}
