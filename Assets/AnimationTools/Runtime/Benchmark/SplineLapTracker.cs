namespace AnimationTools
{
/// <summary>
/// Turns a per-tick stream of normalized spline parameters into "how far round the path has the
/// character actually got", so a run can be ended by lap rather than by a wall-clock guess.
/// </summary>
/// <remarks>
/// A guess is not available: MotionFieldSplineControlInput reports <see cref="float.NaN"/> for its
/// target speed because the policy has no speed model, so lap time cannot be derived from path length. Measuring the character's own progress also keeps the
/// comparison honest — a method that falls behind gets a longer run rather than a truncated lap.
/// <para>
/// Closed splines accumulate signed deltas so the run survives the 1 -&gt; 0 wrap and shrugs off the
/// backward jitter a synthesized root produces every gait cycle. Open splines cannot lap, so
/// progress is just the current parameter and completion means reaching the far end.
/// </para>
/// </remarks>
public sealed class SplineLapTracker
{
    /// <summary>
    /// How close to the end an open spline counts as finished. Short of 1 because the nearest point
    /// to a character that has run past the end clamps to the final knot, which a strict test would
    /// still reject on floating point.
    /// </summary>
    public const float OpenSplineEndT = 0.995f;

    private readonly bool _closed;
    private bool _hasAnchor;
    private float _previousT;

    /// <summary>Laps completed since the last <see cref="Reset"/>, fractional. Negative if the character ran backwards.</summary>
    public float Progress { get; private set; }

    /// <summary>The most recent normalized parameter handed to <see cref="Sample"/>.</summary>
    public float LatestT { get; private set; }

    public SplineLapTracker(bool closed)
    {
        _closed = closed;
    }

    /// <summary>
    /// Drops the accumulated progress and re-anchors on the next sample. Called once the settle time
    /// has elapsed, so the lap is measured over settled motion rather than over the character's
    /// scramble to get up to speed.
    /// </summary>
    public void Reset()
    {
        _hasAnchor = false;
        Progress = 0f;
    }

    public void Sample(float t)
    {
        LatestT = t;

        if (!_hasAnchor)
        {
            _previousT = t;
            _hasAnchor = true;
            return;
        }

        if (_closed)
        {
            // Wrapped into (-0.5, 0.5]: a single tick never covers half the spline, so a delta
            // bigger than that is the seam being crossed, not the character teleporting.
            Progress += WrapDelta(t - _previousT);
        }
        else
        {
            Progress = t;
        }

        _previousT = t;
    }

    /// <summary>Whether the character has covered <paramref name="laps"/> laps (closed) or reached the end (open).</summary>
    public bool HasCompleted(float laps)
    {
        return _closed ? Progress >= laps : LatestT >= OpenSplineEndT;
    }

    /// <summary>Wraps a normalized parameter difference into [-0.5, 0.5].</summary>
    public static float WrapDelta(float delta)
    {
        if (delta > 0.5f) return delta - 1f;
        if (delta < -0.5f) return delta + 1f;
        return delta;
    }
}
}
