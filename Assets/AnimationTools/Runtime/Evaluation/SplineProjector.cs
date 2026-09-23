using Unity.Mathematics;
using UnityEngine.Splines;

namespace AnimationTools
{
/// <summary>
/// Projects a moving point onto a spline while preserving continuity: each query searches only a
/// window around where the point was last time, so the answer tracks along the path instead of
/// jumping to whichever part of the spline happens to be closest.
/// </summary>
/// <remarks>
/// A global <see cref="SplineUtility.GetNearestPoint"/> flips branches where a path crosses itself,
/// and the package has no windowed form, so this mirrors its coarse-to-fine chord search over a
/// window. Normalized parameters are uniform in arc length, so a window of d metres is
/// d / length. See openwiki/animation-tools/path-following-metrics.md.
/// </remarks>
public sealed class SplineProjector
{
    /// <summary>Coarse-to-fine passes per query. Each halves the window around the best chord so far.</summary>
    private const int RefinementPasses = 3;

    /// <summary>Chords per pass floor. Below this the search can step over a fold in the path.</summary>
    private const int MinimumSegments = 8;

    private readonly float _minWindowMeters;
    private readonly float _windowSlack;
    private readonly float _reacquireDistanceMeters;

    private bool _hasAnchor;
    private float _anchorT;
    private float3 _previousPoint;

    /// <param name="minWindowMeters">Smallest window searched, however little the point moved. Covers
    /// a stationary character and the numerical slack around one.</param>
    /// <param name="windowSlack">Multiple of the point's own displacement to search ahead of it, so
    /// the window scales from per-tick steps to coarse recordings.</param>
    /// <param name="reacquireDistanceMeters">Past this distance from the windowed result, give up on
    /// continuity and search globally. Generous, since re-acquiring must not happen at a crossing.</param>
    public SplineProjector(
        float minWindowMeters = 0.5f,
        float windowSlack = 3f,
        float reacquireDistanceMeters = 5f)
    {
        _minWindowMeters = math.max(1e-3f, minWindowMeters);
        _windowSlack = math.max(1f, windowSlack);
        _reacquireDistanceMeters = math.max(0f, reacquireDistanceMeters);
    }

    /// <summary>False until the first <see cref="Project{T}"/>, and again after <see cref="Reset"/>.</summary>
    public bool HasAnchor => _hasAnchor;

    /// <summary>The most recent result, as a normalized spline parameter.</summary>
    public float NormalizedT => _anchorT;

    /// <summary>
    /// Forgets where the point was, so the next query seeds itself with a global search. Call this
    /// whenever the point stops being the same point — a different spline, a respawned character, a
    /// new benchmark run.
    /// </summary>
    public void Reset()
    {
        _hasAnchor = false;
        _anchorT = 0f;
        _previousPoint = default;
    }

    /// <summary>
    /// The normalized parameter of the point on <paramref name="spline"/> nearest
    /// <paramref name="localPoint"/>, constrained to stay near the previous answer.
    /// </summary>
    /// <param name="spline">The path, in its own local space.</param>
    /// <param name="localPoint">The query point, in the same local space.</param>
    public float Project<T>(T spline, float3 localPoint) where T : ISpline
    {
        var length = spline.GetLength();
        if (length < 1e-5f)
        {
            _hasAnchor = true;
            _previousPoint = localPoint;
            _anchorT = 0f;
            return 0f;
        }

        if (!_hasAnchor)
        {
            _anchorT = Global(spline, localPoint);
            _hasAnchor = true;
            _previousPoint = localPoint;
            return _anchorT;
        }

        var displacement = math.distance(localPoint, _previousPoint);
        var forwardMeters = math.max(_minWindowMeters, displacement * _windowSlack);
        var backMeters = math.max(_minWindowMeters * 0.5f, displacement * 0.5f);

        var t = Windowed(spline, localPoint, length, backMeters, forwardMeters, out var distance);

        // Only a point that has lost the path entirely gets to jump. A crossing is close to both
        // branches, so it never trips this.
        if (distance > _reacquireDistanceMeters) t = Global(spline, localPoint);

        _anchorT = t;
        _previousPoint = localPoint;
        return t;
    }

    private static float Global<T>(T spline, float3 localPoint) where T : ISpline
    {
        SplineUtility.GetNearestPoint(spline, localPoint, out float3 _, out var t,
            SplineUtility.PickResolutionMax, 4);
        return t;
    }

    /// <summary>
    /// Coarse-to-fine chord search over <c>[anchor - back, anchor + forward]</c>, in metres. Returns
    /// the normalized parameter of the best point found and, via <paramref name="distance"/>, how far
    /// it was from <paramref name="localPoint"/>.
    /// </summary>
    private float Windowed<T>(
        T spline,
        float3 localPoint,
        float length,
        float backMeters,
        float forwardMeters,
        out float distance) where T : ISpline
    {
        var start = _anchorT - backMeters / length;
        var span = (backMeters + forwardMeters) / length;

        // An open spline cannot be walked off either end, so the window is clipped rather than
        // wrapped. A closed one wraps, and Sample folds the parameter for it.
        if (!spline.Closed)
        {
            var end = math.min(1f, start + span);
            start = math.max(0f, start);
            span = math.max(0f, end - start);
        }
        else if (span >= 1f)
        {
            start = 0f;
            span = 1f;
        }

        var best = start;
        distance = float.PositiveInfinity;

        for (var pass = 0; pass < RefinementPasses; pass++)
        {
            var segments = math.max(MinimumSegments,
                SplineUtility.GetSubdivisionCount(length * span, SplineUtility.PickResolutionMax));

            var passBest = best;
            var passDistance = float.PositiveInfinity;

            for (var i = 0; i <= segments; i++)
            {
                var t = start + span * i / segments;
                var candidate = math.distance(localPoint, spline.EvaluatePosition(Fold(t, spline.Closed)));
                if (candidate >= passDistance) continue;
                passDistance = candidate;
                passBest = t;
            }

            best = passBest;
            distance = passDistance;

            // Narrow to the neighbourhood of the winner and go again. Two chords either side, so the
            // true minimum cannot fall outside the next window.
            var half = span / segments;
            start = best - half;
            span = 2f * half;

            if (span * length < 1e-4f) break;
        }

        return Fold(best, spline.Closed);
    }

    /// <summary>Folds a parameter into [0,1] — wrapping a closed spline, clamping an open one.</summary>
    private static float Fold(float t, bool closed)
    {
        if (!closed) return math.clamp(t, 0f, 1f);
        t -= math.floor(t);
        return t;
    }
}
}
