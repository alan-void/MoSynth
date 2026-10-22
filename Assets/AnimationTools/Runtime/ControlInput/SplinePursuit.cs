using Unity.Mathematics;
using UnityEngine.Splines;

namespace AnimationTools
{
/// <summary>
/// Closed-loop path following, shared by every method's spline input. Each tick the character is
/// projected onto the path; from that point the input can aim a lookahead distance further along,
/// or lay its future trajectory along the path at a target speed.
/// </summary>
/// <remarks>
/// Everything is measured from where the character is, never from a point advancing on its own
/// clock, so a character that falls behind is not asked to catch up and one that strays is steered
/// back. Distances fold through <see cref="SplineFold"/>: near an open path's end every horizon
/// lands on the end point, which asks the character to stop there. The path's container is assumed
/// unscaled, since arc length is measured on the local curve.
/// See openwiki/animation-tools/benchmarking.md.
/// </remarks>
public sealed class SplinePursuit
{
    /// <summary>Arc length ahead of the nearest point to steer toward, in metres, on every input that steers by direction.</summary>
    public const float DefaultLookaheadDistance = 1.5f;

    /// <summary>Speed the trajectory is laid along the path at, in m/s, on every spline input.</summary>
    public const float DefaultTargetSpeed = 1.0f;

    /// <summary>Most the trajectory may start ahead of the nearest point, in metres, on inputs that lead.</summary>
    public const float DefaultMaxLead = 1.0f;

    /// <summary>Arc length a sample's direction is measured over, in metres.</summary>
    private const float DirectionStep = 0.05f;

    /// <summary>Tracks continuously, so a self-crossing path does not flip the projection to the other branch.</summary>
    private readonly SplineProjector _projector = new();

    private Spline _spline;
    private float4x4 _localToWorld;

    /// <summary>Whether the last <see cref="Update"/> found a usable path. Every query below needs it.</summary>
    public bool HasPath { get; private set; }

    /// <summary>The path's length, in metres.</summary>
    public float Length { get; private set; }

    /// <summary>Whether the path loops.</summary>
    public bool Closed { get; private set; }

    /// <summary>Distance along the path of the point nearest the character, in metres.</summary>
    public float NearestDistance { get; private set; }

    /// <summary>World position of the point on the path nearest the character.</summary>
    public float3 NearestWorld { get; private set; }

    /// <summary>
    /// How far ahead of the nearest point the trajectory starts, in metres. Zero unless the caller
    /// runs <see cref="AdvanceLead"/>.
    /// </summary>
    public float Lead { get; private set; }

    /// <summary>Metres the nearest point moved forward along the path in the last update.</summary>
    private float _progress;

    /// <summary>Path left ahead of the character, in metres; infinite on a closed path.</summary>
    public float RemainingDistance =>
        !HasPath ? 0f : Closed ? float.PositiveInfinity : Length - NearestDistance;

    /// <summary>
    /// Forgets where the character was on the path, so the next update searches the whole of it.
    /// Call when the path is replaced or the character is moved onto it.
    /// </summary>
    public void Reset()
    {
        _projector.Reset();
        HasPath = false;
        Lead = 0f;
        _progress = 0f;
    }

    /// <summary>
    /// Projects the character onto <paramref name="spline"/>. Call once per tick before any query.
    /// </summary>
    /// <param name="spline">The path, in its own local space. Null or degenerate leaves <see cref="HasPath"/> false.</param>
    /// <param name="localToWorld">Places the path in the world.</param>
    /// <param name="characterWorld">World position of the character's simulation frame.</param>
    /// <returns><see cref="HasPath"/>.</returns>
    public bool Update(Spline spline, float4x4 localToWorld, float3 characterWorld)
    {
        HasPath = false;
        if (spline == null || spline.Count < 2) return false;

        var length = spline.GetLength();
        if (length < 1e-3f) return false;

        _spline = spline;
        _localToWorld = localToWorld;
        Length = length;
        Closed = spline.Closed;

        var hadAnchor = _projector.HasAnchor;
        var previousDistance = NearestDistance;
        var nearestT = _projector.Project(spline, math.transform(math.inverse(localToWorld), characterWorld));
        NearestDistance = spline.ConvertIndexUnit(nearestT, PathIndexUnit.Normalized, PathIndexUnit.Distance);

        // Measured the short way round, so crossing a closed path's seam is not a lap of progress.
        _progress = 0f;
        if (hadAnchor)
        {
            _progress = NearestDistance - previousDistance;
            if (Closed) _progress -= math.round(_progress / length) * length;
        }
        NearestWorld = math.transform(localToWorld, spline.EvaluatePosition(nearestT));

        HasPath = true;
        return true;
    }

    /// <summary>
    /// Lets the trajectory's start run ahead of the character while it travels slower than
    /// <paramref name="speed"/>, and fall back while it catches up, within [0, <paramref name="maxLead"/>].
    /// Call once per tick after <see cref="Update"/>.
    /// </summary>
    /// <remarks>
    /// The lead is the integral of the speed shortfall, bounded. Without it the request is the same
    /// whatever the character does, so a method that answers a request with slower motion settles
    /// there for good. See openwiki/animation-tools/benchmarking.md.
    /// </remarks>
    public void AdvanceLead(float speed, float deltaTime, float maxLead)
    {
        if (!HasPath) return;
        Lead = math.clamp(Lead + speed * deltaTime - _progress, 0f, math.max(0f, maxLead));
    }

    /// <summary>
    /// Distance along the path a horizon lands at: <paramref name="secondsAhead"/> of travel at
    /// <paramref name="speed"/> from the nearest point plus the <see cref="Lead"/>. Negative looks back.
    /// Not yet folded.
    /// </summary>
    public float DistanceAt(float secondsAhead, float speed) => NearestDistance + Lead + secondsAhead * speed;

    /// <summary>A distance folded onto the path: wrapped on a closed one, clamped to its ends on an open one.</summary>
    public float Fold(float distance) => SplineFold.Distance(distance, Length, Closed);

    /// <summary>The spline's normalized parameter at a distance along it.</summary>
    public float NormalizedAt(float distance) =>
        _spline.ConvertIndexUnit(Fold(distance), PathIndexUnit.Distance, PathIndexUnit.Normalized);

    /// <summary>World position at a distance along the path.</summary>
    public float3 PositionAt(float distance) =>
        math.transform(_localToWorld, _spline.EvaluatePosition(NormalizedAt(distance)));

    /// <summary>
    /// Ground-plane direction of travel at a distance along the path, normalized.
    /// </summary>
    /// <remarks>
    /// Measured as a short step along the path rather than as the analytic tangent, so it cannot
    /// disagree with the positions beside it, and so a linear-tangent corner still has one.
    /// </remarks>
    public float2 DirectionAt(float distance)
    {
        distance = Fold(distance);
        var here = PositionAt(distance);
        var step = (PositionAt(distance + DirectionStep) - here).xz;

        // At the clamped end of an open path the step forward lands on the same point, so measure
        // the last one instead of falling through to an arbitrary default.
        if (math.lengthsq(step) < 1e-12f) step = (here - PositionAt(distance - DirectionStep)).xz;

        return math.normalizesafe(step, new float2(0f, 1f));
    }

    /// <summary>World position <paramref name="lookaheadDistance"/> further along the path than the nearest point.</summary>
    public float3 TargetWorld(float lookaheadDistance) => PositionAt(NearestDistance + lookaheadDistance);

    /// <summary>
    /// Ground-plane world direction from the character toward the lookahead point. Near-zero once an
    /// open path's end is reached, so a caller can tell "arrived" from a heading.
    /// </summary>
    public float3 DesiredWorldDirection(float3 characterWorld, float lookaheadDistance)
    {
        var desired = TargetWorld(lookaheadDistance) - characterWorld;
        desired.y = 0f;
        return desired;
    }

    /// <summary>
    /// <paramref name="speed"/>, ramped down to zero over the last <paramref name="brakingDistance"/>
    /// metres of an open path: the scalar form of the trajectory bunching up at the end.
    /// </summary>
    public float SpeedToward(float speed, float brakingDistance) =>
        Closed ? speed : speed * math.saturate(RemainingDistance / math.max(1e-3f, brakingDistance));
}
}
