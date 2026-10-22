using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

namespace MotionField
{
/// <summary>
/// Pure-pursuit spline following, through <see cref="SplinePursuit"/>: each frame, find the point on
/// the spline nearest the character root, steer toward the point a fixed arc length further along,
/// and ask for the target speed. Shorter lookahead hugs the spline but can oscillate, longer cuts
/// corners.
/// </summary>
public class MotionFieldSplineControlInput : MotionFieldControlInput, IMotionSynthesisSplineControlInput,
    ISteeringTargetSource
{
    [Tooltip("Path to follow. Closed splines loop; on open splines the character slows to a stop " +
             "over the last lookahead distance.")]
    [SerializeField]
    private SplineContainer splineContainer;

    [Tooltip("Arc length ahead of the nearest spline point to steer toward, in meters.")]
    [SerializeField]
    [Min(0.1f)]
    private float lookaheadDistance = SplinePursuit.DefaultLookaheadDistance;

    [Tooltip("Ground speed to ask the motion field for, in m/s. Reachable only within the speed " +
             "range the config's value function was trained over.")]
    [SerializeField]
    [Min(0f)]
    private float speed = SplinePursuit.DefaultTargetSpeed;

    public SplineContainer SplineContainer
    {
        get => splineContainer;
        set
        {
            splineContainer = value;
            // A new path makes the tracked position meaningless, and a benchmark sweep hands this
            // component a different spline for every run.
            _pursuit.Reset();
        }
    }

    public float TargetSpeed => speed;

    /// <summary>
    /// The direction the path sets off in. Pure pursuit steers toward the path and has no notion of
    /// facing apart from it, so travel direction is the whole answer here.
    /// </summary>
    public float3 GetWorldInitDirection() => SplinePathDirection.WorldStartDirection(splineContainer);

    /// <summary>Arc length ahead of the nearest spline point to steer toward, in meters.</summary>
    public float LookaheadDistance { get => lookaheadDistance; set => lookaheadDistance = value; }

    private readonly SplinePursuit _pursuit = new();

    protected override void OnEnable()
    {
        base.OnEnable();
        if (!IsBound) return;

        // Nothing carries over from a previous activation, so the first frame back re-seeds.
        _pursuit.Reset();
    }

    protected override void GetDesiredMotion(out Vector3 worldDirection, out float desiredSpeed)
    {
        worldDirection = Vector3.zero;
        desiredSpeed = 0f;

        var spline = splineContainer != null ? splineContainer.Spline : null;
        if (spline == null)
        {
            _pursuit.Reset();
            return;
        }

        if (!_pursuit.Update(spline, splineContainer.transform.localToWorldMatrix, RootPosition)) return;

        // Near-zero at an open spline's end, so the stage keeps the last heading as it stops.
        worldDirection = _pursuit.DesiredWorldDirection(RootPosition, lookaheadDistance);
        desiredSpeed = _pursuit.SpeedToward(speed, lookaheadDistance);
    }

    /// <summary>The pursuit point: <see cref="LookaheadDistance"/> along the path from the character's projection.</summary>
    public bool TryGetSteeringTarget(out float3 worldPosition)
    {
        worldPosition = _pursuit.HasPath ? _pursuit.TargetWorld(lookaheadDistance) : float3.zero;
        return _pursuit.HasPath;
    }

#if UNITY_EDITOR
    protected override void OnDrawGizmos()
    {
        base.OnDrawGizmos();
        if (!Application.isPlaying || !_pursuit.HasPath) return;
        var target = (Vector3)_pursuit.TargetWorld(lookaheadDistance);
        Gizmos.color = new Color(0.1f, 1.0f, 0.1f, 1.0f);
        Gizmos.DrawSphere((Vector3)_pursuit.NearestWorld + Vector3.up * 0.02f, 0.05f);
        Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f); // prediction purple, per MM gizmo style
        Gizmos.DrawSphere(target + Vector3.up * 0.02f, 0.08f);
        Gizmos.DrawLine(RootPosition + Vector3.up * 0.05f, target + Vector3.up * 0.05f);
    }
#endif
}
}
