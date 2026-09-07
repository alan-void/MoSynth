using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

namespace MotionField
{
/// <summary>
/// Pure-pursuit spline following: each frame, find the point on the spline nearest the character
/// root, then steer toward the point a fixed arc length further along. The policy has no speed
/// control, so the lookahead distance is the only tuning knob: shorter hugs the spline but can
/// oscillate, longer cuts corners.
/// </summary>
public class MotionFieldSplineControlInput : MotionFieldControlInput, IMotionSynthesisSplineControlInput
{
    [Tooltip("Path to follow. Closed splines loop; on open splines the character keeps its last " +
             "heading past the end.")]
    [SerializeField]
    private SplineContainer splineContainer;

    [Tooltip("Arc length ahead of the nearest spline point to steer toward, in meters.")]
    [SerializeField]
    [Min(0.1f)]
    private float lookaheadDistance = 1.5f;

    public SplineContainer SplineContainer
    {
        get => splineContainer;
        set
        {
            splineContainer = value;
            // A new path makes the tracked position meaningless, and a benchmark sweep hands this
            // component a different spline for every run.
            _projector.Reset();
        }
    }

    public float TargetSpeed => float.NaN;

    /// <summary>
    /// The direction the path sets off in. Pure pursuit steers toward the path and has no notion of
    /// facing apart from it, so travel direction is the whole answer here.
    /// </summary>
    public float3 GetWorldInitDirection() => SplinePathDirection.WorldStartDirection(splineContainer);

    /// <summary>Arc length ahead of the nearest spline point to steer toward, in meters. The only tuning knob this policy has.</summary>
    public float LookaheadDistance { get => lookaheadDistance; set => lookaheadDistance = value; }

    /// <summary>
    /// Tracks where on the path the character is. Continuity matters here rather than raw proximity:
    /// on a path that crosses itself, the globally nearest point flips branches at the crossing and
    /// takes the steering target with it.
    /// </summary>
    private readonly SplineProjector _projector = new();

    private Vector3 _nearestWorld;
    private Vector3 _targetWorld;
    private bool _hasTarget;

    protected override void OnEnable()
    {
        base.OnEnable();
        if (!IsBound) return;

        // Nothing carries over from a previous activation, so the first frame back re-seeds.
        _projector.Reset();
    }

    protected override Vector3 GetDesiredWorldDirection()
    {
        _hasTarget = false;
        if (splineContainer == null) return Vector3.zero;

        var spline = splineContainer.Spline;
        if (spline == null || spline.Count < 2) return Vector3.zero;

        var length = spline.GetLength();
        if (length < 1e-3f) return Vector3.zero;

        // The projection works in the spline's local space; the container transform maps it
        // to the world. Lookahead is measured on the local curve, so a scaled container would
        // skew it -- containers are assumed unscaled.
        var splineTransform = splineContainer.transform;
        var localRoot = (float3)splineTransform.InverseTransformPoint(RootPosition);

        var nearestT = _projector.Project(spline, localRoot);
        var nearestLocal = (Vector3)spline.EvaluatePosition(nearestT);

        var nearestDistance = spline.ConvertIndexUnit(
            nearestT, PathIndexUnit.Normalized, PathIndexUnit.Distance);
        var targetDistance = SplineFold.Distance(
            nearestDistance + lookaheadDistance, length, spline.Closed);

        var targetT = spline.ConvertIndexUnit(
            targetDistance, PathIndexUnit.Distance, PathIndexUnit.Normalized);
        var targetLocal = spline.EvaluatePosition(targetT);

        _nearestWorld = splineTransform.TransformPoint(nearestLocal);
        _targetWorld = splineTransform.TransformPoint((Vector3)targetLocal);
        _hasTarget = true;

        var desired = _targetWorld - RootPosition;
        desired.y = 0f;
        return desired; // near-zero at an open spline's end -> stage keeps the last heading
    }

#if UNITY_EDITOR
    protected override void OnDrawGizmos()
    {
        base.OnDrawGizmos();
        if (!Application.isPlaying || !_hasTarget) return;
        Gizmos.color = new Color(0.1f, 1.0f, 0.1f, 1.0f);
        Gizmos.DrawSphere(_nearestWorld + Vector3.up * 0.02f, 0.05f);
        Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f); // prediction purple, per MM gizmo style
        Gizmos.DrawSphere(_targetWorld + Vector3.up * 0.02f, 0.08f);
        Gizmos.DrawLine(RootPosition + Vector3.up * 0.05f, _targetWorld + Vector3.up * 0.05f);
    }
#endif
}
}
