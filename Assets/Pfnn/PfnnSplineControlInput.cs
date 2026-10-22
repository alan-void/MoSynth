using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

namespace Pfnn
{
/// <summary>
/// Follows a spline at a target speed, handing the network the stretch of path ahead of the
/// character as its future trajectory.
/// </summary>
/// <remarks>
/// Closed loop, through <see cref="SplinePursuit"/>: the stretch is laid from the point on the path
/// nearest the character, as <c>SplineControlInput</c> does, so the request never asks it to catch
/// up with a point that ran ahead.
/// </remarks>
public class PfnnSplineControlInput : PfnnControlInput, IMotionSynthesisSplineControlInput, ISteeringTargetSource
{
    [Tooltip("Path to follow. Closed splines loop; on open splines the character stops at the end.")]
    [SerializeField]
    private SplineContainer splineContainer;

    [Tooltip("Speed the trajectory is laid along the spline at, from the character's nearest point, in m/s.")]
    [SerializeField]
    private float speed = SplinePursuit.DefaultTargetSpeed;

    public SplineContainer SplineContainer
    {
        get => splineContainer;
        set
        {
            splineContainer = value;
            // A benchmark sweep hands this component a different spline for every run, and where
            // the character was on the old one means nothing on the new one.
            _pursuit.Reset();
        }
    }

    public float TargetSpeed => speed;

    /// <summary>The direction the path sets off in; a bare path carries no facing of its own.</summary>
    public float3 GetWorldInitDirection() => SplinePathDirection.WorldStartDirection(splineContainer);

    private readonly SplinePursuit _pursuit = new();

    /// <summary>Seconds per synthesis frame, which is what the stage counts horizons in.</summary>
    private float _frameTime;

    protected override void OnEnable()
    {
        base.OnEnable();
        if (!IsBound) return;

        // Nothing carries over from a previous activation, so the first update back re-seeds.
        _pursuit.Reset();
    }

    protected override void OnUpdate()
    {
        _frameTime = 1f / math.max(1f, Synthesizer.synthesisFrameRate);

        var spline = splineContainer != null ? splineContainer.Spline : null;
        if (spline == null)
        {
            _pursuit.Reset();
            return;
        }

        _pursuit.Update(spline, splineContainer.transform.localToWorldMatrix, RootPosition);
    }

    public override bool TryGetFutureSample(int frameOffset, out float2 position,
        out float2 direction)
    {
        position = default;
        direction = default;
        if (!_pursuit.HasPath) return false;

        var distance = _pursuit.DistanceAt(frameOffset * _frameTime, speed);
        position = _pursuit.PositionAt(distance).xz;
        direction = _pursuit.DirectionAt(distance);
        return true;
    }

    /// <summary>The far end of the requested trajectory: the stage's last future horizon along the path.</summary>
    public bool TryGetSteeringTarget(out float3 worldPosition)
    {
        worldPosition = float3.zero;
        if (!_pursuit.HasPath || Stage == null) return false;

        worldPosition = _pursuit.PositionAt(_pursuit.DistanceAt(Stage.HorizonFrames * _frameTime, speed));
        return true;
    }
}
}
