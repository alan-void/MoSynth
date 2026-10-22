using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.Splines;

namespace MotionMatching
{
/// <summary>
/// Follows a spline at a target speed, looping a closed path and stopping at the end of an open one.
/// The trajectory is read straight off the spline rather than simulated, which makes this what
/// <see cref="AnimationTools.PathFollowingMetric"/> drives.
/// </summary>
/// <remarks>
/// Closed loop, through <see cref="SplinePursuit"/>: every horizon is laid along the path from the
/// point nearest the character, plus a lead that grows while the character is slower than
/// <see cref="speed"/> and is capped at <see cref="maxLead"/>, so a slow character is asked for more
/// but never chases a point that ran away. Negative prediction frames look back from that same start.
/// </remarks>
public class SplineControlInput : MotionMatchingControlInput, IMotionSynthesisSplineControlInput, IFrameTarget,
    ISteeringTargetSource
{
    [FormerlySerializedAs("TrajectoryPositionFeatureName")] public string trajectoryPositionFeatureName = "FuturePosition";
    [FormerlySerializedAs("TrajectoryDirectionFeatureName")] public string trajectoryDirectionFeatureName = "FutureDirection";

    [FormerlySerializedAs("SplineContainer")] public SplineContainer splineContainer;

    [Tooltip("Speed the trajectory is laid along the spline at, from the character's nearest point, in m/s.")]
    [FormerlySerializedAs("Speed")] public float speed = SplinePursuit.DefaultTargetSpeed;

    [Tooltip("Most the trajectory may start ahead of the character's nearest point, in metres. The lead " +
             "builds while the character is slower than the target speed; 0 lays it from the nearest point.")]
    [Min(0f)] public float maxLead = SplinePursuit.DefaultMaxLead;

    public virtual SplineContainer SplineContainer
    {
        get => splineContainer;
        set
        {
            splineContainer = value;
            // A benchmark sweep hands this component a different spline for every run, and where
            // the character was on the old one means nothing on the new one.
            Pursuit.Reset();
        }
    }

    public float TargetSpeed => speed;

    /// <summary>The character's projection onto the path, refreshed once per update.</summary>
    protected SplinePursuit Pursuit { get; } = new();

    /// <summary>
    /// The path in its own local space, and the transform placing it in the world. Overridden to
    /// follow a path that does not live in a <see cref="SplineContainer"/>.
    /// </summary>
    /// <remarks>
    /// Check the spline list too: a container with an emptied list throws from <c>Spline</c>'s
    /// evaluation.
    /// </remarks>
    protected virtual bool TryGetPath(out Spline spline, out float4x4 localToWorld)
    {
        spline = splineContainer != null ? splineContainer.Spline : null;
        localToWorld = spline != null ? splineContainer.transform.localToWorldMatrix : float4x4.identity;
        return spline != null;
    }

    private float2 _currentPosition;
    private float2 _currentDirection;

    private float2 _framePosition;
    private float2 _frameDirection;

    private float2[] _predictedPositions;
    private float2[] _predictedDirections;

    private TrajectoryFeaturePair _features;

    private int PredictionCount => _features.PredictionCount;

    protected virtual void Start()
    {
        _features = ResolveTrajectoryFeaturePair(trajectoryPositionFeatureName, trajectoryDirectionFeatureName);

        _predictedPositions = new float2[PredictionCount];
        _predictedDirections = new float2[PredictionCount];
    }

    protected override void OnEnable()
    {
        base.OnEnable();
        if (!IsBound) return;

        // Nothing carries over from a previous activation, so the first update back re-seeds.
        Pursuit.Reset();
    }

    /// <summary>
    /// Projects the character onto the spline and samples it at each prediction horizon from there.
    /// </summary>
    protected override void OnUpdate()
    {
        if (!TryGetPath(out var spline, out var localToWorld))
        {
            Pursuit.Reset();
            return;
        }

        if (!Pursuit.Update(spline, localToWorld, synthesizer.transform.position)) return;
        Pursuit.AdvanceLead(speed, Time.deltaTime, maxLead);

        var nearest = Pursuit.NearestDistance;
        _currentPosition = Pursuit.NearestWorld.xz;
        _currentDirection = SampleDirection(nearest);

        for (var i = 0; i < PredictionCount; i++)
        {
            var distance = Pursuit.DistanceAt(_features.PredictionFrames[i] * DatabaseDeltaTime, speed);
            _predictedPositions[i] = Pursuit.PositionAt(distance).xz;
            _predictedDirections[i] = SampleDirection(distance);
        }

        // One frame of travel past the projection, so a follower landing the character there
        // carries it along the path at the target speed rather than pinning it in place. Without
        // the lead: a follower closes the gap itself, so a lead would only make it jump.
        var frameDistance = Pursuit.NearestDistance + Time.deltaTime * speed;
        _framePosition = Pursuit.PositionAt(frameDistance).xz;
        _frameDirection = SampleDirection(frameDistance);
    }

    /// <summary>
    /// The direction a character following this path should face at a point on it. The only place a
    /// direction enters the input, so the query, the gizmos and <see cref="GetCurrentRotation"/> agree.
    /// </summary>
    /// <remarks>
    /// A bare path has no facing, so this is the direction of travel. See
    /// <see cref="SplinePoseKeypointControlInput"/> for a path that carries facing.
    /// </remarks>
    /// <param name="distance">Distance along the path, in metres; folded by the caller's path.</param>
    /// <returns>A normalized XZ direction.</returns>
    protected virtual float2 SampleDirection(float distance) => Pursuit.DirectionAt(distance);

    /// <summary>
    /// Normalized spline parameter this input predicts for a horizon of <paramref name="frames"/>
    /// database frames ahead of the character's nearest point on the path.
    /// </summary>
    public float GetPredictedSplineT(int frames) =>
        Pursuit.HasPath ? Pursuit.NormalizedAt(Pursuit.DistanceAt(frames * DatabaseDeltaTime, speed)) : 0f;

    public quaternion GetCurrentRotation()
    {
        return Quaternion.LookRotation(new Vector3(_currentDirection.x, 0, _currentDirection.y));
    }

    public override void GetTrajectoryFeature(TrajectoryFeatureChannel feature, int index, Transform character,
        Span<float> span)
    {
        if (!feature.simulationBone) Debug.Assert(false, "Trajectory should be computed using the simulation frame");
        switch (feature.featureType)
        {
            case TrajectoryFeatureChannel.Type.Position:
                WritePlanarPosition(character, GetWorldPredictedPos(index), span);
                break;
            case TrajectoryFeatureChannel.Type.Direction:
                WritePlanarDirection(character, GetWorldPredictedDir(index), span);
                break;
            default:
                Debug.Assert(false, "Unknown feature type: " + feature.featureType);
                break;
        }
    }

    private float2 GetWorldPredictedPos(int index)
    {
        return _predictedPositions[index];
    }

    private float2 GetWorldPredictedDir(int index)
    {
        return _predictedDirections[index];
    }

    /// <summary>The point on the path nearest the character.</summary>
    public override float3 GetPosition()
    {
        return new Vector3(_currentPosition.x, 0, _currentPosition.y);
    }

    public override float3 GetWorldInitPosition()
    {
        return transform.position;
    }

    /// <summary>
    /// The direction the path sets off in, falling back to the Transform's forward only when the path
    /// has none: the Transform is arbitrary for a character spawned onto a path.
    /// </summary>
    public override float3 GetWorldInitDirection()
    {
        var start = SplinePathDirection.WorldStartDirection(splineContainer);
        return math.lengthsq(start) > 0f
            ? start
            : math.normalizesafe(new float3(transform.forward.x, 0, transform.forward.z), math.forward());
    }

    public override float GetTargetSpeed()
    {
        return speed;
    }

    /// <summary>
    /// Where on the path the character should be after this frame, for a stage pulling it onto the
    /// path: its own projection, one frame of travel further on at <see cref="speed"/>.
    /// </summary>
    /// <remarks>
    /// The point moves along the spline while this Transform stays put, so a follower must ask
    /// rather than read the Transform. False until the first update has found a usable path.
    /// </remarks>
    public bool TryGetFrame(out float2 positionXZ, out float yaw)
    {
        positionXZ = _framePosition;
        yaw = math.atan2(_frameDirection.x, _frameDirection.y);
        return Pursuit.HasPath && math.lengthsq(_frameDirection) > 0f;
    }

    /// <summary>The far end of the queried trajectory: the last prediction horizon along the path.</summary>
    public bool TryGetSteeringTarget(out float3 worldPosition)
    {
        worldPosition = float3.zero;
        if (!Pursuit.HasPath || _predictedPositions == null || PredictionCount == 0) return false;

        var lastFrames = _features.PredictionFrames[PredictionCount - 1];
        worldPosition = Pursuit.PositionAt(Pursuit.DistanceAt(lastFrames * DatabaseDeltaTime, speed));
        return true;
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (!Application.isPlaying || !Pursuit.HasPath) return;

        const float heightOffset = 0.01f;

        Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f);
        Vector3 currentPos = (Vector3)GetPosition() + Vector3.up * heightOffset * 2;
        Gizmos.DrawSphere(currentPos, 0.1f);
        GizmosExtensions.DrawLine(currentPos, currentPos + (Quaternion)GetCurrentRotation() * Vector3.forward, 12);
        if (_predictedPositions == null || _predictedPositions.Length != PredictionCount ||
            _predictedDirections == null || _predictedDirections.Length != PredictionCount) return;
        Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f);
        for (var i = 0; i < PredictionCount; i++)
        {
            float2 predictedPosf2 = GetWorldPredictedPos(i);
            Vector3 predictedPos = new Vector3(predictedPosf2.x, heightOffset * 2, predictedPosf2.y);
            Gizmos.DrawSphere(predictedPos, 0.1f);
            float2 dirf2 = GetWorldPredictedDir(i);
            GizmosExtensions.DrawLine(predictedPos, predictedPos + new Vector3(dirf2.x, 0.0f, dirf2.y) * 0.5f, 12);
        }
    }
#endif
}
}
