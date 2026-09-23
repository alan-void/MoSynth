using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.Splines;

namespace MotionMatching
{
/// <summary>
/// Follows a spline at constant speed, looping a closed path and stopping at the end of an open one.
/// The trajectory is read straight off the spline rather than simulated, which makes this what
/// <see cref="AnimationTools.PathFollowingMetric"/> drives.
/// </summary>
/// <remarks>
/// Open loop: the point on the spline advances on its own clock and never reacts to the character,
/// so drift shows up as measurable path-following error. Negative prediction frames need no special
/// case, since the path behind the current point is where this input has been.
/// </remarks>
public class SplineControlInput : MotionMatchingControlInput, IMotionSynthesisSplineControlInput, IFrameTarget
{
    [FormerlySerializedAs("TrajectoryPositionFeatureName")] public string trajectoryPositionFeatureName = "FuturePosition";
    [FormerlySerializedAs("TrajectoryDirectionFeatureName")] public string trajectoryDirectionFeatureName = "FutureDirection";

    [FormerlySerializedAs("SplineContainer")] public SplineContainer splineContainer;

    [Tooltip("Travel speed along the spline, in m/s.")]
    [FormerlySerializedAs("Speed")] public float speed = 1.0f;

    public virtual SplineContainer SplineContainer { get => splineContainer; set => splineContainer = value; }
    public float TargetSpeed => speed;

    /// <summary>
    /// Whether there is a usable path to follow. Check before the evaluation seams below: a container
    /// with an emptied spline list throws from <c>EvaluatePosition</c> and <c>CalculateLength</c>.
    /// </summary>
    protected virtual bool HasPath => splineContainer != null && splineContainer.Spline != null;

    /// <summary>World position on the path at a normalized parameter. Overridden to follow a path that
    /// does not live in a <see cref="SplineContainer"/>.</summary>
    protected virtual float3 SamplePosition(float t) => splineContainer.EvaluatePosition(t);

    /// <summary>The path's world length, or 0 when there is no usable path. Callers must treat 0 as
    /// "do not divide".</summary>
    protected virtual float PathLength => HasPath ? splineContainer.CalculateLength() : 0f;

    /// <summary>Whether the path loops.</summary>
    protected virtual bool IsClosed => HasPath && splineContainer.Spline.Closed;

    /// <summary>
    /// Folds a normalized parameter into the path, against this input's own <see cref="IsClosed"/>.
    /// See <see cref="SplineFold"/> for why an open path clamps.
    /// </summary>
    protected float Fold(float t) => SplineFold.Normalized(t, IsClosed);

    /// <summary>
    /// Position along the spline, normalized to 0..1 and folded by <see cref="Fold"/>. Distances must
    /// be divided by the spline's length before being added to it.
    /// </summary>
    protected float _splineT;

    private float2 _currentPosition;
    private float2 _currentDirection;

    /// <summary>Lookahead for the finite-difference direction, refreshed once per update so
    /// <see cref="SampleDirection"/> does not re-measure the spline's length per sample.</summary>
    private float _directionSampleStep;

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

    /// <summary>
    /// Samples the spline now and at each prediction horizon.
    /// </summary>
    protected override void OnUpdate()
    {
        // A zero-length path would divide to infinity and leave NaN stuck in _splineT; hold instead.
        var pathLength = PathLength;
        if (pathLength <= 1e-5f) return;

        // Spline parameters are normalized, so metres per second becomes spline-fraction per second.
        var normalizedSpeed = speed / pathLength;
        _directionSampleStep = normalizedSpeed * DatabaseDeltaTime * 0.1f;

        var pos = SamplePosition(_splineT);
        _currentPosition = pos.xz;
        _currentDirection = SampleDirection(_splineT, pos);

        for (var i = 0; i < PredictionCount; i++)
        {
            var t = Fold(_splineT + _features.PredictionFrames[i] * normalizedSpeed * DatabaseDeltaTime);
            var predPos = SamplePosition(t);
            _predictedPositions[i] = predPos.xz;
            _predictedDirections[i] = SampleDirection(t, predPos);
        }

        _splineT += normalizedSpeed * Time.deltaTime;
        _splineT = Fold(_splineT);
    }

    /// <summary>
    /// The direction a character following this path should face at a point on it. The only place a
    /// direction enters the input, so the query, the gizmos and <see cref="GetCurrentRotation"/> agree.
    /// </summary>
    /// <remarks>
    /// A bare path has no facing, so this is the direction of travel, measured as a short step along
    /// the path to stay consistent with the predicted positions. See
    /// <see cref="SplinePoseKeypointControlInput"/> for a path that carries facing.
    /// </remarks>
    /// <param name="t">Normalized spline parameter to sample at.</param>
    /// <param name="positionAtT">The spline position there, already evaluated by the caller.</param>
    /// <returns>A normalized XZ direction.</returns>
    protected virtual float2 SampleDirection(float t, float3 positionAtT)
    {
        var ahead = SamplePosition(Fold(t + _directionSampleStep));
        var step = new float2(ahead.x - positionAtT.x, ahead.z - positionAtT.z);

        // At the clamped end of an open path the step forward lands back on the same point, so
        // measure the last one instead of falling through to the arbitrary default direction.
        if (math.lengthsq(step) < 1e-12f)
        {
            var behind = SamplePosition(Fold(t - _directionSampleStep));
            step = new float2(positionAtT.x - behind.x, positionAtT.z - behind.z);
        }

        return math.normalizesafe(step, new float2(0f, 1f));
    }

    /// <summary>
    /// Normalized spline parameter this input predicts for a horizon of <paramref name="frames"/>
    /// database frames ahead of the current position.
    /// </summary>
    public float GetPredictedSplineT(int frames)
    {
        var pathLength = PathLength;
        if (pathLength <= 1e-5f) return _splineT;

        var normalizedSpeed = speed / pathLength;
        return Fold(_splineT + frames * normalizedSpeed * DatabaseDeltaTime);
    }

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
    /// Where on the path this input has got to, for a stage pulling the character onto it.
    /// </summary>
    /// <remarks>
    /// The point travels along the spline while this Transform stays put, so a follower must ask
    /// rather than read the Transform. False until the first update has sampled a usable path.
    /// </remarks>
    public bool TryGetFrame(out float2 positionXZ, out float yaw)
    {
        positionXZ = _currentPosition;
        yaw = math.atan2(_currentDirection.x, _currentDirection.y);
        return HasPath && math.lengthsq(_currentDirection) > 0f;
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (!HasPath) return;

        const float heightOffset = 0.01f;

        if (!Application.isPlaying) return;
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