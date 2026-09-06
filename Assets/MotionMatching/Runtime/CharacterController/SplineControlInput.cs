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
/// The trajectory is read straight off the spline
/// rather than simulated, which makes this what <see cref="AnimationTools.PathFollowingMetric"/>
/// drives.
/// </summary>
/// <remarks>
/// Nothing here reacts to where the character actually is — the point on the spline advances on its
/// own clock, so drift shows up as measurable path-following error instead of being steered out.
/// That is what makes it a measurement tool. <see cref="CrowdSplineControlInput"/> steers.
/// <para>
/// A negative prediction frame — a trajectory feature sampling the past — needs no special case
/// here: the path behind the current point <em>is</em> where this input has been, so the same
/// spline evaluation answers it.
/// </para>
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
    /// Whether there is a usable path to follow. Guards the evaluation seams below: a container whose
    /// spline list has been emptied throws from <c>EvaluatePosition</c> and <c>CalculateLength</c>
    /// rather than answering null, so the check has to happen before the call, not inside it.
    /// </summary>
    protected virtual bool HasPath => splineContainer != null && splineContainer.Spline != null;

    /// <summary>World position on the path at a normalized parameter. The seam a subclass overrides
    /// to follow a path that does not live in a <see cref="SplineContainer"/>.</summary>
    protected virtual float3 SamplePosition(float t) => splineContainer.EvaluatePosition(t);

    /// <summary>The path's world length, or 0 when there is no usable path. Callers must treat 0 as
    /// "do not divide".</summary>
    protected virtual float PathLength => HasPath ? splineContainer.CalculateLength() : 0f;

    /// <summary>Whether the path loops. A seam like <see cref="HasPath"/>: a subclass may follow a
    /// path that does not live in a <see cref="SplineContainer"/>.</summary>
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

    // Features -----------------------------------------------------------------
    private TrajectoryFeaturePair _features;

    private int PredictionCount => _features.PredictionCount;
    // --------------------------------------------------------------------------

    protected virtual void Start()
    {
        _features = ResolveTrajectoryFeaturePair(trajectoryPositionFeatureName, trajectoryDirectionFeatureName);

        _predictedPositions = new float2[PredictionCount];
        _predictedDirections = new float2[PredictionCount];
    }

    /// <summary>
    /// Samples the spline now and at each prediction horizon. Every direction comes from
    /// <see cref="SampleDirection"/>, so the query, the gizmos and <see cref="GetCurrentRotation"/>
    /// can never end up reporting different things.
    /// </summary>
    protected override void OnUpdate()
    {
        // A path with no length would make the next line infinite, math.frac of that is NaN, and the
        // NaN sticks in _splineT and leaves as a non-finite character position. Hold where we are.
        var pathLength = PathLength;
        if (pathLength <= 1e-5f) return;

        // Spline parameters are normalized, so metres per second becomes spline-fraction per second.
        var normalizedSpeed = speed / pathLength;
        _directionSampleStep = normalizedSpeed * DatabaseDeltaTime * 0.1f;

        float3 pos = SamplePosition(_splineT);
        _currentPosition = pos.xz;
        _currentDirection = SampleDirection(_splineT, pos);

        for (int i = 0; i < PredictionCount; i++)
        {
            var t = Fold(_splineT + _features.PredictionFrames[i] * normalizedSpeed * DatabaseDeltaTime);
            float3 predPos = SamplePosition(t);
            _predictedPositions[i] = predPos.xz;
            _predictedDirections[i] = SampleDirection(t, predPos);
        }

        _splineT += normalizedSpeed * Time.deltaTime;
        _splineT = Fold(_splineT);
    }

    /// <summary>
    /// The direction a character following this path should face at a point on it. This is the only
    /// place a direction enters the input, so an override reaches the trajectory query, the debug
    /// gizmos and <see cref="GetCurrentRotation"/> together.
    /// </summary>
    /// <remarks>
    /// A bare path carries no facing of its own, so the direction of travel is the best answer
    /// available here — measured as a short step further along rather than as the analytic tangent,
    /// which keeps it consistent with the predicted positions. A path that does carry facing
    /// overrides this; see <see cref="SplinePoseKeypointControlInput"/>.
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
    /// database frames ahead of the current position. Public because it is the only readable account
    /// of where on its path this input thinks it is, which tools and diagnostics need.
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
        Quaternion rot = Quaternion.LookRotation(new Vector3(_currentDirection.x, 0, _currentDirection.y));
        return rot;
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
    /// The direction the path sets off in. A pure path follower has no notion of facing beyond its
    /// path, so this is the honest answer; the Transform's own forward is arbitrary for a character
    /// that gets spawned onto a path it was never authored against.
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
    /// The point travels along the spline while this component's Transform stays where it was
    /// dropped, so a follower has to ask rather than read that Transform. Answers false until the
    /// first update has sampled a usable path.
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

        // Draw Current Position And Direction
        if (!Application.isPlaying) return;
        Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f);
        Vector3 currentPos = (Vector3)GetPosition() + Vector3.up * heightOffset * 2;
        Gizmos.DrawSphere(currentPos, 0.1f);
        GizmosExtensions.DrawLine(currentPos, currentPos + (Quaternion)GetCurrentRotation() * Vector3.forward, 12);
        // Draw Prediction
        if (_predictedPositions == null || _predictedPositions.Length != PredictionCount ||
            _predictedDirections == null || _predictedDirections.Length != PredictionCount) return;
        Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f);
        for (int i = 0; i < PredictionCount; i++)
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