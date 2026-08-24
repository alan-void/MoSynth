using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.Splines;

namespace MotionMatching
{
/// <summary>
/// Follows a spline at constant speed, looping. The trajectory is read straight off the spline
/// rather than simulated, which makes this what <see cref="AnimationTools.PathFollowingMetric"/>
/// drives.
/// </summary>
/// <remarks>
/// Nothing here reacts to where the character actually is — the point on the spline advances on its
/// own clock, so drift shows up as measurable path-following error instead of being steered out.
/// That is what makes it a measurement tool. <see cref="CrowdSplineControlInput"/> steers.
/// </remarks>
public class SplineControlInput : MotionMatchingControlInput, IMotionSynthesisSplineControlInput
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

    /// <summary>
    /// Position along the spline, normalized to 0..1 and wrapped. Distances must be divided by the
    /// spline's length before being added to it.
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
    private int _trajectoryPosFeatureIndex;
    private int _trajectoryRotFeatureIndex;
    private int[] _trajectoryPosPredictionFrames;
    private int[] _trajectoryRotPredictionFrames;

    private int NumberPredictionPos
    {
        get { return _trajectoryPosPredictionFrames.Length; }
    }

    private int NumberPredictionRot
    {
        get { return _trajectoryRotPredictionFrames.Length; }
    }
    // --------------------------------------------------------------------------

    protected virtual void Start()
    {
        // Get the feature indices
        _trajectoryPosFeatureIndex = -1;
        _trajectoryRotFeatureIndex = -1;
        var mmData = motionSynthesizer.GetMmData();
        for (int i = 0; i < mmData.trajectoryFeatures.Count; ++i)
        {
            if (mmData.trajectoryFeatures[i].name == trajectoryPositionFeatureName)
                _trajectoryPosFeatureIndex = i;
            if (mmData.trajectoryFeatures[i].name == trajectoryDirectionFeatureName)
                _trajectoryRotFeatureIndex = i;
        }

        Debug.Assert(_trajectoryPosFeatureIndex != -1, "Trajectory Position Feature not found");
        Debug.Assert(_trajectoryRotFeatureIndex != -1, "Trajectory Direction Feature not found");

        _trajectoryPosPredictionFrames = mmData.trajectoryFeatures[_trajectoryPosFeatureIndex]
            .predictionFrames;
        _trajectoryRotPredictionFrames = mmData.trajectoryFeatures[_trajectoryRotFeatureIndex]
            .predictionFrames;
        // TODO: generalize this, allow for different number of prediction frames
        Debug.Assert(_trajectoryPosPredictionFrames.Length == _trajectoryRotPredictionFrames.Length,
            "Trajectory Position and Trajectory Direction Prediction Frames must be the same for PathCharacterController");
        for (int i = 0; i < _trajectoryPosPredictionFrames.Length; ++i)
        {
            Debug.Assert(_trajectoryPosPredictionFrames[i] == _trajectoryRotPredictionFrames[i],
                "Trajectory Position and Trajectory Direction Prediction Frames must be the same for PathCharacterController");
        }

        _predictedPositions = new float2[NumberPredictionPos];
        _predictedDirections = new float2[NumberPredictionRot];
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

        for (int i = 0; i < NumberPredictionPos; i++)
        {
            var t = math.frac(_splineT + _trajectoryPosPredictionFrames[i] * normalizedSpeed * DatabaseDeltaTime);
            float3 predPos = SamplePosition(t);
            _predictedPositions[i] = predPos.xz;
            _predictedDirections[i] = SampleDirection(t, predPos);
        }

        _splineT += normalizedSpeed * Time.deltaTime;
        _splineT = math.frac(_splineT);
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
        float3 next = SamplePosition(math.frac(t + _directionSampleStep));
        return math.normalizesafe(new float2(next.x - positionAtT.x, next.z - positionAtT.z), new float2(0f, 1f));
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
        return math.frac(_splineT + frames * normalizedSpeed * DatabaseDeltaTime);
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
                float2 world = GetWorldPredictedPos(index);
                float3 local = character.InverseTransformPoint(new float3(world.x, 0.0f, world.y));
                span[0] = local.x;
                span[1] = local.z;
                break;
            case TrajectoryFeatureChannel.Type.Direction:
                float2 worldDir = GetWorldPredictedDir(index);
                float3 localDir = character.InverseTransformDirection(new Vector3(worldDir.x, 0.0f, worldDir.y));
                span[0] = localDir.x;
                span[1] = localDir.z;
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
        if (_predictedPositions == null || _predictedPositions.Length != NumberPredictionPos ||
            _predictedDirections == null || _predictedDirections.Length != NumberPredictionRot) return;
        Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f);
        for (int i = 0; i < NumberPredictionPos; i++)
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