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

    public SplineContainer SplineContainer { get => splineContainer; set => splineContainer = value; }
    public float TargetSpeed => speed;

    /// <summary>
    /// Position along the spline, normalized to 0..1 and wrapped. Distances must be divided by the
    /// spline's length before being added to it.
    /// </summary>
    private float _splineT;

    private float2 _currentPosition;
    private float2 _currentDirection;
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

    private void Start()
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
    /// Samples the spline now and at each prediction horizon. Directions come from a short step
    /// further along rather than the analytic tangent, keeping position and direction consistent.
    /// </summary>
    protected override void OnUpdate()
    {
        // Spline parameters are normalized, so metres per second becomes spline-fraction per second.
        var normalizedSpeed = speed / splineContainer.CalculateLength();

        // Lookahead used for the finite-difference direction. Small relative to a frame's travel, so
        // it measures the tangent rather than a chord across curvature.
        var directionSampleStep = normalizedSpeed * DatabaseDeltaTime * 0.1f;

        float3 pos = splineContainer.EvaluatePosition(_splineT);
        _currentPosition = pos.xz;
        float3 nextPos = splineContainer.EvaluatePosition(math.frac(_splineT + directionSampleStep));
        _currentDirection = math.normalize(new float2(nextPos.x - pos.x, nextPos.z - pos.z));

        for (int i = 0; i < NumberPredictionPos; i++)
        {
            var t = math.frac(_splineT + _trajectoryPosPredictionFrames[i] * normalizedSpeed * DatabaseDeltaTime);
            float3 predPos = splineContainer.EvaluatePosition(t);
            _predictedPositions[i] = predPos.xz;
            float3 predNextPos = splineContainer.EvaluatePosition(math.frac(t + directionSampleStep));
            _predictedDirections[i] = math.normalize(new float2(predNextPos.x - predPos.x, predNextPos.z - predPos.z));
        }

        _splineT += normalizedSpeed * Time.deltaTime;
        _splineT = math.frac(_splineT);
    }

    public quaternion GetCurrentRotation()
    {
        Quaternion rot = Quaternion.LookRotation(new Vector3(_currentDirection.x, 0, _currentDirection.y));
        return rot;
    }

    public override void GetTrajectoryFeature(TrajectoryFeatureChannel feature, int index, Transform character,
        Span<float> span)
    {
        if (!feature.simulationBone) Debug.Assert(false, "Trajectory should be computed using the SimulationBone");
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

    public override float3 GetWorldInitDirection()
    {
        return math.normalize(new float3(transform.forward.x, 0, transform.forward.z));
    }

    public override float GetTargetSpeed()
    {
        return speed;
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (splineContainer == null) return;

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