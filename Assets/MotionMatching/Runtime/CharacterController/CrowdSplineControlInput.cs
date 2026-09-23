using System;
using System.Collections.Generic;
using AnimationTools;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

namespace MotionMatching
{
/// <summary>
/// Follows a spline through a crowd: <see cref="SplineControlInput"/>'s fixed path, plus
/// <see cref="CrowdControlInput"/>'s avoidance. The path sets where the character is going; steering
/// offsets it sideways around whoever is in the way, and it settles back onto the path once clear.
/// </summary>
/// <remarks>
/// Unlike <see cref="SplineControlInput"/> the point on the spline can wait for the character to
/// catch up (<see cref="UpdateOnlyWhenCharacterMoving"/>), so avoidance does not let the target run
/// away. That makes it a poor path-following benchmark: the reference reacts to what it measures.
/// </remarks>
public class CrowdSplineControlInput : MotionMatchingControlInput, IObstacleAwareCharacterControler, IMotionSynthesisSplineControlInput
{
    public string TrajectoryPositionFeatureName = "FuturePosition";
    public string TrajectoryDirectionFeatureName = "FutureDirection";

    [Header("Crowds")]
    [Tooltip("This character's own obstacle, excluded so it does not steer around itself.")]
    public Obstacle IgnoreObstacle;

    public SplineContainer SplineContainer;

    [Tooltip("Wrap back to the start of the spline on reaching the end.")]
    public bool Loop = true;

    [Tooltip("Travel speed along the spline, in m/s.")]
    public float Speed = 1.0f;

    [Tooltip("Hold the point on the spline back when the character falls behind it, instead of " +
             "advancing on its own clock. Keeps the target reachable after avoidance costs time.")]
    public bool UpdateOnlyWhenCharacterMoving = false;

    [Tooltip(
        "If UpdateOnlyWhenCharacterMoving is true, the character will only move if the distance to the next point is greater than this value")]
    public float DistanceToMove = 1.0f;

    [Tooltip(
        "If UpdateOnlyWhenCharacterMoving is true and DistanceToMove was not reached, the character will resume moving if the distance to the next point is greater than this value")]
    public float DistanceResumeMoving = 0.75f;

    public bool DoSteering = false;
    public float SteeringLookAhead = 4.0f;
    public float SteeringForce = 2.0f;
    public float SteeringSplineForce = 0.5f;
    public float SteeringChangeFactor = 5.0f;
    public float MaximumEllipseLength = 0.9f;

    public bool DebugDraw = true;
    public bool DebugSteering = false;

    /// <summary>Position along the spline, normalized to 0..1.</summary>
    private float _splineT;

    /// <summary>Set while the point on the spline is waiting for the character to catch up.</summary>
    private bool _isStopped;

    /// <summary>Current avoidance force. Smoothed toward the raw steering by SteeringChangeFactor.</summary>
    public float2 Steering { get; private set; }

    /// <summary>Accumulated sideways displacement from the spline, worked off once the way is clear.</summary>
    private float2 _steeringOffset;

    private float2 _currentPosition;
    private float2 _currentDirection;
    private float2[] _predictedPositions;
    private float2[] _predictedDirections;

    private int _trajectoryPosFeatureIndex;
    private int _trajectoryRotFeatureIndex;
    private int[] _trajectoryPosPredictionFrames;
    private int[] _trajectoryRotPredictionFrames;

    private int NumberPredictionPos => _trajectoryPosPredictionFrames.Length;

    private int NumberPredictionRot => _trajectoryRotPredictionFrames.Length;

    private Obstacle[] _obstacles;
    private NativeArray<(float2, float, float2)> _obstaclesCirclesArray;
    private NativeArray<int> _obstaclesCirclesArrayCount;
    private List<List<(Obstacle, bool)>> _candidateCirclesObstacles;
    private NativeArray<(float2, float2, float2)> _obstaclesEllipsesArray;
    private NativeArray<int> _obstaclesEllipsesArrayCount;
    private List<List<(Obstacle, bool)>> _candidateEllipseObstacles;

    private void Start()
    {
        _trajectoryPosFeatureIndex = -1;
        _trajectoryRotFeatureIndex = -1;
        for (int i = 0; i < synthesizer.GetMmData().trajectoryFeatures.Count; ++i)
        {
            if (synthesizer.GetMmData().trajectoryFeatures[i].name == TrajectoryPositionFeatureName)
                _trajectoryPosFeatureIndex = i;
            if (synthesizer.GetMmData().trajectoryFeatures[i].name == TrajectoryDirectionFeatureName)
                _trajectoryRotFeatureIndex = i;
        }

        Debug.Assert(_trajectoryPosFeatureIndex != -1, "Trajectory Position Feature not found");
        Debug.Assert(_trajectoryRotFeatureIndex != -1, "Trajectory Direction Feature not found");

        _trajectoryPosPredictionFrames = synthesizer.GetMmData().trajectoryFeatures[_trajectoryPosFeatureIndex]
            .predictionFrames;
        _trajectoryRotPredictionFrames = synthesizer.GetMmData().trajectoryFeatures[_trajectoryRotFeatureIndex]
            .predictionFrames;
        Debug.Assert(_trajectoryPosPredictionFrames.Length == _trajectoryRotPredictionFrames.Length,
            "Trajectory Position and Trajectory Direction Prediction Frames must be the same for PathCharacterController");
        for (int i = 0; i < _trajectoryPosPredictionFrames.Length; ++i)
        {
            Debug.Assert(_trajectoryPosPredictionFrames[i] == _trajectoryRotPredictionFrames[i],
                "Trajectory Position and Trajectory Direction Prediction Frames must be the same for PathCharacterController");
        }

        _predictedPositions = new float2[NumberPredictionPos];
        _predictedDirections = new float2[NumberPredictionRot];
        _candidateCirclesObstacles = new List<List<(Obstacle, bool)>>();
        for (int i = 0; i < NumberPredictionPos; ++i)
        {
            _candidateCirclesObstacles.Add(new List<(Obstacle, bool)>());
        }

        _candidateEllipseObstacles = new List<List<(Obstacle, bool)>>();
        for (int i = 0; i < NumberPredictionPos; ++i)
        {
            _candidateEllipseObstacles.Add(new List<(Obstacle, bool)>());
        }

        OnObstaclesUpdated(ObstacleManager.Instance.GetObstacles());
    }

    protected override void OnEnable()
    {
        base.OnEnable();
        if (!IsBound) return;

        ObstacleManager.Instance.OnObstaclesUpdated += OnObstaclesUpdated;
    }

    protected override void OnDisable()
    {
        base.OnDisable();
        ObstacleManager.Instance.OnObstaclesUpdated -= OnObstaclesUpdated;
    }

    protected override void OnUpdate()
    {
        float speed = Speed / SplineContainer.CalculateLength();
        float delta = speed * DatabaseDeltaTime * 0.1f;

        void UpdateT()
        {
            _splineT += speed * Time.deltaTime;
            if (Loop)
            {
                _splineT = math.frac(_splineT);
            }
        }

        float GetTDelta(float t, float delta)
        {
            float tDelta = t + delta;
            if (SplineContainer.Spline.Closed)
            {
                tDelta = math.frac(tDelta);
            }
            else if (tDelta > 1.0f)
            {
                tDelta = 1.0f;
            }

            return tDelta;
        }

        float3 pos = SplineContainer.EvaluatePosition(_splineT);
        _currentPosition = pos.xz;
        float3 nextPos = SplineContainer.EvaluatePosition(GetTDelta(_splineT, delta));
        float2 predDir = new(nextPos.x - pos.x, nextPos.z - pos.z);
        if (math.lengthsq(predDir) < 1e-9) predDir = _currentDirection;
        _currentDirection = math.normalize(predDir);

        if (!_isStopped)
        {
            for (int i = 0; i < NumberPredictionPos; i++)
            {
                float t = GetTDelta(_splineT, _trajectoryPosPredictionFrames[i] * speed * DatabaseDeltaTime);
                float3 predPos = SplineContainer.EvaluatePosition(t);
                _predictedPositions[i] = predPos.xz;
                float3 predNextPos = SplineContainer.EvaluatePosition(GetTDelta(t, delta));
                float2 predNextDir = new(predNextPos.x - predPos.x, predNextPos.z - predPos.z);
                if (math.lengthsq(predNextDir) < 1e-9) predNextDir = _currentDirection;
                _predictedDirections[i] = math.normalize(predNextDir);
            }
        }

        if (UpdateOnlyWhenCharacterMoving)
        {
            float2 characterPos = new(synthesizer.RootPosition.x, synthesizer.RootPosition.z);
            float distance = math.length(new float2(nextPos.x - characterPos.x, nextPos.z - characterPos.y));
            float3 deltaPos =
                SplineContainer.EvaluatePosition(GetTDelta(_splineT,
                    _trajectoryPosPredictionFrames[^1] * speed * DatabaseDeltaTime));
            float distanceDelta = math.length(new float2(deltaPos.x - characterPos.x, deltaPos.z - characterPos.y));
            if (distanceDelta < distance || (!_isStopped && distance < DistanceToMove))
            {
                _isStopped = false;
                UpdateT();
            }
            else
            {
                _isStopped = distance > DistanceResumeMoving;
                for (int i = 0; i < NumberPredictionPos; i++)
                {
                    _predictedPositions[i] = math.lerp(_predictedPositions[i], _currentPosition + _steeringOffset,
                        Time.deltaTime);
                }
            }
        }
        else
        {
            UpdateT();
            _isStopped = false;
        }

        if (DoSteering)
        {
            if (_isStopped)
            {
                // The predictions were not refreshed this frame, so take off last frame's offset first.
                for (int i = 0; i < NumberPredictionPos; i++)
                {
                    _predictedPositions[i] -= _steeringOffset;
                }
            }

            float2 steeringPos = new(synthesizer.RootPosition.x, synthesizer.RootPosition.z);
            float2 targetSteering = CrowdControlInput.ComputeSteering(steeringPos,
                new Vector3(_currentDirection.x, 0.0f, _currentDirection.y),
                _obstacles, SteeringLookAhead, SteeringForce, debug: DebugSteering);
            targetSteering += -_steeringOffset * SteeringSplineForce;
            Steering = math.lerp(Steering, targetSteering, Time.deltaTime * SteeringChangeFactor);
            _steeringOffset += Steering * Time.deltaTime;

            for (int i = 0; i < NumberPredictionPos; i++)
            {
                _predictedPositions[i] += _steeringOffset;
            }
        }
    }

    private void OnObstaclesUpdated(List<Obstacle> obstacles)
    {
        _obstacles = new Obstacle[obstacles.Count - (IgnoreObstacle == null ? 0 : 1)];
        int it = 0;
        for (int i = 0; i < obstacles.Count; i++)
        {
            if (obstacles[i] != IgnoreObstacle)
            {
                _obstacles[it] = obstacles[i];
                it += 1;
            }
        }
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

    public (
        NativeArray<(float2, float, float2)>,
        NativeArray<int>,
        NativeArray<(float2, float2, float2)>,
        NativeArray<int>
        ) GetNearbyObstacles(Transform character, float obstacleDistanceThreshold)
    {
        for (int p = 0; p < _candidateCirclesObstacles.Count; p++)
        {
            _candidateCirclesObstacles[p].Clear();
        }

        for (int p = 0; p < _candidateEllipseObstacles.Count; p++)
        {
            _candidateEllipseObstacles[p].Clear();
        }

        int candidateObstaclesCirclesCount = 0;
        int candidateObstaclesEllipsesCount = 0;
        float candidateThreshold = MaximumEllipseLength + obstacleDistanceThreshold;
        for (int p = 0; p < _predictedPositions.Length; p++)
        {
            float3 predPos = synthesizer.GetMainPositionFeature(p);
            for (int i = 0; i < _obstacles.Length; i++)
            {
                Obstacle obs = _obstacles[i];
                (float3 obsPos, bool isEllipse, _) = obs.GetProjWorldPosition(p);
                if (isEllipse)
                {
                    if (math.distance(predPos, obsPos) < candidateThreshold + MaximumEllipseLength)
                    {
                        _candidateEllipseObstacles[p].Add((obs, false));
                        candidateObstaclesEllipsesCount += 1;
                    }

                    if (math.distance(predPos, obs.GetProjWorldPosition(p, forceCurrent: true).Item1) <
                        candidateThreshold + obs.Radius)
                    {
                        _candidateCirclesObstacles[p].Add((obs, true));
                        candidateObstaclesCirclesCount += 1;
                    }
                }
                else
                {
                    if (math.distance(predPos, obsPos) < candidateThreshold + obs.Radius)
                    {
                        _candidateCirclesObstacles[p].Add((obs, false));
                        candidateObstaclesCirclesCount += 1;
                    }
                }
            }
        }

        if (_obstaclesCirclesArray.IsCreated || _obstaclesCirclesArrayCount.IsCreated ||
            _obstaclesEllipsesArray.IsCreated || _obstaclesEllipsesArrayCount.IsCreated)
        {
            _obstaclesCirclesArray.Dispose();
            _obstaclesCirclesArrayCount.Dispose();
            _obstaclesEllipsesArray.Dispose();
            _obstaclesEllipsesArrayCount.Dispose();
        }

        _obstaclesCirclesArrayCount = new NativeArray<int>(_candidateCirclesObstacles.Count, Allocator.TempJob);
        _obstaclesCirclesArray =
            new NativeArray<(float2, float, float2)>(candidateObstaclesCirclesCount, Allocator.TempJob);
        _obstaclesEllipsesArrayCount = new NativeArray<int>(_candidateEllipseObstacles.Count, Allocator.TempJob);
        _obstaclesEllipsesArray =
            new NativeArray<(float2, float2, float2)>(candidateObstaclesEllipsesCount, Allocator.TempJob);
        int itCircle = 0;
        int itEllipse = 0;
        for (int p = 0; p < _predictedPositions.Length; p++)
        {
            _obstaclesCirclesArrayCount[p] = _candidateCirclesObstacles[p].Count;
            for (int i = 0; i < _candidateCirclesObstacles[p].Count; i++)
            {
                (Obstacle obstacle, bool forceCurrent) = _candidateCirclesObstacles[p][i];
                (float3 world, _, _) = obstacle.GetProjWorldPosition(p, forceCurrent: forceCurrent);
                float3 localPos = character.InverseTransformPoint(world);
                _obstaclesCirclesArray[itCircle++] = (new float2(localPos.x, localPos.z),
                    obstacle.Radius,
                    new float2(obstacle.GetMinHeightWorld(), obstacle.GetMaxHeightWorld()));
            }

            _obstaclesEllipsesArrayCount[p] = _candidateEllipseObstacles[p].Count;
            for (int i = 0; i < _candidateEllipseObstacles[p].Count; i++)
            {
                (Obstacle obstacle, bool forceCurrent) = _candidateEllipseObstacles[p][i];
                (float3 world, _, float4 ellipse) = obstacle.GetProjWorldPosition(p, forceCurrent: forceCurrent);
                float3 localPos = character.InverseTransformPoint(world);
                float3 primaryAxis = new(ellipse.x, 0.0f, ellipse.y);
                float3 secondaryAxis = new(ellipse.z, 0.0f, ellipse.w);
                primaryAxis = character.InverseTransformDirection(primaryAxis);
                secondaryAxis = character.InverseTransformDirection(secondaryAxis);
                _obstaclesEllipsesArray[itEllipse++] = (new float2(localPos.x, localPos.z),
                    new float2(primaryAxis.x, primaryAxis.z),
                    new float2(secondaryAxis.x, secondaryAxis.z));
            }
        }

        return (_obstaclesCirclesArray, _obstaclesCirclesArrayCount, _obstaclesEllipsesArray, _obstaclesEllipsesArrayCount);
    }

    private float2 GetWorldPredictedPos(int index)
    {
        return _predictedPositions[index];
    }

    private float2 GetWorldPredictedDir(int index)
    {
        return _predictedDirections[index];
    }

    public override float3 GetWorldInitPosition()
    {
        return SplineContainer.EvaluatePosition(0.0f);
    }

    public override float3 GetWorldInitDirection()
    {
        float3 start = SplineContainer.EvaluatePosition(0.0f);
        float3 delta = SplineContainer.EvaluatePosition(0.01f) - start;
        return math.normalize(new float3(delta.x, 0.0f, delta.z));
    }

    public override float3 GetPosition()
    {
        return new Vector3(_currentPosition.x, 0, _currentPosition.y);
    }

    public override float GetTargetSpeed()
    {
        return math.length(Speed);
    }

    public float2 GetPredictedPosition(int index)
    {
        return _predictedPositions[index];
    }

    SplineContainer IMotionSynthesisSplineControlInput.SplineContainer
    {
        get => SplineContainer;
        set => SplineContainer = value;
    }

    public float TargetSpeed => Speed;

    private void OnDestroy()
    {
        if (_obstaclesCirclesArray.IsCreated) _obstaclesCirclesArray.Dispose();
        if (_obstaclesCirclesArrayCount.IsCreated) _obstaclesCirclesArrayCount.Dispose();
        if (_obstaclesEllipsesArray.IsCreated) _obstaclesEllipsesArray.Dispose();
        if (_obstaclesEllipsesArrayCount.IsCreated) _obstaclesEllipsesArrayCount.Dispose();
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (!DebugDraw) return;

        if (SplineContainer == null) return;

        const float heightOffset = 0.01f;

        if (!Application.isPlaying) return;
        Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f);
        Vector3 currentPos = (Vector3)GetPosition() + Vector3.up * heightOffset * 2;
        Gizmos.DrawSphere(currentPos, 0.1f);
        GizmosExtensions.DrawLine(currentPos, currentPos + (Quaternion)GetCurrentRotation() * Vector3.forward, 12);

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

        if (DoSteering && math.lengthsq(Steering) > 0.0001f)
        {
            Gizmos.color = new Color(0.1f, 0.8f, 0.1f, 1.0f);
            GizmosExtensions.DrawLine(synthesizer.RootPosition,
                synthesizer.RootPosition + new float3(Steering.x, 0.0f, Steering.y) / SteeringForce, 3);
        }
    }
#endif
}
}