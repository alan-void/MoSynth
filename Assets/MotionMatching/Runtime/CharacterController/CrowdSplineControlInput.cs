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
/// The workhorse for populating a scene: give each character its own spline and they walk their
/// routes while negotiating each other at crossings.
/// <para>
/// Unlike <see cref="SplineControlInput"/> the point on the spline can wait for the character to
/// catch up (<see cref="UpdateOnlyWhenCharacterMoving"/>), so avoidance does not let the target run
/// away. That also makes it a poorer path-following benchmark, since the reference reacts to the
/// character it is measuring.
/// </para>
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
    private float T;

    /// <summary>Set while the point on the spline is waiting for the character to catch up.</summary>
    private bool IsStopped;

    /// <summary>Current avoidance force. Smoothed toward the raw steering by SteeringChangeFactor.</summary>
    public float2 Steering { get; private set; }

    /// <summary>Accumulated sideways displacement from the spline, worked off once the way is clear.</summary>
    private float2 SteeringOffset;

    private float2 CurrentPosition;
    private float2 CurrentDirection;
    private float2[] PredictedPositions;
    private float2[] PredictedDirections;

    // Features -----------------------------------------------------------------
    private int TrajectoryPosFeatureIndex;
    private int TrajectoryRotFeatureIndex;
    private int[] TrajectoryPosPredictionFrames;
    private int[] TrajectoryRotPredictionFrames;

    private int NumberPredictionPos
    {
        get { return TrajectoryPosPredictionFrames.Length; }
    }

    private int NumberPredictionRot
    {
        get { return TrajectoryRotPredictionFrames.Length; }
    }

    // Crowds ------------------------------------------------------------------
    private Obstacle[] Obstacles;
    private NativeArray<(float2, float, float2)> ObstaclesCirclesArray;
    private NativeArray<int> ObstaclesCirclesArrayCount;
    private List<List<(Obstacle, bool)>> CandidateCirclesObstacles;
    private NativeArray<(float2, float2, float2)> ObstaclesEllipsesArray;
    private NativeArray<int> ObstaclesEllipsesArrayCount;
    private List<List<(Obstacle, bool)>> CandidateEllipseObstacles;
    // --------------------------------------------------------------------------

    private void Start()
    {
        // Get the feature indices
        TrajectoryPosFeatureIndex = -1;
        TrajectoryRotFeatureIndex = -1;
        for (int i = 0; i < motionSynthesizer.GetMmData().trajectoryFeatures.Count; ++i)
        {
            if (motionSynthesizer.GetMmData().trajectoryFeatures[i].name == TrajectoryPositionFeatureName)
                TrajectoryPosFeatureIndex = i;
            if (motionSynthesizer.GetMmData().trajectoryFeatures[i].name == TrajectoryDirectionFeatureName)
                TrajectoryRotFeatureIndex = i;
        }

        Debug.Assert(TrajectoryPosFeatureIndex != -1, "Trajectory Position Feature not found");
        Debug.Assert(TrajectoryRotFeatureIndex != -1, "Trajectory Direction Feature not found");

        TrajectoryPosPredictionFrames = motionSynthesizer.GetMmData().trajectoryFeatures[TrajectoryPosFeatureIndex]
            .predictionFrames;
        TrajectoryRotPredictionFrames = motionSynthesizer.GetMmData().trajectoryFeatures[TrajectoryRotFeatureIndex]
            .predictionFrames;
        // TODO: generalize this, allow for different number of prediction frames
        Debug.Assert(TrajectoryPosPredictionFrames.Length == TrajectoryRotPredictionFrames.Length,
            "Trajectory Position and Trajectory Direction Prediction Frames must be the same for PathCharacterController");
        for (int i = 0; i < TrajectoryPosPredictionFrames.Length; ++i)
        {
            Debug.Assert(TrajectoryPosPredictionFrames[i] == TrajectoryRotPredictionFrames[i],
                "Trajectory Position and Trajectory Direction Prediction Frames must be the same for PathCharacterController");
        }

        PredictedPositions = new float2[NumberPredictionPos];
        PredictedDirections = new float2[NumberPredictionRot];
        CandidateCirclesObstacles = new List<List<(Obstacle, bool)>>();
        for (int i = 0; i < NumberPredictionPos; ++i)
        {
            CandidateCirclesObstacles.Add(new List<(Obstacle, bool)>());
        }

        CandidateEllipseObstacles = new List<List<(Obstacle, bool)>>();
        for (int i = 0; i < NumberPredictionPos; ++i)
        {
            CandidateEllipseObstacles.Add(new List<(Obstacle, bool)>());
        }

        // Crowds
        OnObstaclesUpdated(ObstacleManager.Instance.GetObstacles());
    }

    private void OnEnable()
    {
        ObstacleManager.Instance.OnObstaclesUpdated += OnObstaclesUpdated;
    }

    private void OnDisable()
    {
        ObstacleManager.Instance.OnObstaclesUpdated -= OnObstaclesUpdated;
    }

    protected override void OnUpdate()
    {
        float speed = Speed / SplineContainer.CalculateLength();
        float delta = speed * DatabaseDeltaTime * 0.1f;

        void updateT()
        {
            T += speed * Time.deltaTime;
            if (Loop)
            {
                T = math.frac(T);
            }
        }

        float getTDelta(float t, float delta)
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

        float3 pos = SplineContainer.EvaluatePosition(T);
        CurrentPosition = pos.xz;
        float3 nextPos = SplineContainer.EvaluatePosition(getTDelta(T, delta));
        float2 predDir = new(nextPos.x - pos.x, nextPos.z - pos.z);
        if (math.lengthsq(predDir) < 1e-9) predDir = CurrentDirection;
        CurrentDirection = math.normalize(predDir);

        if (!IsStopped)
        {
            for (int i = 0; i < NumberPredictionPos; i++)
            {
                float t = getTDelta(T, TrajectoryPosPredictionFrames[i] * speed * DatabaseDeltaTime);
                float3 predPos = SplineContainer.EvaluatePosition(t);
                PredictedPositions[i] = predPos.xz;
                float3 predNextPos = SplineContainer.EvaluatePosition(getTDelta(t, delta));
                float2 predNextDir = new(predNextPos.x - predPos.x, predNextPos.z - predPos.z);
                if (math.lengthsq(predNextDir) < 1e-9) predNextDir = CurrentDirection;
                PredictedDirections[i] = math.normalize(predNextDir);
            }
        }

        if (UpdateOnlyWhenCharacterMoving)
        {
            float2 characterPos = new(motionSynthesizer.RootPosition.x, motionSynthesizer.RootPosition.z);
            float distance = math.length(new float2(nextPos.x - characterPos.x, nextPos.z - characterPos.y));
            float3 deltaPos =
                SplineContainer.EvaluatePosition(getTDelta(T,
                    TrajectoryPosPredictionFrames[^1] * speed * DatabaseDeltaTime));
            float distanceDelta = math.length(new float2(deltaPos.x - characterPos.x, deltaPos.z - characterPos.y));
            if (distanceDelta < distance || (!IsStopped && distance < DistanceToMove))
            {
                IsStopped = false;
                updateT();
            }
            else
            {
                IsStopped = distance > DistanceResumeMoving;
                for (int i = 0; i < NumberPredictionPos; i++)
                {
                    PredictedPositions[i] = math.lerp(PredictedPositions[i], CurrentPosition + SteeringOffset,
                        Time.deltaTime);
                    PredictedDirections[i] = PredictedDirections[i];
                }
            }
        }
        else
        {
            updateT();
            IsStopped = false;
        }

        if (DoSteering)
        {
            if (IsStopped)
            {
                // PredictedPositions was not updated
                for (int i = 0; i < NumberPredictionPos; i++)
                {
                    PredictedPositions[i] -= SteeringOffset;
                }
            }

            float2 steeringPos = new(motionSynthesizer.RootPosition.x, motionSynthesizer.RootPosition.z);
            float2 targetSteering = CrowdControlInput.ComputeSteering(steeringPos,
                new Vector3(CurrentDirection.x, 0.0f, CurrentDirection.y),
                Obstacles, SteeringLookAhead, SteeringForce, debug: DebugSteering);
            targetSteering += -SteeringOffset * SteeringSplineForce;
            Steering = math.lerp(Steering, targetSteering, Time.deltaTime * SteeringChangeFactor);
            SteeringOffset += Steering * Time.deltaTime;

            for (int i = 0; i < NumberPredictionPos; i++)
            {
                PredictedPositions[i] += SteeringOffset;
            }
        }
    }

    private void OnObstaclesUpdated(List<Obstacle> obstacles)
    {
        Obstacles = new Obstacle[obstacles.Count - (IgnoreObstacle == null ? 0 : 1)];
        int it = 0;
        for (int i = 0; i < obstacles.Count; i++)
        {
            if (obstacles[i] != IgnoreObstacle)
            {
                Obstacles[it] = obstacles[i];
                it += 1;
            }
        }
    }

    public quaternion GetCurrentRotation()
    {
        Quaternion rot = Quaternion.LookRotation(new Vector3(CurrentDirection.x, 0, CurrentDirection.y));
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


    public (
        NativeArray<(float2, float, float2)>,
        NativeArray<int>,
        NativeArray<(float2, float2, float2)>,
        NativeArray<int>
        ) GetNearbyObstacles(Transform character, float obstacleDistanceThreshold)
    {
        for (int p = 0; p < CandidateCirclesObstacles.Count; p++)
        {
            CandidateCirclesObstacles[p].Clear();
        }

        for (int p = 0; p < CandidateEllipseObstacles.Count; p++)
        {
            CandidateEllipseObstacles[p].Clear();
        }

        int candidateObstaclesCirclesCount = 0;
        int candidateObstaclesEllipsesCount = 0;
        float candidateThreshold = MaximumEllipseLength + obstacleDistanceThreshold;
        for (int p = 0; p < PredictedPositions.Length; p++)
        {
            float3 predPos = motionSynthesizer.GetMainPositionFeature(p);
            for (int i = 0; i < Obstacles.Length; i++)
            {
                Obstacle obs = Obstacles[i];
                (float3 obsPos, bool isEllipse, _) = obs.GetProjWorldPosition(p);
                if (isEllipse)
                {
                    if (math.distance(predPos, obsPos) < candidateThreshold + MaximumEllipseLength)
                    {
                        CandidateEllipseObstacles[p].Add((obs, false));
                        candidateObstaclesEllipsesCount += 1;
                    }

                    if (math.distance(predPos, obs.GetProjWorldPosition(p, forceCurrent: true).Item1) <
                        candidateThreshold + obs.Radius)
                    {
                        CandidateCirclesObstacles[p].Add((obs, true));
                        candidateObstaclesCirclesCount += 1;
                    }
                }
                else
                {
                    if (math.distance(predPos, obsPos) < candidateThreshold + obs.Radius)
                    {
                        CandidateCirclesObstacles[p].Add((obs, false));
                        candidateObstaclesCirclesCount += 1;
                    }
                }
            }
        }

        if (ObstaclesCirclesArray.IsCreated || ObstaclesCirclesArrayCount.IsCreated ||
            ObstaclesEllipsesArray.IsCreated || ObstaclesEllipsesArrayCount.IsCreated)
        {
            ObstaclesCirclesArray.Dispose();
            ObstaclesCirclesArrayCount.Dispose();
            ObstaclesEllipsesArray.Dispose();
            ObstaclesEllipsesArrayCount.Dispose();
        }

        ObstaclesCirclesArrayCount = new NativeArray<int>(CandidateCirclesObstacles.Count, Allocator.TempJob);
        ObstaclesCirclesArray =
            new NativeArray<(float2, float, float2)>(candidateObstaclesCirclesCount, Allocator.TempJob);
        ObstaclesEllipsesArrayCount = new NativeArray<int>(CandidateEllipseObstacles.Count, Allocator.TempJob);
        ObstaclesEllipsesArray =
            new NativeArray<(float2, float2, float2)>(candidateObstaclesEllipsesCount, Allocator.TempJob);
        int itCircle = 0;
        int itEllipse = 0;
        for (int p = 0; p < PredictedPositions.Length; p++)
        {
            ObstaclesCirclesArrayCount[p] = CandidateCirclesObstacles[p].Count;
            for (int i = 0; i < CandidateCirclesObstacles[p].Count; i++)
            {
                (Obstacle obstacle, bool forceCurrent) = CandidateCirclesObstacles[p][i];
                (float3 world, _, _) = obstacle.GetProjWorldPosition(p, forceCurrent: forceCurrent);
                float3 localPos = character.InverseTransformPoint(world);
                ObstaclesCirclesArray[itCircle++] = (new float2(localPos.x, localPos.z),
                    obstacle.Radius,
                    new float2(obstacle.GetMinHeightWorld(), obstacle.GetMaxHeightWorld()));
            }

            ObstaclesEllipsesArrayCount[p] = CandidateEllipseObstacles[p].Count;
            for (int i = 0; i < CandidateEllipseObstacles[p].Count; i++)
            {
                (Obstacle obstacle, bool forceCurrent) = CandidateEllipseObstacles[p][i];
                (float3 world, _, float4 ellipse) = obstacle.GetProjWorldPosition(p, forceCurrent: forceCurrent);
                float3 localPos = character.InverseTransformPoint(world);
                float3 primaryAxis = new(ellipse.x, 0.0f, ellipse.y);
                float3 secondaryAxis = new(ellipse.z, 0.0f, ellipse.w);
                primaryAxis = character.InverseTransformDirection(primaryAxis);
                secondaryAxis = character.InverseTransformDirection(secondaryAxis);
                ObstaclesEllipsesArray[itEllipse++] = (new float2(localPos.x, localPos.z),
                    new float2(primaryAxis.x, primaryAxis.z),
                    new float2(secondaryAxis.x, secondaryAxis.z));
            }
        }

        return (ObstaclesCirclesArray, ObstaclesCirclesArrayCount, ObstaclesEllipsesArray, ObstaclesEllipsesArrayCount);
    }

    private float2 GetWorldPredictedPos(int index)
    {
        return PredictedPositions[index];
    }

    private float2 GetWorldPredictedDir(int index)
    {
        return PredictedDirections[index];
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
        return new Vector3(CurrentPosition.x, 0, CurrentPosition.y);
    }

    public override float GetTargetSpeed()
    {
        return math.length(Speed);
    }

    public float2 GetPredictedPosition(int index)
    {
        return PredictedPositions[index];
    }

    SplineContainer IMotionSynthesisSplineControlInput.SplineContainer
    {
        get => SplineContainer;
        set => SplineContainer = value;
    }

    public float TargetSpeed => Speed;

    private void OnDestroy()
    {
        if (ObstaclesCirclesArray.IsCreated) ObstaclesCirclesArray.Dispose();
        if (ObstaclesCirclesArrayCount.IsCreated) ObstaclesCirclesArrayCount.Dispose();
        if (ObstaclesEllipsesArray.IsCreated) ObstaclesEllipsesArray.Dispose();
        if (ObstaclesEllipsesArrayCount.IsCreated) ObstaclesEllipsesArrayCount.Dispose();
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (!DebugDraw) return;

        if (SplineContainer == null) return;

        const float heightOffset = 0.01f;

        // Draw Current Position And Direction
        if (!Application.isPlaying) return;
        Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f);
        Vector3 currentPos = (Vector3)GetPosition() + Vector3.up * heightOffset * 2;
        Gizmos.DrawSphere(currentPos, 0.1f);
        GizmosExtensions.DrawLine(currentPos, currentPos + (Quaternion)GetCurrentRotation() * Vector3.forward, 12);

        // Draw Prediction
        if (PredictedPositions == null || PredictedPositions.Length != NumberPredictionPos ||
            PredictedDirections == null || PredictedDirections.Length != NumberPredictionRot) return;
        Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f);
        for (int i = 0; i < NumberPredictionPos; i++)
        {
            float2 predictedPosf2 = GetWorldPredictedPos(i);
            Vector3 predictedPos = new Vector3(predictedPosf2.x, heightOffset * 2, predictedPosf2.y);
            Gizmos.DrawSphere(predictedPos, 0.1f);
            float2 dirf2 = GetWorldPredictedDir(i);
            GizmosExtensions.DrawLine(predictedPos, predictedPos + new Vector3(dirf2.x, 0.0f, dirf2.y) * 0.5f, 12);
        }

        // Draw Steering
        if (DoSteering && math.lengthsq(Steering) > 0.0001f)
        {
            Gizmos.color = new Color(0.1f, 0.8f, 0.1f, 1.0f);
            GizmosExtensions.DrawLine(motionSynthesizer.RootPosition,
                motionSynthesizer.RootPosition + new float3(Steering.x, 0.0f, Steering.y) / SteeringForce, 3);
        }
    }
#endif
}
}