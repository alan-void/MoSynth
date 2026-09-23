using System;
using System.Collections.Generic;
using AnimationTools;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;

namespace MotionMatching
{
/// <summary>
/// <see cref="DirectionControlInput"/> for a crowd: the same stick-driven spring simulation object,
/// with a steering force that pushes it around other moving characters before they are walked into.
/// </summary>
/// <remarks>
/// Two separate mechanisms: <b>steering</b> (<see cref="ComputeSteering"/>) deflects the simulation
/// object around neighbours, while <b>obstacle features</b> (<see cref="GetNearbyObstacles"/>) put
/// nearby obstacles into the query so the search can prefer animations recorded while avoiding
/// something. Either works alone.
/// </remarks>
public class CrowdControlInput : MotionMatchingControlInput, IObstacleAwareCharacterControler
{
    [Header("Crowd")]
    [Tooltip("This character's own obstacle, excluded so it does not steer around itself.")]
    public Obstacle IgnoreObstacle;

    public bool DoSteering = false;

    [Tooltip("How far ahead to look for obstacles to steer around, in metres.")]
    public float SteeringLookAhead = 4.0f;

    [Tooltip("Strength of the sideways avoidance push at closest approach.")]
    public float SteeringForce = 2.0f;

    [Tooltip("Smoothing rate for the steering vector, so avoidance eases in rather than snapping.")]
    public float SteeringChangeFactor = 5.0f;

    [Tooltip("Cap on an obstacle ellipse's semi-axis when packed into the query features, in metres.")]
    public float MaximumEllipseLength = 0.9f;

    [Header("Features")] public string TrajectoryPositionFeatureName = "FuturePosition";

    public string TrajectoryDirectionFeatureName = "FutureDirection";

    [Header("General")] public float MaxSpeed = 1.0f;
    [Range(0.0f, 1.0f)] public float ResponsivenessPositions = 0.75f;
    [Range(0.0f, 1.0f)] public float ResponsivenessDirections = 0.75f;
    public float MinimumVelocityClamp = 0.01f;

    [Tooltip(
        "Controls when to consider that the input has suddenly changed. Used to recompute MotionMatching. -1.0f: Never. 1.0f: Always")]
    [Range(-1.0f, 1.0f)]
    public float InputBigChangeThreshold = 0.5f;

    // Adjustment pulls the synthesized character toward this simulation object.
    [Header("Adjustment")]
    public bool DoAdjustment = true;

    [Range(0.0f, 2.0f)] public float PositionAdjustmentHalflife = 0.1f; // Seconds to close half the gap

    [Range(0.0f, 2.0f)] public float RotationAdjustmentHalflife = 0.1f;

    // Caps each correction to this fraction of the character's own velocity.
    [Range(0.0f, 2.0f)] public float PosMaximumAdjustmentRatio = 0.1f;

    [Range(0.0f, 2.0f)] public float RotMaximumAdjustmentRatio = 0.1f;

    public bool DoClamping = true;

    [Range(0.0f, 2.0f)] public float MaxDistanceMMAndCharacterController = 0.1f; // Metres

    [Header("DEBUG")] public bool DebugCurrent = true;
    public bool DebugPrediction = true;
    public bool DebugClamping = true;
    public bool DebugSteering = false;

    private float2 _inputMovement;

    private bool _orientationFixed;

    private quaternion _desiredRotation;
    private quaternion[] _predictedRotations;
    private float3 _angularVelocity;

    private float3[] _predictedAngularVelocities;

    private float2[] _predictedPosition;
    private float2 _velocity;
    private float2[] _predictedVelocity;
    private float2 _acceleration;

    private float2[] _predictedAcceleration;

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
    public float2 Steering { get; private set; }

    private void Start()
    {
        _trajectoryPosFeatureIndex = -1;
        _trajectoryRotFeatureIndex = -1;
        for (var i = 0; i < synthesizer.GetMmData().trajectoryFeatures.Count; ++i)
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
            "Trajectory Position and Trajectory Direction Prediction Frames must be the same for SpringCharacterController");
        for (var i = 0; i < _trajectoryPosPredictionFrames.Length; ++i)
        {
            Debug.Assert(_trajectoryPosPredictionFrames[i] == _trajectoryRotPredictionFrames[i],
                "Trajectory Position and Trajectory Direction Prediction Frames must be the same for SpringCharacterController");
        }

        _predictedPosition = new float2[NumberPredictionPos];
        _predictedVelocity = new float2[NumberPredictionPos];
        _predictedAcceleration = new float2[NumberPredictionPos];
        _desiredRotation = quaternion.LookRotation(transform.forward, transform.up);
        _predictedRotations = new quaternion[NumberPredictionRot];
        _predictedAngularVelocities = new float3[NumberPredictionRot];
        _candidateCirclesObstacles = new List<List<(Obstacle, bool)>>();
        for (var i = 0; i < NumberPredictionPos; ++i)
        {
            _candidateCirclesObstacles.Add(new List<(Obstacle, bool)>());
        }

        _candidateEllipseObstacles = new List<List<(Obstacle, bool)>>();
        for (var i = 0; i < NumberPredictionPos; ++i)
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

    /// <summary>Feeds in a new movement direction; a sharp enough change forces an immediate search.</summary>
    public void SetMovementDirection(Vector2 movementDirection)
    {
        var prevInputMovement = _inputMovement;
        _inputMovement = movementDirection;
        if (!_orientationFixed && math.length(movementDirection) > 0.0001f)
        {
            var desiredDirection = math.normalize(movementDirection);
            _desiredRotation =
                quaternion.LookRotation(new float3(desiredDirection.x, 0.0f, desiredDirection.y), transform.up);
        }

        if (math.dot(prevInputMovement, _inputMovement) < InputBigChangeThreshold)
        {
            NotifyInputChangedQuickly();
        }
    }

    public void SwapFixOrientation()
    {
        _orientationFixed = !_orientationFixed;
    }

    protected override void OnUpdate()
    {
        quaternion currentRotation = transform.rotation;
        PredictRotations(currentRotation, DatabaseDeltaTime);
        var newRot = ComputeNewRot(currentRotation);
        transform.rotation = newRot;

        float2 currentPos = new(synthesizer.RootPosition.x, synthesizer.RootPosition.z);
        var desiredSpeed = _inputMovement * MaxSpeed;
        if (DoSteering)
        {
            var targetSteering = ComputeSteering(currentPos, transform.forward, _obstacles, SteeringLookAhead,
                SteeringForce, debug: DebugSteering);
            Steering = math.lerp(Steering, targetSteering, Time.deltaTime * SteeringChangeFactor);
            desiredSpeed += Steering;
        }

        PredictPositions(currentPos, desiredSpeed, DatabaseDeltaTime);
        // Called only to advance the velocity spring; the position follows the synthesized root.
        ComputeNewPos(currentPos, desiredSpeed);

        transform.position = new float3(currentPos.x, transform.position.y, currentPos.y);

        if (DoAdjustment) AdjustMotionMatching();
        if (DoClamping) ClampMotionMatching();
    }

    private void PredictRotations(quaternion currentRotation, float averagedDeltaTime)
    {
        for (var i = 0; i < NumberPredictionRot; i++)
        {
            _predictedRotations[i] = currentRotation;
            _predictedAngularVelocities[i] = _angularVelocity;
            Spring.SimpleSpringDamperImplicit(ref _predictedRotations[i], ref _predictedAngularVelocities[i],
                _desiredRotation, 1.0f - ResponsivenessDirections, _trajectoryRotPredictionFrames[i] * averagedDeltaTime);
        }
    }

    /// <summary>
    /// Chains the position spring from horizon to horizon.
    /// <see href="https://theorangeduck.com/page/spring-roll-call#controllers">Spring roll call</see>.
    /// </summary>
    private void PredictPositions(float2 currentPos, float2 desiredSpeed, float averagedDeltaTime)
    {
        var lastPredictionFrames = 0;
        for (var i = 0; i < NumberPredictionPos; ++i)
        {
            if (i == 0)
            {
                _predictedPosition[i] = currentPos;
                _predictedVelocity[i] = _velocity;
                _predictedAcceleration[i] = _acceleration;
            }
            else
            {
                _predictedPosition[i] = _predictedPosition[i - 1];
                _predictedVelocity[i] = _predictedVelocity[i - 1];
                _predictedAcceleration[i] = _predictedAcceleration[i - 1];
            }

            var diffPredictionFrames = _trajectoryPosPredictionFrames[i] - lastPredictionFrames;
            lastPredictionFrames = _trajectoryPosPredictionFrames[i];
            Spring.CharacterPositionUpdate(ref _predictedPosition[i], ref _predictedVelocity[i],
                ref _predictedAcceleration[i],
                desiredSpeed, 1.0f - ResponsivenessPositions, diffPredictionFrames * averagedDeltaTime);
        }
    }

    private quaternion ComputeNewRot(quaternion currentRotation)
    {
        var newRotation = currentRotation;
        Spring.SimpleSpringDamperImplicit(ref newRotation, ref _angularVelocity, _desiredRotation,
            1.0f - ResponsivenessDirections, Time.deltaTime);
        return newRotation;
    }

    private float2 ComputeNewPos(float2 currentPos, float2 desiredSpeed)
    {
        var newPos = currentPos;
        Spring.CharacterPositionUpdate(ref newPos, ref _velocity, ref _acceleration, desiredSpeed,
            1.0f - ResponsivenessPositions, Time.deltaTime);
        return newPos;
    }

    private void AdjustMotionMatching()
    {
        AdjustCharacterPosition();
        AdjustCharacterRotation();
    }

    private void ClampMotionMatching()
    {
        float3 characterController = transform.position;
        var mmPos = synthesizer.RootPosition;
        if (math.distance(characterController, mmPos) > MaxDistanceMMAndCharacterController)
        {
            var newMotionMatchingPos =
                MaxDistanceMMAndCharacterController * math.normalize(mmPos - characterController) + characterController;
            synthesizer.SetPosAdjustment(newMotionMatchingPos - mmPos);
        }
    }

    private void AdjustCharacterPosition()
    {
        float3 characterController = transform.position;
        float3 mmPos = synthesizer.RootPosition;
        var differencePosition = characterController - mmPos;
        var adjustmentPosition =
            Spring.DampAdjustmentImplicit(differencePosition, PositionAdjustmentHalflife, Time.deltaTime);
        var maxLength = PosMaximumAdjustmentRatio * math.length(synthesizer.RootVelocity) * Time.deltaTime;
        if (math.length(adjustmentPosition) > maxLength)
        {
            adjustmentPosition = maxLength * math.normalize(adjustmentPosition);
        }

        synthesizer.SetPosAdjustment(adjustmentPosition);
    }

    private void AdjustCharacterRotation()
    {
        quaternion characterController = transform.rotation;
        quaternion mmRot = synthesizer.RootRotation;
        var differenceRotation = math.mul(math.inverse(mmRot), characterController);
        var adjustmentRotation =
            Spring.DampAdjustmentImplicit(differenceRotation, RotationAdjustmentHalflife, Time.deltaTime);
        var maxLength = RotMaximumAdjustmentRatio * math.length(synthesizer.RootAngularVelocity) *
                        Time.deltaTime;
        if (math.length(MathExtensions.QuaternionToScaledAngleAxis(adjustmentRotation)) > maxLength)
        {
            adjustmentRotation = MathExtensions.QuaternionFromScaledAngleAxis(
                maxLength * math.normalize(MathExtensions.QuaternionToScaledAngleAxis(adjustmentRotation)));
        }

        synthesizer.SetRotAdjustment(adjustmentRotation);
    }

    /// <summary>
    /// Sideways avoidance force for whichever moving obstacle is most immediately in the way.
    /// Shared by <see cref="CrowdControlInput"/> and <see cref="CrowdSplineControlInput"/>, hence
    /// static and parameterized rather than reading fields.
    /// </summary>
    /// <remarks>
    /// Casts a fan of rays ahead. Only the closest hit steers, since averaging several steers nowhere;
    /// the force is perpendicular to forward (a sidestep, not a brake) with a log10 falloff. When the
    /// obstacle is itself steering, this takes the opposite side so the two do not dodge into each
    /// other. Static obstacles are skipped.
    /// </remarks>
    /// <param name="lookAhead">How far ahead to look, in metres, and the range beyond which an
    /// obstacle exerts no force.</param>
    /// <returns>A world-space XZ steering vector, or zero when nothing is in the way.</returns>
    public static float2 ComputeSteering(float2 currentPos, float3 currentForward, Obstacle[] obstacles,
        float lookAhead, float force, float fovAngle = 30.0f, int numRays = 20,
        bool debug = false)
    {
        var bestSteering = float2.zero;
        var closestObstacleDistance = lookAhead;

        var forwardAngle = math.degrees(math.atan2(currentForward.z, currentForward.x));
        var angleIncrement = fovAngle / (numRays - 1);

        for (var i = 0; i < numRays; i++)
        {
            var angle = forwardAngle - fovAngle / 2f + i * angleIncrement;
            float2 rayDirection = new(math.cos(math.radians(angle)), math.sin(math.radians(angle)));

            if (debug)
            {
                Debug.DrawRay(new Vector3(currentPos.x, 0.0f, currentPos.y),
                    new float3(rayDirection.x, 0.0f, rayDirection.y) * lookAhead, Color.black);
            }

            var resHitDistance = float.MaxValue;
            Obstacle resObstacle = null;

            for (var j = 0; j < obstacles.Length; j++)
            {
                if (obstacles[j].IsStatic) continue;

                if (obstacles[j].Intersect(currentPos, rayDirection, out var hitPoint1, out var hitDistance1,
                        out var hitPoint2, out var hitDistance2))
                {
                    var hitDistance = math.min(hitDistance1, hitDistance2);

                    if (hitDistance < resHitDistance && hitDistance < lookAhead)
                    {
                        resHitDistance = hitDistance;
                        resObstacle = obstacles[j];
                    }
                }
            }

            if (resHitDistance < lookAhead)
            {
                float2 forwardProj = new(currentForward.x, currentForward.z);
                var localSteering = math.normalize(forwardProj) * force *
                                    math.max(0.0f, math.log10(1.0f - (resHitDistance / lookAhead)) + 1.0f);
                localSteering = new float2(-localSteering.y, localSteering.x);

                if (resObstacle != null && resObstacle.GetCurrentSteering(out var obsSteering))
                {
                    obsSteering = math.normalize(obsSteering);
                    var dot1 = math.dot(math.normalize(localSteering), obsSteering);
                    var dot2 = math.dot(math.normalize(-localSteering), obsSteering);
                    if (dot1 > dot2)
                    {
                        localSteering = -localSteering;
                    }
                }

                if (resHitDistance < closestObstacleDistance)
                {
                    closestObstacleDistance = resHitDistance;
                    bestSteering = localSteering;
                }
            }
        }

        return bestSteering;
    }

    private void OnObstaclesUpdated(List<Obstacle> obstacles)
    {
        _obstacles = new Obstacle[obstacles.Count - (IgnoreObstacle == null ? 0 : 1)];
        var it = 0;
        for (var i = 0; i < obstacles.Count; i++)
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
        return transform.rotation;
    }

    public override void GetTrajectoryFeature(TrajectoryFeatureChannel feature, int index, Transform character,
        Span<float> span)
    {
        if (!feature.simulationBone) Debug.Assert(false, "Trajectory should be computed using the simulation frame");
        switch (feature.featureType)
        {
            case TrajectoryFeatureChannel.Type.Position:
                var world = _predictedPosition[index];
                float3 local = character.InverseTransformPoint(new float3(world.x, 0.0f, world.y));
                span[0] = local.x;
                span[1] = local.z;
                break;
            case TrajectoryFeatureChannel.Type.Direction:
                var dirProjected = GetWorldSpaceDirectionPrediction(index);
                float3 localDir =
                    character.InverseTransformDirection(new Vector3(dirProjected.x, 0.0f, dirProjected.y));
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
        for (var p = 0; p < _candidateCirclesObstacles.Count; p++)
        {
            _candidateCirclesObstacles[p].Clear();
        }

        for (var p = 0; p < _candidateEllipseObstacles.Count; p++)
        {
            _candidateEllipseObstacles[p].Clear();
        }

        var candidateObstaclesCirclesCount = 0;
        var candidateObstaclesEllipsesCount = 0;
        var candidateThreshold = MaximumEllipseLength + obstacleDistanceThreshold;
        for (var p = 0; p < _predictedPosition.Length; p++)
        {
            float3 predPos = synthesizer.GetMainPositionFeature(p);
            for (var i = 0; i < _obstacles.Length; i++)
            {
                var obs = _obstacles[i];
                (var obsPos, var isEllipse, _) = obs.GetProjWorldPosition(p);
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
        var itCircle = 0;
        var itEllipse = 0;
        for (var p = 0; p < _predictedPosition.Length; p++)
        {
            _obstaclesCirclesArrayCount[p] = _candidateCirclesObstacles[p].Count;
            for (var i = 0; i < _candidateCirclesObstacles[p].Count; i++)
            {
                (var obstacle, var forceCurrent) = _candidateCirclesObstacles[p][i];
                (var world, _, _) = obstacle.GetProjWorldPosition(p, forceCurrent: forceCurrent);
                float3 localPos = character.InverseTransformPoint(world);
                _obstaclesCirclesArray[itCircle++] = (new float2(localPos.x, localPos.z),
                    obstacle.Radius,
                    new float2(obstacle.GetMinHeightWorld(), obstacle.GetMaxHeightWorld()));
            }

            _obstaclesEllipsesArrayCount[p] = _candidateEllipseObstacles[p].Count;
            for (var i = 0; i < _candidateEllipseObstacles[p].Count; i++)
            {
                (var obstacle, var forceCurrent) = _candidateEllipseObstacles[p][i];
                (var world, _, var ellipse) = obstacle.GetProjWorldPosition(p, forceCurrent: forceCurrent);
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

    private float2 GetWorldSpaceDirectionPrediction(int index)
    {
        var dir = math.mul(_predictedRotations[index], new float3(0, 0, 1));
        return math.normalize(new float2(dir.x, dir.z));
    }

    public override float3 GetWorldInitPosition()
    {
        return transform.position;
    }

    public override float3 GetWorldInitDirection()
    {
        return transform.forward;
    }

    public override float3 GetPosition()
    {
        return transform.position;
    }

    public override float GetTargetSpeed()
    {
        return math.length(_predictedVelocity[^1]);
    }

    public float2 GetPredictedPosition(int index)
    {
        return _predictedPosition[index];
    }

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
        const float radius = 0.05f;
        const float vectorReduction = 0.5f;
        const float verticalOffset = 0.05f;
        var transformPos = (Vector3)GetPosition() + Vector3.up * verticalOffset;
        if (DebugCurrent)
        {
            Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f);
            Gizmos.DrawSphere(transformPos, radius);
            GizmosExtensions.DrawArrow(transformPos,
                transformPos + ((Quaternion)GetCurrentRotation() * Vector3.forward) * vectorReduction, 0.1f,
                thickness: 2);
        }

        if (_predictedPosition == null || _predictedRotations == null) return;

        if (DebugPrediction)
        {
            Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f);
            for (var i = 0; i < _predictedPosition.Length; ++i)
            {
                float3 predictedPos = new(_predictedPosition[i].x, verticalOffset, _predictedPosition[i].y);
                var predictedDir = GetWorldSpaceDirectionPrediction(i);
                float3 predictedDir3D = new(predictedDir.x, 0.0f, predictedDir.y);
                Gizmos.DrawSphere(predictedPos, radius);
                GizmosExtensions.DrawArrow(predictedPos, predictedPos + predictedDir3D * vectorReduction, 0.1f,
                    thickness: 2);
            }

            if (DoSteering && math.lengthsq(Steering) > 0.0001f)
            {
                Gizmos.color = new Color(0.1f, 0.8f, 0.1f, 1.0f);
                GizmosExtensions.DrawArrow(transformPos,
                    transformPos + new Vector3(Steering.x, 0.0f, Steering.y) * vectorReduction, 0.1f, thickness: 2);
            }
        }

        if (DebugClamping)
        {
            if (DoClamping)
            {
                Gizmos.color = new Color(0.1f, 1.0f, 0.1f, 1.0f);
                GizmosExtensions.DrawWireCircle(transformPos, MaxDistanceMMAndCharacterController, quaternion.identity);
            }
        }
    }
#endif
}
}