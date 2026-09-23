using System;
using AnimationTools;
using UnityEngine;
using Unity.Mathematics;

namespace MotionMatching
{
/// <summary>
/// <see cref="DirectionControlInput"/> plus the physical world: the simulation object is swept
/// against colliders so it cannot walk through walls, and dropped onto the floor so it follows terrain.
/// </summary>
/// <remarks>
/// Both raycasts act on the simulation object, never on the synthesized character, so the trajectory
/// handed to the search already respects the environment.
/// <para>
/// The only control input that calls the reconciliation API (<see cref="AdjustMotionMatching"/>).
/// </para>
/// </remarks>
public class CollisionsSpringControlInput : MotionMatchingControlInput
{
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

    [Header("Height & Collisions")]
    [Tooltip("Character height in metres. Sets how high the wall probe rides and how far the floor probe reaches.")]
    public float ApproximatedPlayerHeight = 2.0f;

    [Tooltip("How far in metres the simulation object is held off a wall it hits — roughly the character's radius.")]
    public float CollisionClearance = 0.75f;
    [Header("DEBUG")] public bool DebugCurrent = true;
    public bool DebugPrediction = true;
    public bool DebugClamping = true;

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

        var desiredSpeed = _inputMovement * MaxSpeed;
        var currentPos = new float2(transform.position.x, transform.position.z);
        PredictPositions(currentPos, desiredSpeed, DatabaseDeltaTime);
        var newPos = ComputeNewPos(currentPos, desiredSpeed);

        if (math.lengthsq(_velocity) > MinimumVelocityClamp * MinimumVelocityClamp)
        {
            newPos = CheckCollision(newPos, currentPos);
            transform.position = new float3(newPos.x, transform.position.y, newPos.y);
            transform.rotation = newRot;
        }

        if (DoAdjustment) AdjustMotionMatching();
        if (DoClamping) ClampMotionMatching();

        UpdateHeight();
    }

    /// <summary>
    /// Pulls <paramref name="nextPos"/> out of any wall in the way, leaving
    /// <see cref="CollisionClearance"/> of gap.
    /// </summary>
    /// <remarks>
    /// The ray starts a step <em>behind</em> the current position, so an object already slightly
    /// inside geometry still sees the surface instead of starting past it. Surfaces facing more up
    /// than sideways are ignored — those are floors, handled by <see cref="UpdateHeight"/>.
    /// </remarks>
    private float2 CheckCollision(float2 nextPos, float2 currentPos)
    {
        var height = transform.position.y + ApproximatedPlayerHeight * 0.1f;
        var nPos = new Vector3(nextPos.x, height, nextPos.y);
        var cPos = new Vector3(currentPos.x, height, currentPos.y);
        var dir = nPos - cPos;
        var mag = dir.magnitude;
        dir.Normalize();
        if (Physics.Raycast(cPos - dir * mag, dir, out var hit, mag * 2 + CollisionClearance) &&
            Vector3.Dot(hit.normal, Vector3.up) < 0.5f)
        {
            nextPos = new float2(hit.point.x, hit.point.z) - math.normalize(nextPos - currentPos) * CollisionClearance;
        }

        return nextPos;
    }

    /// <summary>
    /// Snaps the simulation object down onto the floor, so it follows terrain. With nothing beneath
    /// it, falls back to y = 0.
    /// </summary>
    private void UpdateHeight()
    {
        var floorY = 0.0f;
        var origin = transform.position + Vector3.up * (ApproximatedPlayerHeight * 0.5f);
        if (Physics.Raycast(origin, Vector3.down, out var hit, ApproximatedPlayerHeight * 0.55f))
        {
            floorY = hit.point.y;
        }

        var pos = transform.position;
        pos.y = floorY;
        transform.position = pos;
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
    /// Chains the position spring from horizon to horizon, then pulls each prediction out of walls.
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

        var prev = currentPos;
        for (var i = 0; i < NumberPredictionPos; ++i)
        {
            _predictedPosition[i] = CheckCollision(_predictedPosition[i], prev);
            prev = _predictedPosition[i];
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
        var mmPos = synthesizer.RootPosition;
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
        var mmRot = synthesizer.RootRotation;
        var differenceRotation = math.mul(math.inverse(mmRot), characterController);
        var adjustmentRotation =
            Spring.DampAdjustmentImplicit(differenceRotation, RotationAdjustmentHalflife, Time.deltaTime);
        var maxLength = RotMaximumAdjustmentRatio * math.length(synthesizer.RootAngularVelocity) * Time.deltaTime;
        if (math.length(MathExtensions.QuaternionToScaledAngleAxis(adjustmentRotation)) > maxLength)
        {
            adjustmentRotation = MathExtensions.QuaternionFromScaledAngleAxis(
                maxLength * math.normalize(MathExtensions.QuaternionToScaledAngleAxis(adjustmentRotation)));
        }

        synthesizer.SetRotAdjustment(adjustmentRotation);
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
            GizmosExtensions.DrawLine(transformPos,
                transformPos + ((Quaternion)GetCurrentRotation() * Vector3.forward) * vectorReduction, 3);
        }

        if (_predictedPosition == null || _predictedRotations == null) return;

        if (DebugPrediction)
        {
            Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f);
            for (var i = 0; i < _predictedPosition.Length; ++i)
            {
                var predictedPos = new float3(_predictedPosition[i].x, transformPos.y, _predictedPosition[i].y);
                var predictedDir = GetWorldSpaceDirectionPrediction(i);
                var predictedDir3D = new float3(predictedDir.x, 0.0f, predictedDir.y);
                Gizmos.DrawSphere(predictedPos, radius);
                GizmosExtensions.DrawLine(predictedPos, predictedPos + predictedDir3D * vectorReduction, 3);
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