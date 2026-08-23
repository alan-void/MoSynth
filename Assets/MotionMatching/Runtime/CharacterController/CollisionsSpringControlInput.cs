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
/// Both raycasts act on the simulation object, never on the synthesized character. That way the
/// trajectory handed to the search already respects the environment, so it picks an animation that
/// turns along the wall rather than one that walks into it.
/// <para>
/// Also the only control input that wires up the reconciliation calls
/// (<see cref="AdjustMotionMatching"/>), so it is the one to look at when that API is finished.
/// </para>
/// </remarks>
public class CollisionsSpringControlInput : MotionMatchingControlInput
{
    // Features ----------------------------------------------------------
    [Header("Features")] public string TrajectoryPositionFeatureName = "FuturePosition";

    public string TrajectoryDirectionFeatureName = "FutureDirection";

    // General ----------------------------------------------------------
    [Header("General")] public float MaxSpeed = 1.0f;
    [Range(0.0f, 1.0f)] public float ResponsivenessPositions = 0.75f;
    [Range(0.0f, 1.0f)] public float ResponsivenessDirections = 0.75f;
    public float MinimumVelocityClamp = 0.01f;

    [Tooltip(
        "Controls when to consider that the input has suddenly changed. Used to recompute MotionMatching. -1.0f: Never. 1.0f: Always")]
    [Range(-1.0f, 1.0f)]
    public float InputBigChangeThreshold = 0.5f;

    // Adjustment & Clamping --------------------------------------------
    [Header("Adjustment")] // Move Simulation Bone towards the Simulation Object (motion matching towards character controller)
    public bool DoAdjustment = true;

    [Range(0.0f, 2.0f)]
    public float
        PositionAdjustmentHalflife =
            0.1f; // Time needed to move half of the distance between MotionMatching and the CharacterController

    [Range(0.0f, 2.0f)] public float RotationAdjustmentHalflife = 0.1f;

    [Range(0.0f, 2.0f)]
    public float
        PosMaximumAdjustmentRatio =
            0.1f; // Ratio between the adjustment and the character's velocity to clamp the adjustment

    [Range(0.0f, 2.0f)]
    public float
        RotMaximumAdjustmentRatio =
            0.1f; // Ratio between the adjustment and the character's velocity to clamp the adjustment

    public bool DoClamping = true;

    [Range(0.0f, 2.0f)]
    public float
        MaxDistanceMMAndCharacterController = 0.1f; // Max distance between MotionMatching and the CharacterController

    // Height & Collisions -----------------------------------------------
    [Header("Height & Collisions")]
    [Tooltip("Character height in metres. Sets how high the wall probe rides and how far the floor probe reaches.")]
    public float ApproximatedPlayerHeight = 2.0f;

    [Tooltip("How far in metres the simulation object is held off a wall it hits — roughly the character's radius.")]
    public float CollisionClearance = 0.75f;
    [Header("DEBUG")] public bool DebugCurrent = true;
    public bool DebugPrediction = true;
    public bool DebugClamping = true;
    // --------------------------------------------------------------------------

    // PRIVATE ------------------------------------------------------------------
    // Input --------------------------------------------------------------------
    private float2 InputMovement;

    private bool OrientationFixed;

    // Rotation and Predicted Rotation ------------------------------------------
    private quaternion DesiredRotation; // Desired Rotation/Direction
    private quaternion[] PredictedRotations;
    private float3 AngularVelocity;

    private float3[] PredictedAngularVelocities;

    // Position and Predicted Position ------------------------------------------
    private float2[] PredictedPosition;
    private float2 Velocity;
    private float2[] PredictedVelocity;
    private float2 Acceleration;

    private float2[] PredictedAcceleration;

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
    // --------------------------------------------------------------------------

    // FUNCTIONS ---------------------------------------------------------------
    private void Start()
    {
        // Get the feature indices
        TrajectoryPosFeatureIndex = -1;
        TrajectoryRotFeatureIndex = -1;
        for (var i = 0; i < motionSynthesizer.GetMmData().trajectoryFeatures.Count; ++i)
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
        // TODO: generalize this... allow different number of prediction frames for different features
        Debug.Assert(TrajectoryPosPredictionFrames.Length == TrajectoryRotPredictionFrames.Length,
            "Trajectory Position and Trajectory Direction Prediction Frames must be the same for SpringCharacterController");
        for (var i = 0; i < TrajectoryPosPredictionFrames.Length; ++i)
        {
            Debug.Assert(TrajectoryPosPredictionFrames[i] == TrajectoryRotPredictionFrames[i],
                "Trajectory Position and Trajectory Direction Prediction Frames must be the same for SpringCharacterController");
        }

        PredictedPosition = new float2[NumberPredictionPos];
        PredictedVelocity = new float2[NumberPredictionPos];
        PredictedAcceleration = new float2[NumberPredictionPos];
        DesiredRotation = quaternion.LookRotation(transform.forward, transform.up);
        PredictedRotations = new quaternion[NumberPredictionRot];
        PredictedAngularVelocities = new float3[NumberPredictionRot];
    }

    // Input a change in the movement direction
    public void SetMovementDirection(Vector2 movementDirection)
    {
        var prevInputMovement = InputMovement;
        InputMovement = movementDirection;
        // Desired Rotation
        if (!OrientationFixed && math.length(movementDirection) > 0.0001f)
        {
            var desiredDirection = math.normalize(movementDirection);
            DesiredRotation =
                quaternion.LookRotation(new float3(desiredDirection.x, 0.0f, desiredDirection.y), transform.up);
        }

        // Input Changed Quickly
        if (math.dot(prevInputMovement, InputMovement) < InputBigChangeThreshold)
        {
            NotifyInputChangedQuickly();
        }
    }

    public void SwapFixOrientation()
    {
        OrientationFixed = !OrientationFixed;
    }

    protected override void OnUpdate()
    {
        // Rotations
        quaternion currentRotation = transform.rotation;
        PredictRotations(currentRotation, DatabaseDeltaTime);
        // Update Current Rotation
        var newRot = ComputeNewRot(currentRotation);

        // Positions
        var desiredSpeed = InputMovement * MaxSpeed;
        var currentPos = new float2(transform.position.x, transform.position.z);
        // Predict
        PredictPositions(currentPos, desiredSpeed, DatabaseDeltaTime);
        // Update Current Position
        var newPos = ComputeNewPos(currentPos, desiredSpeed);

        // Update Character Controller
        if (math.lengthsq(Velocity) > MinimumVelocityClamp * MinimumVelocityClamp)
        {
            newPos = CheckCollision(newPos, currentPos);
            // Update Transform
            transform.position = new float3(newPos.x, transform.position.y, newPos.y);
            transform.rotation = newRot;
        }

        // Adjust MotionMatching to pull the Character towards the Character Controller
        if (DoAdjustment) AdjustMotionMatching();
        if (DoClamping) ClampMotionMatching();

        // Adjust Height
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
            // Init Predicted values
            PredictedRotations[i] = currentRotation;
            PredictedAngularVelocities[i] = AngularVelocity;
            // Predict
            Spring.SimpleSpringDamperImplicit(ref PredictedRotations[i], ref PredictedAngularVelocities[i],
                DesiredRotation, 1.0f - ResponsivenessDirections, TrajectoryRotPredictionFrames[i] * averagedDeltaTime);
        }
    }

    /* https://theorangeduck.com/page/spring-roll-call#controllers */
    private void PredictPositions(float2 currentPos, float2 desiredSpeed, float averagedDeltaTime)
    {
        var lastPredictionFrames = 0;
        for (var i = 0; i < NumberPredictionPos; ++i)
        {
            if (i == 0)
            {
                PredictedPosition[i] = currentPos;
                PredictedVelocity[i] = Velocity;
                PredictedAcceleration[i] = Acceleration;
            }
            else
            {
                PredictedPosition[i] = PredictedPosition[i - 1];
                PredictedVelocity[i] = PredictedVelocity[i - 1];
                PredictedAcceleration[i] = PredictedAcceleration[i - 1];
            }

            var diffPredictionFrames = TrajectoryPosPredictionFrames[i] - lastPredictionFrames;
            lastPredictionFrames = TrajectoryPosPredictionFrames[i];
            Spring.CharacterPositionUpdate(ref PredictedPosition[i], ref PredictedVelocity[i],
                ref PredictedAcceleration[i],
                desiredSpeed, 1.0f - ResponsivenessPositions, diffPredictionFrames * averagedDeltaTime);
        }

        // Check collisions
        var prev = currentPos;
        for (var i = 0; i < NumberPredictionPos; ++i)
        {
            PredictedPosition[i] = CheckCollision(PredictedPosition[i], prev);
            prev = PredictedPosition[i];
        }
    }

    private quaternion ComputeNewRot(quaternion currentRotation)
    {
        var newRotation = currentRotation;
        Spring.SimpleSpringDamperImplicit(ref newRotation, ref AngularVelocity, DesiredRotation,
            1.0f - ResponsivenessDirections, Time.deltaTime);
        return newRotation;
    }

    private float2 ComputeNewPos(float2 currentPos, float2 desiredSpeed)
    {
        var newPos = currentPos;
        Spring.CharacterPositionUpdate(ref newPos, ref Velocity, ref Acceleration, desiredSpeed,
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
        // Clamp Position
        float3 characterController = transform.position;
        var mmPos = motionSynthesizer.RootPosition;
        if (math.distance(characterController, mmPos) > MaxDistanceMMAndCharacterController)
        {
            var newMotionMatchingPos =
                MaxDistanceMMAndCharacterController * math.normalize(mmPos - characterController) + characterController;
            motionSynthesizer.SetPosAdjustment(newMotionMatchingPos - mmPos);
        }
    }

    private void AdjustCharacterPosition()
    {
        float3 characterController = transform.position;
        var mmRot = motionSynthesizer.RootPosition;
        var differencePosition = characterController - mmRot;
        // Damp the difference using the adjustment halflife and dt
        var adjustmentPosition =
            Spring.DampAdjustmentImplicit(differencePosition, PositionAdjustmentHalflife, Time.deltaTime);
        // Clamp adjustment if the length is greater than the character velocity
        // multiplied by the ratio
        var maxLength = PosMaximumAdjustmentRatio * math.length(motionSynthesizer.RootVelocity) * Time.deltaTime;
        if (math.length(adjustmentPosition) > maxLength)
        {
            adjustmentPosition = maxLength * math.normalize(adjustmentPosition);
        }

        // Move the simulation bone towards the simulation object
        motionSynthesizer.SetPosAdjustment(adjustmentPosition);
    }

    private void AdjustCharacterRotation()
    {
        quaternion characterController = transform.rotation;
        var mmRot = motionSynthesizer.RootRotation;
        // Find the difference in rotation (from character to simulation object)
        // Note: if numerically unstable, try quaternion.Normalize(quaternion.Inverse(characterController) * motionMatching)
        var differenceRotation = math.mul(math.inverse(mmRot), characterController);
        // Damp the difference using the adjustment halflife and dt
        var adjustmentRotation =
            Spring.DampAdjustmentImplicit(differenceRotation, RotationAdjustmentHalflife, Time.deltaTime);
        // Clamp adjustment if the length is greater than the character angular velocity
        // multiplied by the ratio
        var maxLength = RotMaximumAdjustmentRatio * math.length(motionSynthesizer.RootAngularVelocity) * Time.deltaTime;
        if (math.length(MathExtensions.QuaternionToScaledAngleAxis(adjustmentRotation)) > maxLength)
        {
            adjustmentRotation = MathExtensions.QuaternionFromScaledAngleAxis(maxLength *
                                                                              math.normalize(
                                                                                  MathExtensions
                                                                                      .QuaternionToScaledAngleAxis(
                                                                                          adjustmentRotation)));
        }

        // Rotate the simulation bone towards the simulation object
        motionSynthesizer.SetRotAdjustment(adjustmentRotation);
    }

    public quaternion GetCurrentRotation()
    {
        return transform.rotation;
    }

    public override void GetTrajectoryFeature(TrajectoryFeatureChannel feature, int index, Transform character,
        Span<float> span)
    {
        if (!feature.simulationBone) Debug.Assert(false, "Trajectory should be computed using the SimulationBone");
        switch (feature.featureType)
        {
            case TrajectoryFeatureChannel.Type.Position:
                var world = PredictedPosition[index];
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
        var dir = math.mul(PredictedRotations[index], new float3(0, 0, 1));
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
        return math.length(PredictedVelocity[^1]);
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
            // Draw Current Position & Velocity
            Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f);
            Gizmos.DrawSphere(transformPos, radius);
            GizmosExtensions.DrawLine(transformPos,
                transformPos + ((Quaternion)GetCurrentRotation() * Vector3.forward) * vectorReduction, 3);
        }

        if (PredictedPosition == null || PredictedRotations == null) return;

        if (DebugPrediction)
        {
            // Draw Predicted Position & Velocity
            Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f);
            for (var i = 0; i < PredictedPosition.Length; ++i)
            {
                var predictedPos = new float3(PredictedPosition[i].x, transformPos.y, PredictedPosition[i].y);
                var predictedDir = GetWorldSpaceDirectionPrediction(i);
                var predictedDir3D = new float3(predictedDir.x, 0.0f, predictedDir.y);
                Gizmos.DrawSphere(predictedPos, radius);
                GizmosExtensions.DrawLine(predictedPos, predictedPos + predictedDir3D * vectorReduction, 3);
            }
        }

        if (DebugClamping)
        {
            // Draw Clamp Circle
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