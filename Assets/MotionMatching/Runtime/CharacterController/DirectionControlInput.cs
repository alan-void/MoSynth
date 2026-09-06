using System;
using AnimationTools;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using UnityEngine.Serialization;

namespace MotionMatching
{
/// <summary>
/// Stick/WASD control, and the simplest example of the model described on
/// <see cref="MotionMatchingControlInput"/>.
/// </summary>
/// <remarks>
/// A spring moves this component's Transform toward the velocity the stick asks for, smoothing the
/// jerky input into something a body could do. Running that same spring further ahead, with no new
/// input, is what produces the predicted trajectory.
/// <para>
/// Maths from <a href="https://theorangeduck.com/page/spring-roll-call#controllers">spring roll
/// call</a>.
/// </para>
/// </remarks>
public class DirectionControlInput : MotionMatchingControlInput, IMotionSynthesisDirectionControlInput
{
    [Header("Features")]
    [Tooltip("Name of the database trajectory feature holding future positions.")]
    public string trajectoryPositionFeatureName = "FuturePosition";

    [Tooltip("Name of the database trajectory feature holding future facing directions.")]
    public string trajectoryDirectionFeatureName = "FutureDirection";

    [Header("General")]
    [Tooltip("Speed at full stick deflection, in m/s.")]
    public float maxSpeed = 1.0f;

    [Tooltip("How hard the position spring pulls toward the desired velocity. 1 = snap, 0 = ignore input.")]
    [Range(0.0f, 1.0f)] public float responsivenessPositions = 0.75f;

    [Tooltip("How hard the facing spring turns toward the input direction. 1 = snap, 0 = never turn.")]
    [Range(0.0f, 1.0f)] public float responsivenessDirections = 0.75f;

    [Tooltip("Speed below which the simulation object stops being moved at all, so it settles cleanly.")]
    public float minimumVelocityClamp = 0.01f;

    [Tooltip(
        "Controls when to consider that the input has suddenly changed. Used to recompute MotionMatching. -1.0f: Never. 1.0f: Always")]
    [Range(-1.0f, 1.0f)]
    public float inputBigChangeThreshold = 0.5f;

    // Settings for the reconciliation methods further down, which are not currently called.

    [Tooltip("Time to close half the gap between the synthesized character and this simulation object.")]
    [Range(0.0f, 2.0f)] public float positionAdjustmentHalflife = 0.1f;

    [Tooltip("As positionAdjustmentHalflife, for facing.")]
    [FormerlySerializedAs("rotationAdjustmentHalflife")] [Range(0.0f, 2.0f)]
    public float rotationAdjustmentHalfLife = 0.1f;

    [Tooltip("Caps the per-frame correction to this fraction of the character's own travel, so it " +
             "never outruns the animation and looks like sliding.")]
    [Range(0.0f, 2.0f)] public float posMaximumAdjustmentRatio = 0.1f;

    [Tooltip("As posMaximumAdjustmentRatio, against angular velocity.")]
    [Range(0.0f, 2.0f)] public float rotMaximumAdjustmentRatio = 0.1f;

    public bool doClamping = true;

    [Tooltip("Hard leash: how far the synthesized character may drift from this simulation object, in metres.")]
    [Range(0.0f, 2.0f)] public float maxDistanceMmAndCharacterController = 0.1f;

    [Header("DEBUG")] public bool debugCurrent = true;
    public bool debugPrediction = true;
    public bool debugClamping = true;

    // --- Input ----------------------------------------------------------------------------------

    /// <summary>Latest stick/WASD vector, in the XZ plane. Length scales speed up to <see cref="maxSpeed"/>.</summary>
    private float2 _inputMovement;

    /// <summary>While set, the character keeps its facing and strafes instead of turning to face travel.</summary>
    private bool _orientationFixed;

    // --- Facing: current spring state, then one predicted state per horizon ----------------------

    /// <summary>Where the facing spring is being pulled toward, i.e. the input direction.</summary>
    private quaternion _desiredRotation;

    private quaternion[] _predictedRotations;
    private float3 _angularVelocity;
    private float3[] _predictedAngularVelocities;

    // --- Position: current spring state, then one predicted state per horizon --------------------

    private float2[] _predictedPosition;
    private float2 _velocity;
    private float2[] _predictedVelocity;
    private float2 _acceleration;
    private float2[] _predictedAcceleration;

    // --- Resolved database feature layout, cached in Start --------------------------------------

    private TrajectoryFeaturePair _features;

    // --- Past horizons --------------------------------------------------------------------------

    /// <summary>
    /// Where the simulation object has been, for the negative prediction frames of a trajectory
    /// feature that samples the past. Null while the database asks for no history.
    /// </summary>
    private TrajectoryHistory _history;

    /// <summary>
    /// Samples per second the history is sized to hold. The spring is stepped once per rendered
    /// frame, so a render rate above this quietly shortens the window it can answer for.
    /// </summary>
    private const float HistorySampleRate = 240f;

    private int PredictionCount => _features.PredictionCount;

    /// <summary>Resolves the named trajectory features and sizes the prediction arrays.</summary>
    private void Start()
    {
        _features = ResolveTrajectoryFeaturePair(trajectoryPositionFeatureName, trajectoryDirectionFeatureName);

        _predictedPosition = new float2[PredictionCount];
        _predictedVelocity = new float2[PredictionCount];
        _predictedAcceleration = new float2[PredictionCount];
        _desiredRotation = quaternion.LookRotation(transform.forward, transform.up);
        _predictedRotations = new quaternion[PredictionCount];
        _predictedAngularVelocities = new float3[PredictionCount];

        var mmData = motionSynthesizer.GetMmData();
        var historyFrames = mmData.MaximumFramesHistory;
        if (historyFrames > 0)
        {
            var historySeconds = historyFrames * mmData.GetOrImportPoseSet().FrameTime;
            _history = new TrajectoryHistory(Mathf.CeilToInt(historySeconds * HistorySampleRate) + 2);
        }
    }

    /// <summary>
    /// Feeds in a new movement direction; nothing here polls the device. A sharp enough reversal
    /// forces an immediate search, so the character does not keep running the wrong way.
    /// </summary>
    public void SetMovementDirection(Vector2 movementDirection)
    {
        var prevInputMovement = _inputMovement;
        _inputMovement = movementDirection;
        // Desired Rotation
        if (!_orientationFixed && math.length(movementDirection) > 0.0001f)
        {
            var desiredDirection = math.normalize(movementDirection);
            _desiredRotation =
                quaternion.LookRotation(new float3(desiredDirection.x, 0.0f, desiredDirection.y), transform.up);
        }

        // Input Changed Quickly
        if (math.dot(prevInputMovement, _inputMovement) < inputBigChangeThreshold)
        {
            NotifyInputChangedQuickly();
        }
    }

    /// <summary>Toggles strafing: facing stops following the movement direction.</summary>
    public void SwapFixOrientation()
    {
        _orientationFixed = !_orientationFixed;
    }

    /// <summary>Advances the simulation object one frame and refreshes the predicted trajectory.</summary>
    protected override void OnUpdate()
    {
        // Rotations
        quaternion currentRotation = transform.rotation;
        PredictRotations(currentRotation, DatabaseDeltaTime);
        // Update Current Rotation
        var newRot = ComputeNewRot(currentRotation);

        // Positions
        var desiredSpeed = _inputMovement * maxSpeed;
        var currentPos = new float2(transform.position.x, transform.position.z);
        // Predict
        PredictPositions(currentPos, desiredSpeed, DatabaseDeltaTime);
        // Update Current Position
        var newPos = ComputeNewPos(currentPos, desiredSpeed);

        // Update Character Controller
        if (math.lengthsq(_velocity) > minimumVelocityClamp * minimumVelocityClamp)
        {
            // Update Transform
            transform.position = new float3(newPos.x, transform.position.y, newPos.y);
            transform.rotation = newRot;
        }

        RecordHistory();

        // if (DoClamping) ClampMotionMatching();
    }

    /// <summary>
    /// Appends where the simulation object ended up this frame, which is what a negative prediction
    /// frame is answered from.
    /// </summary>
    private void RecordHistory()
    {
        if (_history == null) return;

        var position = transform.position;
        var forward = transform.forward;
        _history.Record(Time.time, PlanarPosition(transform), PlanarForward(transform));
    }

    /// <summary>
    /// Where the simulation object was <paramref name="framesBack"/> database frames ago, falling
    /// back to where it is now for the first second or so of a run, before the history reaches that
    /// far back — which is what a character that had been standing still would have recorded anyway.
    /// </summary>
    private void GetPastState(int framesBack, out float2 position, out float2 forward)
    {
        if (_history != null &&
            _history.TrySample(Time.time - framesBack * DatabaseDeltaTime, out position, out forward))
        {
            return;
        }

        position = PlanarPosition(transform);
        forward = PlanarForward(transform);
    }

    /// <summary>
    /// Facing at each horizon. The same spring as <see cref="ComputeNewRot"/>, jumped straight to
    /// each horizon in one step — the implicit form is exact at any step size. Past horizons hold
    /// the current facing; <see cref="GetTrajectoryFeature"/> answers those from the history.
    /// </summary>
    private void PredictRotations(quaternion currentRotation, float averagedDeltaTime)
    {
        for (var i = 0; i < PredictionCount; i++)
        {
            // Init Predicted values
            _predictedRotations[i] = currentRotation;
            _predictedAngularVelocities[i] = _angularVelocity;
            if (_features.PredictionFrames[i] < 0) continue;

            // Predict
            Spring.SimpleSpringDamperImplicit(ref _predictedRotations[i], ref _predictedAngularVelocities[i],
                _desiredRotation, 1.0f - responsivenessDirections,
                _features.PredictionFrames[i] * averagedDeltaTime);
        }
    }

    /// <summary>
    /// Position at each horizon. Unlike facing, these must be chained — each horizon continues from
    /// the previous one, because this spring carries acceleration and cannot be jumped. Past
    /// horizons hold the current state and are skipped by the chain;
    /// <see cref="GetTrajectoryFeature"/> answers those from the history.
    /// </summary>
    /// <remarks>Horizons must be in ascending order, since the chain steps by their differences.</remarks>
    /* https://theorangeduck.com/page/spring-roll-call#controllers */
    private void PredictPositions(float2 currentPos, float2 desiredSpeed, float averagedDeltaTime)
    {
        var lastPredictionFrames = 0;
        var position = currentPos;
        var velocity = _velocity;
        var acceleration = _acceleration;

        for (var i = 0; i < PredictionCount; ++i)
        {
            var predictionFrames = _features.PredictionFrames[i];
            if (predictionFrames < 0)
            {
                _predictedPosition[i] = currentPos;
                _predictedVelocity[i] = _velocity;
                _predictedAcceleration[i] = _acceleration;
                continue;
            }

            Spring.CharacterPositionUpdate(ref position, ref velocity, ref acceleration,
                desiredSpeed, 1.0f - responsivenessPositions,
                (predictionFrames - lastPredictionFrames) * averagedDeltaTime);
            lastPredictionFrames = predictionFrames;

            _predictedPosition[i] = position;
            _predictedVelocity[i] = velocity;
            _predictedAcceleration[i] = acceleration;
        }
    }

    private quaternion ComputeNewRot(quaternion currentRotation)
    {
        var newRotation = currentRotation;
        Spring.SimpleSpringDamperImplicit(ref newRotation, ref _angularVelocity, _desiredRotation,
            1.0f - responsivenessDirections, Time.deltaTime);
        return newRotation;
    }

    private float2 ComputeNewPos(float2 currentPos, float2 desiredSpeed)
    {
        var newPos = currentPos;
        Spring.CharacterPositionUpdate(ref newPos, ref _velocity, ref _acceleration, desiredSpeed,
            1.0f - responsivenessPositions, Time.deltaTime);
        return newPos;
    }

    // Pulling the synthesized character back to the simulation object, which the search alone does
    // not keep in sync. None of the three run today: they need the unimplemented adjustment API on
    // MotionSynthesisComponent -- see the comment there.

    /// <summary>Hard leash, capped at <see cref="maxDistanceMmAndCharacterController"/>.</summary>
    private void ClampMotionMatching()
    {
        // Clamp Position
        float3 characterController = transform.position;
        var mmPos = motionSynthesizer.RootPosition;
        if (math.distance(characterController, mmPos) > maxDistanceMmAndCharacterController)
        {
            float3 newMotionMatchingPos =
                maxDistanceMmAndCharacterController * math.normalize(mmPos - characterController) +
                characterController;
            motionSynthesizer.SetPosAdjustment(newMotionMatchingPos - mmPos);
        }
    }

    /// <summary>
    /// Soft pull toward the simulation object, damped and capped relative to how far the character
    /// is actually travelling, so the correction hides in the motion instead of looking like sliding.
    /// </summary>
    private void AdjustCharacterPosition()
    {
        float3 characterController = transform.position;
        var mmPos = motionSynthesizer.RootPosition;
        var differencePosition = characterController - mmPos;
        // Damp the difference using the adjustment halflife and dt
        var adjustmentPosition =
            Spring.DampAdjustmentImplicit(differencePosition, positionAdjustmentHalflife, Time.deltaTime);
        // Clamp adjustment if the length is greater than the character velocity
        // multiplied by the ratio
        var maxLength = posMaximumAdjustmentRatio * math.length(motionSynthesizer.RootVelocity) * Time.deltaTime;
        if (math.length(adjustmentPosition) > maxLength)
        {
            adjustmentPosition = maxLength * math.normalize(adjustmentPosition);
        }

        // Move the simulation bone towards the simulation object
        motionSynthesizer.SetPosAdjustment(adjustmentPosition);
    }

    /// <summary>As <see cref="AdjustCharacterPosition"/>, for facing.</summary>
    private void AdjustCharacterRotation()
    {
        quaternion characterController = transform.rotation;
        var mmRot = motionSynthesizer.RootRotation;
        // Find the difference in rotation (from character to simulation object)
        // Note: if numerically unstable, try quaternion.Normalize(quaternion.Inverse(characterController) * motionMatching)
        var differenceRotation = math.mul(math.inverse(mmRot), characterController);
        // Damp the difference using the adjustment halflife and dt
        var adjustmentRotation =
            Spring.DampAdjustmentImplicit(differenceRotation, rotationAdjustmentHalfLife, Time.deltaTime);
        // Clamp adjustment if the length is greater than the character angular velocity
        // multiplied by the ratio
        var maxLength = rotMaximumAdjustmentRatio * math.length(motionSynthesizer.RootAngularVelocity) * Time.deltaTime;
        if (math.length(MathExtensions.QuaternionToScaledAngleAxis(adjustmentRotation)) > maxLength)
        {
            adjustmentRotation = MathExtensions.QuaternionFromScaledAngleAxis(
                maxLength * math.normalize(
                    MathExtensions.QuaternionToScaledAngleAxis(adjustmentRotation)));
        }

        // Rotate the simulation bone towards the simulation object
        motionSynthesizer.SetRotAdjustment(adjustmentRotation);
    }

    public quaternion GetCurrentRotation()
    {
        return transform.rotation;
    }

    public override float3 GetPosition()
    {
        return transform.position;
    }

    /// <summary>
    /// One horizon of one trajectory feature, converted into the simulation bone's frame. Y is
    /// dropped — the trajectory is a ground-plane path, so each value is two floats.
    /// </summary>
    /// <remarks>
    /// A negative horizon is answered from the recorded history rather than the spring; see
    /// <see cref="TrajectoryHistory"/>. Before enough has been recorded it falls back to the current
    /// state, which is what a character that has been standing still would have recorded anyway.
    /// </remarks>
    // TODO: the trajectory construction should be inside the animation system
    // and not the character controller. Move it
    public override void GetTrajectoryFeature(
        TrajectoryFeatureChannel feature, int index,
        Transform character, Span<float> output
    )
    {
        if (feature.name == "FutureSphere")
        {
            output[0] = 0.0f;
            return;
        }

        if (!feature.simulationBone) Debug.Assert(false, "Trajectory should be computed using the simulation frame");

        var framesBack = -feature.predictionFrames[index];

        switch (feature.featureType)
        {
            case TrajectoryFeatureChannel.Type.Position:
            {
                float2 world;
                if (framesBack > 0) GetPastState(framesBack, out world, out _);
                else world = _predictedPosition[index];

                WritePlanarPosition(character, world, output);
                break;
            }
            case TrajectoryFeatureChannel.Type.Direction:
            {
                float2 dirProjected;
                if (framesBack > 0) GetPastState(framesBack, out _, out dirProjected);
                else dirProjected = GetWorldSpaceDirectionPrediction(index);

                WritePlanarDirection(character, dirProjected, output);
                break;
            }
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
        if (debugCurrent)
        {
            // Draw Current Position & Velocity
            Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f);
            Gizmos.DrawSphere(transformPos, radius);
            GizmosExtensions.DrawLine(transformPos,
                transformPos + ((Quaternion)GetCurrentRotation() * Vector3.forward) * vectorReduction, 3);
        }

        if (_predictedPosition == null || _predictedRotations == null) return;

        if (debugPrediction)
        {
            // Draw Predicted Position & Velocity
            Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f);
            for (var i = 0; i < _predictedPosition.Length; ++i)
            {
                var predictedPos = new float3(_predictedPosition[i].x, verticalOffset, _predictedPosition[i].y);
                var predictedDir = GetWorldSpaceDirectionPrediction(i);
                var predictedDir3D = new float3(predictedDir.x, 0.0f, predictedDir.y);
                Gizmos.DrawSphere(predictedPos, radius);
                GizmosExtensions.DrawLine(predictedPos, predictedPos + predictedDir3D * vectorReduction, 3);
            }
        }

        if (debugClamping)
        {
            // Draw Clamp Circle
            if (doClamping)
            {
                Gizmos.color = new Color(0.1f, 1.0f, 0.1f, 1.0f);
                GizmosExtensions.DrawWireCircle(transformPos, maxDistanceMmAndCharacterController, quaternion.identity);
            }
        }
    }
#endif
}
}