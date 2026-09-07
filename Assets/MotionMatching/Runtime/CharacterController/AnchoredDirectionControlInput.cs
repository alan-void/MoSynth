using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// Stick/WASD control whose trajectory starts at the character rather than at a simulation object
/// of its own. Same springs as <see cref="DirectionControlInput"/>; what differs is where the
/// prediction is measured from.
/// </summary>
/// <remarks>
/// <see cref="DirectionControlInput"/> integrates a position and asks the search to chase it, so
/// any difference between the two accumulates and the character is permanently catching up. Here
/// the origin is re-read off the character every frame and only the velocity springs carry state,
/// so the query says "from where you are, go like this" and there is no position that can drift.
/// The trade is that this input cannot be used to *place* the character — see
/// <see cref="RootFollowStage"/> for that, and do not use both on one character.
/// <para>
/// The construction is <c>PfnnDirectionControlInput</c>'s, over motion matching's authored horizons.
/// Maths from <a href="https://theorangeduck.com/page/spring-roll-call#controllers">spring roll
/// call</a>.
/// </para>
/// </remarks>
public class AnchoredDirectionControlInput : MotionMatchingControlInput, IMotionSynthesisDirectionControlInput
{
    [Header("Features")]
    [Tooltip("Name of the database trajectory feature holding future positions.")]
    public string trajectoryPositionFeatureName = "FuturePosition";

    [Tooltip("Name of the database trajectory feature holding future facing directions.")]
    public string trajectoryDirectionFeatureName = "FutureDirection";

    [Header("General")]
    [Tooltip("Speed at full stick deflection, in m/s.")]
    public float maxSpeed = 1.0f;

    [Tooltip("Time to close half the gap between the requested velocity and the one being asked " +
             "for. How fast the trajectory may re-aim, as opposed to how fast the body follows it.")]
    [Range(0.01f, 1f)] public float steeringHalfLife = 0.25f;

    [Tooltip("Time to close half the gap between the current velocity and the requested one.")]
    [Range(0.01f, 1f)] public float velocityHalfLife = 0.2f;

    [Tooltip("Time to close half the gap between the current facing and the direction of travel.")]
    [Range(0.01f, 1f)] public float facingHalfLife = 0.15f;

    [Tooltip("Speed below which the character is treated as stopped, so it settles cleanly.")]
    public float minimumSpeed = 0.01f;

    [Tooltip(
        "Controls when to consider that the input has suddenly changed. Used to recompute MotionMatching. -1.0f: Never. 1.0f: Always")]
    [Range(-1.0f, 1.0f)]
    public float inputBigChangeThreshold = 0.5f;

    [Header("DEBUG")] public bool debugCurrent = true;
    public bool debugPrediction = true;

    // --- Input ----------------------------------------------------------------------------------

    /// <summary>Latest stick/WASD vector, in the XZ plane. Length scales speed up to <see cref="maxSpeed"/>.</summary>
    private float2 _inputMovement;

    /// <summary>While set, the character keeps its facing and strafes instead of turning to face travel.</summary>
    private bool _orientationFixed;

    /// <summary>The facing held while strafing, so the request does not turn the character.</summary>
    private float2 _fixedFacing;

    // --- Spring state. There is deliberately no position here ------------------------------------

    /// <summary>The requested velocity, rate-limited so a stepped stick cannot step the trajectory.</summary>
    private float2 _goalVelocity;

    private float2 _velocity;
    private float2 _acceleration;

    // --- Predictions, in world space, refreshed once per frame ------------------------------------

    private float2[] _predictedPositions;
    private float2[] _predictedDirections;

    private TrajectoryFeaturePair _features;

    private int PredictionCount => _features.PredictionCount;

    /// <summary>
    /// Where the character has been, for the negative prediction frames of a trajectory feature
    /// that samples the past. Null while the database asks for no history.
    /// </summary>
    private TrajectoryHistory _history;

    /// <summary>
    /// Samples per second the history is sized to hold. Recorded once per rendered frame, so a
    /// render rate above this quietly shortens the window it can answer for.
    /// </summary>
    private const float HistorySampleRate = 240f;

    /// <summary>Resolves the named trajectory features and sizes the prediction arrays.</summary>
    private void Start()
    {
        _features = ResolveTrajectoryFeaturePair(trajectoryPositionFeatureName, trajectoryDirectionFeatureName);

        _predictedPositions = new float2[PredictionCount];
        _predictedDirections = new float2[PredictionCount];
        _fixedFacing = PlanarForward(Synthesizer.transform);

        var mmData = synthesizer.GetMmData();
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
        var previousInput = _inputMovement;
        _inputMovement = new float2(movementDirection.x, movementDirection.y);
        if (math.lengthsq(_inputMovement) > 1f) _inputMovement = math.normalize(_inputMovement);

        if (math.dot(previousInput, _inputMovement) < inputBigChangeThreshold)
        {
            NotifyInputChangedQuickly();
        }
    }

    /// <summary>Toggles strafing: facing stops following the movement direction.</summary>
    public void SwapFixOrientation()
    {
        _orientationFixed = !_orientationFixed;
        if (_orientationFixed) _fixedFacing = PlanarForward(Synthesizer.transform);
    }

    /// <summary>Advances the springs and refreshes the predicted trajectory from where the character is.</summary>
    protected override void OnUpdate()
    {
        var deltaTime = Time.deltaTime;

        // The request is rate-limited before the body sees it, because a keyboard delivers it as a
        // step: a horizon a second out has converged onto the goal, so it would inherit that step
        // whole while the samples beside the character did not move at all.
        _goalVelocity = TrajectorySteering.DampToward(_goalVelocity, _inputMovement * maxSpeed,
            steeringHalfLife, deltaTime);

        // A velocity goal, which is what a stick gives. The travel is discarded: where the
        // character is comes from the character, not from integrating this.
        var travelled = float2.zero;
        Spring.CharacterPositionUpdate(ref travelled, ref _velocity, ref _acceleration, _goalVelocity,
            velocityHalfLife, deltaTime);
        if (math.length(_velocity) < minimumSpeed) _velocity = float2.zero;

        var characterTransform = Synthesizer.transform;
        var origin = PlanarPosition(characterTransform);
        var facing = _orientationFixed ? _fixedFacing : PlanarForward(characterTransform);

        for (var i = 0; i < PredictionCount; i++)
        {
            var horizon = _features.PredictionFrames[i];
            if (horizon < 0)
            {
                // Answered from the history in GetTrajectoryFeature; hold the current state so a
                // gizmo reading these has something honest to draw.
                _predictedPositions[i] = origin;
                _predictedDirections[i] = facing;
                continue;
            }

            var offset = TrajectorySteering.PredictOffset(_velocity, _acceleration, _goalVelocity,
                horizon, DatabaseDeltaTime, velocityHalfLife, out var horizonVelocity);
            _predictedPositions[i] = origin + offset;

            var travel = _orientationFixed ? _fixedFacing : math.normalizesafe(horizonVelocity, facing);
            _predictedDirections[i] =
                TrajectorySteering.PredictFacing(facing, travel, horizon, DatabaseDeltaTime, facingHalfLife);
        }

        _history?.Record(Time.time, origin, facing);
    }

    /// <summary>
    /// Where the character was <paramref name="framesBack"/> database frames ago, falling back to
    /// where it is now for the first second or so of a run, before the history reaches that far
    /// back — which is what a character that had been standing still would have recorded anyway.
    /// </summary>
    private void GetPastState(int framesBack, out float2 position, out float2 forward)
    {
        if (_history != null &&
            _history.TrySample(Time.time - framesBack * DatabaseDeltaTime, out position, out forward))
        {
            return;
        }

        var characterTransform = Synthesizer.transform;
        position = PlanarPosition(characterTransform);
        forward = PlanarForward(characterTransform);
    }

    public override void GetTrajectoryFeature(TrajectoryFeatureChannel feature, int index,
        Transform character, Span<float> span)
    {
        if (!feature.simulationBone) Debug.Assert(false, "Trajectory should be computed using the simulation frame");

        var framesBack = -feature.predictionFrames[index];

        switch (feature.featureType)
        {
            case TrajectoryFeatureChannel.Type.Position:
            {
                float2 world;
                if (framesBack > 0) GetPastState(framesBack, out world, out _);
                else world = _predictedPositions[index];

                WritePlanarPosition(character, world, span);
                break;
            }
            case TrajectoryFeatureChannel.Type.Direction:
            {
                float2 world;
                if (framesBack > 0) GetPastState(framesBack, out _, out world);
                else world = _predictedDirections[index];

                WritePlanarDirection(character, world, span);
                break;
            }
            default:
                Debug.Assert(false, "Unknown feature type: " + feature.featureType);
                break;
        }
    }

    public override float3 GetPosition() => Synthesizer.transform.position;

    public override float3 GetWorldInitPosition() => Synthesizer.transform.position;

    public override float3 GetWorldInitDirection() => Synthesizer.transform.forward;

    public override float GetTargetSpeed() => math.length(_goalVelocity);

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (!Application.isPlaying || Synthesizer == null) return;

        const float radius = 0.05f;
        const float vectorReduction = 0.5f;
        const float verticalOffset = 0.05f;

        var characterTransform = Synthesizer.transform;
        var origin = (Vector3)characterTransform.position + Vector3.up * verticalOffset;

        if (debugCurrent)
        {
            Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f);
            Gizmos.DrawSphere(origin, radius);
            GizmosExtensions.DrawLine(origin, origin + characterTransform.forward * vectorReduction, 3);
        }

        if (!debugPrediction || _predictedPositions == null) return;

        Gizmos.color = new Color(0.6f, 0.3f, 0.8f, 1.0f);
        for (var i = 0; i < _predictedPositions.Length; i++)
        {
            var position = new Vector3(_predictedPositions[i].x, verticalOffset, _predictedPositions[i].y);
            var direction = new Vector3(_predictedDirections[i].x, 0f, _predictedDirections[i].y);
            Gizmos.DrawSphere(position, radius);
            GizmosExtensions.DrawLine(position, position + direction * vectorReduction, 3);
        }
    }
#endif
}
}
