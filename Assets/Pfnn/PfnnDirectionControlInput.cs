using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace Pfnn
{
/// <summary>
/// Stick or WASD control: a spring smooths the requested velocity into something a body could do,
/// and running that same spring further ahead with no new input is the future trajectory.
/// </summary>
/// <remarks>
/// The same construction as <c>DirectionControlInput</c>, on the shared <see cref="Spring"/> from
/// Daniel Holden's <a href="https://theorangeduck.com/page/spring-roll-call#controllers">spring
/// roll call</a>. A PFNN reads the whole predicted path, which is why the request itself is also
/// rate-limited — see <see cref="steeringHalfLife"/>.
/// </remarks>
public class PfnnDirectionControlInput : PfnnControlInput, IMotionSynthesisDirectionControlInput
{
    [Tooltip("Speed at full stick deflection, in m/s.")]
    public float maxSpeed = 1.0f;

    [Tooltip("Time to close half the gap between the requested velocity and the one being asked " +
             "for. How fast the trajectory may re-aim, as opposed to how fast the body follows it.")]
    [Range(0.01f, 1f)]
    public float steeringHalfLife = 0.25f;

    [Tooltip("Time to close half the gap between the current velocity and the requested one.")]
    [Range(0.01f, 1f)]
    public float velocityHalfLife = 0.2f;

    [Tooltip("Time to close half the gap between the current facing and the direction of travel.")]
    [Range(0.01f, 1f)]
    public float facingHalfLife = 0.15f;

    [Tooltip("How far off the character's own facing the requested trajectory may bend, in " +
             "degrees. The cone turns with the character, so a sharper request is spent as a " +
             "sustained turn instead of a path the network was never trained on. 180 lifts it.")]
    [Range(15f, 180f)]
    public float maxRequestAngle = 90f;

    [Tooltip("Speed below which the character is treated as stopped, so it settles cleanly.")]
    public float minimumSpeed = 0.01f;

    /// <summary>How long the character may go undriven before saying so, in seconds.</summary>
    private const float UndrivenWarningDelay = 5f;

    private float2 _desiredDirection;
    private float2 _goalVelocity;
    private float2 _velocity;
    private float2 _acceleration;
    private float2 _facing;
    private bool _everDriven;
    private bool _warnedUndriven;
    private float _undrivenSeconds;

    protected override void Awake()
    {
        base.Awake();

        // Start from the character's real facing, or the first frame asks it to turn to world +z.
        var forward = synthesizer != null ? synthesizer.transform.forward : Vector3.forward;
        _facing = math.normalizesafe(new float2(forward.x, forward.z), new float2(0f, 1f));
    }

    /// <summary>Set by an input script; a zero vector asks the character to stop.</summary>
    public void SetMovementDirection(Vector2 movementDirection)
    {
        _everDriven = true;
        _desiredDirection = new float2(movementDirection.x, movementDirection.y);
        if (math.lengthsq(_desiredDirection) > 1f) _desiredDirection = math.normalize(_desiredDirection);
    }

    protected override void OnUpdate()
    {
        WarnIfNothingIsDriving();

        var deltaTime = Time.deltaTime;

        // The network extrapolates badly on paths bending further than its training data did.
        // See openwiki/pfnn/training-and-checkpoints.md.
        var request = TrajectorySteering.ClampToCone(_desiredDirection * maxSpeed, CharacterForward(),
            math.radians(maxRequestAngle));

        // Rate-limited because a keyboard delivers the request as a step, which the far trajectory
        // samples would otherwise inherit whole while the near ones do not move.
        _goalVelocity = TrajectorySteering.DampToward(_goalVelocity, request,
            steeringHalfLife, deltaTime);

        // A velocity-goal controller: a position spring aimed at a velocity settles at a speed that
        // depends on the frame rate rather than on the request. See openwiki/pfnn/pfnn-stage.md.
        var travelled = float2.zero;
        Spring.CharacterPositionUpdate(ref travelled, ref _velocity, ref _acceleration, _goalVelocity,
            velocityHalfLife, deltaTime);

        if (math.length(_velocity) < minimumSpeed) _velocity = float2.zero;
        else
        {
            _facing = TrajectorySteering.DampFacing(_facing, math.normalizesafe(_velocity, _facing),
                facingHalfLife, deltaTime);
        }
    }

    /// <summary>
    /// The character's facing on the ground plane: the axis the stage measures the trajectory window
    /// against, so the one the cone is centred on.
    /// </summary>
    private float2 CharacterForward()
    {
        var forward = synthesizer.transform.forward;
        return math.normalizesafe(new float2(forward.x, forward.z), _facing);
    }

    /// <summary>
    /// Warns once if no input has arrived: an unwired driver otherwise looks like a healthy
    /// character that nobody has steered yet.
    /// </summary>
    private void WarnIfNothingIsDriving()
    {
        if (_everDriven || _warnedUndriven) return;

        _undrivenSeconds += Time.deltaTime;
        if (_undrivenSeconds < UndrivenWarningDelay) return;

        Debug.LogWarning(
            $"[PFNN] No movement input has reached '{name}' in {UndrivenWarningDelay:0} seconds. " +
            "If nothing has touched the controls yet, that is expected; otherwise there is no " +
            "UserInput in the scene to publish the move action.", this);
        _warnedUndriven = true;
    }

    public override bool TryGetFutureSample(int frameOffset, out float2 position,
        out float2 direction)
    {
        // Never driven is the base contract's "nothing to say"; the stage substitutes the
        // character's own position and facing.
        if (!_everDriven)
        {
            position = default;
            direction = default;
            return false;
        }

        var origin = RootPosition;
        var frameTime = 1f / math.max(1f, Synthesizer.synthesisFrameRate);

        // The same springs run on with no new input, read by value: only OnUpdate advances them.
        var offset = TrajectorySteering.PredictOffset(_velocity, _acceleration, _goalVelocity, frameOffset,
            frameTime, velocityHalfLife, out var horizonVelocity);
        position = new float2(origin.x, origin.z) + offset;

        var travel = math.normalizesafe(horizonVelocity, _facing);
        direction = TrajectorySteering.PredictFacing(_facing, travel, frameOffset, frameTime, facingHalfLife);
        return true;
    }
}
}
