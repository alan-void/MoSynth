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
/// The construction is the one <c>DirectionControlInput</c> documents, and the maths is the shared
/// <see cref="Spring"/> — from Daniel Holden's <a
/// href="https://theorangeduck.com/page/spring-roll-call#controllers">spring roll call</a>. What
/// differs is only how much of it the network is shown: a PFNN reads the whole predicted path
/// rather than a handful of authored horizons, which is also why the request itself is rate-limited
/// and not only the body's response to it — see <see cref="steeringHalfLife"/>.
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

        // The facing has to start where the character is actually pointing. Starting it at world
        // +z would ask a character facing any other way to turn round on its very first frame.
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

        // Held inside a cone around the character's own facing before it is damped, because the
        // network extrapolates badly on a path that bends further than its training data ever did.
        // See openwiki/pfnn/training-and-checkpoints.md.
        var request = TrajectorySteering.ClampToCone(_desiredDirection * maxSpeed, CharacterForward(),
            math.radians(maxRequestAngle));

        // The request is rate-limited before the body ever sees it, because a keyboard delivers it
        // as a step. A trajectory sample a second ahead has converged onto the goal, so it inherits
        // any step in the goal whole, while the samples near the character do not move at all.
        _goalVelocity = TrajectorySteering.DampToward(_goalVelocity, request,
            steeringHalfLife, deltaTime);

        // The controller for a *velocity* goal, which is what a stick gives. A position spring aimed
        // at a velocity has a fixed point that depends on the step size rather than on the request:
        // it settled at 3.26 m/s for maxSpeed 1 at 30 fps, and drifted with the frame rate.
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
    /// The character's own facing on the ground plane, which is the axis the stage measures the
    /// trajectory window against — so it is the axis the cone has to be centred on.
    /// </summary>
    private float2 CharacterForward()
    {
        var forward = synthesizer.transform.forward;
        return math.normalizesafe(new float2(forward.x, forward.z), _facing);
    }

    /// <summary>
    /// Nothing in this component polls input, so a character whose driver was never wired up simply
    /// stands still, and every component involved reports itself healthy. Saying so once is the only
    /// signal that distinguishes it from a character nobody has steered yet.
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
        // Never driven is the base contract's "nothing to say": the stage then substitutes the
        // character's own position and facing, which is the same answer this would give but leaves
        // the two states distinguishable at the seam.
        if (!_everDriven)
        {
            position = default;
            direction = default;
            return false;
        }

        var origin = RootPosition;
        var frameTime = 1f / math.max(1f, Synthesizer.synthesisFrameRate);

        // The same springs, run on with no new input. Their own state must not move, so this reads
        // it by value -- what the character does is decided in OnUpdate, once.
        var offset = TrajectorySteering.PredictOffset(_velocity, _acceleration, _goalVelocity, frameOffset,
            frameTime, velocityHalfLife, out var horizonVelocity);
        position = new float2(origin.x, origin.z) + offset;

        var travel = math.normalizesafe(horizonVelocity, _facing);
        direction = TrajectorySteering.PredictFacing(_facing, travel, frameOffset, frameTime, facingHalfLife);
        return true;
    }
}
}
