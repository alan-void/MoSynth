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
/// rather than a handful of authored horizons.
/// </remarks>
public class PfnnDirectionControlInput : PfnnControlInput, IMotionSynthesisDirectionControlInput
{
    [Tooltip("Speed at full stick deflection, in m/s.")]
    public float maxSpeed = 1.0f;

    [Tooltip("Time to close half the gap between the current velocity and the requested one.")]
    [Range(0.01f, 1f)]
    public float velocityHalfLife = 0.2f;

    [Tooltip("Time to close half the gap between the current facing and the direction of travel.")]
    [Range(0.01f, 1f)]
    public float facingHalfLife = 0.15f;

    [Tooltip("Speed below which the character is treated as stopped, so it settles cleanly.")]
    public float minimumSpeed = 0.01f;

    private float2 _desiredDirection;
    private float2 _velocity;
    private float2 _facing = new(0f, 1f);

    /// <summary>Set by an input script; a zero vector asks the character to stop.</summary>
    public void SetMovementDirection(Vector2 movementDirection)
    {
        _desiredDirection = new float2(movementDirection.x, movementDirection.y);
        if (math.lengthsq(_desiredDirection) > 1f) _desiredDirection = math.normalize(_desiredDirection);
    }

    protected override void OnUpdate()
    {
        var goal = _desiredDirection * maxSpeed;
        var position = float2.zero;
        Spring.SimpleSpringDamperImplicit(ref position, ref _velocity, goal, velocityHalfLife,
            Time.deltaTime);

        if (math.length(_velocity) < minimumSpeed) _velocity = float2.zero;
        else _facing = math.normalizesafe(_velocity, _facing);
    }

    public override bool TryGetFutureSample(int frameOffset, out float2 position,
        out float2 direction)
    {
        var origin = RootPosition;
        var frameTime = 1f / math.max(1f, Synthesizer.synthesisFrameRate);

        // The same spring, run on with no new input. Its own state must not move, so the prediction
        // works on copies -- what the character does is decided in OnUpdate, once.
        var predictedVelocity = _velocity;
        var predictedOffset = float2.zero;
        var goal = _desiredDirection * maxSpeed;

        for (var frame = 0; frame < frameOffset; frame++)
        {
            var spring = float2.zero;
            Spring.SimpleSpringDamperImplicit(ref spring, ref predictedVelocity, goal,
                velocityHalfLife, frameTime);
            predictedOffset += predictedVelocity * frameTime;
        }

        position = new float2(origin.x, origin.z) + predictedOffset;

        var travel = math.normalizesafe(predictedVelocity, _facing);
        // The facing lags the travel direction by its own spring, which is what stops a character
        // snapping round the instant the stick moves.
        var blend = 1f - math.pow(0.5f, frameOffset * frameTime / facingHalfLife);
        direction = math.normalizesafe(math.lerp(_facing, travel, blend), new float2(0f, 1f));
        return true;
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (!Application.isPlaying || Stage == null) return;

        Gizmos.color = new Color(1f, 0.6f, 0.1f);
        for (var frames = 0; frames <= 30; frames += 5)
        {
            if (!TryGetFutureSample(frames, out var position, out var direction)) continue;
            var world = new Vector3(position.x, RootPosition.y + 0.05f, position.y);
            Gizmos.DrawSphere(world, 0.04f);
            Gizmos.DrawLine(world, world + new Vector3(direction.x, 0f, direction.y) * 0.2f);
        }
    }
#endif
}
}
