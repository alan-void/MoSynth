using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// The maths a stick-driven control input uses to turn a movement request into the future half
/// of a predicted trajectory. Pure, so the continuity these functions exist to guarantee
/// can be measured without standing up a character.
/// </summary>
/// <remarks>
/// Everything here is the same exponential the shared <see cref="Spring"/> implements, from Daniel
/// Holden's <a href="https://theorangeduck.com/page/spring-roll-call">spring roll call</a>. The
/// window reaches a full second ahead, so a sample that far out is essentially the goal itself: any
/// step in the goal arrives at the far end of the trajectory undamped. Rate-limiting the goal is
/// what keeps the whole window continuous — see <c>openwiki/pfnn/pfnn-stage.md</c>.
/// </remarks>
public static class TrajectorySteering
{
    /// <summary>Moves <paramref name="current"/> a fixed fraction of the way to the target.</summary>
    public static float2 DampToward(float2 current, float2 target, float halfLife, float dt) =>
        current + Spring.DampAdjustmentImplicit(target - current, halfLife, dt);

    /// <summary>
    /// As <see cref="DampToward"/>, for a heading held as a unit vector. Damping the vector rather
    /// than the angle takes the shortest arc without any wrap handling, but its result is no longer
    /// unit length.
    /// </summary>
    public static float2 DampFacing(float2 facing, float2 travel, float halfLife, float dt) =>
        math.normalizesafe(DampToward(facing, travel, halfLife, dt), facing);

    /// <summary>
    /// Where the character would have got to <paramref name="frameOffset"/> frames from now if the
    /// request stopped changing: the same controller <c>OnUpdate</c> runs, carried on from where it
    /// stands.
    /// </summary>
    /// <remarks>
    /// Stepped a frame at a time rather than jumped straight to the horizon. The implicit form is
    /// exact at any step size but <c>Spring</c>'s exponential is an approximation, and jumping a
    /// full second in one call drifts ~2 cm from the path the character will actually take — which
    /// is the whole point of the prediction. Same reasoning as <c>DirectionControlInput</c>'s
    /// <c>PredictPositions</c>.
    /// </remarks>
    /// <param name="endVelocity">The velocity at that horizon, i.e. the direction of travel there.</param>
    public static float2 PredictOffset(float2 velocity, float2 acceleration, float2 goalVelocity,
        int frameOffset, float frameTime, float velocityHalfLife, out float2 endVelocity)
    {
        var offset = float2.zero;
        for (var frame = 0; frame < frameOffset; frame++)
        {
            Spring.CharacterPositionUpdate(ref offset, ref velocity, ref acceleration, goalVelocity,
                velocityHalfLife, frameTime);
        }

        endVelocity = velocity;
        return offset;
    }

    /// <summary>
    /// <paramref name="request"/> turned back toward <paramref name="reference"/> until it lies
    /// within <paramref name="maxAngle"/> radians of it, at its original length.
    /// </summary>
    /// <remarks>
    /// A network answers a trajectory that bends further from the character's facing than its
    /// training data ever did by extrapolating, which is not a graceful failure — see
    /// <c>openwiki/pfnn/training-and-checkpoints.md</c>. Holding the request inside a cone that
    /// turns with the character keeps it on ground the model has seen, and spends a reversal as a
    /// sustained turn rather than as one step the pose has to absorb.
    /// </remarks>
    /// <param name="reference">A unit vector the cone is centred on, normally the character's facing.</param>
    public static float2 ClampToCone(float2 request, float2 reference, float maxAngle)
    {
        var length = math.length(request);
        if (length < 1e-6f) return request;

        var direction = request / length;
        if (math.dot(direction, reference) >= math.cos(maxAngle)) return request;

        // Directly astern is the one request with two equally good answers, and an unstable choice
        // there would dither instead of turning. Taking the sign of an exact zero decides it, and
        // the first frame of the turn makes the cross product unambiguous from then on.
        var cross = reference.x * direction.y - reference.y * direction.x;
        var angle = cross >= 0f ? maxAngle : -maxAngle;
        var sin = math.sin(angle);
        var cos = math.cos(angle);

        return new float2(reference.x * cos - reference.y * sin,
                          reference.x * sin + reference.y * cos) * length;
    }

    /// <summary>
    /// The facing at a horizon: the same damper as <see cref="DampFacing"/>, jumped straight to that
    /// horizon in one step, which is exact for an exponential.
    /// </summary>
    public static float2 PredictFacing(float2 facing, float2 travel, int frameOffset, float frameTime,
        float facingHalfLife)
    {
        var blend = 1f - math.pow(0.5f, frameOffset * frameTime / math.max(1e-5f, facingHalfLife));
        return math.normalizesafe(math.lerp(facing, travel, blend), new float2(0f, 1f));
    }
}
}
