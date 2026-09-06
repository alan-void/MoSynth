using Unity.Mathematics;

namespace Pfnn
{
/// <summary>
/// Where the character has actually been, kept so the past half of the network's trajectory window
/// can be filled with history rather than with intent.
/// </summary>
/// <remarks>
/// During training both halves of the window are the same thing — the path the character really
/// took, sampled either side of the query frame. At runtime only the future is a wish, so only the
/// future comes from the control input. Filling the past from the control input too would tell the
/// network the character had already been following the requested path, which is exactly the case
/// the training data never contains.
/// </remarks>
public class PfnnTrajectory
{
    private readonly float2[] _positions;
    private readonly float[] _yaws;
    private int _newest;
    private bool _seeded;

    /// <param name="frames">How far back the window reaches, in synthesis frames.</param>
    public PfnnTrajectory(int frames)
    {
        var capacity = math.max(1, frames) + 1;
        _positions = new float2[capacity];
        _yaws = new float[capacity];
    }

    public int Capacity => _positions.Length;

    /// <summary>Fill the whole history with one pose, for a character that has just spawned.</summary>
    /// <remarks>
    /// A standing start is the honest seed: it is what the window looks like for a character that
    /// has not moved, which is also what the network sees for the stationary frames it trained on.
    /// </remarks>
    public void Seed(float2 position, float yaw)
    {
        for (var i = 0; i < _positions.Length; i++)
        {
            _positions[i] = position;
            _yaws[i] = yaw;
        }

        _newest = 0;
        _seeded = true;
    }

    /// <summary>Record where the character is now.</summary>
    public void Push(float2 position, float yaw)
    {
        if (!_seeded)
        {
            Seed(position, yaw);
            return;
        }

        _newest = (_newest + 1) % _positions.Length;
        _positions[_newest] = position;
        _yaws[_newest] = yaw;
    }

    /// <summary>
    /// Where the character was <paramref name="framesAgo"/> synthesis frames ago, in world space.
    /// </summary>
    /// <remarks>
    /// Clamped to the oldest sample held rather than extrapolated, which matches how
    /// <c>TrainingSet.trajectory_window</c> answers an offset that would leave the clip: it repeats
    /// the last real sample, which reads as a character that was standing still.
    /// </remarks>
    public void Sample(int framesAgo, out float2 position, out float yaw)
    {
        var clamped = math.clamp(framesAgo, 0, _positions.Length - 1);
        var index = _newest - clamped;
        if (index < 0) index += _positions.Length;

        position = _positions[index];
        yaw = _yaws[index];
    }

    /// <summary>The heading a world-space forward direction represents.</summary>
    /// <remarks>
    /// Unity is y-up and left-handed with the character facing +z, so a heading of theta is the
    /// direction (sin theta, cos theta) in (x, z). The same convention as
    /// <c>simulation_frame.frame_yaw</c>, which is what the training data was measured with.
    /// </remarks>
    public static float YawOf(float2 direction) => math.atan2(direction.x, direction.y);

    /// <summary>A world ground-plane point, in the frame at <paramref name="origin"/>.</summary>
    public static float2 ToFrame(float2 world, float2 origin, float frameYaw)
    {
        var offset = world - origin;
        math.sincos(frameYaw, out var sin, out var cos);
        return new float2(cos * offset.x - sin * offset.y, sin * offset.x + cos * offset.y);
    }

    /// <summary>A world heading, as a unit direction in the frame with heading <paramref name="frameYaw"/>.</summary>
    public static float2 DirectionInFrame(float yaw, float frameYaw)
    {
        var relative = yaw - frameYaw;
        return new float2(math.sin(relative), math.cos(relative));
    }

    /// <summary>Inverse of <see cref="ToFrame"/>.</summary>
    public static float2 FromFrame(float2 local, float2 origin, float frameYaw) =>
        origin + RotateOutOfFrame(local, frameYaw);

    /// <summary>Inverse of <see cref="DirectionInFrame"/>, as a world direction rather than a yaw.</summary>
    public static float2 DirectionFromFrame(float2 localDirection, float frameYaw) =>
        RotateOutOfFrame(localDirection, frameYaw);

    private static float2 RotateOutOfFrame(float2 local, float frameYaw)
    {
        math.sincos(frameYaw, out var sin, out var cos);
        return new float2(cos * local.x + sin * local.y, -sin * local.x + cos * local.y);
    }
}
}
