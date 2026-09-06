using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// Something a character can be made to follow: a ground-plane position and a facing, answered
/// each tick. A component on a <c>RootFollowStage</c>'s target implements this when the target's
/// Transform is not where the target actually is — a spline follower, say, whose point on the path
/// never moves its own object.
/// </summary>
public interface IFrameTarget
{
    /// <summary>Where the target is now and which way it faces, or false to leave the character alone this tick.</summary>
    /// <param name="positionXZ">World ground-plane position.</param>
    /// <param name="yaw">World heading in radians, the convention of <see cref="SimulationFrame.Yaw"/>.</param>
    bool TryGetFrame(out float2 positionXZ, out float yaw);
}
}
