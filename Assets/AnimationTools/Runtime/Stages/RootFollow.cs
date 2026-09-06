using Unity.Mathematics;

namespace AnimationTools
{
/// <summary>
/// How much the character frame's own motion has to be changed this tick for the character to end
/// up on a target it is following. Pure, so the arithmetic can be checked against the integration
/// it is written for without standing up a scene.
/// </summary>
/// <remarks>
/// The character accumulates movement by integrating the frame velocity a pose carries (see
/// <see cref="SimulationFrame.Advance"/>). So "land on the target" is not a position to write
/// anywhere — it is a velocity that, integrated over one tick, arrives there. That is what this
/// returns, and it is why a correction can be blended, rate-limited, or turned off without any
/// other part of the pipeline knowing it happened.
/// </remarks>
public static class RootFollow
{
    /// <summary>
    /// The frame velocity and yaw rate that carry the character from where it is toward where the
    /// target is, on top of the motion the pose was already going to produce.
    /// </summary>
    /// <param name="frameVelocity">Frame-local ground velocity the pose carries, m/s.</param>
    /// <param name="frameYawRate">Yaw rate the pose carries, radians/s.</param>
    /// <param name="positionHalfLife">Time to close half the remaining gap; 0 closes it this tick.</param>
    /// <param name="rotationHalfLife">As <paramref name="positionHalfLife"/>, for facing.</param>
    /// <param name="maxCorrectionSpeed">Cap on the correction alone, m/s. 0 is uncapped.</param>
    /// <param name="maxCorrectionYawRate">Cap on the facing correction alone, radians/s. 0 is uncapped.</param>
    public static void Solve(float2 ownerPositionXZ, float ownerYaw, float2 targetPositionXZ, float targetYaw,
        float3 frameVelocity, float frameYawRate, float positionHalfLife, float rotationHalfLife,
        float maxCorrectionSpeed, float maxCorrectionYawRate, float dt,
        out float3 correctedFrameVelocity, out float correctedYawRate)
    {
        correctedFrameVelocity = frameVelocity;
        correctedYawRate = frameYawRate;
        if (dt <= 0f) return;

        // What the pose alone moves the character this tick, and what is left over after it.
        var animated = RotateY(frameVelocity, ownerYaw).xz * dt;
        var residual = targetPositionXZ - ownerPositionXZ - animated;
        var residualYaw = SimulationFrame.SignedYawDelta(ownerYaw, targetYaw) - frameYawRate * dt;

        var correction = Spring.DampAdjustmentImplicit(residual, positionHalfLife, dt);
        var correctionYaw = Spring.DampAdjustmentImplicit(residualYaw, rotationHalfLife, dt);

        if (maxCorrectionSpeed > 0f)
        {
            var limit = maxCorrectionSpeed * dt;
            if (math.lengthsq(correction) > limit * limit)
            {
                correction = math.normalize(correction) * limit;
            }
        }

        if (maxCorrectionYawRate > 0f)
        {
            var limit = maxCorrectionYawRate * dt;
            correctionYaw = math.clamp(correctionYaw, -limit, limit);
        }

        // Back into the frame the velocity channels are expressed in. Height is left alone: the
        // frame is ground-projected, so nothing routed through it can move the character vertically.
        var correctionFrameLocal = RotateY(new float3(correction.x, 0f, correction.y), -ownerYaw);
        correctedFrameVelocity = frameVelocity + correctionFrameLocal / dt;
        correctedYawRate = frameYawRate + correctionYaw / dt;
    }

    private static float3 RotateY(float3 v, float yaw) => math.mul(quaternion.RotateY(yaw), v);
}
}
