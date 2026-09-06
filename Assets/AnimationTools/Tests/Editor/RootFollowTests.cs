using NUnit.Framework;
using Unity.Mathematics;

namespace AnimationTools.Tests
{
/// <summary>
/// What a root correction has to satisfy: the character, once the component has integrated the
/// velocity this produces, is where the target said.
/// </summary>
/// <remarks>
/// The contract is written against <see cref="SimulationFrame.Advance"/> rather than against the
/// numbers themselves, because the correction is only meaningful as the input to that integration.
/// </remarks>
public class RootFollowTests
{
    private const float FrameTime = 1f / 30f;

    private static readonly float2 OwnerPosition = new(0f, 0f);
    private static readonly float2 TargetPosition = new(1f, 0.5f);
    private const float TargetYaw = 0.5236f; // 30 degrees

    /// <summary>A walking pose: a metre per second forwards, not turning.</summary>
    private static readonly float3 FrameVelocity = new(0f, 0f, 1f);

    private static void SolveAndAdvance(float ownerYaw, float positionHalfLife, float rotationHalfLife,
        float maxSpeed, float maxYawRate, out float2 landedPosition, out float landedYaw)
    {
        RootFollow.Solve(OwnerPosition, ownerYaw, TargetPosition, TargetYaw, FrameVelocity, 0f,
            positionHalfLife, rotationHalfLife, maxSpeed, maxYawRate, FrameTime,
            out var velocity, out var yawRate);

        SimulationFrame.Advance(new float3(OwnerPosition.x, 0f, OwnerPosition.y),
            quaternion.RotateY(ownerYaw), velocity, yawRate, FrameTime,
            out var position, out var rotation);

        landedPosition = position.xz;
        landedYaw = SimulationFrame.Yaw(rotation);
    }

    /// <summary>The whole point of the snap mode: no lag at all, whatever the animation was doing.</summary>
    [Test]
    public void ZeroHalfLife_LandsExactlyOnTheTarget()
    {
        SolveAndAdvance(0f, 0f, 0f, 0f, 0f, out var position, out var yaw);

        Assert.That(math.distance(position, TargetPosition), Is.LessThan(1e-4f));
        Assert.That(math.abs(SimulationFrame.SignedYawDelta(yaw, TargetYaw)), Is.LessThan(1e-4f));
    }

    /// <summary>The correction is measured from the character's own frame, not from world axes.</summary>
    [Test]
    public void ZeroHalfLife_LandsOnTheTarget_FromAnyFacing()
    {
        SolveAndAdvance(math.radians(115f), 0f, 0f, 0f, 0f, out var position, out var yaw);

        Assert.That(math.distance(position, TargetPosition), Is.LessThan(1e-4f));
        Assert.That(math.abs(SimulationFrame.SignedYawDelta(yaw, TargetYaw)), Is.LessThan(1e-4f));
    }

    /// <summary>A long half-life is the soft mode: the animation still leads, the gap closes slowly.</summary>
    [Test]
    public void ALongHalfLife_ClosesOnlyPartOfTheGap()
    {
        SolveAndAdvance(0f, 0.5f, 0.5f, 0f, 0f, out var position, out _);

        var animatedOnly = OwnerPosition + FrameVelocity.xz * FrameTime;
        var closed = math.distance(position, animatedOnly);
        var gap = math.distance(TargetPosition, animatedOnly);

        Assert.That(closed, Is.GreaterThan(0f), "some of the gap should close");
        Assert.That(closed, Is.LessThan(gap * 0.2f), "but nowhere near all of it in one tick");
    }

    /// <summary>Nothing to correct means the pose passes through untouched.</summary>
    [Test]
    public void AnAlreadyMatchingTarget_LeavesTheAnimationAlone()
    {
        var animated = OwnerPosition + FrameVelocity.xz * FrameTime;

        RootFollow.Solve(OwnerPosition, 0f, animated, 0f, FrameVelocity, 0f, 0f, 0f, 0f, 0f, FrameTime,
            out var velocity, out var yawRate);

        Assert.That(math.distance(velocity, FrameVelocity), Is.LessThan(1e-5f));
        Assert.That(yawRate, Is.EqualTo(0f).Within(1e-5f));
    }

    /// <summary>The cap is what keeps a correction hidden inside the animation's own travel.</summary>
    [Test]
    public void TheSpeedCap_LimitsTheCorrectionAlone()
    {
        const float maxCorrectionSpeed = 0.25f;
        RootFollow.Solve(OwnerPosition, 0f, TargetPosition, TargetYaw, FrameVelocity, 0f, 0f, 0f,
            maxCorrectionSpeed, 0f, FrameTime, out var velocity, out _);

        var correction = velocity - FrameVelocity;
        Assert.That(math.length(correction), Is.LessThanOrEqualTo(maxCorrectionSpeed + 1e-4f));
        Assert.That(math.length(correction), Is.GreaterThan(maxCorrectionSpeed * 0.99f),
            "a gap this large should saturate the cap");
    }

    [Test]
    public void TheYawCap_LimitsTheFacingCorrection()
    {
        var maxCorrectionYawRate = math.radians(90f);
        RootFollow.Solve(OwnerPosition, 0f, TargetPosition, TargetYaw, FrameVelocity, 0f, 0f, 0f, 0f,
            maxCorrectionYawRate, FrameTime, out _, out var yawRate);

        Assert.That(math.abs(yawRate), Is.LessThanOrEqualTo(maxCorrectionYawRate + 1e-4f));
    }

    /// <summary>
    /// A character facing just short of -180 degrees and a target just past it are 20 degrees apart,
    /// not 340. Taking the long way round would spin the character on the spot.
    /// </summary>
    [Test]
    public void TheFacingCorrection_TakesTheShortWayRound()
    {
        var ownerYaw = math.radians(-170f);
        var targetYaw = math.radians(170f);

        RootFollow.Solve(OwnerPosition, ownerYaw, OwnerPosition, targetYaw, float3.zero, 0f, 0f, 0f,
            0f, 0f, FrameTime, out _, out var yawRate);

        Assert.That(math.degrees(yawRate * FrameTime), Is.EqualTo(-20f).Within(1e-3f));
    }

    /// <summary>Height is not the frame's to move, so a target above the character must not lift it.</summary>
    [Test]
    public void TheCorrection_NeverMovesTheCharacterVertically()
    {
        RootFollow.Solve(OwnerPosition, 0f, TargetPosition, TargetYaw, FrameVelocity, 0f, 0f, 0f, 0f,
            0f, FrameTime, out var velocity, out _);

        Assert.That(velocity.y, Is.EqualTo(0f).Within(1e-6f));
    }

    /// <summary>A zero timestep has no velocity that means anything, so nothing is changed.</summary>
    [Test]
    public void AZeroTimestep_LeavesTheAnimationAlone()
    {
        RootFollow.Solve(OwnerPosition, 0f, TargetPosition, TargetYaw, FrameVelocity, 0.5f, 0f, 0f, 0f,
            0f, 0f, out var velocity, out var yawRate);

        Assert.That(math.distance(velocity, FrameVelocity), Is.EqualTo(0f).Within(1e-6f));
        Assert.That(yawRate, Is.EqualTo(0.5f).Within(1e-6f));
    }
}
}
