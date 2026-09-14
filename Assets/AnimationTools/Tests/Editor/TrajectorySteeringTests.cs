using NUnit.Framework;
using Unity.Mathematics;

namespace AnimationTools.Tests
{
/// <summary>
/// The continuity the future half of the trajectory window depends on.
/// </summary>
/// <remarks>
/// A window sample a second ahead has converged onto the requested velocity, so it inherits any
/// step in that request whole while the samples beside the character do not move at all — which
/// reads as the far end of the trajectory teleporting. A keyboard delivers the request as a step,
/// so the request itself has to be rate-limited. These measure that, and that the controller a
/// request is fed into settles on the speed that was actually asked for.
/// </remarks>
public class TrajectorySteeringTests
{
    private const float FrameTime = 1f / 30f;
    private const float VelocityHalfLife = 0.2f;
    private const float FacingHalfLife = 0.15f;

    /// <summary>The far end of the window: one second ahead at 30 Hz.</summary>
    private const int FarHorizon = 30;

    private static readonly float2 Forward = new(0f, 1f);
    private static readonly float2 Left = new(-1f, 0f);

    /// <summary>
    /// Walks a character settled at <see cref="Forward"/> through a 90-degree change of request,
    /// and reports how far the far horizon moved in the worst single frame.
    /// </summary>
    private static float WorstFarHorizonStep(float steeringHalfLife, int frames,
        out float2 finalSample)
    {
        var goal = Forward;
        var velocity = Forward;
        var acceleration = float2.zero;

        var previous = TrajectorySteering.PredictOffset(velocity, acceleration, goal, FarHorizon, FrameTime,
            VelocityHalfLife, out _);
        var worst = 0f;

        for (var frame = 0; frame < frames; frame++)
        {
            goal = TrajectorySteering.DampToward(goal, Left, steeringHalfLife, FrameTime);

            var travelled = float2.zero;
            Spring.CharacterPositionUpdate(ref travelled, ref velocity, ref acceleration, goal,
                VelocityHalfLife, FrameTime);

            var sample = TrajectorySteering.PredictOffset(velocity, acceleration, goal, FarHorizon,
                FrameTime, VelocityHalfLife, out _);
            worst = math.max(worst, math.distance(sample, previous));
            previous = sample;
        }

        finalSample = previous;
        return worst;
    }

    /// <summary>
    /// Without the damper the request steps, and one second of trajectory steps with it. This is
    /// the behaviour being fixed; it is asserted so that the bound below cannot pass vacuously.
    /// </summary>
    [Test]
    public void AnUndampedRequest_TeleportsTheFarHorizon()
    {
        var worst = WorstFarHorizonStep(0f, 90, out _);
        Assert.That(worst, Is.GreaterThan(1f),
            "an undamped request should move the far horizon by about the whole second of travel");
    }

    [Test]
    public void ADampedRequest_MovesTheFarHorizonSmoothly()
    {
        var worst = WorstFarHorizonStep(0.25f, 90, out _);
        Assert.That(worst, Is.LessThan(0.12f),
            "the far horizon should sweep rather than jump");
    }

    /// <summary>Smoothing must delay the new heading, not lose it.</summary>
    [Test]
    public void ADampedRequest_StillReachesTheRequestedTrajectory()
    {
        WorstFarHorizonStep(0.25f, 90, out var finalSample);
        Assert.That(math.distance(finalSample, Left), Is.LessThan(0.01f));
    }

    /// <summary>
    /// A one-second horizon of a one metre per second request is one metre of travel. The spring
    /// this replaced was a position spring handed a velocity as its goal, so it settled at 3.26 m/s
    /// for the same request, and on a value that moved with the frame rate.
    /// </summary>
    [TestCase(30f)]
    [TestCase(60f)]
    [TestCase(144f)]
    public void TheVelocityController_SettlesOnTheRequestedSpeed(float frameRate)
    {
        var frameTime = 1f / frameRate;
        var velocity = float2.zero;
        var acceleration = float2.zero;

        for (var frame = 0; frame < frameRate * 5f; frame++)
        {
            var travelled = float2.zero;
            Spring.CharacterPositionUpdate(ref travelled, ref velocity, ref acceleration, Forward,
                VelocityHalfLife, frameTime);
        }

        Assert.That(math.length(velocity), Is.EqualTo(1f).Within(1e-3f));
    }

    /// <summary>
    /// The facing used to be assigned outright from the velocity, so a reversal — where the velocity
    /// passes through the stopped deadzone — flipped it in a single frame.
    /// </summary>
    [Test]
    public void TheFacing_TurnsThroughAReversalRatherThanFlipping()
    {
        var travel = new float2(math.sin(math.radians(170f)), math.cos(math.radians(170f)));
        var facing = Forward;
        var worst = 0f;

        for (var frame = 0; frame < 120; frame++)
        {
            var next = TrajectorySteering.DampFacing(facing, travel, FacingHalfLife, FrameTime);
            worst = math.max(worst, AngleBetween(facing, next));
            facing = next;
        }

        Assert.That(worst, Is.LessThan(12f), "the facing should turn, not snap");
        Assert.That(AngleBetween(facing, travel), Is.LessThan(1f), "and should get there");
    }

    /// <summary>A horizon further out is further round toward the direction of travel.</summary>
    [Test]
    public void PredictedFacing_LeadsFurtherTheFurtherAhead()
    {
        var travel = new float2(1f, 0f);
        var previous = 0f;

        for (var horizon = 0; horizon <= FarHorizon; horizon += 5)
        {
            var predicted = TrajectorySteering.PredictFacing(Forward, travel, horizon, FrameTime,
                FacingHalfLife);
            var turned = AngleBetween(Forward, predicted);
            Assert.That(turned, Is.GreaterThanOrEqualTo(previous - 1e-3f));
            previous = turned;
        }

        Assert.That(previous, Is.GreaterThan(85f), "a full second should be all but there");
    }

    [Test]
    public void ARequestInsideTheConeIsLeftExactlyAsItWas()
    {
        var request = new float2(0.4f, 0.9f);

        var clamped = TrajectorySteering.ClampToCone(request, Forward, math.radians(90f));

        Assert.That(clamped.x, Is.EqualTo(request.x).Within(1e-5f));
        Assert.That(clamped.y, Is.EqualTo(request.y).Within(1e-5f));
    }

    [Test]
    public void ARequestOutsideTheConeComesBackToItsEdgeAtTheSameSpeed()
    {
        var request = Left * 2f;

        var clamped = TrajectorySteering.ClampToCone(request, Forward, math.radians(45f));

        Assert.That(AngleBetween(Forward, clamped), Is.EqualTo(45f).Within(0.01f));
        Assert.That(math.length(clamped), Is.EqualTo(2f).Within(1e-4f),
            "the cone decides the heading, not how fast the character was asked to go");
    }

    [Test]
    public void ARequestComesBackOnTheSideItWasAskedFrom()
    {
        // Turning the shorter way round is the whole point: bending a left request back through
        // straight ahead would answer a turn with its mirror image.
        var fromTheLeft = TrajectorySteering.ClampToCone(Left, Forward, math.radians(45f));
        var fromTheRight = TrajectorySteering.ClampToCone(-Left, Forward, math.radians(45f));

        Assert.That(Cross(Forward, fromTheLeft), Is.GreaterThan(0f), "a left request came back right");
        Assert.That(Cross(Forward, fromTheRight), Is.LessThan(0f), "a right request came back left");
    }

    [Test]
    public void AReversalIsClampedRatherThanPassedThroughWholesale()
    {
        // The measured failure this exists for: past about 90 degrees off its facing the network
        // stops tracking the request and inverts its own turn.
        var clamped = TrajectorySteering.ClampToCone(-Forward, Forward, math.radians(90f));

        Assert.That(AngleBetween(Forward, clamped), Is.EqualTo(90f).Within(0.01f));
    }

    [Test]
    public void AReversalPicksASideAndKeepsTurningThatWay()
    {
        // Directly astern has two equally good answers, and dithering between them would leave the
        // character shuffling in place instead of turning round. Once the first frame commits, the
        // cone travels with the character and the rest of the turn follows it.
        var facing = Forward;
        var turned = 0f;

        for (var frame = 0; frame < 200 && turned < 179f; frame++)
        {
            var clamped = TrajectorySteering.ClampToCone(-Forward, facing, math.radians(90f));
            facing = TrajectorySteering.DampFacing(facing, math.normalizesafe(clamped),
                FacingHalfLife, FrameTime);
            turned = AngleBetween(Forward, facing);
        }

        Assert.That(turned, Is.GreaterThan(179f), "the character should come all the way round");
    }

    [Test]
    public void AConeOfHalfATurnLetsEveryRequestThrough()
    {
        foreach (var request in new[] { Forward, Left, -Forward, -Left })
        {
            var clamped = TrajectorySteering.ClampToCone(request, Forward, math.radians(180f));
            Assert.That(AngleBetween(request, clamped), Is.LessThan(0.01f), $"{request} was bent");
        }
    }

    [Test]
    public void AStandingRequestIsLeftAloneRatherThanGivenAHeading()
    {
        var clamped = TrajectorySteering.ClampToCone(float2.zero, Forward, math.radians(90f));

        Assert.That(math.length(clamped), Is.EqualTo(0f).Within(1e-6f));
    }

    private static float Cross(float2 a, float2 b) => a.x * b.y - a.y * b.x;

    private static float AngleBetween(float2 a, float2 b) =>
        math.degrees(math.acos(math.clamp(math.dot(math.normalizesafe(a), math.normalizesafe(b)), -1f, 1f)));
}
}
