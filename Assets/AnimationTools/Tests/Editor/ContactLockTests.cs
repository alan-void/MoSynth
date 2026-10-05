using NUnit.Framework;
using Unity.Mathematics;

namespace AnimationTools.Tests
{
/// <summary>
/// What a contact lock has to satisfy: a planted bone stays put while the animation slides under
/// it, and every way out of or back into a lock is continuous.
/// </summary>
public class ContactLockTests
{
    private const float DeltaTime = 1f / 60f;
    private const float HalfLife = 0.1f;
    private const float NoDistanceLimit = 0f;

    /// <summary>The animation sliding a foot forwards, a centimetre per tick.</summary>
    private static float3 Sliding(int tick) => new(0.01f * tick, 0f, 0f);

    /// <summary>Locks at tick 0 and holds for <paramref name="ticks"/> ticks of sliding.</summary>
    private static ContactLock HoldWhileSliding(int ticks, out float3 lockPosition)
    {
        var contactLock = new ContactLock();
        lockPosition = contactLock.Step(true, Sliding(0), NoDistanceLimit, HalfLife, DeltaTime);
        for (var tick = 1; tick < ticks; tick++)
        {
            contactLock.Step(true, Sliding(tick), NoDistanceLimit, HalfLife, DeltaTime);
        }
        return contactLock;
    }

    [Test]
    public void Latch_HoldsWhileTheAnimationMoves()
    {
        var contactLock = new ContactLock();
        var lockPosition = contactLock.Step(true, Sliding(0), NoDistanceLimit, HalfLife, DeltaTime);

        for (var tick = 1; tick < 20; tick++)
        {
            var output = contactLock.Step(true, Sliding(tick), NoDistanceLimit, HalfLife, DeltaTime);
            Assert.That(math.distance(output, lockPosition), Is.LessThan(1e-6f));
        }
        Assert.That(contactLock.IsLocked, Is.True);
    }

    [Test]
    public void NoContact_FollowsTheAnimation()
    {
        var contactLock = new ContactLock();
        for (var tick = 0; tick < 10; tick++)
        {
            var output = contactLock.Step(false, Sliding(tick), NoDistanceLimit, HalfLife, DeltaTime);
            Assert.That(math.distance(output, Sliding(tick)), Is.LessThan(1e-6f));
        }
    }

    /// <summary>
    /// After contact ends the residual shrinks every tick and never jumps — including on the
    /// release tick itself.
    /// </summary>
    [Test]
    public void ContactEnd_ReleasesAndDecaysContinuously()
    {
        const int heldTicks = 20;
        var contactLock = HoldWhileSliding(heldTicks, out var lockPosition);

        var releaseOutput = contactLock.Step(false, Sliding(heldTicks), NoDistanceLimit, HalfLife, DeltaTime);
        var initialResidual = math.distance(lockPosition, Sliding(heldTicks));
        Assert.That(contactLock.IsLocked, Is.False);
        Assert.That(math.distance(releaseOutput, lockPosition), Is.LessThan(0.05f * initialResidual));

        var previousResidual = math.distance(releaseOutput, Sliding(heldTicks));
        for (var tick = heldTicks + 1; tick < heldTicks + 120; tick++)
        {
            var output = contactLock.Step(false, Sliding(tick), NoDistanceLimit, HalfLife, DeltaTime);
            var residual = math.distance(output, Sliding(tick));
            Assert.That(residual, Is.LessThanOrEqualTo(previousResidual));
            previousResidual = residual;
        }
        Assert.That(previousResidual, Is.LessThan(1e-3f));
    }

    /// <summary>The animation dragging the bone too far away releases it even though contact holds.</summary>
    [Test]
    public void StrayingPastMaxDistance_Releases()
    {
        const float maxLockDistance = 0.105f;
        var contactLock = new ContactLock();
        var lockPosition = contactLock.Step(true, Sliding(0), maxLockDistance, HalfLife, DeltaTime);

        var previousOutput = lockPosition;
        var releasedAt = -1;
        for (var tick = 1; tick < 30; tick++)
        {
            var output = contactLock.Step(true, Sliding(tick), maxLockDistance, HalfLife, DeltaTime);
            if (releasedAt < 0 && !contactLock.IsLocked) releasedAt = tick;
            Assert.That(math.distance(output, previousOutput), Is.LessThan(0.03f));
            previousOutput = output;
        }

        // A centimetre per tick: the eleventh tick is the first past 10.5 cm.
        Assert.That(releasedAt, Is.EqualTo(11));
        Assert.That(contactLock.IsLocked, Is.False);
    }

    /// <summary>Touching down again mid-release latches where the bone is shown, so it does not pop.</summary>
    [Test]
    public void RelatchMidRelease_LatchesAtTheOutput()
    {
        const int heldTicks = 20;
        var contactLock = HoldWhileSliding(heldTicks, out _);

        // Released, with the animation standing still while the residual eases out.
        var standing = Sliding(heldTicks);
        var output = float3.zero;
        for (var tick = 0; tick < 5; tick++)
        {
            output = contactLock.Step(false, standing, NoDistanceLimit, HalfLife, DeltaTime);
        }
        Assert.That(math.distance(output, standing), Is.GreaterThan(0.01f), "the residual should not be gone yet");

        var relatched = contactLock.Step(true, standing, NoDistanceLimit, HalfLife, DeltaTime);
        Assert.That(contactLock.IsLocked, Is.True);
        Assert.That(math.distance(relatched, output), Is.LessThan(1e-6f));
    }

    [Test]
    public void Reset_ForgetsTheLockAndResidual()
    {
        var contactLock = HoldWhileSliding(20, out _);
        contactLock.Reset();

        var output = contactLock.Step(false, Sliding(20), NoDistanceLimit, HalfLife, DeltaTime);
        Assert.That(contactLock.IsLocked, Is.False);
        Assert.That(math.distance(output, Sliding(20)), Is.LessThan(1e-6f));
    }
}
}
