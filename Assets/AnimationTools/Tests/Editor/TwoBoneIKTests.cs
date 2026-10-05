using NUnit.Framework;
using Unity.Mathematics;

namespace AnimationTools.Tests
{
/// <summary>
/// What a two-bone solve has to satisfy, measured on the positions its rotations produce: the end
/// reaches the target, the bones keep their lengths, and the knee stays on the side it bent to.
/// </summary>
public class TwoBoneIKTests
{
    private const float Tolerance = 1e-3f;

    /// <summary>A leg in the YZ plane, knee slightly forward of the hip-ankle line.</summary>
    private static readonly float3 Hip = new(0f, 1f, 0f);
    private static readonly float3 Knee = new(0f, 0.5f, 0.05f);
    private static readonly float3 Ankle = new(0f, 0f, 0f);

    private static readonly quaternion HipRotation = quaternion.Euler(0.3f, -0.7f, 0.2f);
    private static readonly quaternion KneeRotation = quaternion.Euler(-0.4f, 0.1f, 0.9f);

    /// <summary>
    /// Runs the solve and moves the chain by the rotation each joint gained, which is what a rig
    /// would show.
    /// </summary>
    private static void SolveAndPose(float3 a, float3 b, float3 c, float3 target, float3 fallbackBendDirection,
        out float3 solvedB, out float3 solvedC)
    {
        TwoBoneIK.Solve(a, b, c, HipRotation, KneeRotation, target, fallbackBendDirection,
            out var newRotationA, out var newRotationB);

        var deltaA = math.mul(newRotationA, math.inverse(HipRotation));
        var deltaB = math.mul(newRotationB, math.inverse(KneeRotation));
        solvedB = a + math.mul(deltaA, b - a);
        solvedC = solvedB + math.mul(deltaB, c - b);
    }

    [Test]
    public void ReachableTarget_IsReached()
    {
        var target = new float3(0.1f, 0.2f, 0.15f);
        SolveAndPose(Hip, Knee, Ankle, target, math.forward(), out _, out var solvedC);

        Assert.That(math.distance(solvedC, target), Is.LessThan(Tolerance));
    }

    [Test]
    public void BoneLengths_ArePreserved()
    {
        SolveAndPose(Hip, Knee, Ankle, new float3(0.1f, 0.2f, 0.15f), math.forward(),
            out var solvedB, out var solvedC);

        Assert.That(math.distance(Hip, solvedB), Is.EqualTo(math.distance(Hip, Knee)).Within(1e-4f));
        Assert.That(math.distance(solvedB, solvedC), Is.EqualTo(math.distance(Knee, Ankle)).Within(1e-4f));
    }

    /// <summary>Out of reach, the chain straightens towards the target instead of failing.</summary>
    [Test]
    public void UnreachableTarget_ExtendsTheChainTowardsIt()
    {
        var target = new float3(0f, -2f, 0.5f);
        SolveAndPose(Hip, Knee, Ankle, target, math.forward(), out _, out var solvedC);

        var chainLength = math.distance(Hip, Knee) + math.distance(Knee, Ankle);
        Assert.That(math.distance(Hip, solvedC), Is.EqualTo(chainLength).Within(2e-3f));
        Assert.That(math.dot(math.normalize(solvedC - Hip), math.normalize(target - Hip)),
            Is.GreaterThan(0.9999f));
    }

    /// <summary>A target in the chain's plane keeps the knee in that plane and on the side it was.</summary>
    [Test]
    public void BentChain_StaysInItsPlane()
    {
        // The fallback points sideways; a bent chain must ignore it.
        SolveAndPose(Hip, Knee, Ankle, new float3(0f, 0.3f, 0.2f), math.right(), out var solvedB, out _);

        Assert.That(solvedB.x, Is.EqualTo(0f).Within(1e-4f));
        Assert.That(solvedB.z, Is.GreaterThan(0f));
    }

    /// <summary>A straight chain has no plane, so the fallback decides which way the knee goes.</summary>
    [Test]
    public void StraightChain_BendsTowardsTheFallback()
    {
        var straightKnee = new float3(0f, 0.5f, 0f);
        SolveAndPose(Hip, straightKnee, Ankle, new float3(0f, 0.3f, 0f), math.forward(),
            out var solvedB, out var solvedC);

        Assert.That(solvedB.z, Is.GreaterThan(0.1f));
        Assert.That(math.abs(solvedB.x), Is.LessThan(1e-4f));
        Assert.That(math.distance(solvedC, new float3(0f, 0.3f, 0f)), Is.LessThan(Tolerance));
    }
}
}
