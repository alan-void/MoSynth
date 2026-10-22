using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;

namespace Style.Tests
{
/// <summary>
/// The arithmetic the style stage edits a pose with. A mistake here does not throw; it restyles the
/// character wrongly, or turns it, or moves it.
/// </summary>
public class StyleCompositionTests
{
    private const float Tolerance = 1e-4f;

    private static readonly quaternion Posture = quaternion.EulerXYZ(0.3f, -0.2f, 0.1f);
    private static readonly quaternion Dynamics = quaternion.EulerXYZ(-0.1f, 0.4f, 0.25f);

    private static void AssertSameRotation(quaternion expected, quaternion actual, string message = null)
    {
        // q and -q are the same rotation.
        var dot = math.abs(math.dot(expected.value, actual.value));
        Assert.AreEqual(1f, dot, Tolerance, message);
    }

    private static void AssertClose(float3 expected, float3 actual, string message = null)
    {
        Assert.Less(math.length(expected - actual), Tolerance, $"{message} expected {expected}, got {actual}");
    }

    // --- Composition -----------------------------------------------------------------------------

    [Test]
    public void ZeroIntensities_AreTheIdentity()
    {
        var delta = StyleComposition.ComposeDelta(Posture, Dynamics, 0f, 0f);
        AssertSameRotation(quaternion.identity, delta);

        var input = quaternion.EulerXYZ(0.7f, 1.1f, -0.4f);
        AssertSameRotation(input, math.mul(input, delta));
    }

    [Test]
    public void UnitIntensities_ComposePostureThenDynamics()
    {
        var delta = StyleComposition.ComposeDelta(Posture, Dynamics, 1f, 1f);
        AssertSameRotation(math.mul(Posture, Dynamics), delta);
    }

    [Test]
    public void Scale_AboveOne_TurnsFurtherAboutTheSameAxis()
    {
        AssertSameRotation(math.mul(Posture, Posture), StyleComposition.Scale(Posture, 2f));
        AssertSameRotation(Posture, StyleComposition.Scale(Posture, 1f));
    }

    [Test]
    public void Scale_TakesTheShortWayRound()
    {
        var negated = new quaternion(-Posture.value);
        AssertSameRotation(StyleComposition.Scale(Posture, 0.5f), StyleComposition.Scale(negated, 0.5f));
    }

    [Test]
    public void Crossfade_EndsOnEachSide()
    {
        AssertSameRotation(Posture, StyleComposition.Crossfade(Posture, Dynamics, 0f));
        AssertSameRotation(Dynamics, StyleComposition.Crossfade(Posture, Dynamics, 1f));
    }

    // --- Yaw strip -------------------------------------------------------------------------------

    private static IEnumerable<float3> ForwardAxes()
    {
        yield return math.forward();
        // A rig whose root bone points its local Y along the character's facing, as Mixamo hips do.
        yield return math.up();
        yield return math.normalize(new float3(0.2f, 0.1f, 1f));
    }

    [Test]
    public void StripYaw_KeepsTheFrameHeading([ValueSource(nameof(ForwardAxes))] float3 axis)
    {
        var original = math.mul(quaternion.RotateY(0.8f), quaternion.EulerXYZ(0.05f, 0f, -0.03f));
        var delta = quaternion.EulerXYZ(0.2f, 0.5f, -0.15f);
        var modified = math.mul(original, delta);

        var stripped = StyleComposition.StripYaw(original, modified, axis);

        AssertClose(StyleComposition.FlattenedForward(original, axis),
            StyleComposition.FlattenedForward(stripped, axis), "heading");
    }

    [Test]
    public void StripYaw_KeepsPitchAndRoll([ValueSource(nameof(ForwardAxes))] float3 axis)
    {
        var original = quaternion.RotateY(-1.2f);
        var modified = math.mul(original, quaternion.EulerXYZ(0.3f, 0.6f, 0.2f));

        var stripped = StyleComposition.StripYaw(original, modified, axis);

        // Only a turn about world up separates the two, so every direction keeps its height.
        var correction = math.mul(stripped, math.inverse(modified));
        Assert.AreEqual(0f, correction.value.x, Tolerance);
        Assert.AreEqual(0f, correction.value.z, Tolerance);

        foreach (var direction in new[] { math.right(), math.up(), math.forward() })
        {
            Assert.AreEqual(math.rotate(modified, direction).y, math.rotate(stripped, direction).y,
                Tolerance);
        }
    }

    [Test]
    public void StripYaw_OfAWorldYaw_RestoresTheOriginal()
    {
        var original = quaternion.EulerXYZ(0.1f, 0.4f, 0f);
        var modified = math.mul(quaternion.RotateY(0.7f), original);

        AssertSameRotation(original, StyleComposition.StripYaw(original, modified, math.forward()));
    }

    // --- Region weights --------------------------------------------------------------------------

    //   0 root
    //   ├─ 1 spine
    //   │  ├─ 2 head
    //   │  └─ 3 arm
    //   └─ 4 hip
    //      └─ 5 knee
    private static readonly int[] Parents = { -1, 0, 1, 1, 0, 4 };

    private static float[] Resolve(Dictionary<int, float> listed)
    {
        var weights = new float[Parents.Length];
        StyleComposition.ResolveRegionWeights(Parents, listed, weights);
        return weights;
    }

    [Test]
    public void RegionWeights_DefaultToOne()
    {
        CollectionAssert.AreEqual(new[] { 1f, 1f, 1f, 1f, 1f, 1f }, Resolve(new Dictionary<int, float>()));
        CollectionAssert.AreEqual(new[] { 1f, 1f, 1f, 1f, 1f, 1f }, Resolve(null));
    }

    [Test]
    public void RegionWeights_TakeTheNearestListedAncestorOrSelf()
    {
        var weights = Resolve(new Dictionary<int, float> { { 1, 0.5f }, { 4, 0f } });
        CollectionAssert.AreEqual(new[] { 1f, 0.5f, 0.5f, 0.5f, 0f, 0f }, weights);
    }

    [Test]
    public void RegionWeights_ADeeperEntryOverridesItsAncestor()
    {
        var weights = Resolve(new Dictionary<int, float> { { 0, 2f }, { 3, 1f } });
        CollectionAssert.AreEqual(new[] { 2f, 2f, 2f, 1f, 2f, 2f }, weights);
    }

    // --- Root rewrite ----------------------------------------------------------------------------

    [Test]
    public void RootRewrite_AtZeroWeight_IsTheIdentity()
    {
        var velocity = new float3(0.2f, 0f, 1.4f);
        AssertClose(velocity, StyleComposition.StylizeRootVelocity(velocity, 0.6f,
            new float3(0.3f, 0f, -0.1f), 0f));
        Assert.AreEqual(0.35f, StyleComposition.StylizeYawRate(0.35f, 0.8f, 0f), Tolerance);
    }

    [Test]
    public void RootRewrite_AtFullWeight_ScalesAndAddsTheResidual()
    {
        var velocity = new float3(0.2f, 0f, 1.4f);
        var residual = new float3(0.3f, 0f, -0.1f);
        AssertClose(velocity * 0.6f + residual,
            StyleComposition.StylizeRootVelocity(velocity, 0.6f, residual, 1f));
        Assert.AreEqual(0.35f + 0.8f, StyleComposition.StylizeYawRate(0.35f, 0.8f, 1f), Tolerance);
    }

    [Test]
    public void RootRewrite_AtHalfWeight_MovesHalfWayToTheSpeedRatio()
    {
        var velocity = new float3(0f, 0f, 2f);
        AssertClose(new float3(0f, 0f, 2f * 0.8f),
            StyleComposition.StylizeRootVelocity(velocity, 0.6f, float3.zero, 0.5f));
    }
}
}
