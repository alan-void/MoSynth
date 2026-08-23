using Unity.Mathematics;
using NUnit.Framework;

namespace AnimationTools.Tests
{
public class MotionQualityMetricsCalculatorTests
{
    private static float[] UniformTimes(int count, float dt)
    {
        var times = new float[count];
        for (var i = 0; i < count; i++) times[i] = i * dt;
        return times;
    }

    [Test]
    public void PlantedFoot_FootskateIsZero()
    {
        var times = UniformTimes(5, 0.1f);
        var root = new float3[5];
        for (var i = 0; i < 5; i++) root[i] = new float3(times[i], 0f, 0f);

        var leftFoot = new float3[5];
        for (var i = 0; i < 5; i++) leftFoot[i] = new float3(0.2f, 0f, 0.5f);
        var leftContacts = new[] { true, true, true, true, true };

        var rightFoot = new float3[5];
        for (var i = 0; i < 5; i++) rightFoot[i] = new float3(-0.2f, 0f, 0.5f);
        var rightContacts = new[] { false, false, false, false, false };

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, leftFoot, rightFoot, leftContacts, rightContacts, null, times, 0f);

        Assert.AreEqual(5, result.framesEvaluated);
        Assert.AreEqual(0f, result.footskatePerMeter, 1e-5f);
        Assert.AreEqual(1f, result.contactFraction, 1e-5f);
    }

    [Test]
    public void SlidingFoot_FootskateMatchesKnownDistance()
    {
        var times = UniformTimes(5, 0.1f);
        var root = new float3[5];
        for (var i = 0; i < 5; i++) root[i] = new float3(times[i], 0f, 0f); // 1 m/s -> 0.4 m total travel

        var leftFoot = new float3[5];
        for (var i = 0; i < 5; i++) leftFoot[i] = new float3(0.05f * i, 0f, 0f); // slides 0.05 m/step
        var leftContacts = new[] { true, true, true, true, true };

        var rightFoot = new float3[5];
        var rightContacts = new[] { false, false, false, false, false };

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, leftFoot, rightFoot, leftContacts, rightContacts, null, times, 0f);

        // totalSlip = 4 * 0.05 = 0.2, totalRootTravelXZ = 4 * 0.1 = 0.4
        Assert.AreEqual(0.5f, result.footskatePerMeter, 1e-4f);
        // totalContactSeconds = 4 * 0.1 = 0.4
        Assert.AreEqual(0.5f, result.meanFootskateSpeed, 1e-4f);
    }

    [Test]
    public void SettleTime_ExcludesEarlyFrames()
    {
        var times = UniformTimes(11, 0.1f); // 0 .. 1.0
        var root = new float3[11];
        for (var i = 0; i < 11; i++) root[i] = new float3(times[i], 0f, 0f);

        var leftFoot = new float3[11];
        var leftContacts = new bool[11];
        for (var i = 0; i < 11; i++)
        {
            leftContacts[i] = true;
            leftFoot[i] = times[i] < 0.5f
                ? new float3(10f * math.sin(i), 0f, 0f) // wild slide before settle
                : new float3(0.3f, 0f, 0.5f); // planted after settle
        }

        var rightFoot = new float3[11];
        var rightContacts = new bool[11];

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, leftFoot, rightFoot, leftContacts, rightContacts, null, times, 0.5f);

        Assert.AreEqual(6, result.framesEvaluated);
        Assert.Less(result.framesEvaluated, times.Length);
        Assert.AreEqual(0f, result.footskatePerMeter, 1e-5f);
    }

    [Test]
    public void ConstantVelocityRoot_JerkNearZero()
    {
        var times = UniformTimes(20, 0.05f);
        var root = new float3[20];
        for (var i = 0; i < 20; i++) root[i] = new float3(times[i], 0f, 0f);

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, null, null, null, null, null, times, 0f);

        Assert.AreEqual(0f, result.rootJerkMean, 1e-3f);
        Assert.AreEqual(0f, result.rootJerkP95, 1e-3f);
    }

    [Test]
    public void PositionPop_JerkP95AboveMean()
    {
        var times = UniformTimes(10, 0.1f);
        var root = new float3[10];
        for (var i = 0; i < 10; i++) root[i] = new float3(times[i], 0f, 0f);
        root[5] += new float3(1f, 0f, 0f); // single-frame pop

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, null, null, null, null, null, times, 0f);

        Assert.Greater(result.rootJerkP95, 0f);
        Assert.Greater(result.rootJerkP95, result.rootJerkMean);
    }

    [Test]
    public void NullFootArrays_FootskateNaN_JerkStillValid()
    {
        var times = UniformTimes(5, 0.1f);
        var root = new float3[5];
        for (var i = 0; i < 5; i++) root[i] = new float3(times[i], 0f, 0f);

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, null, null, null, null, null, times, 0f);

        Assert.IsNaN(result.footskatePerMeter);
        Assert.IsNaN(result.meanFootskateSpeed);
        Assert.IsNaN(result.contactFraction);
        Assert.IsFalse(float.IsNaN(result.rootJerkMean));
        Assert.IsFalse(float.IsNaN(result.rootJerkP95));
    }

    [Test]
    public void FewerThanFourFrames_ReturnsZeroEvaluated()
    {
        var times = new[] { 0f, 0.1f, 0.2f };
        var root = new[] { float3.zero, float3.zero, float3.zero };

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, null, null, null, null, null, times, 0f);

        Assert.AreEqual(0, result.framesEvaluated);
        Assert.IsNaN(result.footskatePerMeter);
        Assert.IsNaN(result.meanFootskateSpeed);
        Assert.IsNaN(result.contactFraction);
        Assert.IsNaN(result.rootJerkMean);
        Assert.IsNaN(result.rootJerkP95);
        Assert.IsNaN(result.discontinuitiesPerSecond);
    }

    [Test]
    public void DiscontinuitiesPerSecond_MatchesKnownCountOverSpan()
    {
        var times = UniformTimes(11, 0.1f); // span 0 .. 1.0
        var root = new float3[11];

        var discontinuities = new bool[11];
        discontinuities[2] = true;
        discontinuities[5] = true;
        discontinuities[8] = true;

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, null, null, null, null, discontinuities, times, 0f);

        Assert.AreEqual(3f, result.discontinuitiesPerSecond, 1e-4f);
    }

    [Test]
    public void NeverInContact_FootskateIsUnavailableRatherThanZero()
    {
        // A stage that writes no foot contacts must not be scored as having perfect foot planting:
        // zero slip over zero contact frames is missing data, not a good result.
        var times = UniformTimes(10, 1f / 30f);
        var root = new float3[10];
        var foot = new float3[10];
        var contacts = new bool[10];
        for (var i = 0; i < 10; i++)
        {
            root[i] = new float3(i * 0.1f, 0f, 0f);
            foot[i] = new float3(i * 0.1f, 0f, 0f);
        }

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, foot, foot, contacts, contacts, null, times, 0f);

        Assert.IsNaN(result.footskatePerMeter);
        Assert.IsNaN(result.meanFootskateSpeed);
        Assert.AreEqual(0f, result.contactFraction, 1e-6f);
    }
}
}
