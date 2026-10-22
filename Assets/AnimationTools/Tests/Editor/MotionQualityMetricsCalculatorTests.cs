using Unity.Mathematics;
using NUnit.Framework;

namespace AnimationTools.Tests
{
public class MotionQualityMetricsCalculatorTests
{
    private static readonly GroundContactSettings Settings = new();

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
            root, new[] { leftFoot, rightFoot }, new[] { leftContacts, rightContacts }, null, times, 0f, Settings);

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
            root, new[] { leftFoot, rightFoot }, new[] { leftContacts, rightContacts }, null, times, 0f, Settings);

        // totalSlip = 4 * 0.05 = 0.2, totalRootTravelXZ = 4 * 0.1 = 0.4
        Assert.AreEqual(0.5f, result.footskatePerMeter, 1e-4f);
        // Both feet sit at their contact height throughout, so 0.8 weighted contact-seconds.
        Assert.AreEqual(0.25f, result.meanFootskateSpeed, 1e-4f);

        Assert.AreEqual(0.5f, result.flagFootskatePerMeter, 1e-4f);
        // Only the left foot is flagged: 0.4 flagged contact-seconds.
        Assert.AreEqual(0.5f, result.flagMeanFootskateSpeed, 1e-4f);
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
            root, new[] { leftFoot, rightFoot }, new[] { leftContacts, rightContacts }, null, times, 0.5f, Settings);

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
            root, null, null, null, times, 0f, Settings);

        Assert.AreEqual(0f, result.rootJerkMean, 1e-3f);
        Assert.AreEqual(0f, result.rootJerkP95, 1e-3f);
        Assert.AreEqual(0f, result.rootVerticalJerkMean, 1e-3f);
        Assert.AreEqual(0f, result.rootVerticalJerkP95, 1e-3f);
    }

    [Test]
    public void PositionPop_JerkP95AboveMean()
    {
        var times = UniformTimes(10, 0.1f);
        var root = new float3[10];
        for (var i = 0; i < 10; i++) root[i] = new float3(times[i], 0f, 0f);
        root[5] += new float3(1f, 0f, 0f); // single-frame pop

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, null, null, null, times, 0f, Settings);

        Assert.Greater(result.rootJerkP95, 0f);
        Assert.Greater(result.rootJerkP95, result.rootJerkMean);
        // The pop is horizontal, so the vertical channel sees none of it.
        Assert.AreEqual(0f, result.rootVerticalJerkP95, 1e-3f);
    }

    [Test]
    public void VerticalBob_IsVerticalJerkOnly_AtTheAnalyticAmplitude()
    {
        // 1/64 s steps keep times and the 1 m/s x-track exact in float, so ground-plane jerk is exactly 0.
        const float dt = 1f / 64f;
        const float amplitude = 0.025f;
        const float frequency = 2f;
        var times = UniformTimes(256, dt); // 4 s, 8 bob cycles
        var root = new float3[256];
        for (var i = 0; i < 256; i++)
        {
            root[i] = new float3(times[i], amplitude * math.sin(2f * math.PI * frequency * times[i]), 0f);
        }

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, null, null, null, times, 0f, Settings);

        Assert.AreEqual(0f, result.rootJerkMean, 1e-3f);
        Assert.AreEqual(0f, result.rootJerkP95, 1e-3f);

        // y = A sin(wt) has |jerk| = A w^3 |cos(wt)|: peak A w^3, mean (2/pi) A w^3.
        var omega = 2f * math.PI * frequency;
        var peak = amplitude * omega * omega * omega; // ~49.6 m/s^3
        Assert.AreEqual(peak, result.rootVerticalJerkP95, 0.03f * peak);
        Assert.AreEqual(2f / math.PI * peak, result.rootVerticalJerkMean, 0.03f * peak);
    }

    [Test]
    public void HorizontalVelocityStep_RegistersAsGroundPlaneJerk()
    {
        const float dt = 1f / 64f;
        var times = UniformTimes(40, dt);
        var root = new float3[40];
        for (var i = 1; i < 40; i++)
        {
            // 1 m/s, stepping to 2 m/s at frame 20, at a constant height.
            var speed = i <= 20 ? 1f : 2f;
            root[i] = root[i - 1] + new float3(0f, 0f, speed * dt);
        }

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, null, null, null, times, 0f, Settings);

        // A 1 m/s step over one tick is a jerk of 1 / dt^2 ~ 4096 m/s^3 on the samples straddling it.
        Assert.Greater(result.rootJerkP95, 1000f);
        Assert.AreEqual(0f, result.rootVerticalJerkMean, 1e-3f);
    }

    [Test]
    public void NullFootArrays_FootskateNaN_JerkStillValid()
    {
        var times = UniformTimes(5, 0.1f);
        var root = new float3[5];
        for (var i = 0; i < 5; i++) root[i] = new float3(times[i], 0f, 0f);

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, null, null, null, times, 0f, Settings);

        Assert.IsNaN(result.footskatePerMeter);
        Assert.IsNaN(result.meanFootskateSpeed);
        Assert.IsNaN(result.contactFraction);
        Assert.IsNaN(result.flagFootskatePerMeter);
        Assert.IsNaN(result.flagContactFraction);
        Assert.IsNaN(result.contactFlagAgreement);
        Assert.IsFalse(float.IsNaN(result.rootJerkMean));
        Assert.IsFalse(float.IsNaN(result.rootJerkP95));
        Assert.IsFalse(float.IsNaN(result.rootVerticalJerkMean));
    }

    [Test]
    public void FewerThanFourFrames_ReturnsZeroEvaluated()
    {
        var times = new[] { 0f, 0.1f, 0.2f };
        var root = new[] { float3.zero, float3.zero, float3.zero };

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, null, null, null, times, 0f, Settings);

        Assert.AreEqual(0, result.framesEvaluated);
        Assert.IsNaN(result.footskatePerMeter);
        Assert.IsNaN(result.meanFootskateSpeed);
        Assert.IsNaN(result.contactFraction);
        Assert.IsNaN(result.flagFootskatePerMeter);
        Assert.IsNaN(result.flagMeanFootskateSpeed);
        Assert.IsNaN(result.flagContactFraction);
        Assert.IsNaN(result.contactFlagAgreement);
        Assert.IsNaN(result.rootJerkMean);
        Assert.IsNaN(result.rootJerkP95);
        Assert.IsNaN(result.rootVerticalJerkMean);
        Assert.IsNaN(result.rootVerticalJerkP95);
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
            root, null, null, discontinuities, times, 0f, Settings);

        Assert.AreEqual(3f, result.discontinuitiesPerSecond, 1e-4f);
    }

    [Test]
    public void NeverFlagged_FlagFootskateIsUnavailable_GeometricStillMeasuresTheSlide()
    {
        // A stage that writes no foot contacts must not be scored as having perfect foot planting by
        // its own flags, and the geometric metric must still see feet sliding along the ground.
        var times = UniformTimes(10, 1f / 30f);
        var root = new float3[10];
        var foot = new float3[10];
        var contacts = new bool[10];
        for (var i = 0; i < 10; i++)
        {
            root[i] = new float3(i * 0.1f, 0f, 0f);
            foot[i] = new float3(i * 0.1f, 0f, 0f); // 3 m/s, on the ground
        }

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, new[] { foot, foot }, new[] { contacts, contacts }, null, times, 0f, Settings);

        Assert.IsNaN(result.flagFootskatePerMeter);
        Assert.IsNaN(result.flagMeanFootskateSpeed);
        Assert.AreEqual(0f, result.flagContactFraction, 1e-6f);

        // Two feet each sliding as far as the root travels.
        Assert.AreEqual(2f, result.footskatePerMeter, 1e-4f);
        Assert.AreEqual(3f, result.meanFootskateSpeed, 1e-3f);
        // Too fast to be in stance, so never in geometric contact, which the all-false flags match.
        Assert.AreEqual(0f, result.contactFraction, 1e-6f);
        Assert.AreEqual(1f, result.contactFlagAgreement, 1e-6f);
    }

    [Test]
    public void ThreeContactBones_SlipSumsOverEveryBoneAndAnyContactCounts()
    {
        var times = UniformTimes(5, 0.1f);
        var root = new float3[5];
        var planted = new float3[5];
        var sliding = new float3[5];
        for (var i = 0; i < 5; i++)
        {
            root[i] = new float3(times[i], 0f, 0f); // 0.4 m total travel
            sliding[i] = new float3(0.05f * i, 0f, 0f); // 0.2 m slip
        }

        var always = new[] { true, true, true, true, true };
        var never = new bool[5];
        var once = new[] { false, false, true, false, false };

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, new[] { planted, sliding, planted }, new[] { always, always, once }, null, times, 0f, Settings);

        Assert.AreEqual(0.5f, result.flagFootskatePerMeter, 1e-4f);
        // Two bones flagged across four intervals: 0.2 m over 0.8 contact-seconds.
        Assert.AreEqual(0.25f, result.flagMeanFootskateSpeed, 1e-4f);
        Assert.AreEqual(1f, result.flagContactFraction, 1e-5f);

        // Every bone stays on the ground: 0.2 m over 1.2 weighted contact-seconds.
        Assert.AreEqual(0.5f, result.footskatePerMeter, 1e-4f);
        Assert.AreEqual(0.2f / 1.2f, result.meanFootskateSpeed, 1e-4f);
        Assert.AreEqual(1f, result.contactFraction, 1e-5f);

        var sparse = MotionQualityMetricsCalculator.Evaluate(
            root, new[] { planted, sliding, planted }, new[] { never, never, once }, null, times, 0f, Settings);
        Assert.AreEqual(0.2f, sparse.flagContactFraction, 1e-5f);
        Assert.AreEqual(1f, sparse.contactFraction, 1e-5f);
    }

    [Test]
    public void GeometricFootskate_PlantedFootThatDoesNotSlide_IsZero()
    {
        var times = UniformTimes(31, 1f / 30f);
        var root = new float3[31];
        var foot = new float3[31];
        for (var i = 0; i < 31; i++)
        {
            root[i] = new float3(times[i], 0f, 0f);
            foot[i] = new float3(0.1f, 0.04f, 0.3f);
        }

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, new[] { foot }, null, null, times, 0f, Settings);

        Assert.AreEqual(0f, result.footskatePerMeter, 1e-6f);
        Assert.AreEqual(0f, result.meanFootskateSpeed, 1e-6f);
        Assert.AreEqual(1f, result.contactFraction, 1e-6f);
        Assert.IsNaN(result.contactFlagAgreement);
    }

    [Test]
    public void GeometricFootskate_PlantedFootSlidingAtKnownSpeed_MatchesThatSpeed()
    {
        var times = UniformTimes(31, 1f / 30f); // 1 s
        var root = new float3[31];
        var foot = new float3[31];
        for (var i = 0; i < 31; i++)
        {
            root[i] = new float3(times[i], 0f, 0f); // 1 m of travel
            foot[i] = new float3(0.2f * times[i], 0f, 0f); // slides at 0.2 m/s on the ground
        }

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, new[] { foot }, null, null, times, 0f, Settings);

        Assert.AreEqual(0.2f, result.footskatePerMeter, 1e-4f);
        Assert.AreEqual(0.2f, result.meanFootskateSpeed, 1e-4f);
    }

    [Test]
    public void GeometricFootskate_RaisedSwingFootMovingFast_ContributesNothing()
    {
        var times = UniformTimes(30, 1f / 30f);
        var root = new float3[30];
        var foot = new float3[30];
        for (var i = 0; i < 30; i++)
        {
            root[i] = new float3(times[i], 0f, 0f);
            // Planted and still for a third of the run, then lifted 10 cm and swung at 3 m/s.
            foot[i] = i < 10
                ? new float3(0f, 0f, 0f)
                : new float3(0.1f * (i - 9), 0.1f, 0f);
        }

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, new[] { foot }, null, null, times, 0f, Settings);

        Assert.AreEqual(0f, result.footskatePerMeter, 1e-6f);
        Assert.AreEqual(0f, result.meanFootskateSpeed, 1e-6f);
        Assert.AreEqual(9f / 30f, result.contactFraction, 1e-6f);
    }

    [Test]
    public void GeometricFootskate_IsIdenticalWhateverTheFlagsSay()
    {
        var times = UniformTimes(30, 1f / 30f);
        var root = new float3[30];
        var foot = new float3[30];
        var correct = new bool[30];
        var allTrue = new bool[30];
        var allFalse = new bool[30];
        for (var i = 0; i < 30; i++)
        {
            root[i] = new float3(times[i], 0f, 0f);
            var planted = i < 15;
            foot[i] = planted
                ? new float3(0.05f * times[i], 0f, 0f) // slow slide while down
                : new float3(0.05f * times[i] + 0.1f * (i - 14), 0.1f, 0f); // swing
            correct[i] = planted;
            allTrue[i] = true;
        }

        var reference = MotionQualityMetricsCalculator.Evaluate(
            root, new[] { foot }, null, null, times, 0f, Settings);
        Assert.Greater(reference.footskatePerMeter, 0f);

        foreach (var flags in new[] { correct, allTrue, allFalse })
        {
            var result = MotionQualityMetricsCalculator.Evaluate(
                root, new[] { foot }, new[] { flags }, null, times, 0f, Settings);

            Assert.AreEqual(reference.footskatePerMeter, result.footskatePerMeter);
            Assert.AreEqual(reference.meanFootskateSpeed, result.meanFootskateSpeed);
            Assert.AreEqual(reference.contactFraction, result.contactFraction);
        }
    }

    [Test]
    public void ContactFlagAgreement_MatchesKnownMismatchCount()
    {
        var times = UniformTimes(20, 1f / 30f);
        var root = new float3[20];
        var liftingFoot = new float3[20];
        var plantedFoot = new float3[20];
        var lateFlags = new bool[20];
        var plantedFlags = new bool[20];
        for (var i = 0; i < 20; i++)
        {
            root[i] = new float3(times[i], 0f, 0f);
            // Down for frames 0-9, then held 10 cm up without moving horizontally.
            liftingFoot[i] = new float3(0f, i < 10 ? 0f : 0.1f, 0f);
            plantedFoot[i] = new float3(0.3f, 0f, 0f);
            lateFlags[i] = i < 15; // still flagged for 5 frames after lifting
            plantedFlags[i] = true;
        }

        var result = MotionQualityMetricsCalculator.Evaluate(
            root, new[] { liftingFoot, plantedFoot }, new[] { lateFlags, plantedFlags }, null, times, 0f, Settings);

        // 15 of 20 frames agree on the lifting foot, all 20 on the planted one.
        Assert.AreEqual((0.75f + 1f) / 2f, result.contactFlagAgreement, 1e-6f);
    }

    [Test]
    public void ContactHeight_ConstantToeOffsetIsTakenAsTheGround()
    {
        // A planted toe joint sits above y = 0. Measured from y = 0, an 8 cm offset would put the
        // weight at zero and hide the slide; measured from the bone's own contact height, it counts.
        var times = UniformTimes(31, 1f / 30f);
        var root = new float3[31];
        var groundFoot = new float3[31];
        var offsetFoot = new float3[31];
        for (var i = 0; i < 31; i++)
        {
            root[i] = new float3(times[i], 0f, 0f);
            groundFoot[i] = new float3(0.1f * times[i], 0f, 0f);
            offsetFoot[i] = new float3(0.1f * times[i], 0.08f, 0f);
        }

        var onGround = MotionQualityMetricsCalculator.Evaluate(
            root, new[] { groundFoot }, null, null, times, 0f, Settings);
        var offset = MotionQualityMetricsCalculator.Evaluate(
            root, new[] { offsetFoot }, null, null, times, 0f, Settings);

        Assert.AreEqual(0.1f, offset.footskatePerMeter, 1e-4f);
        Assert.AreEqual(onGround.footskatePerMeter, offset.footskatePerMeter, 1e-6f);
        Assert.AreEqual(onGround.meanFootskateSpeed, offset.meanFootskateSpeed, 1e-6f);
        // 0.1 m/s is under the stance speed threshold.
        Assert.AreEqual(1f, offset.contactFraction, 1e-6f);
    }
}
}
