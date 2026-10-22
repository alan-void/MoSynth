using System.IO;
using System.Text;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// The causal gait phase a stage reads from contact flags.
/// </summary>
/// <remarks>
/// The trainer runs a Python port of the same rule over the database's contacts, so the golden test
/// at the bottom is what keeps the network fed the phase it was trained on.
/// </remarks>
public class OnlineGaitPhaseTests
{
    private const float Dt = 1f / 30f;
    private const float Tolerance = 1e-4f;

    private static OnlineGaitPhase Snapping() => new() { correctionHalfLife = 0f };

    private static void Step(OnlineGaitPhase gait, bool left = false, bool right = false, float speed = 1f) =>
        gait.Update(Dt, left, right, speed);

    /// <summary>Ticks with neither foot changing, so no footfall can occur.</summary>
    private static void Idle(OnlineGaitPhase gait, int ticks, float speed = 1f)
    {
        for (var i = 0; i < ticks; i++) Step(gait, speed: speed);
    }

    private static float PhaseDistance(float a, float b)
    {
        var delta = math.abs(a - b) % GaitPhase.Tau;
        return math.min(delta, GaitPhase.Tau - delta);
    }

    [Test]
    public void FirstUpdate_StartsAtZeroAtTheStandingRate()
    {
        var gait = new OnlineGaitPhase { standingPeriod = 2f };
        gait.Update(Dt, true, false, 0f);

        Assert.AreEqual(0f, gait.Phase, Tolerance);
        Assert.AreEqual(GaitPhase.Tau / 2f, gait.Rate, Tolerance);
    }

    [Test]
    public void FootAlreadyDownAtStart_IsNotAFootfall()
    {
        var gait = Snapping();
        gait.Update(Dt, false, true, 1f);
        Step(gait, right: true);

        // Extrapolated one tick at the standing rate, not snapped back to the right-foot target.
        Assert.AreEqual(GaitPhase.Tau * Dt, gait.Phase, Tolerance);
    }

    [Test]
    public void RightFootfall_ZeroesTheCycle_LeftFootfall_IsHalfACycle()
    {
        var gait = Snapping();
        Idle(gait, 10);

        Step(gait, right: true);
        Assert.AreEqual(0f, PhaseDistance(gait.Phase, 0f), Tolerance);

        Idle(gait, 14);
        Step(gait, left: true);
        Assert.AreEqual(0f, PhaseDistance(gait.Phase, math.PI), Tolerance);
    }

    [Test]
    public void AlternatingFootfalls_SetHalfACycleOverTheInterval()
    {
        var gait = Snapping();
        Idle(gait, 10);
        Step(gait, right: true);
        Idle(gait, 9);
        Step(gait, left: true);

        // Half a cycle per footfall over ten ticks; distinct from the standing rate it started at.
        Assert.AreEqual(0.5f * GaitPhase.Tau / (10 * Dt), gait.Rate, 1e-3f);
    }

    [Test]
    public void SameFootTwice_CountsAWholeCycle()
    {
        var gait = Snapping();
        Idle(gait, 10);
        Step(gait, right: true);
        Idle(gait, 19);
        Step(gait, right: true);

        Assert.AreEqual(GaitPhase.Tau / (20 * Dt), gait.Rate, 1e-3f);
    }

    [Test]
    public void ContactChatter_WithinTheStepInterval_IsIgnored()
    {
        var gait = Snapping();
        Idle(gait, 10);
        Step(gait, right: true);
        Idle(gait, 9);
        Step(gait, left: true);
        var rate = gait.Rate;

        // Down again two ticks later: a same-foot footfall would clamp the rate to its maximum.
        Step(gait);
        Step(gait, left: true);

        Assert.AreEqual(rate, gait.Rate, Tolerance);
    }

    [Test]
    public void FootfallsTooCloseTogether_ClampTheRate()
    {
        var gait = Snapping();
        Idle(gait, 10);
        Step(gait, right: true);
        Step(gait, left: true);

        Assert.AreEqual(gait.maxRate, gait.Rate, Tolerance);
    }

    [Test]
    public void StandingStill_FallsBackToTheStandingRate_OnlyAfterTheTimeout()
    {
        var gait = Snapping();
        Idle(gait, 10);
        Step(gait, right: true);
        Idle(gait, 9);
        Step(gait, left: true);
        var walkingRate = gait.Rate;
        Assert.AreNotEqual(GaitPhase.Tau, walkingRate, "the walking rate must differ for this test");

        Idle(gait, 29, speed: 0f);
        Assert.AreEqual(walkingRate, gait.Rate, Tolerance, "inside the timeout");

        Idle(gait, 2, speed: 0f);
        Assert.AreEqual(GaitPhase.Tau / gait.standingPeriod, gait.Rate, Tolerance, "past the timeout");
    }

    [Test]
    public void MovingWithoutFootfalls_KeepsTheLastStrideRate()
    {
        var gait = Snapping();
        Idle(gait, 10);
        Step(gait, right: true);
        Idle(gait, 9);
        Step(gait, left: true);
        var walkingRate = gait.Rate;

        Idle(gait, 60, speed: 1f);
        Assert.AreEqual(walkingRate, gait.Rate, Tolerance);
    }

    [Test]
    public void Correction_IsSpreadOverTime_AndConvergesOnTheSnappedPhase()
    {
        var smooth = new OnlineGaitPhase { correctionHalfLife = 0.1f };
        var snapped = Snapping();

        void Both(bool left = false, bool right = false)
        {
            Step(smooth, left, right);
            Step(snapped, left, right);
        }

        for (var i = 0; i < 10; i++) Both();
        var before = smooth.Phase;
        Both(right: true);

        var extrapolated = before + smooth.Rate * Dt;
        var error = PhaseDistance(extrapolated, 0f);
        Assert.Greater(error, 0.5f, "the footfall must land well off the extrapolated phase");
        Assert.Less(PhaseDistance(smooth.Phase, extrapolated), 0.5f * error, "no snap at the footfall");

        for (var i = 0; i < 30; i++) Both();
        Assert.Less(PhaseDistance(smooth.Phase, snapped.Phase), 1e-2f);
    }

    [Test]
    public void Phase_StaysInsideTheHalfOpenCycle()
    {
        var gait = new OnlineGaitPhase();
        for (var i = 0; i < 2000; i++)
        {
            Step(gait, left: i % 17 == 3, right: i % 17 == 11, speed: i % 400 < 200 ? 1f : 0f);
            Assert.GreaterOrEqual(gait.Phase, 0f);
            Assert.Less(gait.Phase, GaitPhase.Tau);
        }
    }

    // --- Golden vectors shared with the Python port ------------------------------------------------

    private static string GoldenPath =>
        Path.Combine(Application.dataPath, "../Python/tests/fixtures/online_gait_phase_golden.json");

    [Test]
    public void MatchesThePythonPort_OnTheGoldenVectors()
    {
        if (!File.Exists(GoldenPath))
        {
            Assert.Ignore($"No golden vectors at {GoldenPath}. The Python side generates them " +
                          "(Python/tests); run its suite first.");
        }

        var golden = JObject.Parse(File.ReadAllText(GoldenPath));
        var parameters = golden["params"] as JObject ?? new JObject();
        var gait = new OnlineGaitPhase();
        gait.standingSpeed = Param(parameters, nameof(gait.standingSpeed), gait.standingSpeed);
        gait.standingPeriod = Param(parameters, nameof(gait.standingPeriod), gait.standingPeriod);
        gait.minStepInterval = Param(parameters, nameof(gait.minStepInterval), gait.minStepInterval);
        gait.correctionHalfLife = Param(parameters, nameof(gait.correctionHalfLife), gait.correctionHalfLife);
        gait.minRate = Param(parameters, nameof(gait.minRate), gait.minRate);
        gait.maxRate = Param(parameters, nameof(gait.maxRate), gait.maxRate);
        gait.standTimeout = Param(parameters, nameof(gait.standTimeout), gait.standTimeout);

        var dt = golden.Value<float>("dt");
        var inputs = (JArray)golden["inputs"];
        var expected = (JArray)golden["expected"];
        Assert.AreEqual(inputs.Count, expected.Count, "one expected row per input row");

        for (var i = 0; i < inputs.Count; i++)
        {
            var input = (JArray)inputs[i];
            gait.Update(dt, input[0].Value<float>() > 0.5f, input[1].Value<float>() > 0.5f,
                input[2].Value<float>());

            var row = (JArray)expected[i];
            var phase = row[0].Value<float>();
            var rate = row[1].Value<float>();

            Assert.Less(PhaseDistance(gait.Phase, phase), 1e-3f, $"phase at step {i}");
            Assert.AreEqual(rate, gait.Rate, 1e-3f, $"rate at step {i}");
        }
    }

    /// <summary>
    /// A tunable from the fixture, under its C# name or the snake_case one Python writes; the default
    /// when the fixture leaves it out.
    /// </summary>
    private static float Param(JObject parameters, string name, float fallback)
    {
        var token = parameters[name] ?? parameters[SnakeCase(name)];
        return token == null ? fallback : token.Value<float>();
    }

    private static string SnakeCase(string camelCase)
    {
        var builder = new StringBuilder();
        foreach (var character in camelCase)
        {
            if (char.IsUpper(character))
            {
                builder.Append('_').Append(char.ToLowerInvariant(character));
            }
            else
            {
                builder.Append(character);
            }
        }

        return builder.ToString();
    }
}
}
