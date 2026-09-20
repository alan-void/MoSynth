using System;
using AnimationTools;
using UnityEngine;

namespace Lmm
{
// Benchmark overrides for the learned matching arm. See the note in the MotionField equivalent for
// why these live beside the thing they poke rather than in AnimationTools.

/// <summary>
/// Switches which of the three networks are running, so one character prefab covers every ablation.
/// </summary>
/// <remarks>
/// This is what makes the staging measurable rather than merely sequential: the same prefab, the
/// same database and the same query, with one more piece of the classic matcher replaced each time.
/// </remarks>
[Serializable]
public sealed class LmmModeOverride : BenchmarkOverride
{
    [Tooltip("How much of the method runs. A mode whose networks the checkpoint does not carry is " +
             "refused at Init rather than silently downgraded.")]
    public LmmMode mode = LmmMode.DecompressorOnly;

    public override string Describe() => $"lmm={mode}";

    public override void Apply(GameObject characterInstance) =>
        LmmOverrideTarget.ForEach(characterInstance, nameof(LmmModeOverride), stage => stage.mode = mode);
}

/// <summary>
/// Changes how often the database is searched, the same quality/cost dial
/// <c>SearchIntervalOverride</c> turns for the classic matcher.
/// </summary>
/// <remarks>
/// It means something different once the stepper exists: for the classic matcher a long interval
/// means playing the database further before reconsidering, while for a learned one it means
/// trusting the stepper further. Sweeping it against the classic matcher's is how the difference is
/// measured rather than assumed.
/// </remarks>
[Serializable]
public sealed class LmmSearchIntervalOverride : BenchmarkOverride
{
    [Tooltip("Seconds between searches. 0 searches every tick.")]
    [Min(0f)] public float searchInterval = 10f / 60f;

    public override string Describe() => $"lmmSearchInterval={searchInterval:0.###}";

    public override void Apply(GameObject characterInstance) =>
        LmmOverrideTarget.ForEach(characterInstance, nameof(LmmSearchIntervalOverride),
            stage => stage.searchInterval = searchInterval);
}

/// <summary>
/// Turns off the discontinuity a jump raises, so downstream blending does not re-anchor.
/// </summary>
/// <remarks>
/// Deliberately available as a measurement, because it is the comparison's most flattering mistake.
/// A learned matcher has no frame jumps by construction, so leaving the flag down makes it smoother
/// than the classic matcher on exactly the transitions both of them take — an advantage that comes
/// from not reporting the jump rather than from not making it.
/// </remarks>
[Serializable]
public sealed class LmmPoseDiscontinuityOverride : BenchmarkOverride
{
    public bool raisePoseDiscontinuity = true;

    public override string Describe() => $"lmmDiscontinuity={raisePoseDiscontinuity}";

    public override void Apply(GameObject characterInstance) =>
        LmmOverrideTarget.ForEach(characterInstance, nameof(LmmPoseDiscontinuityOverride),
            stage => stage.raisePoseDiscontinuity = raisePoseDiscontinuity);
}

/// <summary>Shared lookup, so each override above is only the knob it turns.</summary>
internal static class LmmOverrideTarget
{
    internal static void ForEach(GameObject characterInstance, string overrideName,
        Action<LmmStage> apply)
    {
        var synthesizer = characterInstance.GetComponentInChildren<MotionSynthesisComponent>(true);
        if (synthesizer == null)
        {
            Debug.LogWarning($"{overrideName}: no MotionSynthesisComponent under \"{characterInstance.name}\".");
            return;
        }

        var matched = 0;
        foreach (var stage in synthesizer.stages)
        {
            if (stage is not LmmStage lmm) continue;
            apply(lmm);
            matched++;
        }

        if (matched == 0)
            Debug.LogWarning($"{overrideName}: no LmmStage on \"{characterInstance.name}\".");
    }
}
}
