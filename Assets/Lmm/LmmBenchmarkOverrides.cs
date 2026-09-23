using System;
using AnimationTools;
using UnityEngine;

namespace Lmm
{
/// <summary>
/// Switches which of the three networks are running, so one character prefab covers every ablation.
/// </summary>
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
/// With the stepper running, a longer interval means trusting the stepper further rather than
/// playing the database further.
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
/// For measurement only: with the flag down the learned matcher looks smoother than the classic one
/// because it stops reporting jumps, not because it stops making them.
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
