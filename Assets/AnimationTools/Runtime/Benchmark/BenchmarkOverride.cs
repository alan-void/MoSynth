using System;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// A tweak applied to a freshly spawned benchmark character before it wakes up, so one character
/// prefab can stand in for several "methods" without a prefab variant per knob.
/// </summary>
/// <remarks>
/// Concrete overrides live next to the thing they poke — a motion field policy override belongs in
/// the MotionField assembly, which depends on this one and not the other way round. The
/// <c>[SerializeReference]</c> list on a benchmark method picks them up wherever they are defined,
/// the same way <see cref="MoSynthStage"/> and <see cref="RecorderChannel"/> are picked up.
/// <para>
/// Apply runs while the character root is still inactive, so anything set here is what Awake sees.
/// </para>
/// </remarks>
[Serializable]
public abstract class BenchmarkOverride
{
    /// <summary>Short "knob=value" text for the run label and the report. Keep it one field wide.</summary>
    public abstract string Describe();

    /// <summary>
    /// Applies this override to the spawned character. The instance is inactive and its components
    /// have not had Awake called, so fields may be written directly.
    /// </summary>
    public abstract void Apply(GameObject characterInstance);
}

/// <summary>Runs the character's pipeline at a different tick rate.</summary>
[Serializable]
public sealed class SynthesisFrameRateOverride : BenchmarkOverride
{
    [Tooltip("Synthesis ticks per second. 0 = uncapped (one tick per rendered frame).")]
    [Min(0f)] public float synthesisFrameRate = 30f;

    public override string Describe() => $"fps={synthesisFrameRate:0.##}";

    public override void Apply(GameObject characterInstance)
    {
        var synthesizer = characterInstance.GetComponentInChildren<MotionSynthesisComponent>(true);
        if (synthesizer == null)
        {
            Debug.LogWarning($"{nameof(SynthesisFrameRateOverride)}: no MotionSynthesisComponent under \"{characterInstance.name}\".");
            return;
        }

        synthesizer.synthesisFrameRate = synthesisFrameRate;
    }
}

/// <summary>Enables or disables one stage by concrete type name, e.g. to benchmark with and without inertialization.</summary>
[Serializable]
public sealed class StageEnabledOverride : BenchmarkOverride
{
    [Tooltip("Concrete stage type name, without namespace (e.g. \"Inertialization\").")]
    public string stageTypeName = "";

    public bool enabled = true;

    public override string Describe() => $"{stageTypeName}={(enabled ? "on" : "off")}";

    public override void Apply(GameObject characterInstance)
    {
        var synthesizer = characterInstance.GetComponentInChildren<MotionSynthesisComponent>(true);
        if (synthesizer == null)
        {
            Debug.LogWarning($"{nameof(StageEnabledOverride)}: no MotionSynthesisComponent under \"{characterInstance.name}\".");
            return;
        }

        var matched = 0;
        foreach (var stage in synthesizer.stages)
        {
            if (stage == null || stage.GetType().Name != stageTypeName) continue;
            stage.isEnabled = enabled;
            matched++;
        }

        if (matched == 0)
            Debug.LogWarning($"{nameof(StageEnabledOverride)}: no stage named \"{stageTypeName}\" on \"{characterInstance.name}\".");
    }
}
}
