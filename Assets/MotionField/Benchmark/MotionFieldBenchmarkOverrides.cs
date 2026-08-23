using System;
using AnimationTools;
using UnityEngine;

namespace MotionField
{
// Benchmark overrides for the motion field. They live here rather than in AnimationTools because
// AnimationTools is the assembly this one depends on, not the other way round; the
// [SerializeReference] list on a BenchmarkMethod picks them up wherever they are declared.
//
// Deliberately absent: overrides for MotionFieldConfig hyperparameters (kNeighbors, tugRatio, the
// per-bone weights). Those live on a shared ScriptableObject, so writing them would dirty the asset
// and leak into every later run in the sweep. Sweeping them needs a config asset per variant.

/// <summary>Runs the character's motion field under a different policy.</summary>
[Serializable]
public sealed class MotionFieldPolicyOverride : BenchmarkOverride
{
    public MotionFieldStage.Policy policy = MotionFieldStage.Policy.Optimal;

    public override string Describe() => $"policy={policy}";

    public override void Apply(GameObject characterInstance)
    {
        var synthesizer = characterInstance.GetComponentInChildren<MotionSynthesisComponent>(true);
        if (synthesizer == null)
        {
            Debug.LogWarning($"{nameof(MotionFieldPolicyOverride)}: no MotionSynthesisComponent under \"{characterInstance.name}\".");
            return;
        }

        var matched = 0;
        foreach (var stage in synthesizer.stages)
        {
            if (stage is not MotionFieldStage field) continue;
            field.policy = policy;
            matched++;
        }

        if (matched == 0)
            Debug.LogWarning($"{nameof(MotionFieldPolicyOverride)}: no MotionFieldStage on \"{characterInstance.name}\".");
    }
}

/// <summary>
/// Changes how far ahead the pure-pursuit spline follower aims. The only steering knob the motion
/// field arm has, so it is the one worth sweeping.
/// </summary>
[Serializable]
public sealed class MotionFieldLookaheadOverride : BenchmarkOverride
{
    [Min(0.1f)] public float lookaheadDistance = 1.5f;

    public override string Describe() => $"lookahead={lookaheadDistance:0.##}";

    public override void Apply(GameObject characterInstance)
    {
        var input = characterInstance.GetComponentInChildren<MotionFieldSplineControlInput>(true);
        if (input == null)
        {
            Debug.LogWarning($"{nameof(MotionFieldLookaheadOverride)}: no MotionFieldSplineControlInput under \"{characterInstance.name}\".");
            return;
        }

        input.LookaheadDistance = lookaheadDistance;
    }
}

/// <summary>Starts the field from a different database state, to check a method is not just lucky in its start pose.</summary>
[Serializable]
public sealed class MotionFieldStartStateOverride : BenchmarkOverride
{
    [Min(0)] public int startStateIndex;

    public override string Describe() => $"startState={startStateIndex}";

    public override void Apply(GameObject characterInstance)
    {
        var synthesizer = characterInstance.GetComponentInChildren<MotionSynthesisComponent>(true);
        if (synthesizer == null) return;

        foreach (var stage in synthesizer.stages)
        {
            if (stage is MotionFieldStage field) field.startStateIndex = startStateIndex;
        }
    }
}
}
