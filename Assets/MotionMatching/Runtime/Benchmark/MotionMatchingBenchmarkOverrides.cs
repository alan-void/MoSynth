using System;
using AnimationTools;
using UnityEngine;

namespace MotionMatching
{
// Benchmark overrides for the motion matching arm. See the note in the MotionField equivalent for
// why these live beside the thing they poke rather than in AnimationTools.

/// <summary>Changes the constant pace the spline follower travels at, which is also the target the velocity error is measured against.</summary>
[Serializable]
public sealed class SplineSpeedOverride : BenchmarkOverride
{
    [Min(0.01f)] public float speed = 1f;

    public override string Describe() => $"speed={speed:0.##}";

    public override void Apply(GameObject characterInstance)
    {
        var input = characterInstance.GetComponentInChildren<SplineControlInput>(true);
        if (input == null)
        {
            Debug.LogWarning($"{nameof(SplineSpeedOverride)}: no SplineControlInput under \"{characterInstance.name}\".");
            return;
        }

        input.speed = speed;
    }
}

/// <summary>
/// Changes how often the database is searched. The main quality/cost dial for motion matching:
/// searching every tick tracks better and costs more, which is exactly what the sweep exists to
/// quantify.
/// </summary>
[Serializable]
public sealed class SearchIntervalOverride : BenchmarkOverride
{
    [Tooltip("Seconds between searches. 0 searches every tick.")]
    [Min(0f)] public float searchInterval = 10f / 60f;

    public override string Describe() => $"searchInterval={searchInterval:0.###}";

    public override void Apply(GameObject characterInstance)
    {
        var synthesizer = characterInstance.GetComponentInChildren<MotionSynthesisComponent>(true);
        if (synthesizer == null)
        {
            Debug.LogWarning($"{nameof(SearchIntervalOverride)}: no MotionSynthesisComponent under \"{characterInstance.name}\".");
            return;
        }

        var matched = 0;
        foreach (var stage in synthesizer.stages)
        {
            if (stage is not MotionMatchingStage mm) continue;
            mm.searchInterval = searchInterval;
            matched++;
        }

        if (matched == 0)
            Debug.LogWarning($"{nameof(SearchIntervalOverride)}: no MotionMatchingStage on \"{characterInstance.name}\".");
    }
}
}
