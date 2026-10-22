using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using Object = UnityEngine.Object;

namespace AnimationTools.Tests
{
/// <summary>
/// Per-seed checkpoint names, and how a sweep expands each method into one run per seed.
/// </summary>
public class SeededBenchmarkTests
{
    [Serializable]
    private sealed class SeededTestStage : MoSynthStage, ISeededStage
    {
        public bool loadsCheckpoint = true;

        public int? CheckpointSeed { get; set; }
        public bool LoadsCheckpoint => loadsCheckpoint;

        public override void Init(MotionSynthesisComponent motionSynthesisComponent)
        {
        }

        public override bool Apply(PoseBuffer pose, float deltaTime) => true;
    }

    [Serializable]
    private sealed class PlainTestStage : MoSynthStage
    {
        public override void Init(MotionSynthesisComponent motionSynthesisComponent)
        {
        }

        public override bool Apply(PoseBuffer pose, float deltaTime) => true;
    }

    /// <summary>Stands in for an override like a greedy motion field policy, which loads no checkpoint.</summary>
    [Serializable]
    private sealed class NoCheckpointOverride : BenchmarkOverride
    {
        public override string Describe() => "checkpoint=off";

        public override void Apply(GameObject characterInstance)
        {
            foreach (var stage in characterInstance.GetComponentInChildren<MotionSynthesisComponent>(true).stages)
            {
                if (stage is SeededTestStage seeded) seeded.loadsCheckpoint = false;
            }
        }
    }

    private readonly List<GameObject> _templates = new();

    [TearDown]
    public void TearDown()
    {
        foreach (var template in _templates) Object.DestroyImmediate(template);
        _templates.Clear();
    }

    private GameObject Character(MoSynthStage stage)
    {
        var template = new GameObject("SeedTestCharacter");
        template.SetActive(false);
        template.AddComponent<MotionSynthesisComponent>().stages = new List<MoSynthStage> { stage };
        _templates.Add(template);
        return template;
    }

    private BenchmarkMethod Method(string name, MoSynthStage stage, params int[] seeds) => new()
    {
        name = name,
        characterPrefab = Character(stage),
        seeds = new List<int>(seeds)
    };

    // --- Checkpoint names -------------------------------------------------------------------

    [Test]
    public void TheSeedGoesBeforeTheCompoundExtension()
    {
        Assert.AreEqual("Walk.s42.pfnn.npz", SeededCheckpoint.FileName("Walk", 42, ".pfnn.npz"));
        Assert.AreEqual("LMM_YBot.s7.lmm.npz", SeededCheckpoint.FileName("LMM_YBot", 7, ".lmm.npz"));
    }

    [Test]
    public void ThePathIsTheFileNameInsideTheFolder()
    {
        Assert.AreEqual(Path.Combine("Fields", "MF", "MF.s43.mffield.npz"),
            SeededCheckpoint.Path(Path.Combine("Fields", "MF"), "MF", 43, ".mffield.npz"));
    }

    [Test]
    public void DifferentSeedsNeverShareAFile()
    {
        Assert.AreNotEqual(SeededCheckpoint.FileName("A", 4, ".pfnn.npz"),
            SeededCheckpoint.FileName("A", 42, ".pfnn.npz"));
    }

    // --- Sweep expansion --------------------------------------------------------------------

    [Test]
    public void AMethodWithoutSeedsRunsOnceUnseeded()
    {
        var config = ScriptableObject.CreateInstance<SynthesisBenchmarkConfig>();
        config.methods.Add(Method("Mm", new PlainTestStage()));

        var runs = config.ExpandRuns();

        Assert.AreEqual(1, runs.Count);
        Assert.IsNull(runs[0].Seed);
        Assert.AreEqual("Mm", runs[0].Label);
        Object.DestroyImmediate(config);
    }

    [Test]
    public void AMethodRunsOncePerOwnSeedInOrder()
    {
        var config = ScriptableObject.CreateInstance<SynthesisBenchmarkConfig>();
        config.methods.Add(Method("Pfnn", new SeededTestStage(), 44, 42));
        config.methods.Add(Method("Mm", new PlainTestStage()));

        var runs = config.ExpandRuns();

        Assert.AreEqual(3, runs.Count);
        Assert.AreEqual(("Pfnn", 44), (runs[0].Method.name, runs[0].Seed.Value));
        Assert.AreEqual(("Pfnn", 42), (runs[1].Method.name, runs[1].Seed.Value));
        Assert.AreEqual("Pfnn_s44", runs[0].Label);
        Assert.IsNull(runs[2].Seed);
        Object.DestroyImmediate(config);
    }

    [Test]
    public void SweepSeedsReplaceTheSeedsOfSeededMethodsOnly()
    {
        var config = ScriptableObject.CreateInstance<SynthesisBenchmarkConfig>();
        config.methods.Add(Method("Lmm", new SeededTestStage(), 1));
        config.methods.Add(Method("Mm", new PlainTestStage()));

        var runs = config.ExpandRuns(new[] { 42, 43, 44 });

        CollectionAssert.AreEqual(new int?[] { 42, 43, 44, null }, runs.ConvertAll(r => r.Seed));
        Assert.AreEqual("Mm", runs[3].Method.name);
        Object.DestroyImmediate(config);
    }

    [Test]
    public void AnOverrideThatTurnsTheCheckpointOffRunsOnceUnderSweepSeeds()
    {
        var config = ScriptableObject.CreateInstance<SynthesisBenchmarkConfig>();
        var greedy = Method("MfGreedy", new SeededTestStage());
        greedy.overrides.Add(new NoCheckpointOverride());
        config.methods.Add(greedy);

        var runs = config.ExpandRuns(new[] { 42, 43 });

        Assert.AreEqual(1, runs.Count);
        Assert.IsNull(runs[0].Seed);
        Object.DestroyImmediate(config);
    }

    [Test]
    public void ADisabledSeededStageDoesNotCount()
    {
        var method = Method("Off", new SeededTestStage { isEnabled = false });
        Assert.IsFalse(method.LoadsSeededCheckpoint());
    }

    [Test]
    public void ProbingTheCheckpointLeavesThePrefabAlone()
    {
        var stage = new SeededTestStage();
        var method = Method("Lmm", stage);
        method.overrides.Add(new NoCheckpointOverride());

        method.LoadsSeededCheckpoint();

        Assert.IsTrue(stage.loadsCheckpoint);
    }

    [Test]
    public void ARepeatedSeedFailsValidation()
    {
        var config = ScriptableObject.CreateInstance<SynthesisBenchmarkConfig>();
        config.methods.Add(Method("Pfnn", new SeededTestStage(), 42, 42));

        Assert.IsFalse(config.TryValidate(out var error));
        StringAssert.Contains("seed twice", error);
        Object.DestroyImmediate(config);
    }
}
}
