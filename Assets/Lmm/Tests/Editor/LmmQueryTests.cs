using MotionMatching;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace Lmm.Tests
{
/// <summary>
/// The search metric a learned matcher is trained under, and the database it is allowed to learn.
/// </summary>
/// <remarks>
/// The trajectory half of the query itself needs no test here: both stages call the one
/// <see cref="MotionMatchingQuery.FillTrajectory"/>, so they ask the same question by construction
/// rather than by assertion. What is not guaranteed by construction is the *weights* — those are
/// expanded once in C# for the search and once in Python for training, and the two agreeing is
/// what stops the projector approximating a search nobody runs.
/// </remarks>
public class LmmQueryTests
{
    /// <summary>
    /// The checked-in demo asset, or an ignored test. A FeatureSet needs a real pose database,
    /// which needs real clips — there is no synthetic stand-in for that.
    /// </summary>
    private static MotionMatchingData LoadDemoData()
    {
        var data = AssetDatabase.LoadAssetAtPath<MotionMatchingData>(
            "Assets/Animation/LafanCorrected/MotionMatching/MM_LafanCorrected.asset");
        if (data == null || !data.TryValidate(out _))
        {
            Assert.Ignore("The demo MotionMatchingData is not available.");
        }

        return data;
    }

    private static LmmConfig ConfigFor(MotionMatchingData data)
    {
        var config = ScriptableObject.CreateInstance<LmmConfig>();
        config.name = "TestLmmConfig";
        config.mmData = data;

        // One distinct weight per definition, so a slice written at the wrong offset shows up as
        // the wrong number rather than as the same number.
        config.featureWeights.Clear();
        for (var i = 0; i < config.FeatureDefinitionCount; i++) config.featureWeights.Add(i + 1f);

        return config;
    }

    [Test]
    public void ExpandFeatureWeights_GivesEveryFloatOfADefinitionThatDefinitionsWeight()
    {
        var data = LoadDemoData();
        var featureSet = data.GetOrImportFeatureSet();
        Assert.IsNotNull(featureSet, "The demo asset produced no feature set.");

        var config = ConfigFor(data);
        var weights = new float[featureSet.FeatureSize];
        config.ExpandFeatureWeights(featureSet, weights);

        for (var i = 0; i < data.trajectoryFeatures.Count; i++)
        {
            var floats = featureSet.GetTrajectoryFeatureFloatCount(i);
            for (var p = 0; p < featureSet.GetPredictionCount(i); p++)
            {
                var offset = featureSet.GetTrajectoryFeatureOffset(i, p);
                for (var f = 0; f < floats; f++)
                {
                    Assert.AreEqual(i + 1f, weights[offset + f],
                        $"trajectory feature {i}, prediction {p}, float {f}");
                }
            }
        }

        for (var i = 0; i < data.poseFeatures.Count; i++)
        {
            var offset = featureSet.PoseOffset + i * FeatureSet.FloatsPerPoseFeature;
            for (var f = 0; f < FeatureSet.FloatsPerPoseFeature; f++)
            {
                Assert.AreEqual(data.trajectoryFeatures.Count + i + 1f, weights[offset + f],
                    $"pose feature {i}, float {f}");
            }
        }

        Object.DestroyImmediate(config);
    }

    [Test]
    public void ExpandFeatureWeights_LeavesNoFloatUnwritten()
    {
        var data = LoadDemoData();
        var featureSet = data.GetOrImportFeatureSet();
        Assert.IsNotNull(featureSet);

        var config = ConfigFor(data);

        // Sentinel: a float the expansion never reaches would weight that channel out of the search
        // entirely, and a search missing a channel still returns plausible frames.
        var weights = new float[featureSet.FeatureSize];
        for (var i = 0; i < weights.Length; i++) weights[i] = float.NaN;
        config.ExpandFeatureWeights(featureSet, weights);

        for (var i = 0; i < weights.Length; i++)
        {
            Assert.IsFalse(float.IsNaN(weights[i]), $"float {i} was never written");
        }

        Object.DestroyImmediate(config);
    }

    [Test]
    public void ExpandFeatureWeights_DefaultsAMissingEntryToOneRatherThanZero()
    {
        var data = LoadDemoData();
        var featureSet = data.GetOrImportFeatureSet();
        Assert.IsNotNull(featureSet);

        var config = ConfigFor(data);
        config.featureWeights.Clear();

        var weights = new float[featureSet.FeatureSize];
        config.ExpandFeatureWeights(featureSet, weights);

        for (var i = 0; i < weights.Length; i++)
        {
            Assert.AreEqual(1f, weights[i], $"float {i}");
        }

        Object.DestroyImmediate(config);
    }

    [Test]
    public void TryValidate_RefusesAConfigWithNoDatabase()
    {
        var config = ScriptableObject.CreateInstance<LmmConfig>();

        Assert.IsFalse(config.TryValidate(out var error));
        StringAssert.Contains("MotionMatchingData", error);

        Object.DestroyImmediate(config);
    }

    [Test]
    public void TryValidate_RefusesAPoseOnlyDatabase()
    {
        // A MotionMatchingData with no feature channels bakes no .mmfeatures, so there is no query
        // vector to learn against — the one thing a learned matcher cannot do without.
        var data = ScriptableObject.CreateInstance<MotionMatchingData>();
        data.name = "PoseOnly";

        var config = ScriptableObject.CreateInstance<LmmConfig>();
        config.mmData = data;

        Assert.IsFalse(config.TryValidate(out var error));
        StringAssert.Contains("feature channels", error);

        Object.DestroyImmediate(config);
        Object.DestroyImmediate(data);
    }

    [Test]
    public void TryValidate_AcceptsTheDemoDatabase()
    {
        var config = ConfigFor(LoadDemoData());

        Assert.IsTrue(config.TryValidate(out var error), error);

        Object.DestroyImmediate(config);
    }
}
}
