using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;

namespace MotionMatching.Tests
{
/// <summary>
/// Round-tripping and staleness detection for <c>.mmfeatures</c>. The format carries no version
/// byte by design, so the schema block is the only thing standing between a stale file and a
/// database of plausible numbers read off the wrong offsets — these tests are that guarantee.
/// </summary>
public class FeatureSerializerTests
{
    private string _directory;

    [SetUp]
    public void SetUp()
    {
        _directory = Path.Combine(Path.GetTempPath(), "mmfeatures-tests-" + Path.GetRandomFileName());
        Directory.CreateDirectory(_directory);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, true);
    }

    /// <summary>
    /// The checked-in demo asset, or an ignored test. Building a FeatureSet needs a real pose
    /// database, which needs real clips — there is no synthetic stand-in for that.
    /// </summary>
    private static MotionMatchingData LoadDemoData()
    {
        var data = AssetDatabase.LoadAssetAtPath<MotionMatchingData>(
            "Assets/Animation/LafanCorrected/MotionMatching/MM_LafanCorrected.asset");
        if (data == null || !data.TryValidate(out _)) Assert.Ignore("The demo MotionMatchingData is not available.");
        return data;
    }

    /// <summary>An asset whose feature configuration is deliberately not the demo asset's.</summary>
    private static MotionMatchingData CreateDifferentlyConfiguredData()
    {
        var data = ScriptableObject.CreateInstance<MotionMatchingData>();
        data.name = "DifferentlyConfigured";
        data.trajectoryFeatures.Add(new TrajectoryFeatureChannel
        {
            name = "SomethingElse",
            predictionFrames = new[] { 20 },
            zeroY = true
        });
        return data;
    }

    [Test]
    public void RoundTrip_PreservesVectorsAndStatistics()
    {
        var mmData = LoadDemoData();
        var featureSet = mmData.GetOrImportFeatureSet();
        Assert.IsNotNull(featureSet, "The demo asset produced no feature set.");

        new FeatureSerializer().Serialize(featureSet, mmData, _directory, "roundtrip");

        Assert.IsTrue(new FeatureSerializer().Deserialize(_directory, "roundtrip", mmData, out var restored));
        Assert.AreEqual(featureSet.NumberFeatureVectors, restored.NumberFeatureVectors);
        Assert.AreEqual(featureSet.FeatureSize, restored.FeatureSize);
        Assert.AreEqual(featureSet.PoseOffset, restored.PoseOffset);

        for (var i = 0; i < featureSet.FeatureSize; i++)
        {
            Assert.That(restored.GetMean(i), Is.EqualTo(featureSet.GetMean(i)).Within(1e-6f), $"mean[{i}]");
            Assert.That(restored.GetStandardDeviation(i),
                Is.EqualTo(featureSet.GetStandardDeviation(i)).Within(1e-6f), $"std[{i}]");
        }

        // Spot-check a spread of frames rather than all several thousand.
        for (var frame = 0; frame < featureSet.NumberFeatureVectors; frame += 97)
        {
            Assert.AreEqual(featureSet.IsValidFeature(frame), restored.IsValidFeature(frame), $"valid[{frame}]");

            var original = featureSet.GetFeatureVector(frame);
            var read = restored.GetFeatureVector(frame);
            for (var f = 0; f < original.Length; f++)
            {
                Assert.That(read[f], Is.EqualTo(original[f]).Within(1e-6f), $"frame {frame} float {f}");
            }
        }
    }

    [Test]
    public void Deserialize_RejectsAFileWrittenForADifferentFeatureConfiguration()
    {
        var mmData = LoadDemoData();
        var featureSet = mmData.GetOrImportFeatureSet();
        Assert.IsNotNull(featureSet, "The demo asset produced no feature set.");

        new FeatureSerializer().Serialize(featureSet, mmData, _directory, "stale");

        var other = CreateDifferentlyConfiguredData();
        try
        {
            LogAssert.ignoreFailingMessages = true;
            Assert.IsFalse(new FeatureSerializer().Deserialize(_directory, "stale", other, out var restored),
                "A file describing another configuration must be refused, not read off the wrong offsets.");
            Assert.IsNull(restored);
        }
        finally
        {
            LogAssert.ignoreFailingMessages = false;
            Object.DestroyImmediate(other);
        }
    }

    [Test]
    public void Deserialize_WithNoFile_ReportsFailureQuietly()
    {
        var mmData = LoadDemoData();

        Assert.IsFalse(new FeatureSerializer().Deserialize(_directory, "absent", mmData, out var restored));
        Assert.IsNull(restored);
    }

    [Test]
    public void Deserialize_RejectsATruncatedFile()
    {
        var mmData = LoadDemoData();
        var featureSet = mmData.GetOrImportFeatureSet();
        Assert.IsNotNull(featureSet, "The demo asset produced no feature set.");

        var path = Path.Combine(_directory, "truncated.mmfeatures");
        new FeatureSerializer().Serialize(featureSet, mmData, _directory, "truncated");

        var bytes = File.ReadAllBytes(path);
        File.WriteAllBytes(path, bytes[..(bytes.Length / 2)]);

        try
        {
            LogAssert.ignoreFailingMessages = true;
            Assert.IsFalse(new FeatureSerializer().Deserialize(_directory, "truncated", mmData, out var restored));
            Assert.IsNull(restored);
        }
        finally
        {
            LogAssert.ignoreFailingMessages = false;
        }
    }
}
}
