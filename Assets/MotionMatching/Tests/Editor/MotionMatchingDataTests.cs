using NUnit.Framework;
using UnityEngine;

namespace MotionMatching.Tests
{
/// <summary>
/// How far a feature configuration reaches either side of the query frame, which is what decides
/// which poses can carry a valid feature vector.
/// </summary>
public class MotionMatchingDataTests
{
    private MotionMatchingData _mmData;

    [SetUp]
    public void SetUp() => _mmData = ScriptableObject.CreateInstance<MotionMatchingData>();

    [TearDown]
    public void TearDown() => Object.DestroyImmediate(_mmData);

    private void AddTrajectoryFeature(params int[] predictionFrames)
    {
        _mmData.trajectoryFeatures.Add(new TrajectoryFeatureChannel
        {
            name = $"feature{_mmData.trajectoryFeatures.Count}",
            predictionFrames = predictionFrames
        });
    }

    [Test]
    public void WithNoFeatures_NeitherDirectionReaches()
    {
        Assert.AreEqual(0, _mmData.MaximumFramesPrediction);
        Assert.AreEqual(0, _mmData.MaximumFramesHistory);
    }

    [Test]
    public void ReachIsTakenAcrossEveryFeature()
    {
        AddTrajectoryFeature(20, 40);
        AddTrajectoryFeature(60);

        Assert.AreEqual(60, _mmData.MaximumFramesPrediction);
    }

    [Test]
    public void UnsortedPredictionFramesStillReportTheFurthest()
    {
        AddTrajectoryFeature(60, 20, 40);

        Assert.AreEqual(60, _mmData.MaximumFramesPrediction);
    }

    [Test]
    public void NegativePredictionFramesCountAsHistory()
    {
        AddTrajectoryFeature(-40, -20, 20, 40);

        Assert.AreEqual(40, _mmData.MaximumFramesPrediction);
        Assert.AreEqual(40, _mmData.MaximumFramesHistory);
    }

    [Test]
    public void APurelyPastTrajectoryNeedsNoLookahead()
    {
        AddTrajectoryFeature(-60, -30);

        Assert.AreEqual(0, _mmData.MaximumFramesPrediction);
        Assert.AreEqual(60, _mmData.MaximumFramesHistory);
    }
}
}
