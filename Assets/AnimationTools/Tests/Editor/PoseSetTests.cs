using NUnit.Framework;

namespace AnimationTools.Tests
{
/// <summary>
/// Clip bookkeeping on <see cref="PoseSet"/>: poses from every clip share one array, so the
/// interesting behaviour is all about where one clip ends and the next begins.
/// </summary>
public class PoseSetTests
{
    private const float FrameTime = 1f / 30f;

    private PoseSet _poseSet;

    /// <summary>Two clips of ten frames each: clip 0 is [0, 10), clip 1 is [10, 20).</summary>
    [SetUp]
    public void SetUp()
    {
        _poseSet = new PoseSet();
        _poseSet.SetSkeleton(TestSkeletons.CreateChain3());
        _poseSet.BeginClip(10, FrameTime);
        _poseSet.BeginClip(10, FrameTime);
    }

    [TearDown]
    public void TearDown()
    {
        _poseSet.Dispose();
        TestSkeletons.DestroyAll();
    }

    [Test]
    public void GetClipContaining_ReportsTheClipThePoseBelongsTo()
    {
        Assert.AreEqual(0, _poseSet.GetClipContaining(9).Start);
        Assert.AreEqual(10, _poseSet.GetClipContaining(10).Start);
    }

    [Test]
    public void IsPoseValidForPrediction_RejectsPosesWhoseLookaheadLeavesTheClip()
    {
        Assert.IsTrue(_poseSet.IsPoseValidForPrediction(5, framesAhead: 4));
        Assert.IsFalse(_poseSet.IsPoseValidForPrediction(6, framesAhead: 4),
            "Frame 10 is the next clip, not this one's future.");
    }

    [Test]
    public void IsPoseValidForPrediction_RejectsPosesWhoseHistoryLeavesTheClip()
    {
        Assert.IsTrue(_poseSet.IsPoseValidForPrediction(14, framesAhead: 0, framesBehind: 4));
        Assert.IsFalse(_poseSet.IsPoseValidForPrediction(13, framesAhead: 0, framesBehind: 4),
            "Frame 9 is the previous clip, not this one's past.");
    }

    [Test]
    public void IsPoseValidForPrediction_WithNoWindow_AcceptsEveryPose()
    {
        for (var i = 0; i < 20; i++)
        {
            Assert.IsTrue(_poseSet.IsPoseValidForPrediction(i, framesAhead: 0));
        }
    }
}
}
