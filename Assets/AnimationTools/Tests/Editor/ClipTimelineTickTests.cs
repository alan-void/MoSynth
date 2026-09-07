using AnimationTools.Editor;
using NUnit.Framework;

namespace AnimationTools.Tests
{
/// <summary>Ruler tick spacing across the whole zoom range.</summary>
public class ClipTimelineTickTests
{
    [Test]
    public void MajorStepIsNeverZero()
    {
        foreach (var pixelsPerFrame in new[] { 0.001f, 0.05f, 1f, 7f, 40f })
        {
            ClipTimelineTicks.Choose(pixelsPerFrame, out var major, out _);
            Assert.Greater(major, 0, $"at {pixelsPerFrame} px/frame");
        }
    }

    [Test]
    public void MajorStepShrinksAsYouZoomIn()
    {
        var previous = int.MaxValue;

        foreach (var pixelsPerFrame in new[] { 0.05f, 0.2f, 1f, 4f, 12f, 40f })
        {
            ClipTimelineTicks.Choose(pixelsPerFrame, out var major, out _);
            Assert.LessOrEqual(major, previous, $"at {pixelsPerFrame} px/frame");
            previous = major;
        }
    }

    [Test]
    public void MajorTicksStayAtLeastSixtyPixelsApartWhereverPossible()
    {
        foreach (var pixelsPerFrame in new[] { 0.05f, 0.2f, 1f, 4f, 12f, 40f })
        {
            ClipTimelineTicks.Choose(pixelsPerFrame, out var major, out _);
            Assert.GreaterOrEqual(major * pixelsPerFrame, 60f, $"at {pixelsPerFrame} px/frame");
        }
    }

    [Test]
    public void MinorStepEitherDividesTheMajorOrIsAbsent()
    {
        foreach (var pixelsPerFrame in new[] { 0.05f, 0.2f, 1f, 4f, 12f, 40f })
        {
            ClipTimelineTicks.Choose(pixelsPerFrame, out var major, out var minor);
            if (minor == 0) continue;

            Assert.Less(minor, major, $"at {pixelsPerFrame} px/frame");
            Assert.AreEqual(0, major % minor, $"at {pixelsPerFrame} px/frame");
        }
    }

    [Test]
    public void FirstTickRoundsUpToTheStep()
    {
        Assert.AreEqual(0, ClipTimelineTicks.FirstTickAtOrAfter(0, 25));
        Assert.AreEqual(25, ClipTimelineTicks.FirstTickAtOrAfter(1, 25));
        Assert.AreEqual(25, ClipTimelineTicks.FirstTickAtOrAfter(25, 25));
        Assert.AreEqual(50, ClipTimelineTicks.FirstTickAtOrAfter(26, 25));
    }

    [Test]
    public void FirstTickHandlesTheFramesBeforeTheClip()
    {
        // Panning is unfenced, so the ruler starts left of frame 0 and the loop has to begin on a
        // real multiple of the step. Integer remainders go negative here, which is the trap.
        Assert.AreEqual(-25, ClipTimelineTicks.FirstTickAtOrAfter(-26, 25));
        Assert.AreEqual(-25, ClipTimelineTicks.FirstTickAtOrAfter(-25, 25));
        Assert.AreEqual(0, ClipTimelineTicks.FirstTickAtOrAfter(-24, 25));
    }

    [Test]
    public void LabelsSwitchBetweenFramesAndSeconds()
    {
        Assert.AreEqual("120", ClipTimelineTicks.Label(120, 1f / 30f, false));
        StringAssert.EndsWith("s", ClipTimelineTicks.Label(120, 1f / 30f, true));
    }

    [Test]
    public void LabelsCountBackwardsBeforeTheClipStarts()
    {
        Assert.AreEqual("-120", ClipTimelineTicks.Label(-120, 1f / 30f, false));
        StringAssert.StartsWith("-", ClipTimelineTicks.Label(-120, 1f / 30f, true));
    }
}
}
