using AnimationTools.Editor;
using NUnit.Framework;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// The timeline's frame-to-pixel mapping.
/// </summary>
/// <remarks>
/// The anchored-zoom invariant is the one worth stating as a test rather than reading off the
/// implementation: the frame under the cursor must not move while the wheel turns, and every other
/// zoom bug in a timeline is a variation on breaking it.
/// </remarks>
public class ClipTimeAxisTests
{
    private const int FrameCount = 1000;
    private const float Tolerance = 1e-3f;

    private static readonly Rect View = new(0f, 0f, 500f, 60f);

    private static ClipTimeAxis Axis(float pixelsPerFrame = 4f, float scrollFrames = 300f)
    {
        var axis = new ClipTimeAxis();
        axis.Prepare(View, FrameCount);
        axis.pixelsPerFrame = pixelsPerFrame;
        axis.scrollFrames = scrollFrames;
        return axis;
    }

    [Test]
    public void FrameAndPixelRoundTrip()
    {
        var axis = Axis();

        foreach (var frame in new[] { 0f, 12.5f, 300f, 412f, 999f })
        {
            Assert.AreEqual(frame, axis.XToFrame(axis.FrameToX(frame)), Tolerance);
        }
    }

    [Test]
    public void PrepareSeedsZoomToFitTheWholeClip()
    {
        var axis = new ClipTimeAxis();
        axis.Prepare(View, FrameCount);

        Assert.AreEqual(View.width / FrameCount, axis.pixelsPerFrame, Tolerance);
        Assert.AreEqual(0f, axis.scrollFrames, Tolerance);
    }

    [Test]
    public void PrepareIgnoresAPlaceholderWidth()
    {
        // A pane is laid out at a placeholder width before it is measured, and seeding the zoom from
        // that once drew a whole clip into one pixel with no way back. The zoom floor is now
        // absolute rather than "the clip must fit", so ignoring the placeholder is the only thing
        // standing between that bug and the user.
        var axis = new ClipTimeAxis();
        axis.Prepare(new Rect(0f, 0f, 1f, 60f), FrameCount);
        Assert.AreEqual(0f, axis.pixelsPerFrame, 1e-6f);

        axis.Prepare(View, FrameCount);

        Assert.AreEqual(axis.FitPixelsPerFrame(FrameCount), axis.pixelsPerFrame, Tolerance);
        Assert.AreEqual(FrameCount, axis.VisibleFrameSpan, 0.5f);
    }

    [Test]
    public void PrepareLeavesADeliberateZoomAlone()
    {
        var axis = Axis(pixelsPerFrame: 8f, scrollFrames: 100f);

        axis.Prepare(View, FrameCount);

        Assert.AreEqual(8f, axis.pixelsPerFrame, Tolerance);
        Assert.AreEqual(100f, axis.scrollFrames, Tolerance);
    }

    [Test]
    public void ZoomKeepsTheFrameUnderThePivotInPlace()
    {
        foreach (var pivotX in new[] { 40f, 250f, 460f })
        {
            foreach (var factor in new[] { 0.6f, 0.85f, 1.2f, 1.9f })
            {
                var axis = Axis();
                var before = axis.XToFrame(pivotX);

                axis.ZoomAt(pivotX, factor, FrameCount);

                Assert.AreEqual(before, axis.XToFrame(pivotX), Tolerance,
                    $"pivot {pivotX}, factor {factor}");
            }
        }
    }

    [Test]
    public void ZoomingOutGoesWellPastTheEndOfTheClip()
    {
        // Fitting the clip is a default, not a limit: there are reasons to want room beyond the
        // clip, and Home is always the way back.
        var axis = Axis();
        for (var i = 0; i < 6; i++) axis.ZoomAt(250f, 0.5f, FrameCount);

        Assert.Less(axis.pixelsPerFrame, axis.FitPixelsPerFrame(FrameCount));
        Assert.Greater(axis.VisibleFrameSpan, FrameCount);
    }

    [Test]
    public void ZoomStopsAtTheFloorAndAtTheMaximum()
    {
        var axis = Axis();
        for (var i = 0; i < 200; i++) axis.ZoomAt(250f, 0.5f, FrameCount);
        Assert.AreEqual(ClipTimeAxis.MinPixelsPerFrame, axis.pixelsPerFrame, 1e-6f);

        for (var i = 0; i < 200; i++) axis.ZoomAt(250f, 2f, FrameCount);
        Assert.AreEqual(ClipTimeAxis.MaxPixelsPerFrame, axis.pixelsPerFrame, Tolerance);
    }

    [Test]
    public void AClipShorterThanTheViewCanStillBeShownWhole()
    {
        // Fitting five frames into 500px already exceeds the normal ceiling, so the ceiling has to
        // rise to the fit rather than crop the clip.
        var axis = new ClipTimeAxis();
        axis.Prepare(View, 5);

        var fitted = axis.pixelsPerFrame;
        Assert.Greater(fitted, ClipTimeAxis.MaxPixelsPerFrame);

        axis.ZoomAt(250f, 2f, 5);
        Assert.AreEqual(fitted, axis.pixelsPerFrame, Tolerance);

        axis.ZoomAt(250f, 0.25f, 5);
        Assert.Less(axis.pixelsPerFrame, fitted);
    }

    [Test]
    public void PanningIsNotFencedByTheClip()
    {
        // The clip is content to look at, not a wall to bump into. A view that has run off the end
        // of it is one you are allowed to hold; the ruler keeps numbering into it and Home is the
        // way back.
        var axis = Axis();

        axis.PanPixels(-1_000_000f);
        Assert.Less(axis.LeftEdgeFrame, -1000f);

        axis.PanPixels(2_000_000f);
        Assert.Greater(axis.LeftEdgeFrame, FrameCount + 1000f);
    }

    [Test]
    public void TheEdgeFramesReportWhatIsOnScreenEvenOutsideTheClip()
    {
        // The ruler draws from these, which is what lets it label negative frames; the clamped pair
        // beside them is what indexes per-frame data and must never leave the clip.
        var axis = Axis(pixelsPerFrame: 4f, scrollFrames: -200f);

        Assert.AreEqual(-200f, axis.LeftEdgeFrame, Tolerance);
        Assert.AreEqual(-200f + View.width / 4f, axis.RightEdgeFrame, Tolerance);

        Assert.AreEqual(0, axis.FirstVisibleFrame(FrameCount));
        Assert.GreaterOrEqual(axis.LastVisibleFrame(FrameCount), 0);
    }

    [Test]
    public void PanMovesByWholePixels()
    {
        var axis = Axis();
        var frameAtLeftEdge = axis.XToFrame(View.xMin);

        axis.PanPixels(40f);

        Assert.AreEqual(frameAtLeftEdge + 40f / axis.pixelsPerFrame, axis.scrollFrames, Tolerance);
    }

    [Test]
    public void FrameRangeFitsTheRequestedSpan()
    {
        var axis = Axis();
        axis.FrameRange(200, 299, FrameCount);

        Assert.AreEqual(200f, axis.scrollFrames, Tolerance);
        Assert.AreEqual(100f, axis.VisibleFrameSpan, Tolerance);
    }

    [Test]
    public void VisibleBoundsBracketTheViewAndStayInsideTheClip()
    {
        var axis = Axis();

        var first = axis.FirstVisibleFrame(FrameCount);
        var last = axis.LastVisibleFrame(FrameCount);

        Assert.LessOrEqual(first, Mathf.CeilToInt(axis.scrollFrames));
        Assert.GreaterOrEqual(last, Mathf.FloorToInt(axis.scrollFrames + axis.VisibleFrameSpan));
        Assert.GreaterOrEqual(first, 0);
        Assert.Less(last, FrameCount);
    }

    [Test]
    public void VisibleBoundsSurviveAZeroWidthView()
    {
        // A pane can be laid out at zero width for a frame before it is measured.
        var axis = new ClipTimeAxis();
        axis.Prepare(new Rect(0f, 0f, 0f, 0f), FrameCount);

        Assert.AreEqual(0, axis.FirstVisibleFrame(FrameCount));
        Assert.Less(axis.LastVisibleFrame(FrameCount), FrameCount);
    }
}
}
