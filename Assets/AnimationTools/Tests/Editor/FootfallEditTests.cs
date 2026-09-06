using System.Collections.Generic;
using AnimationTools.Editor;
using NUnit.Framework;

namespace AnimationTools.Tests
{
/// <summary>
/// Picking and editing footfall anchors on the clip editor's timeline.
/// </summary>
/// <remarks>
/// Both failure modes here are silent. <c>GaitPhase</c> drops any anchor that is not strictly later
/// than the one before it, and <see cref="GaitPhaseComponent.OnValidate"/> drops any anchor outside
/// the slice, neither with a message - so an unsorted or out-of-range result loses data that nothing
/// reports.
/// </remarks>
public class FootfallEditTests
{
    private const int SliceFrameCount = 100;

    private static List<GaitPhase.Footfall> Footfalls(params (int frame, GaitPhase.Foot foot)[] entries)
    {
        var footfalls = new List<GaitPhase.Footfall>();
        foreach (var (frame, foot) in entries) footfalls.Add(new GaitPhase.Footfall(frame, foot));
        return footfalls;
    }

    private static HashSet<int> Selection(params int[] indices) => new(indices);

    [Test]
    public void MovePreservesRelativeOffsets()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Left),
            (30, GaitPhase.Foot.Right));

        var moved = FootfallEdits.Move(footfalls, Selection(0, 2), 5, SliceFrameCount);

        Assert.AreEqual(15, moved[0].frame);
        Assert.AreEqual(20, moved[1].frame);
        Assert.AreEqual(35, moved[2].frame);
    }

    [Test]
    public void MoveStopsAtTheSliceEdgesInsteadOfPushingAnchorsOff()
    {
        var footfalls = Footfalls((2, GaitPhase.Foot.Right), (95, GaitPhase.Foot.Left));

        var left = FootfallEdits.Move(footfalls, Selection(0, 1), -50, SliceFrameCount);
        Assert.AreEqual(0, left[0].frame);
        Assert.AreEqual(93, left[1].frame);

        var right = FootfallEdits.Move(footfalls, Selection(0, 1), 50, SliceFrameCount);
        Assert.AreEqual(6, right[0].frame);
        Assert.AreEqual(99, right[1].frame);
    }

    [Test]
    public void MoveNeverLosesAnAnchor()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Left),
            (30, GaitPhase.Foot.Right));

        var moved = FootfallEdits.Move(footfalls, Selection(0), 25, SliceFrameCount);

        Assert.AreEqual(3, moved.Count);
    }

    [Test]
    public void MoveReturnsAnAscendingList()
    {
        // Dragging the first anchor past the second must reorder, not leave a descending pair that
        // the phase evaluation would silently drop.
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Left));

        var moved = FootfallEdits.Move(footfalls, Selection(0), 25, SliceFrameCount);

        Assert.AreEqual(20, moved[0].frame);
        Assert.AreEqual(35, moved[1].frame);
    }

    [Test]
    public void ClampDeltaIsZeroForAnEmptySelection()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right));
        Assert.AreEqual(0, FootfallEdits.ClampDelta(footfalls, Selection(), 12, SliceFrameCount));
    }

    [Test]
    public void AddChoosesTheFootOppositeTheAnchorBefore()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Left));

        var added = FootfallEdits.Add(footfalls, 25, SliceFrameCount);

        Assert.AreEqual(3, added.Count);
        Assert.AreEqual(25, added[2].frame);
        Assert.AreEqual(GaitPhase.Foot.Right, added[2].foot);
    }

    [Test]
    public void AddBeforeEveryAnchorDefaultsToLeft()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right));

        var added = FootfallEdits.Add(footfalls, 2, SliceFrameCount);

        Assert.AreEqual(2, added[0].frame);
        Assert.AreEqual(GaitPhase.Foot.Left, added[0].foot);
    }

    [Test]
    public void AddClampsIntoTheSlice()
    {
        var added = FootfallEdits.Add(Footfalls(), 5000, SliceFrameCount);
        Assert.AreEqual(SliceFrameCount - 1, added[0].frame);
    }

    [Test]
    public void DeleteRemovesExactlyTheSelection()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Left),
            (30, GaitPhase.Foot.Right));

        var remaining = FootfallEdits.Delete(footfalls, Selection(1));

        Assert.AreEqual(2, remaining.Count);
        Assert.AreEqual(10, remaining[0].frame);
        Assert.AreEqual(30, remaining[1].frame);
    }

    [Test]
    public void RepeatedFeetFindsEveryMissedContact()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Right),
            (30, GaitPhase.Foot.Left), (40, GaitPhase.Foot.Left));

        CollectionAssert.AreEqual(new[] { 1, 3 }, FootfallEdits.RepeatedFeet(footfalls));
    }

    [Test]
    public void PickTakesTheNearestAnchorInsideTheTolerance()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (40, GaitPhase.Foot.Left));

        // The anchors sit at x = 50 and x = 200, and the tolerance is 6px either side.
        Assert.AreEqual(0, FootfallHitTester.Pick(footfalls, 52f, FrameToX));
        Assert.AreEqual(1, FootfallHitTester.Pick(footfalls, 197f, FrameToX));
    }

    [Test]
    public void PickReturnsNothingBeyondTheTolerance()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right));
        Assert.AreEqual(-1, FootfallHitTester.Pick(footfalls, 200f, FrameToX));
    }

    [Test]
    public void PickIsStableWhenTwoAnchorsShareAPixel()
    {
        // Zoomed far out, adjacent frames land on the same column; the earlier anchor wins so the
        // choice does not flicker as the cursor moves within that pixel.
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (11, GaitPhase.Foot.Left));

        Assert.AreEqual(0, FootfallHitTester.Pick(footfalls, 1.02f, frame => frame * 0.1f));
    }

    [Test]
    public void PickRangeCollectsEveryAnchorInsideTheBand()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Left),
            (30, GaitPhase.Foot.Right));

        var picked = new List<int>();
        FootfallHitTester.PickRange(footfalls, 75f, 125f, FrameToX, picked);

        CollectionAssert.AreEqual(new[] { 1 }, picked);
    }

    [Test]
    public void PickRangeAcceptsABandDraggedRightToLeft()
    {
        var footfalls = Footfalls((10, GaitPhase.Foot.Right), (20, GaitPhase.Foot.Left));

        var picked = new List<int>();
        FootfallHitTester.PickRange(footfalls, 125f, 25f, FrameToX, picked);

        CollectionAssert.AreEqual(new[] { 0, 1 }, picked);
    }

    private static float FrameToX(int sliceFrame) => sliceFrame * 5f;
}
}
