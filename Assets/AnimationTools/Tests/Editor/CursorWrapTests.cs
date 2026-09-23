using AnimationTools.Editor;
using NUnit.Framework;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// Where a panned cursor is sent when it leaves the panel it is panning.
/// </summary>
/// <remarks>
/// Verifies the jump is exactly one panel across, since the drag subtracts the same figure from the
/// next event's delta. Moving the OS cursor itself is a native call and is not tested.
/// </remarks>
public class CursorWrapTests
{
    private static readonly Rect Panel = new(100f, 40f, 300f, 120f);

    [Test]
    public void ACursorInsideThePanelIsLeftAlone()
    {
        Assert.AreEqual(Vector2.zero, CursorWrap.Offset(Panel, new Vector2(250f, 100f)));
        Assert.AreEqual(Vector2.zero, CursorWrap.Offset(Panel, Panel.min));
        Assert.AreEqual(Vector2.zero, CursorWrap.Offset(Panel, Panel.max));
    }

    [Test]
    public void LeavingOneEdgeSendsItExactlyOnePanelToTheOther()
    {
        Assert.AreEqual(new Vector2(-Panel.width, 0f),
            CursorWrap.Offset(Panel, new Vector2(Panel.xMax + 3f, 100f)));

        Assert.AreEqual(new Vector2(Panel.width, 0f),
            CursorWrap.Offset(Panel, new Vector2(Panel.xMin - 3f, 100f)));

        Assert.AreEqual(new Vector2(0f, -Panel.height),
            CursorWrap.Offset(Panel, new Vector2(250f, Panel.yMax + 3f)));

        Assert.AreEqual(new Vector2(0f, Panel.height),
            CursorWrap.Offset(Panel, new Vector2(250f, Panel.yMin - 3f)));
    }

    [Test]
    public void LeavingByACornerWrapsBothAxes()
    {
        Assert.AreEqual(new Vector2(-Panel.width, Panel.height),
            CursorWrap.Offset(Panel, new Vector2(Panel.xMax + 8f, Panel.yMin - 8f)));
    }

    [Test]
    public void TheWrapLandsBackInsideThePanel()
    {
        // A jump of exactly the panel's width from just outside one edge lands just inside the
        // other, which is why one wrap is always enough.
        var outside = new Vector2(Panel.xMax + 2f, Panel.yMin - 2f);
        var landed = outside + CursorWrap.Offset(Panel, outside);

        Assert.IsTrue(Panel.Contains(landed), $"{landed} should be inside {Panel}");
    }
}
}
