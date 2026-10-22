using NUnit.Framework;
using Style.Editor;

namespace Style.Tests
{
/// <summary>
/// Reading 100STYLE's clip names and frame cuts.
/// </summary>
public class HundredStyleClipSetupTests
{
    private static readonly string[] Styles = { "Neutral", "Old", "WildArms" };

    [TestCase("Neutral_FW", "Neutral", "FW")]
    [TestCase("WildArms_TR2", "WildArms", "TR2")]
    [TestCase("Armature_Old_id", "Old", "ID")]
    [TestCase("ybot_hundredstyle_Old_SR", "Old", "SR")]
    [TestCase("neutral_BW", "Neutral", "BW")]
    public void ClipNames_Parse(string clipName, string style, string gait)
    {
        Assert.IsTrue(HundredStyleClipSetup.TryParseClipName(clipName, Styles, out var parsedStyle,
            out var parsedGait));
        Assert.AreEqual(style, parsedStyle);
        Assert.AreEqual(gait, parsedGait);
    }

    [TestCase("Neutral")]
    [TestCase("Neutral_XX")]
    [TestCase("Unknown_FW")]
    [TestCase("Neutral_FW_mirror")]
    [TestCase("")]
    public void ClipNames_ThatDoNotFit_AreRefused(string clipName)
    {
        Assert.IsFalse(HundredStyleClipSetup.TryParseClipName(clipName, Styles, out _, out _));
    }

    [Test]
    public void FrameCuts_SkipNotApplicable()
    {
        var cuts = HundredStyleClipSetup.ParseFrameCuts(new[]
        {
            "STYLE_NAME,FW_START,FW_STOP,TR3_START,TR3_STOP",
            "Old,351,7411,N/A,N/A",
        });

        Assert.IsTrue(cuts["Old"].TryGetValue("FW", out var cut));
        Assert.AreEqual((351, 7411), cut);
        Assert.IsFalse(cuts["Old"].ContainsKey("TR3"));
    }

    [Test]
    public void Descriptions_KeepCommasInsideQuotes()
    {
        var descriptions = HundredStyleClipSetup.ParseDescriptions(new[]
        {
            "Style Name,Description,Stochastic,Symmetric,Notes",
            "Akimbo,Hands on hips,No,Yes,",
            "Old,\"Hunched, slow\",No,Yes,",
        });

        Assert.AreEqual("Hands on hips", descriptions["Akimbo"]);
        Assert.AreEqual("Hunched, slow", descriptions["Old"]);
    }

    [Test]
    public void Slice_IsScaledFromTheSourceRate()
    {
        Assert.IsTrue(HundredStyleClipSetup.TryScaleSlice(300, 6000, 30f, 4000, out var start, out var end));
        Assert.AreEqual(150, start);
        Assert.AreEqual(3000, end);
    }

    [Test]
    public void Slice_IsClampedToTheClip_AndRefusedWhenNothingIsLeft()
    {
        Assert.IsTrue(HundredStyleClipSetup.TryScaleSlice(300, 6000, 60f, 4000, out _, out var end));
        Assert.AreEqual(4000, end);

        Assert.IsFalse(HundredStyleClipSetup.TryScaleSlice(9000, 9500, 60f, 4000, out _, out _));
    }

    [Test]
    public void SliceEnd_IsCappedFromItsStart_AndLeftAloneWhenShorter()
    {
        Assert.AreEqual(1500, HundredStyleClipSetup.CapSliceEnd(300, 6000, 1200));
        Assert.AreEqual(900, HundredStyleClipSetup.CapSliceEnd(300, 900, 1200));
    }
}
}
