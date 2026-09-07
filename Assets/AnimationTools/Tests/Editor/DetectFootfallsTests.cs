using AnimationTools.Editor;
using NUnit.Framework;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// Which clips a bulk footfall detection touches, and what it says it did.
/// </summary>
/// <remarks>
/// The decision is the part worth pinning: anchors are hand-correctable and nothing records that a
/// clip was corrected, so a run that quietly redetects a clip holding anchors destroys work with no
/// way to notice. Selection gathering and the prompt itself are editor UI and are not covered here.
/// </remarks>
public class DetectFootfallsTests
{
    private AnnotatedAnimationClip _clip;

    [SetUp]
    public void SetUp() => _clip = ScriptableObject.CreateInstance<AnnotatedAnimationClip>();

    [TearDown]
    public void TearDown() => Object.DestroyImmediate(_clip);

    private GaitPhaseComponent AddPhase()
    {
        var phase = new GaitPhaseComponent();
        _clip.components.Add(phase);
        return phase;
    }

    [Test]
    public void AClipWithNoComponentIsACandidate()
    {
        Assert.IsTrue(DetectFootfallsMenu.ShouldDetect(_clip, overwriteExisting: false));
    }

    [Test]
    public void AComponentWithNoAnchorsIsACandidate()
    {
        AddPhase();

        Assert.IsTrue(DetectFootfallsMenu.ShouldDetect(_clip, overwriteExisting: false));
    }

    [Test]
    public void AComponentHoldingAnchorsIsLeftAloneUnlessOverwriting()
    {
        AddPhase().footfalls.Add(new GaitPhase.Footfall(10, GaitPhase.Foot.Right));

        Assert.IsFalse(DetectFootfallsMenu.ShouldDetect(_clip, overwriteExisting: false),
            "anchors may have been corrected by hand");
        Assert.IsTrue(DetectFootfallsMenu.ShouldDetect(_clip, overwriteExisting: true));
    }

    [Test]
    public void ANullClipIsNeverACandidate()
    {
        Assert.IsFalse(DetectFootfallsMenu.ShouldDetect(null, overwriteExisting: true));
    }

    [Test]
    public void ARunThatDidNothingSaysSo()
    {
        Assert.AreEqual("Nothing to do.", new DetectFootfallsMenu.Report().Summary());
    }

    [Test]
    public void TheSummaryNamesEveryOutcome()
    {
        var report = new DetectFootfallsMenu.Report
        {
            Detected = 24, Seeded = 3, Skipped = 70, Failed = 1
        };

        var summary = report.Summary();

        StringAssert.Contains("24 detected", summary);
        StringAssert.Contains("3 of those", summary);
        StringAssert.Contains("70 skipped", summary);
        StringAssert.Contains("1 failed", summary);
    }

    [Test]
    public void ACancelledRunSaysWhatItKept()
    {
        var report = new DetectFootfallsMenu.Report { Detected = 5, Cancelled = true };

        StringAssert.Contains("5 detected", report.Summary());
        StringAssert.Contains("Cancelled", report.Summary());
    }
}
}
