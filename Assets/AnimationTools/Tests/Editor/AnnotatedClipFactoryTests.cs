using AnimationTools.Editor;
using NUnit.Framework;

namespace AnimationTools.Tests
{
/// <summary>
/// The pure halves of the annotated-clip factory: which takes a filter selects, and where each one's
/// asset lands.
/// </summary>
/// <remarks>
/// Verifies a clip maps to the same path every time, so a batch re-run updates assets in place
/// rather than minting new ones that orphan the configs' GUID references.
/// </remarks>
public class AnnotatedClipFactoryTests
{
    private static readonly string[] Takes =
    {
        "dataset-2_walk_normal_001",
        "dataset-2_walk_normal_002",
        "dataset-2_walk_active_001",
        "dataset-2_run_normal_001",
    };

    [Test]
    public void EmptyFilterSelectsEverything()
    {
        Assert.AreEqual(Takes.Length, AnnotatedClipFactory.SelectClipNames(Takes, "").Count);
        Assert.AreEqual(Takes.Length, AnnotatedClipFactory.SelectClipNames(Takes, null).Count);
        Assert.AreEqual(Takes.Length, AnnotatedClipFactory.SelectClipNames(Takes, "   ").Count);
    }

    [Test]
    public void FilterSelectsMatchingNamesInOrder()
    {
        var selected = AnnotatedClipFactory.SelectClipNames(Takes, "walk_normal");

        Assert.AreEqual(2, selected.Count);
        Assert.AreEqual("dataset-2_walk_normal_001", selected[0]);
        Assert.AreEqual("dataset-2_walk_normal_002", selected[1]);
    }

    [Test]
    public void FilterIsCaseInsensitiveAndTrimmed()
    {
        Assert.AreEqual(2, AnnotatedClipFactory.SelectClipNames(Takes, "WALK_Normal").Count);
        Assert.AreEqual(2, AnnotatedClipFactory.SelectClipNames(Takes, "  walk_normal  ").Count);
    }

    [Test]
    public void NoMatchSelectsNothing()
    {
        Assert.IsEmpty(AnnotatedClipFactory.SelectClipNames(Takes, "jump"));
    }

    [Test]
    public void AssetPathIsForwardSlashedUnderTheFolder()
    {
        Assert.AreEqual("Assets/Animation/Pfnn/Bandai/take_001.asset",
            AnnotatedClipFactory.AssetPathFor("Assets/Animation/Pfnn/Bandai", "take_001"));
    }

    [Test]
    public void AssetPathNormalizesTheFolder()
    {
        const string expected = "Assets/Animation/Bandai/take_001.asset";

        Assert.AreEqual(expected, AnnotatedClipFactory.AssetPathFor("Assets/Animation/Bandai/", "take_001"));
        Assert.AreEqual(expected, AnnotatedClipFactory.AssetPathFor(@"Assets\Animation\Bandai", "take_001"));
    }

    [Test]
    public void AssetPathReplacesCharactersAFileNameCannotHold()
    {
        Assert.AreEqual("Assets/A/take_1_2.asset", AnnotatedClipFactory.AssetPathFor("Assets/A", "take:1|2"));
    }

    [Test]
    public void AssetPathIsStableForTheSameClip()
    {
        var first = AnnotatedClipFactory.AssetPathFor("Assets/A", "dataset-2_walk_normal_001");
        var second = AnnotatedClipFactory.AssetPathFor("Assets/A", "dataset-2_walk_normal_001");

        Assert.AreEqual(first, second);
    }
}
}
