using System.Collections.Generic;
using AnimationTools;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// Coverage for <see cref="AnimationClipBaker.TryValidateClip"/>, which answers whether a clip will
/// actually drive a skeleton's bones. Nothing else in the pipeline can tell: the bake hands the clip
/// to an instantiated rig and lets SampleAnimation match curve paths, and unmatched paths are
/// silently ignored rather than reported.
/// </summary>
public class AnimationClipBakerTests
{
    private readonly List<Object> _created = new();

    [TearDown]
    public void TearDown()
    {
        foreach (var created in _created)
        {
            if (created != null) Object.DestroyImmediate(created);
        }

        _created.Clear();
        TestSkeletons.DestroyAll();
    }

    /// <summary>A clip whose only curves are Transform rotations at the given paths.</summary>
    private AnimationClip ClipAnimating(params string[] paths)
    {
        var clip = new AnimationClip { name = "test clip" };
        _created.Add(clip);

        var curve = AnimationCurve.Linear(0f, 0f, 1f, 1f);
        foreach (var path in paths)
        {
            AnimationUtility.SetEditorCurve(clip,
                EditorCurveBinding.FloatCurve(path, typeof(Transform), "m_LocalRotation.x"), curve);
        }

        return clip;
    }

    [Test]
    public void TryValidateClip_ClipAnimatesTheSkeletonsBones_Accepts()
    {
        var skeleton = TestSkeletons.CreateChain3();
        var clip = ClipAnimating("", "spine", "spine/head");

        Assert.IsTrue(AnimationClipBaker.TryValidateClip(clip, skeleton, out var error), error);
        Assert.IsNull(error);
    }

    /// <summary>
    /// The regression guard against over-strictness. Rigs routinely leave leaf bones uncurved — the
    /// project's own clips animate 72 of 85 bones, the rest being FBX "_end" markers — so demanding
    /// full coverage would reject every asset in this repository.
    /// </summary>
    [Test]
    public void TryValidateClip_ClipAnimatesOnlySomeBones_Accepts()
    {
        var skeleton = TestSkeletons.CreateChain3();
        var clip = ClipAnimating("spine");

        Assert.IsTrue(AnimationClipBaker.TryValidateClip(clip, skeleton, out var error), error);
    }

    [Test]
    public void TryValidateClip_ClipBelongsToAnotherRig_RejectsAndNamesTheClip()
    {
        var skeleton = TestSkeletons.CreateChain3();
        var clip = ClipAnimating("hips", "hips/thigh");

        Assert.IsFalse(AnimationClipBaker.TryValidateClip(clip, skeleton, out var error));
        StringAssert.Contains("test clip", error);
        StringAssert.Contains("root", error);
    }

    /// <summary>
    /// An importer writes curve paths relative to the asset's main object, not to whichever bone the
    /// skeleton happens to start at, and the bake instantiates that main object. So a skeleton
    /// rooted part-way down a rig must match the full path from the top.
    /// </summary>
    [Test]
    public void TryValidateClip_SkeletonRootedBelowTheRig_MatchesRigRelativePaths()
    {
        var skeleton = TestSkeletons.CreateChain3();
        var rig = new GameObject("rig");
        _created.Add(rig);
        skeleton.Root.SetParent(rig.transform, false);

        Assert.IsTrue(
            AnimationClipBaker.TryValidateClip(ClipAnimating("root/spine"), skeleton, out var error), error);
    }

    /// <summary>The other half of the convention: skeleton-root-relative paths are not the form
    /// SampleAnimation matches, so they must not be mistaken for a compatible clip.</summary>
    [Test]
    public void TryValidateClip_SkeletonRootedBelowTheRig_RejectsSkeletonRelativePaths()
    {
        var skeleton = TestSkeletons.CreateChain3();
        var rig = new GameObject("rig");
        _created.Add(rig);
        skeleton.Root.SetParent(rig.transform, false);

        Assert.IsFalse(AnimationClipBaker.TryValidateClip(ClipAnimating("spine"), skeleton, out _));
    }

    /// <summary>
    /// The divergence that produced the original console spam: a skeleton whose cached bone list no
    /// longer describes the rig underneath it. The bake bails with a bare null here, so this check
    /// is the only thing that can say why.
    /// </summary>
    [Test]
    public void TryValidateClip_RigChangedSinceTheSkeletonWasBuilt_ReportsBothCounts()
    {
        var skeleton = TestSkeletons.CreateChain3();
        Assert.AreEqual(3, skeleton.BoneCount, "the cached bone list has to exist before the rig moves");

        var extraBone = new GameObject("jaw").transform;
        extraBone.SetParent(skeleton.Root, false);

        Assert.IsFalse(AnimationClipBaker.TryValidateClip(ClipAnimating(""), skeleton, out var error));
        StringAssert.Contains("4", error);
        StringAssert.Contains("3", error);
    }

    /// <summary>Unset clip or skeleton is someone else's message to report — see
    /// <see cref="SkeletonAnimation.TryValidate"/>, whose earlier checks own those cases.</summary>
    [Test]
    public void TryValidateClip_NothingToCheck_Accepts()
    {
        Assert.IsTrue(AnimationClipBaker.TryValidateClip(null, TestSkeletons.CreateChain3(), out _));
        Assert.IsTrue(AnimationClipBaker.TryValidateClip(ClipAnimating(""), new Skeleton(), out _));
    }
}
}
