using AnimationTools;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Tests
{
public class SkeletonTests
{
    private const float Tolerance = 1e-4f;

    [TearDown]
    public void TearDown()
    {
        TestSkeletons.DestroyAll();
    }

    // --- Construction and ordering ---------------------------------------------------------

    [Test]
    public void Unset_ReportsNoBones()
    {
        var skeleton = new Skeleton();

        Assert.IsFalse(skeleton.IsSet);
        Assert.AreEqual(0, skeleton.BoneCount);
        Assert.AreEqual(0, skeleton.ContentHash);
        Assert.AreEqual(-1, skeleton.IndexOf((Transform)null));
    }

    [Test]
    public void BoneOrder_IsPreorderDepthFirstFromRoot()
    {
        var skeleton = TestSkeletons.CreateBranch4();

        Assert.AreEqual(4, skeleton.BoneCount);
        Assert.AreEqual("root", skeleton.GetBone(0).Name);
        Assert.AreEqual("spine", skeleton.GetBone(1).Name);
        Assert.AreEqual("leftHand", skeleton.GetBone(2).Name);
        Assert.AreEqual("rightHand", skeleton.GetBone(3).Name);
    }

    [Test]
    public void RootIsBoneZero_AndHasNoParent()
    {
        var skeleton = TestSkeletons.CreateChain3();

        Assert.AreSame(skeleton.Root, skeleton.GetBone(0).Transform);
        Assert.AreEqual(-1, skeleton.GetParentIndex(0));
        Assert.IsNull(skeleton.GetParent(0));
    }

    [Test]
    public void EveryBonesParentPrecedesIt()
    {
        var skeleton = TestSkeletons.CreateBranch4();

        for (var i = 1; i < skeleton.BoneCount; i++)
        {
            Assert.Less(skeleton.GetParentIndex(i), i, $"bone {i} violates the depth-first invariant");
        }
    }

    [Test]
    public void ParentIndices_MatchTheHierarchy()
    {
        var skeleton = TestSkeletons.CreateBranch4();

        Assert.AreEqual(0, skeleton.GetParentIndex(1)); // spine  -> root
        Assert.AreEqual(1, skeleton.GetParentIndex(2)); // leftHand  -> spine
        Assert.AreEqual(1, skeleton.GetParentIndex(3)); // rightHand -> spine
        Assert.AreEqual("spine", skeleton.GetParent(2).Name);
    }

    [Test]
    public void RestPose_ReadsTheTransformOffsets()
    {
        var skeleton = TestSkeletons.CreateChain3();

        Assert.AreEqual(0f, skeleton.GetBone(0).RestLocalPosition.y, Tolerance);
        Assert.AreEqual(1f, skeleton.GetBone(1).RestLocalPosition.y, Tolerance);
        Assert.AreEqual(1f, skeleton.GetBone(2).RestLocalPosition.y, Tolerance);
    }

    // --- Lookup ------------------------------------------------------------------------------

    [Test]
    public void TryFindByName_FindsAnExistingBone()
    {
        var skeleton = TestSkeletons.CreateChain3();

        Assert.IsTrue(skeleton.TryFindByName("spine", out var index));
        Assert.AreEqual(1, index);
        Assert.AreEqual(1, skeleton.IndexOfName("spine"));
    }

    [Test]
    public void TryFindByName_MissesAnAbsentBone()
    {
        var skeleton = TestSkeletons.CreateChain3();

        Assert.IsFalse(skeleton.TryFindByName("tail", out var index));
        Assert.AreEqual(-1, index);
        Assert.AreEqual(-1, skeleton.IndexOfName("tail"));
    }

    [Test]
    public void IndexOfTransform_ResolvesBonesAndRejectsOutsiders()
    {
        var skeleton = TestSkeletons.CreateChain3();
        var outsider = new GameObject("outsider").transform;

        try
        {
            Assert.AreEqual(2, skeleton.IndexOf(skeleton.GetBone(2).Transform));
            Assert.AreEqual(-1, skeleton.IndexOf(outsider));
        }
        finally
        {
            Object.DestroyImmediate(outsider.gameObject);
        }
    }

    // --- Bone ids ----------------------------------------------------------------------------

    [Test]
    public void IndexOfId_TreatsZeroAsUnset()
    {
        var skeleton = TestSkeletons.CreateChain3();

        Assert.AreEqual(-1, skeleton.IndexOfId(0));
    }

    [Test]
    public void IndexOfId_RejectsAnIdPastTheEnd()
    {
        var skeleton = TestSkeletons.CreateChain3();

        Assert.AreEqual(-1, skeleton.IndexOfId(skeleton.BoneCount + 1));
    }

    [Test]
    public void IndexOfId_RoundTripsWithGetBoneId()
    {
        var skeleton = TestSkeletons.CreateChain3();

        for (var i = 0; i < skeleton.BoneCount; i++)
        {
            Assert.AreEqual(i, skeleton.IndexOfId(skeleton.GetBoneId(i)));
        }

        Assert.AreEqual(0, skeleton.IndexOfId(1));
    }

    // --- SkeletonData ------------------------------------------------------------------------

    [Test]
    public void GetSkeletonData_MirrorsTheHierarchyAndRestPose()
    {
        var skeleton = TestSkeletons.CreateBranch4();
        var data = skeleton.GetSkeletonData();

        Assert.AreEqual(skeleton.BoneCount, data.BoneCount);
        for (var i = 0; i < skeleton.BoneCount; i++)
        {
            Assert.AreEqual(skeleton.GetParentIndex(i), data.ParentIndices[i]);
            Assert.AreEqual(skeleton.GetBone(i).RestLocalPosition.y, data.RestLocalPositions[i].y, Tolerance);
        }
    }

    [Test]
    public void GetSkeletonData_IsSharedAcrossInstancesOverTheSameRoot()
    {
        var skeleton = TestSkeletons.CreateBranch4();
        var alias = new Skeleton(skeleton.Root);

        var data = skeleton.GetSkeletonData();
        var aliasData = alias.GetSkeletonData();

        Assert.AreEqual(data.BoneCount, aliasData.BoneCount);
        Assert.IsTrue(data.ParentIndices.Equals(aliasData.ParentIndices),
            "instances over the same root must share one Domain-allocated SkeletonData");
    }

    // --- Structural comparison ----------------------------------------------------------------

    [Test]
    public void ContentHash_AgreesForIdenticalStructures()
    {
        var a = TestSkeletons.CreateChain3();
        var b = TestSkeletons.CreateChain3();

        Assert.AreEqual(a.ContentHash, b.ContentHash);
    }

    [Test]
    public void ContentHash_DiffersForDifferentStructures()
    {
        var chain = TestSkeletons.CreateChain3();
        var branch = TestSkeletons.CreateBranch4();

        Assert.AreNotEqual(chain.ContentHash, branch.ContentHash);
    }

    [Test]
    public void StructurallyEqual_IgnoresRestPose()
    {
        var a = TestSkeletons.CreateChain3();
        var b = TestSkeletons.Build(
            new TestSkeletons.BoneSpec("root", -1, float3.zero),
            new TestSkeletons.BoneSpec("spine", 0, new float3(0f, 7f, 0f)),
            new TestSkeletons.BoneSpec("head", 1, new float3(0f, 9f, 0f)));

        Assert.IsTrue(Skeleton.StructurallyEqual(a, b));
    }

    [Test]
    public void StructurallyEqual_IsFalseOnDifferentNames()
    {
        var a = TestSkeletons.CreateChain3();
        var b = TestSkeletons.Build(
            new TestSkeletons.BoneSpec("root", -1, float3.zero),
            new TestSkeletons.BoneSpec("neck", 0, new float3(0f, 1f, 0f)),
            new TestSkeletons.BoneSpec("head", 1, new float3(0f, 1f, 0f)));

        Assert.IsFalse(Skeleton.StructurallyEqual(a, b));
    }

    [Test]
    public void StructurallyEqual_IsFalseOnDifferentParents()
    {
        var chain = TestSkeletons.CreateChain3();
        var flat = TestSkeletons.Build(
            new TestSkeletons.BoneSpec("root", -1, float3.zero),
            new TestSkeletons.BoneSpec("spine", 0, new float3(0f, 1f, 0f)),
            new TestSkeletons.BoneSpec("head", 0, new float3(0f, 1f, 0f)));

        Assert.IsFalse(Skeleton.StructurallyEqual(chain, flat));
    }

    [Test]
    public void StructurallyEqual_IsNullSafe()
    {
        var skeleton = TestSkeletons.CreateChain3();

        Assert.IsTrue(Skeleton.StructurallyEqual(null, null));
        Assert.IsTrue(Skeleton.StructurallyEqual(skeleton, skeleton));
        Assert.IsFalse(Skeleton.StructurallyEqual(skeleton, null));
        Assert.IsFalse(Skeleton.StructurallyEqual(null, skeleton));
    }

    // --- MatchesFrom: a skeleton embedded under an extra ancestor -------------------------------

    [Test]
    public void MatchesFrom_AcceptsASkeletonNestedUnderAnExtraRoot()
    {
        // The same three bones, carrying one extra ancestor above them.
        var nested = TestSkeletons.Build(
            new TestSkeletons.BoneSpec("armature", -1, float3.zero),
            new TestSkeletons.BoneSpec("root", 0, float3.zero),
            new TestSkeletons.BoneSpec("spine", 1, new float3(0f, 1f, 0f)),
            new TestSkeletons.BoneSpec("head", 2, new float3(0f, 1f, 0f)));
        var bare = TestSkeletons.CreateChain3();

        Assert.IsTrue(nested.MatchesFrom(1, bare));
    }

    [Test]
    public void MatchesFrom_RejectsAMismatchedBoneCount()
    {
        var nested = TestSkeletons.Build(
            new TestSkeletons.BoneSpec("armature", -1, float3.zero),
            new TestSkeletons.BoneSpec("root", 0, float3.zero),
            new TestSkeletons.BoneSpec("spine", 1, new float3(0f, 1f, 0f)));
        var bare = TestSkeletons.CreateChain3();

        Assert.IsFalse(nested.MatchesFrom(1, bare));
    }

    [Test]
    public void MatchesFrom_RejectsADifferentlyNamedBone()
    {
        var nested = TestSkeletons.Build(
            new TestSkeletons.BoneSpec("armature", -1, float3.zero),
            new TestSkeletons.BoneSpec("root", 0, float3.zero),
            new TestSkeletons.BoneSpec("neck", 1, new float3(0f, 1f, 0f)),
            new TestSkeletons.BoneSpec("head", 2, new float3(0f, 1f, 0f)));
        var bare = TestSkeletons.CreateChain3();

        Assert.IsFalse(nested.MatchesFrom(1, bare));
    }

    [Test]
    public void MatchesFrom_RejectsAReparentedBone()
    {
        var nested = TestSkeletons.Build(
            new TestSkeletons.BoneSpec("armature", -1, float3.zero),
            new TestSkeletons.BoneSpec("root", 0, float3.zero),
            new TestSkeletons.BoneSpec("spine", 1, new float3(0f, 1f, 0f)),
            new TestSkeletons.BoneSpec("head", 1, new float3(0f, 1f, 0f)));
        var bare = TestSkeletons.CreateChain3();

        Assert.IsFalse(nested.MatchesFrom(1, bare));
    }

    [Test]
    public void MatchesFrom_ZeroIsPlainStructuralEquality()
    {
        var a = TestSkeletons.CreateChain3();
        var b = TestSkeletons.CreateChain3();

        Assert.IsTrue(a.MatchesFrom(0, b));
        Assert.IsFalse(a.MatchesFrom(0, TestSkeletons.CreateBranch4()));
    }

    // --- SkeletonBone as a standalone reference ------------------------------------------------

    [Test]
    public void SkeletonBone_ResolvesItsOwnIndexAndParent()
    {
        var skeleton = TestSkeletons.CreateChain3();
        var reference = new SkeletonBone(skeleton, skeleton.GetBone(2).Transform);

        Assert.AreEqual(2, reference.Index);
        Assert.AreEqual(1, reference.ParentIndex);
        Assert.AreEqual("spine", reference.Parent.Name);
        Assert.AreEqual("head", reference.Name);
        Assert.IsTrue(reference.IsSet);
    }

    [Test]
    public void SkeletonBone_Unset_ResolvesToNothing()
    {
        var reference = new SkeletonBone();

        Assert.IsFalse(reference.IsSet);
        Assert.AreEqual(-1, reference.Index);
        Assert.AreEqual(-1, reference.ParentIndex);
        Assert.IsNull(reference.Parent);
        Assert.IsNull(reference.Name);
    }

    [Test]
    public void SkeletonBone_ResolveIndex_FallsBackToNameAcrossRigs()
    {
        var picked = TestSkeletons.CreateChain3();
        var otherRig = TestSkeletons.CreateChain3();
        var reference = new SkeletonBone(picked, picked.GetBone(1).Transform);

        // A different rig shares no Transforms, so only the name match can resolve it.
        Assert.AreEqual(-1, otherRig.IndexOf(reference.Transform));
        Assert.AreEqual(1, reference.ResolveIndex(otherRig));
    }

    [Test]
    public void SkeletonBone_ResolveIndex_ReturnsMinusOneWhenNoBoneMatches()
    {
        var picked = TestSkeletons.CreateChain3();
        var branch = TestSkeletons.CreateBranch4();
        var reference = new SkeletonBone(picked, picked.GetBone(2).Transform); // "head"

        Assert.AreEqual(-1, reference.ResolveIndex(branch));
        Assert.AreEqual(-1, reference.ResolveIndex(null));
    }
}
}
