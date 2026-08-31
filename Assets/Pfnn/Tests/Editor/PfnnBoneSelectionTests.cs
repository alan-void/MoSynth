using AnimationTools;
using AnimationTools.Tests;
using NUnit.Framework;
using Pfnn.Editor;
using Unity.Mathematics;
using UnityEngine;

namespace Pfnn.Tests
{
/// <summary>
/// Choosing which bones the network predicts, on the config and against a checkpoint.
/// </summary>
/// <remarks>
/// The selection is the one thing that has to mean the same on both sides of the PythonNET
/// boundary. Python refuses a selection it cannot make sense of; these are the C# half — that the
/// asset never records a selection Python would refuse, and that a checkpoint trained against a
/// different rig is caught at load rather than run.
/// </remarks>
public class PfnnBoneSelectionTests
{
    private Skeleton _skeleton;
    private PfnnConfig _config;

    /// <summary>A rig with a hand, two finger bones under it, and a leaf tip — the shapes the
    /// default selection is about.</summary>
    [SetUp]
    public void SetUp()
    {
        _skeleton = TestSkeletons.Build(
            new TestSkeletons.BoneSpec("root", -1, float3.zero),
            new TestSkeletons.BoneSpec("Model:Spine", 0, new float3(0f, 1f, 0f)),
            new TestSkeletons.BoneSpec("Model:LeftArm", 1, new float3(-0.2f, 0.3f, 0f)),
            new TestSkeletons.BoneSpec("Model:LeftHand", 2, new float3(-0.4f, 0f, 0f)),
            new TestSkeletons.BoneSpec("Model:LeftHandIndex1", 3, new float3(-0.1f, 0f, 0f)),
            new TestSkeletons.BoneSpec("Model:LeftHandIndex2", 4, new float3(-0.05f, 0f, 0f)),
            new TestSkeletons.BoneSpec("Model:Head", 1, new float3(0f, 0.4f, 0f)),
            new TestSkeletons.BoneSpec("Model:Head_end", 6, new float3(0f, 0.2f, 0f)));

        _config = ScriptableObject.CreateInstance<PfnnConfig>();
    }

    [TearDown]
    public void TearDown()
    {
        Object.DestroyImmediate(_config);
        TestSkeletons.DestroyAll();
    }

    private int IndexOf(string boneName) => _skeleton.IndexOfName(boneName);

    // --- The config's own bookkeeping ------------------------------------------------------------

    [Test]
    public void EveryBoneIsPredictedByDefault()
    {
        for (var i = 0; i < _skeleton.BoneCount; i++)
        {
            Assert.That(_config.IsPredicted(_skeleton.GetBone(i).Name), Is.True);
        }
    }

    [Test]
    public void ExcludingThenIncludingLeavesNoRowBehind()
    {
        // The list is sparse on purpose: absent means predicted. A row saying "predicted" would be
        // a second way to express the default, and the two could disagree.
        _config.SetPredicted("Model:Head", false);
        _config.SetPredicted("Model:Head", true);

        Assert.That(_config.excludedBones, Is.Empty);
        Assert.That(_config.HasExcludedBones, Is.False);
    }

    [Test]
    public void ExcludingTheSameBoneTwiceDoesNotDuplicateIt()
    {
        _config.SetPredicted("Model:Head", false);
        _config.SetPredicted("Model:Head", false);

        Assert.That(_config.excludedBones.Count, Is.EqualTo(1));
    }

    // --- Subtrees ---------------------------------------------------------------------------------

    [Test]
    public void ASubtreeIsTheBoneAndEverythingBelowIt()
    {
        var subtree = PfnnBoneSelection.Subtree(_skeleton, IndexOf("Model:LeftHand"));

        CollectionAssert.AreEquivalent(
            new[] { IndexOf("Model:LeftHand"), IndexOf("Model:LeftHandIndex1"), IndexOf("Model:LeftHandIndex2") },
            subtree);
    }

    [Test]
    public void ASubtreeDoesNotReachASibling()
    {
        var subtree = PfnnBoneSelection.Subtree(_skeleton, IndexOf("Model:LeftArm"));

        Assert.That(subtree, Has.No.Member(IndexOf("Model:Head")));
    }

    // --- The default selection --------------------------------------------------------------------

    [Test]
    public void TheDefaultHoldsBackFingersAndLeavesAndNothingElse()
    {
        PfnnDefaultBoneSelection.Apply(_config, _skeleton);

        CollectionAssert.AreEquivalent(
            new[] { "Model:LeftHandIndex1", "Model:LeftHandIndex2", "Model:Head_end" },
            _config.excludedBones);
    }

    [Test]
    public void TheDefaultKeepsTheHandItself()
    {
        // It is where the arm ends. Dropping it would take the wrist out of the model.
        PfnnDefaultBoneSelection.Apply(_config, _skeleton);

        Assert.That(_config.IsPredicted("Model:LeftHand"), Is.True);
    }

    [Test]
    public void TheDefaultLeavesASelectionPythonWillAccept()
    {
        // Closed under parent: every kept bone's parent is kept. select_bones refuses anything else,
        // because a rotation needs its parent's frame to be applied in.
        PfnnDefaultBoneSelection.Apply(_config, _skeleton);

        for (var i = 1; i < _skeleton.BoneCount; i++)
        {
            if (!_config.IsPredicted(_skeleton.GetBone(i).Name)) continue;

            var parent = _skeleton.GetBone(_skeleton.GetParentIndex(i)).Name;
            Assert.That(_config.IsPredicted(parent), Is.True,
                $"{_skeleton.GetBone(i).Name} is kept but its parent {parent} is not");
        }
    }

    [Test]
    public void ApplyingTheDefaultTwiceIsTheSameAsApplyingItOnce()
    {
        PfnnDefaultBoneSelection.Apply(_config, _skeleton);
        var first = _config.excludedBones.ToArray();

        PfnnDefaultBoneSelection.Apply(_config, _skeleton);

        CollectionAssert.AreEqual(first, _config.excludedBones);
    }

    // --- Binding a checkpoint to a rig ------------------------------------------------------------

    [Test]
    public void CheckpointBonesResolveToSkeletonIndices()
    {
        var names = new[] { "root", "Model:Spine", "Model:Head" };

        Assert.That(PfnnBoneSelection.TryResolve(names, _skeleton, out var indices, out var error),
            Is.True, error);
        CollectionAssert.AreEqual(
            new[] { IndexOf("root"), IndexOf("Model:Spine"), IndexOf("Model:Head") }, indices);
    }

    [Test]
    public void ACheckpointNamingABoneThisRigLacksIsRefused()
    {
        var names = new[] { "root", "Model:Tail" };

        Assert.That(PfnnBoneSelection.TryResolve(names, _skeleton, out _, out var error), Is.False);
        Assert.That(error, Does.Contain("Model:Tail"));
    }

    [Test]
    public void ACheckpointWhoseBonesAreOutOfOrderIsRefused()
    {
        // Same names, different hierarchy. Nothing about the names alone would catch this, and a
        // model fed a reordered pose produces bad motion rather than an error.
        var names = new[] { "Model:Head", "Model:Spine" };

        Assert.That(PfnnBoneSelection.TryResolve(names, _skeleton, out _, out var error), Is.False);
        Assert.That(error, Does.Contain("different skeleton"));
    }
}
}
