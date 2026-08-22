using System.Collections.Generic;
using AnimationTools;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace MotionMatching.Tests
{
/// <summary>
/// Shared fixtures for the MotionMatching-coupled edit-mode test suite: a small synthetic
/// MM skeleton/pose pair for fast deterministic tests, plus an opt-in loader for the real
/// demo pose database so a handful of tests can also run against production data.
/// </summary>
static class MmTestData
{
    private static readonly List<GameObject> _created = new();

    /// <summary>SimulationBone(0) -&gt; Hips(1) -&gt; {Spine(2), LeftFoot(3) -&gt; LeftToe(4)}.</summary>
    /// <remarks>A skeleton is a Transform tree, so this creates real GameObjects; suites that
    /// call it must call <see cref="DestroyAll"/> from their <c>[TearDown]</c>.</remarks>
    public static Skeleton BuildSkeleton()
    {
        var simulationBone = new GameObject("SimulationBone").transform;

        var hips = NewBone("Hips", simulationBone, new float3(0f, 1f, 0f));
        NewBone("Spine", hips, new float3(0f, 0.2f, 0f));
        var leftFoot = NewBone("LeftFoot", hips, new float3(0.2f, -0.9f, 0f));
        NewBone("LeftToe", leftFoot, new float3(0f, -0.1f, 0.15f));

        _created.Add(simulationBone.gameObject);
        return new Skeleton(simulationBone);
    }

    private static Transform NewBone(string name, Transform parent, float3 localPosition)
    {
        var transform = new GameObject(name).transform;
        transform.SetParent(parent, false);
        transform.localPosition = localPosition;
        transform.localRotation = Quaternion.identity;
        return transform;
    }

    /// <summary>Destroys every rig built by <see cref="BuildSkeleton"/> and drops the
    /// derived-data caches keyed on their now-dead root Transforms.</summary>
    public static void DestroyAll()
    {
        foreach (var gameObject in _created)
        {
            if (gameObject != null) Object.DestroyImmediate(gameObject);
        }

        _created.Clear();
        Skeleton.InvalidateAll();
    }

    /// <summary>
    /// Fills <paramref name="pose"/> with a deterministic pseudo-random pose over
    /// <paramref name="skeleton"/>. Positions are only meaningful for the root (0) and hips (1),
    /// matching <see cref="PoseExtractor"/>'s convention -- every other joint holds its rest
    /// offset. Rotations are uniformly distributed normalized quaternions; velocities/angular
    /// velocities are uniform in [-2, 2]. Contacts are left untouched (Allocate zero-initializes
    /// them, i.e. false).
    /// </summary>
    public static void FillRandomPose(PoseBuffer pose, Skeleton skeleton, uint seed)
    {
        var random = new Unity.Mathematics.Random(seed == 0 ? 1u : seed);
        var boneCount = skeleton.BoneCount;
        var positions = pose.Positions;
        var rotations = pose.Rotations;
        var velocities = pose.Velocities;
        var angularVelocities = pose.AngularVelocities;

        for (var i = 0; i < boneCount; i++)
        {
            positions[i] = i <= 1 ? random.NextFloat3(-2f, 2f) : skeleton.GetBone(i).RestLocalPosition;
            rotations[i] = random.NextQuaternionRotation();
            velocities[i] = random.NextFloat3(-2f, 2f);
            angularVelocities[i] = random.NextFloat3(-2f, 2f);
        }
    }

    /// <summary>
    /// Loads the checked-in demo pose database. Returns false (rather than throwing) on any
    /// failure so demo-guarded tests can <c>Assert.Ignore</c> when it isn't available.
    /// </summary>
    public static bool TryLoadDemoPose(out PoseSet poseSet, out Skeleton skeleton)
    {
        poseSet = null;
        skeleton = null;

        var data = AssetDatabase.LoadAssetAtPath<MotionMatchingData>(
            "Assets/Animation/MotionMatching/MotionMatchingData.asset");
        if (data == null) return false;

        try
        {
            poseSet = data.GetOrImportPoseSet();
        }
        catch
        {
            return false;
        }

        if (poseSet == null || poseSet.NumberPoses == 0) return false;

        skeleton = poseSet.Skeleton;
        return skeleton != null;
    }
}
}
