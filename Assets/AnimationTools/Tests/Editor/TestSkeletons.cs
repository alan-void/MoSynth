using System.Collections.Generic;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Tests
{
/// <summary>
/// Factories for small <see cref="Skeleton"/> instances shared across the AnimationTools
/// edit-mode test suite. Bone ids follow <see cref="Skeleton.GetBoneId"/>'s index + 1
/// convention, so the constants below double as both index and id helpers for the fixtures.
/// </summary>
/// <remarks>
/// A skeleton is a Transform tree, so each fixture creates real GameObjects. Every suite that
/// builds one must call <see cref="DestroyAll"/> from its <c>[TearDown]</c>.
/// </remarks>
public static class TestSkeletons
{
    public const int RootId = 1;
    public const int SpineId = 2;
    public const int HeadId = 3;
    public const int LeftHandId = 3;
    public const int RightHandId = 4;

    /// <summary>A bone spec: name, index of its parent in the same array (-1 for the root),
    /// and its rest offset from that parent.</summary>
    public readonly struct BoneSpec
    {
        public readonly string Name;
        public readonly int ParentIndex;
        public readonly float3 LocalPosition;

        public BoneSpec(string name, int parentIndex, float3 localPosition)
        {
            Name = name;
            ParentIndex = parentIndex;
            LocalPosition = localPosition;
        }
    }

    private static readonly List<GameObject> _created = new();

    /// <summary>
    /// Materializes <paramref name="bones"/> as a GameObject hierarchy and returns a skeleton over
    /// its root. Specs must be in depth-first order with every parent ahead of its children, which
    /// is what makes the resulting DFS bone order match the spec order.
    /// </summary>
    public static Skeleton Build(params BoneSpec[] bones)
    {
        var transforms = new Transform[bones.Length];

        for (var i = 0; i < bones.Length; i++)
        {
            var spec = bones[i];
            var transform = new GameObject(spec.Name).transform;
            transform.SetParent(spec.ParentIndex < 0 ? null : transforms[spec.ParentIndex], false);
            transform.localPosition = spec.LocalPosition;
            transform.localRotation = Quaternion.identity;
            transforms[i] = transform;
        }

        _created.Add(transforms[0].gameObject);
        return new Skeleton(transforms[0]);
    }

    /// <summary>root(1) -&gt; spine(2) -&gt; head(3), each offset (0,1,0) from its parent.</summary>
    public static Skeleton CreateChain3() => Build(
        new BoneSpec("root", -1, float3.zero),
        new BoneSpec("spine", 0, new float3(0f, 1f, 0f)),
        new BoneSpec("head", 1, new float3(0f, 1f, 0f)));

    /// <summary>root(1) -&gt; spine(2), spine branches into leftHand(3) and rightHand(4).</summary>
    public static Skeleton CreateBranch4() => Build(
        new BoneSpec("root", -1, float3.zero),
        new BoneSpec("spine", 0, new float3(0f, 1f, 0f)),
        new BoneSpec("leftHand", 1, new float3(-1f, 0f, 0f)),
        new BoneSpec("rightHand", 1, new float3(1f, 0f, 0f)));

    /// <summary>Destroys every rig built by this class and drops the derived-data caches
    /// that were keyed on their now-dead root Transforms.</summary>
    public static void DestroyAll()
    {
        foreach (var gameObject in _created)
        {
            if (gameObject != null) Object.DestroyImmediate(gameObject);
        }

        _created.Clear();
        Skeleton.InvalidateAll();
    }
}
}
