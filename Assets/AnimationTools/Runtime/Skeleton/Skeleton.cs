using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// A bone hierarchy defined by a Transform tree: the skeleton is the preorder depth-first walk of
/// <see cref="Root"/> and its descendants, so bone 0 is always the root and every bone's parent has
/// a lower index. Serializes as that single root reference; the bone list is derived from it.
/// </summary>
/// <remarks>
/// The rest pose is read straight off the Transforms, so a skeleton must point at an <em>asset</em>
/// rig — an imported FBX or a prefab, which sits at its import pose. A skeleton built over a scene
/// rig that something animates reports the current pose as the rest pose, which silently corrupts
/// FK. The synthesis pipeline keeps the two apart: its skeleton comes from an asset, and the scene
/// rig it drives is bound to that skeleton by <see cref="SkeletonBoneOverrides"/>.
/// <para>
/// Everything under <see cref="Root"/> becomes a bone, so a rig must have nothing but bones beneath
/// its skeleton root — a mesh node or an attachment point parented there shifts every index after it.
/// </para>
/// </remarks>
[Serializable]
public sealed class Skeleton
{
    [SerializeField] private Transform root;

    public Skeleton()
    {
    }

    public Skeleton(Transform root)
    {
        this.root = root;
    }

    /// <summary>Bone 0, and the start of the depth-first walk that defines every other bone.</summary>
    public Transform Root => root;

    public bool IsSet => root != null;

    /// <summary>Diagnostic label; not used for lookups or equality.</summary>
    public string Name => root != null ? root.name : "<unset>";

    public int BoneCount => IsSet ? GetDerived().Bones.Count : 0;

    /// <summary>
    /// Hash over each bone's (name, parentIndex). Two skeletons with the same
    /// <see cref="ContentHash"/> are very likely (though not guaranteed) structurally identical;
    /// use <see cref="StructurallyEqual"/> to confirm.
    /// </summary>
    public int ContentHash => IsSet ? GetDerived().ContentHash : 0;

    public SkeletonBone GetBone(int index) => GetDerivedOrThrow().Bones[index];

    /// <summary>-1 for bone 0.</summary>
    public int GetParentIndex(int index) => GetDerivedOrThrow().ParentIndices[index];

    /// <summary>Null for bone 0.</summary>
    public SkeletonBone GetParent(int index)
    {
        var parentIndex = GetParentIndex(index);
        return parentIndex < 0 ? null : GetBone(parentIndex);
    }

    /// <summary>Returns -1 when the Transform is not a bone of this skeleton.</summary>
    public int IndexOf(Transform bone)
    {
        if (bone == null || !IsSet) return -1;
        return GetDerived().IndexByTransform.TryGetValue(bone, out var index) ? index : -1;
    }

    public int IndexOf(SkeletonBone bone) => bone == null ? -1 : IndexOf(bone.Transform);

    public bool TryFindByName(string boneName, out int index)
    {
        if (IsSet)
        {
            var bones = GetDerived().Bones;
            for (var i = 0; i < bones.Count; i++)
            {
                if (bones[i].Name != boneName) continue;
                index = i;
                return true;
            }
        }

        index = -1;
        return false;
    }

    /// <summary>Returns -1 if no bone with this name exists.</summary>
    public int IndexOfName(string boneName) => TryFindByName(boneName, out var index) ? index : -1;

    /// <summary>
    /// Returns -1 for an unset id (0) or one outside the valid range. Bone ids follow the
    /// index + 1 convention so 0 unambiguously means "unset".
    /// </summary>
    public int IndexOfId(int boneId)
    {
        if (boneId < 1 || boneId > BoneCount) return -1;
        return boneId - 1;
    }

    public int GetBoneId(int index) => index + 1;

    /// <summary>
    /// Character-space rotation of one bone in the rest pose, found by walking up its parent chain.
    /// "Character space" matches <see cref="SkeletonData.CharacterSpaceRotation"/>: the frame of bone 0's parent.
    /// </summary>
    public quaternion RestCharacterRotation(int boneIndex)
    {
        var derived = GetDerivedOrThrow();
        var rotation = quaternion.identity;

        while (boneIndex != -1)
        {
            rotation = math.mul(derived.Bones[boneIndex].RestLocalRotation, rotation);
            boneIndex = derived.ParentIndices[boneIndex];
        }

        return rotation;
    }

    /// <summary>
    /// The bone-local axis that points along <paramref name="characterAxis"/> in the rest pose.
    /// A bone's local frame carries whatever roll the rig was authored with, so the axis that means
    /// "forward" differs per rig and must be read from the rest pose rather than assumed.
    /// </summary>
    public float3 RestLocalAxis(int boneIndex, float3 characterAxis) =>
        math.mul(math.inverse(RestCharacterRotation(boneIndex)), characterAxis);

    /// <summary>
    /// Returns the unmanaged mirror of this skeleton, built once and shared by every
    /// <see cref="Skeleton"/> over the same <see cref="Root"/>. Backing arrays use
    /// <see cref="Allocator.Domain"/>: they live for the domain's lifetime and are cleared
    /// automatically on domain reload, so callers must not Dispose them.
    /// </summary>
    public SkeletonData GetSkeletonData()
    {
        var derived = GetDerivedOrThrow();
        if (derived.HasData && derived.Data.IsCreated) return derived.Data;

        var count = derived.Bones.Count;
        var parentIndices = new NativeArray<int>(count, Allocator.Domain);
        var restLocalPositions = new NativeArray<float3>(count, Allocator.Domain);
        var restLocalRotations = new NativeArray<quaternion>(count, Allocator.Domain);

        for (var i = 0; i < count; i++)
        {
            var bone = derived.Bones[i];
            parentIndices[i] = derived.ParentIndices[i];
            restLocalPositions[i] = bone.RestLocalPosition;
            restLocalRotations[i] = bone.RestLocalRotation;
        }

        derived.Data = new SkeletonData
        {
            ParentIndices = parentIndices,
            RestLocalPositions = restLocalPositions,
            RestLocalRotations = restLocalRotations,
            BoneCount = count
        };
        derived.HasData = true;

        return derived.Data;
    }

    /// <summary>
    /// True when both skeletons have the same bone count and every bone matches by name and
    /// parentIndex. Rest pose is not compared. Null-safe: two nulls (or the same reference)
    /// are considered equal.
    /// </summary>
    public static bool StructurallyEqual(Skeleton a, Skeleton b)
    {
        if (ReferenceEquals(a, b)) return true;
        if (a == null || b == null) return false;
        if (a.Root == b.Root) return true;

        return a.MatchesFrom(0, b);
    }

    /// <summary>
    /// True when this skeleton's bones from <paramref name="startIndex"/> onward match every bone
    /// of <paramref name="other"/> by name and by parent, with parent indices shifted by
    /// <paramref name="startIndex"/>. That is: whether <paramref name="other"/> is embedded in this
    /// skeleton starting at that bone, so a rig nested under extra ancestors can be compared
    /// against the bare hierarchy. <paramref name="startIndex"/> 0 is plain structural equality.
    /// </summary>
    public bool MatchesFrom(int startIndex, Skeleton other)
    {
        if (other == null || startIndex < 0) return false;
        if (BoneCount - startIndex != other.BoneCount) return false;
        if (other.BoneCount == 0) return true;

        var derived = GetDerived();
        var otherDerived = other.GetDerived();

        for (var i = 0; i < otherDerived.Bones.Count; i++)
        {
            if (derived.Bones[startIndex + i].Name != otherDerived.Bones[i].Name) return false;

            var otherParent = otherDerived.ParentIndices[i];
            var expectedParent = otherParent < 0 ? startIndex - 1 : otherParent + startIndex;
            if (derived.ParentIndices[startIndex + i] != expectedParent) return false;
        }

        return true;
    }

    /// <summary>
    /// Returns every Transform under <paramref name="root"/> (root included) in preorder
    /// depth-first order: a transform is visited before its children, and children are visited in
    /// child-index order. This is the project-wide skeleton bone order.
    /// </summary>
    public static List<Transform> CollectTransformsDfs(Transform root)
    {
        var result = new List<Transform>();
        if (root == null) return result;

        CollectRecursive(root, result);
        return result;
    }

    private static void CollectRecursive(Transform transform, List<Transform> results)
    {
        results.Add(transform);
        for (var i = 0; i < transform.childCount; i++)
        {
            CollectRecursive(transform.GetChild(i), results);
        }
    }

    // --- Derived data ------------------------------------------------------------------------
    //
    // The bone list is a pure function of the root Transform, and a SkeletonBone serializes its
    // owning Skeleton by value — so an asset with several bone-reference fields holds several
    // Skeleton instances over one rig. Caching by root rather than per instance means they share
    // one walk, one parent index array, and one Domain-allocated SkeletonData.

    private sealed class DerivedData
    {
        public List<SkeletonBone> Bones;
        public int[] ParentIndices;
        public Dictionary<Transform, int> IndexByTransform;
        public int ContentHash;
        public SkeletonData Data;
        public bool HasData;
    }

    private static readonly Dictionary<Transform, DerivedData> _derivedByRoot = new();

    [NonSerialized] private DerivedData _derived;
    [NonSerialized] private Transform _derivedRoot;

    private DerivedData GetDerivedOrThrow()
    {
        if (!IsSet) throw new InvalidOperationException("Skeleton has no root Transform assigned.");
        return GetDerived();
    }

    private DerivedData GetDerived()
    {
        if (_derived != null && _derivedRoot == root) return _derived;

        if (!_derivedByRoot.TryGetValue(root, out var derived))
        {
            derived = Build(root);
            _derivedByRoot[root] = derived;
        }

        _derived = derived;
        _derivedRoot = root;
        return derived;
    }

    private static DerivedData Build(Transform root)
    {
        var transforms = CollectTransformsDfs(root);
        var count = transforms.Count;

        var indexByTransform = new Dictionary<Transform, int>(count);
        for (var i = 0; i < count; i++)
        {
            // A duplicate would mean the same Transform appeared twice in a DFS walk, which the
            // hierarchy makes impossible; indexing directly keeps the first wins rule explicit.
            indexByTransform[transforms[i]] = i;
        }

        // One owner shared by every bone: it resolves its own derived data lazily, by which point
        // this build has been stored in _derivedByRoot.
        var owner = new Skeleton(root);
        var bones = new List<SkeletonBone>(count);
        var parentIndices = new int[count];

        for (var i = 0; i < count; i++)
        {
            var transform = transforms[i];
            bones.Add(new SkeletonBone(owner, transform));

            if (i == 0)
            {
                parentIndices[i] = -1;
                continue;
            }

            var parent = transform.parent;
            if (parent == null || !indexByTransform.TryGetValue(parent, out var parentIndex))
                throw new InvalidOperationException(
                    $"Bone \"{transform.name}\" at index {i} of skeleton \"{root.name}\" has no parent within the skeleton.");

            parentIndices[i] = parentIndex;
        }

        return new DerivedData
        {
            Bones = bones,
            ParentIndices = parentIndices,
            IndexByTransform = indexByTransform,
            ContentHash = ComputeContentHash(bones, parentIndices)
        };
    }

    private static int ComputeContentHash(List<SkeletonBone> bones, int[] parentIndices)
    {
        unchecked
        {
            var hash = 17;
            for (var i = 0; i < bones.Count; i++)
            {
                hash = hash * 397 ^ (bones[i].Name?.GetHashCode() ?? 0);
                hash = hash * 397 ^ parentIndices[i];
            }

            return hash;
        }
    }

    /// <summary>
    /// Drops the cached bone list for this skeleton's root, so the next access re-walks the
    /// hierarchy. Call after the rig changes underneath a live skeleton — a reimport, or an edit
    /// to the bone tree — since neither the bone list nor <see cref="ContentHash"/> notices on its
    /// own, and <see cref="PoseLayout"/>'s cache is keyed on that hash.
    /// </summary>
    public void Invalidate()
    {
        _derived = null;
        _derivedRoot = null;
        if (root != null) _derivedByRoot.Remove(root);
    }

    /// <summary>Drops every cached bone list. Editor-side reset for a bulk reimport.</summary>
    public static void InvalidateAll() => _derivedByRoot.Clear();
}
}
