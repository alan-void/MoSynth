using System;
using System.Collections.Generic;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Binds a <see cref="Skeleton"/> to a different rig: the skeleton describes an asset rig at its
/// rest pose, while <see cref="Root"/> is the live hierarchy that actually gets posed. Bones are
/// matched by name, with per-bone overrides for a rig that names a bone differently or keeps it
/// outside the walk under <see cref="Root"/>.
/// </summary>
[Serializable]
public sealed class SkeletonBoneOverrides
{
    /// <summary>Escape hatch for a skeleton bone that has no same-named Transform under
    /// <see cref="Root"/>, or that should resolve somewhere else entirely.</summary>
    [Serializable]
    public struct Entry
    {
        [Tooltip("The bone in the skeleton being bound.")]
        public Transform sourceBone;

        [Tooltip("The Transform in this rig it should drive.")]
        public Transform target;
    }

    [SerializeField] private Transform root;
    [SerializeField] private List<Entry> overrides = new();

    public Transform Root => root;
    public bool IsSet => root != null;

    /// <summary>Runtime fallback for when no root was assigned in the inspector.</summary>
    public void SetRoot(Transform newRoot)
    {
        root = newRoot;
    }

    /// <summary>
    /// Finds the Transform for a bone by name: overrides are consulted first, then a DFS search
    /// (root included) for an exact name match. Null when no match is found in either.
    /// </summary>
    public Transform FindBone(string boneName)
    {
        foreach (var boneOverride in overrides)
        {
            if (boneOverride.target == null || boneOverride.sourceBone == null) continue;
            if (boneOverride.sourceBone.name == boneName) return boneOverride.target;
        }

        return root == null ? null : FindBoneRecursive(root, boneName);
    }

    private static Transform FindBoneRecursive(Transform transform, string boneName)
    {
        if (transform.name == boneName) return transform;

        for (var i = 0; i < transform.childCount; i++)
        {
            var found = FindBoneRecursive(transform.GetChild(i), boneName);
            if (found != null) return found;
        }

        return null;
    }

    /// <summary>
    /// Resolves every bone of <paramref name="skeleton"/> to a Transform in this rig. Slot 0 is
    /// <paramref name="indexZeroOverride"/> when given, else <see cref="Root"/>. Missing bones log
    /// an error and leave their slot null; the caller decides how severe that is. Duplicate bone
    /// names in the hierarchy that the skeleton references also log a warning, since the first DFS
    /// match silently wins.
    /// </summary>
    public Transform[] Bind(Skeleton skeleton, Transform indexZeroOverride = null)
    {
        var result = new Transform[skeleton.BoneCount];
        if (skeleton.BoneCount == 0) return result;

        result[0] = indexZeroOverride != null ? indexZeroOverride : root;

        var transforms = Skeleton.CollectTransformsDfs(root);
        var nameCounts = new Dictionary<string, int>();
        foreach (var transform in transforms)
        {
            nameCounts.TryGetValue(transform.name, out var count);
            nameCounts[transform.name] = count + 1;
        }

        for (var i = 1; i < skeleton.BoneCount; i++)
        {
            var sourceBone = skeleton.GetBone(i);

            var bone = FindOverride(sourceBone.Transform);
            if (bone == null)
            {
                var boneName = sourceBone.Name;
                bone = FindBone(boneName);
                if (bone == null)
                {
                    Debug.LogError($"SkeletonBoneOverrides could not find bone \"{boneName}\" under root \"{(root != null ? root.name : "<none>")}\".");
                    continue;
                }

                if (nameCounts.TryGetValue(boneName, out var count) && count > 1)
                {
                    Debug.LogWarning($"SkeletonBoneOverrides found {count} transforms named \"{boneName}\" under root \"{(root != null ? root.name : "<none>")}\"; using the first depth-first match.");
                }
            }

            result[i] = bone;
        }

        return result;
    }

    private Transform FindOverride(Transform sourceBone)
    {
        if (sourceBone == null) return null;

        foreach (var boneOverride in overrides)
        {
            if (boneOverride.sourceBone == sourceBone && boneOverride.target != null)
                return boneOverride.target;
        }

        return null;
    }
}
}
