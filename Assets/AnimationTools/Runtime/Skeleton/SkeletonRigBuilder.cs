using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Materializes a <see cref="Skeleton"/> as a plain GameObject hierarchy at its rest pose.
/// Runtime-only (no UnityEditor APIs), so it can be used at build time for e.g. instantiating a
/// skeleton with no imported rig asset backing it.
/// </summary>
public static class SkeletonRigBuilder
{
    /// <summary>
    /// Creates one GameObject per bone, parented to match <paramref name="skeleton"/>'s
    /// hierarchy (bone 0 parented under <paramref name="parent"/> when given), with local
    /// position/rotation set from the rest pose. Returns the created Transform per bone index.
    /// </summary>
    public static Transform[] CreateHierarchyWithMap(Skeleton skeleton, Transform parent = null)
    {
        var transforms = new Transform[skeleton.BoneCount];

        for (var i = 0; i < skeleton.BoneCount; i++)
        {
            var bone = skeleton.GetBone(i);
            var boneTransform = new GameObject(bone.Name).transform;
            boneTransform.SetParent(i == 0 ? parent : transforms[skeleton.GetParentIndex(i)], false);
            boneTransform.localPosition = bone.RestLocalPosition;
            boneTransform.localRotation = bone.RestLocalRotation;
            transforms[i] = boneTransform;
        }

        return transforms;
    }

    /// <summary>Same as <see cref="CreateHierarchyWithMap"/>, returning only the root Transform.</summary>
    public static Transform CreateHierarchy(Skeleton skeleton, Transform parent = null)
    {
        var transforms = CreateHierarchyWithMap(skeleton, parent);
        return transforms.Length > 0 ? transforms[0] : null;
    }
}
}
