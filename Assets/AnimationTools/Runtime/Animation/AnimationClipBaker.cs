using System;
using System.Collections.Generic;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Samples an <see cref="AnimationClip"/> against an instantiated rig and bakes discrete frames
/// in the <see cref="PoseLayout.CreateFullPose"/>(skeleton, false, false) float layout. Runtime-
/// legal (<see cref="AnimationClip.SampleAnimation"/>), so player-build pose extraction keeps
/// working.
/// </summary>
/// <remarks>
/// Humanoid (muscle-curve) clips are unsupported: they produce no transform motion on a plain
/// hierarchy. Only the root bone's translation is baked per frame; every other bone's position
/// stays at its skeleton rest offset, since <see cref="PoseFK"/> reconstructs non-root positions
/// from the rest pose and rotations.
/// </remarks>
public static class AnimationClipBaker
{
    /// <summary>
    /// Returns null when the hierarchy under <paramref name="skeleton"/>'s root has a different
    /// bone count than the skeleton itself; use <see cref="SkeletonAnimation.TryValidate"/> to
    /// report why.
    /// </summary>
    public static float[] Bake(AnimationClip clip, Skeleton skeleton, int frameCount, float frameTime)
    {
        var skeletonRoot = skeleton.Root;

        // SampleAnimation matches the clip's curve paths against the hierarchy under the object it
        // is handed, and an importer writes those paths relative to the asset's main object — the
        // FBX root, or the container a .bvh import puts its bones under. That object is the
        // skeleton root's topmost ancestor, so it is derived rather than stored: a second field
        // could disagree with the skeleton, this cannot.
        var rig = skeletonRoot.root;

        var path = new List<int>();
        var walker = skeletonRoot;
        while (walker != null && walker != rig)
        {
            path.Add(walker.GetSiblingIndex());
            walker = walker.parent;
        }

        path.Reverse();

        var instance = UnityEngine.Object.Instantiate(rig);
        try
        {
            SetHideFlagsRecursive(instance.transform);
            instance.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);

            var chain = new List<Transform> { instance.transform };
            var instanceRootBone = instance.transform;
            foreach (var siblingIndex in path)
            {
                instanceRootBone = instanceRootBone.GetChild(siblingIndex);
                chain.Add(instanceRootBone);
            }

            var transforms = Skeleton.CollectTransformsDfs(instanceRootBone);
            if (transforms.Count != skeleton.BoneCount) return null;

            WarnOnNonUnitScaleOnce(chain, transforms, rig.name);

            var layout = PoseLayout.CreateFullPose(skeleton, false, false);
            var d = layout.Data;

            // One-frame template with every bone's rest position; slot 0 (root) is overwritten
            // per frame below, identity rotations elsewhere are fine since every rotation slot
            // is written per frame too.
            var template = new float[layout.FloatCount];
            for (var b = 0; b < skeleton.BoneCount; b++)
            {
                var restPosition = skeleton.GetBone(b).RestLocalPosition;
                template[d.PositionStart + b * 3 + 0] = restPosition.x;
                template[d.PositionStart + b * 3 + 1] = restPosition.y;
                template[d.PositionStart + b * 3 + 2] = restPosition.z;
            }

            var result = new float[frameCount * layout.FloatCount];
            for (var f = 0; f < frameCount; f++)
            {
                var frameBase = f * layout.FloatCount;
                Array.Copy(template, 0, result, frameBase, layout.FloatCount);

                clip.SampleAnimation(instance.gameObject, f * frameTime);

                var rootPosition = instanceRootBone.position;
                var positionBase = frameBase + d.PositionStart;
                result[positionBase + 0] = rootPosition.x;
                result[positionBase + 1] = rootPosition.y;
                result[positionBase + 2] = rootPosition.z;

                var rootRotation = instanceRootBone.rotation;
                var rootRotationBase = frameBase + d.RotationStart;
                result[rootRotationBase + 0] = rootRotation.x;
                result[rootRotationBase + 1] = rootRotation.y;
                result[rootRotationBase + 2] = rootRotation.z;
                result[rootRotationBase + 3] = rootRotation.w;

                for (var i = 1; i < skeleton.BoneCount; i++)
                {
                    var localRotation = transforms[i].localRotation;
                    var rotationBase = frameBase + d.RotationStart + i * 4;
                    result[rotationBase + 0] = localRotation.x;
                    result[rotationBase + 1] = localRotation.y;
                    result[rotationBase + 2] = localRotation.z;
                    result[rotationBase + 3] = localRotation.w;
                }
            }

            return result;
        }
        finally
        {
            // instance is a Transform; Unity refuses to destroy a Transform component, and the
            // clone is flagged HideAndDontSave, so getting this wrong leaks the whole rig.
            UnityEngine.Object.DestroyImmediate(instance.gameObject);
        }
    }

    private static void SetHideFlagsRecursive(Transform transform)
    {
        transform.gameObject.hideFlags = HideFlags.HideAndDontSave;
        for (var i = 0; i < transform.childCount; i++)
        {
            SetHideFlagsRecursive(transform.GetChild(i));
        }
    }

    private const float ScaleTolerance = 1e-4f;

    private static void WarnOnNonUnitScaleOnce(List<Transform> chain, List<Transform> boneTransforms, string rigName)
    {
        foreach (var transform in chain)
        {
            if (!HasNonUnitScale(transform)) continue;
            Debug.LogWarning($"AnimationClipBaker: \"{transform.name}\" in rig \"{rigName}\" has non-unit local scale; baked FK is rigid and will be incorrect.");
            return;
        }

        foreach (var transform in boneTransforms)
        {
            if (!HasNonUnitScale(transform)) continue;
            Debug.LogWarning($"AnimationClipBaker: bone \"{transform.name}\" in rig \"{rigName}\" has non-unit local scale; baked FK is rigid and will be incorrect.");
            return;
        }
    }

    private static bool HasNonUnitScale(Transform transform)
    {
        var scale = transform.localScale;
        return Mathf.Abs(scale.x - 1f) > ScaleTolerance
            || Mathf.Abs(scale.y - 1f) > ScaleTolerance
            || Mathf.Abs(scale.z - 1f) > ScaleTolerance;
    }
}
}
