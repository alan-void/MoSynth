using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Serialization;

namespace MotionMatching
{
/// <summary>
/// Plays a <see cref="SkeletonAnimation"/> on a bone hierarchy under this GameObject and draws it with Gizmos.
/// </summary>
public class BvhVisualiser : MonoBehaviour
{
    [FormerlySerializedAs("skeletalAnimation")] [FormerlySerializedAs("bvhAnimation")] public SkeletonAnimation skeletonAnimation;
    public bool play;
    public float spheresRadius = 0.1f;

    private Transform[] _skeletonBoneTransforms;

    private float _currentFrameTime;

    [SerializeField] private int currentFrame;

    private void Awake()
    {
        if (skeletonAnimation == null || skeletonAnimation.Skeleton == null || skeletonAnimation.FrameCount == 0) return;
        SetupSkeleton();
        UpdateSkeletonTransforms();
    }

    private void SetupSkeleton()
    {
        _skeletonBoneTransforms = EnsureSkeletonHierarchy(skeletonAnimation.Skeleton);
        for (var i = 0; i < _skeletonBoneTransforms.Length; i++)
        {
            _skeletonBoneTransforms[i].localPosition = skeletonAnimation.Skeleton.GetBone(i).RestLocalPosition;
        }
    }

    private void Update()
    {
        if (skeletonAnimation == null || skeletonAnimation.Skeleton == null || skeletonAnimation.FrameCount == 0) return;

        if (!play) return;

        _currentFrameTime = currentFrame + math.frac(_currentFrameTime);
        _currentFrameTime += Time.deltaTime / skeletonAnimation.FrameTime;
        currentFrame = (int)math.floor(_currentFrameTime);

        if (currentFrame >= skeletonAnimation.FrameCount)
        {
            currentFrame = 0;
        }

        UpdateSkeletonTransforms();
    }

    private void UpdateSkeletonTransforms()
    {
        var frame = skeletonAnimation.GetFrame(currentFrame);
        _skeletonBoneTransforms[0].localPosition = (Vector3)(float3)frame.Positions[0];
        var rotations = frame.Rotations;
        for (var i = 0; i < rotations.Length; i++)
        {
            _skeletonBoneTransforms[i].localRotation = rotations[i];
        }
    }

    private void OnValidate()
    {
        if (skeletonAnimation == null || skeletonAnimation.Skeleton == null || skeletonAnimation.FrameCount == 0) return;
        SetupSkeleton();
        UpdateSkeletonTransforms();
    }

    /// <summary>
    /// Ensures the GameObject hierarchy matches the provided Skeleton structure.
    /// Existing bones are preserved; missing bones are created at their proper local offsets.
    /// </summary>
    private Transform[] EnsureSkeletonHierarchy(Skeleton skeleton)
    {
        if (skeleton == null || skeleton.BoneCount == 0)
        {
            return Array.Empty<Transform>();
        }

        var boneCount = skeleton.BoneCount;
        var boneTransforms = new Transform[boneCount];

        for (var i = 0; i < boneCount; i++)
        {
            GetOrCreateBone(i);
        }

        return boneTransforms;

        // Resolves parents first, so a bone's parent always exists before the bone is looked up.
        Transform GetOrCreateBone(int boneIndex)
        {
            if (boneIndex < 0 || boneIndex >= boneCount)
            {
                return this.transform;
            }

            if (boneTransforms[boneIndex])
            {
                return boneTransforms[boneIndex];
            }

            var bone = skeleton.GetBone(boneIndex);
            var parentIndex = skeleton.GetParentIndex(boneIndex);

            var parentTransform = this.transform;
            if (parentIndex >= 0)
            {
                parentTransform = GetOrCreateBone(parentIndex);
            }

            var existingBone = parentTransform.Find(bone.Name);

            if (existingBone != null)
            {
                boneTransforms[boneIndex] = existingBone;
            }
            else
            {
                var newBone = new GameObject(bone.Name);
                newBone.transform.SetParent(parentTransform, false);

                boneTransforms[boneIndex] = newBone.transform;
            }

            return boneTransforms[boneIndex];
        }
    }

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        if (_skeletonBoneTransforms == null || skeletonAnimation == null) return;

        Gizmos.color = Color.red;
        for (var i = 1; i < _skeletonBoneTransforms.Length; i++)
        {
            var t = _skeletonBoneTransforms[i];
            GizmosExtensions.DrawLine(t.parent.position, t.position, 3);
        }

        Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f);
        foreach (var t in _skeletonBoneTransforms)
        {
            if (t.name == "End Site") continue;
            Gizmos.DrawWireSphere(t.position, spheresRadius);
        }
    }
#endif
}
}