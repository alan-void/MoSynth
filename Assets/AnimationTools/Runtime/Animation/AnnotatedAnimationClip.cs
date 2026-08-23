using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using Unity.Mathematics;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AnimationTools
{
/// <summary>
/// Stores annotation using tags and other information for a raw animation clip.
/// To be used by Motion Synthesis Systems.
/// </summary>
[CreateAssetMenu(fileName = "AnimationData", menuName = "MotionMatching/AnimationData")]
public class AnnotatedAnimationClip : SkeletonAnimation
{

    [Min(0)] [Tooltip("Start frame of the animation clip. 0 indexing, inclusive.")]
    public int startFrame;

    [Min(0)] [Tooltip("End frame of the animation clip. 0 indexing, exclusive.")]
    public int endFrame;

    /// Number of frames in the [startFrame, endFrame) slice, clamped defensively — a serialized
    /// endFrame can exceed the animation's frame count until OnValidate re-runs.
    public new int FrameCount => !HasClip ? 0 : Math.Max(0, Math.Min(endFrame, base.FrameCount) - startFrame);

    /// Frame view offset by startFrame. Never Dispose the returned buffer.
    public new PoseBuffer GetFrame(int frameIndex) => PoseSequence.GetFrame(startFrame + frameIndex);

    protected override void OnValidate()
    {
        base.OnValidate();

        if (!HasClip)
            return;

        if (startFrame >= base.FrameCount)
            startFrame = base.FrameCount;

        if (endFrame >= base.FrameCount)
            endFrame = base.FrameCount;
    }

    /// <summary>
    /// Character-space (accumulated parent-chain) rotation of a bone at a sliced frame index.
    /// Identity when the asset has no resolvable skeleton.
    /// </summary>
    public quaternion GetWorldRotation(int boneIndex, int frameIndex)
    {
        var skeleton = Skeleton;
        if (skeleton == null) return quaternion.identity;

        return skeleton.GetSkeletonData().CharacterSpaceRotation(GetFrame(frameIndex), boneIndex);
    }

    [Serializable]
    public struct Tag
    {
        public string name;
        public int[] start; // Each element with index i, where, 0 <= i <= Start.Length == End.Length
        public int[] end; // represents a range. That is, for an arbitrary i -> [Start[i], End[i]]
    }

#if UNITY_EDITOR
    public void SaveEditor()
    {
        EditorUtility.SetDirty(this);
    }
#endif
}
}
