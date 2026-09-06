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

    /// <summary>
    /// Annotation attached to this clip — see <see cref="AnimationClipComponent"/>.
    /// </summary>
    /// <remarks>
    /// Initialised here rather than seeded by the creation menu, because
    /// <c>[CreateAssetMenu]</c> makes assets without going through it.
    /// </remarks>
    [SerializeReference] [SubclassSelector]
    public List<AnimationClipComponent> components = new();

    /// Number of frames in the [startFrame, endFrame) slice, clamped defensively — a serialized
    /// endFrame can exceed the animation's frame count until OnValidate re-runs.
    public new int FrameCount => !HasClip ? 0 : Math.Max(0, Math.Min(endFrame, base.FrameCount) - startFrame);

    /// Frame view offset by startFrame. Never Dispose the returned buffer.
    public new PoseBuffer GetFrame(int frameIndex) => PoseSequence.GetFrame(startFrame + frameIndex);

    /// <summary>
    /// The first component of type <typeparamref name="T"/>, or null. Entries deserialize as null
    /// when their type was renamed or removed, so this skips them.
    /// </summary>
    public T GetComponent<T>() where T : AnimationClipComponent
    {
        foreach (var component in components)
        {
            if (component is T match) return match;
        }

        return null;
    }

    /// <inheritdoc cref="GetComponent{T}"/>
    public bool TryGetComponent<T>(out T component) where T : AnimationClipComponent
    {
        component = GetComponent<T>();
        return component != null;
    }

    protected override void OnValidate()
    {
        base.OnValidate();

        if (!HasClip)
            return;

        if (startFrame >= base.FrameCount)
            startFrame = base.FrameCount;

        if (endFrame >= base.FrameCount)
            endFrame = base.FrameCount;

        // Without this a start past the end survives, FrameCount reports 0, and every lane in the
        // clip editor bails on its own emptiness guard - a blank window with nothing to explain it.
        if (startFrame > endFrame)
            startFrame = endFrame;

        // After the clamp, so a component that indexes into the clip sees the range it will
        // actually be asked about. A component must not delete annotation for falling outside it:
        // trimming changes what is extracted, not what is true about the animation.
        foreach (var component in components)
        {
            component?.OnValidate(this);
        }
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
