using System;
using Unity.Collections;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// A clip-backed skeletal animation: wraps a <see cref="UnityEngine.AnimationClip"/> together with
/// the <see cref="AnimationTools.Skeleton"/> it is sampled against, whose root Transform is the
/// single source of truth for where the bones are. Frames are lazily baked (see
/// <see cref="AnimationClipBaker"/>) into the AnimationTools pose format on first access to
/// <see cref="PoseSequence"/>.
/// </summary>
public class SkeletonAnimation : ScriptableObject
{
    [SerializeField] private AnimationClip clip;

    [Tooltip("The rig's root bone; the skeleton is that Transform and everything beneath it. Drop " +
             "the imported model here, then use the dropdown to reach a bone deeper in it — a " +
             "model exposes only its topmost bone to the Project window. It must come from an " +
             "imported rig asset, since rest pose is read live off the Transforms and a scene rig " +
             "reports whatever pose it is currently animated to.")]
    [SerializeField]
    private Skeleton skeleton = new();

    [Tooltip(
        "The bone from which root motion is derived. Currently the x and z motion of this bone is transferred to the root of the animated character.")]
    [BoneFrom(nameof(skeleton))]
    [SerializeField]
    private SkeletonBone rootMotionBone = new();

    public AnimationClip Clip => clip;
    public bool HasClip => clip != null;

    /// <summary>The skeleton's bone 0; null when no resolvable skeleton root is assigned.</summary>
    public Transform RootBone => skeleton != null && IsResolvable(skeleton.Root) ? skeleton.Root : null;

    /// <summary>
    /// A serialized reference whose target no longer resolves is <em>missing</em>, not null: Unity
    /// reports it as non-null on purpose, so touching it throws a diagnostic rather than silently
    /// behaving as null. Inspectors reach <see cref="RootBone"/> and <see cref="TryValidate"/> every
    /// repaint, so neither may be the thing that throws.
    /// </summary>
    private static bool IsResolvable(Transform transform)
    {
        if (transform == null) return false;

        try
        {
            _ = transform.childCount;
            return true;
        }
        catch (MissingReferenceException)
        {
            return false;
        }
    }

    /// <summary>The bone whose motion drives the character root; the skeleton root when unset.</summary>
    public Transform RootMotionBone =>
        rootMotionBone != null && rootMotionBone.IsSet ? rootMotionBone.Transform : RootBone;

    public float FrameTime => clip != null ? 1f / clip.frameRate : 0f;

    // Plain arithmetic; must not materialize the sequence, since OnValidate and inspectors call
    // this every repaint.
    public int FrameCount => FrameCountOf(clip);

    /// <summary>How many frames a clip bakes to, without an asset having to hold it first.</summary>
    public static int FrameCountOf(AnimationClip animationClip) =>
        animationClip == null
            ? 0
            : Mathf.Max(1, Mathf.RoundToInt(animationClip.length * animationClip.frameRate) + 1);

    /// <summary>
    /// Null when the asset is not configured well enough to produce one — see
    /// <see cref="TryValidate"/> for why. Called every Inspector repaint, so it never logs.
    /// </summary>
    public Skeleton Skeleton => RootBone != null ? skeleton : null;

    /// <summary>
    /// Checks that this asset can actually be baked: a clip, and a skeleton root that both resolves
    /// and belongs to an imported asset. Returns false with a message suitable for an Inspector
    /// HelpBox. Never logs — inspectors call it every repaint.
    /// </summary>
    public bool TryValidate(out string error)
    {
        if (clip == null)
        {
            error = "No animation clip assigned.";
            return false;
        }

        var rootBone = RootBone;
        if (rootBone == null)
        {
            error = skeleton == null || skeleton.Root == null
                ? "No skeleton assigned. Drop the imported model on the Skeleton field, then pick " +
                  "the root bone; the skeleton is that Transform and everything beneath it."
                : "The assigned skeleton root no longer resolves; reassign it.";
            return false;
        }

#if UNITY_EDITOR
        // Rest pose is read live off these Transforms, so a scene rig hands back whatever pose it
        // happens to be animated to. Nothing downstream can detect that, hence the check here.
        if (!UnityEditor.EditorUtility.IsPersistent(rootBone))
        {
            error = $"Skeleton root \"{rootBone.name}\" is a scene object, so its rest pose is " +
                    "whatever it is currently posed to. Assign the bone from the imported rig asset.";
            return false;
        }
#endif

        if (!skeleton.TryValidateRestPose(out error)) return false;

        // Last, because it is the only check that has to read the clip's curves.
        return TryValidateClipCached(out error);
    }

    [NonSerialized] private int _validatedClipKey;
    [NonSerialized] private bool _hasValidatedClip;
    [NonSerialized] private string _clipValidationError;

    /// <summary>
    /// <see cref="AnimationClipBaker.TryValidateClip"/>, memoized. Reading curve bindings walks
    /// every curve in the clip — 730 of them for a walk cycle — and inspectors reach
    /// <see cref="TryValidate"/> every repaint, once per clip in a MotionFieldConfig's list. The
    /// answer only moves when the clip or the bone tree does, so key on both rather than relying on
    /// <see cref="ClearRuntimeCaches"/>, which a rig reimport does not trigger.
    /// </summary>
    /// <remarks>
    /// The live bone count is part of the key because <see cref="Skeleton.ContentHash"/> is derived
    /// from the skeleton's own cached bone list, and so cannot move in the one case the check most
    /// needs to catch: a rig whose bones changed underneath a skeleton that has not noticed yet.
    /// </remarks>
    private bool TryValidateClipCached(out string error)
    {
        // Hashing the clip reference rather than its instance id: Object.GetHashCode is the same
        // identity without the deprecation that GetInstanceID now carries.
        var key = HashCode.Combine(clip, skeleton.ContentHash,
            Skeleton.CollectTransformsDfs(skeleton.Root).Count);
        if (!_hasValidatedClip || key != _validatedClipKey)
        {
            AnimationClipBaker.TryValidateClip(clip, skeleton, out _clipValidationError);
            _validatedClipKey = key;
            _hasValidatedClip = true;
        }

        error = _clipValidationError;
        return error == null;
    }

    [NonSerialized] private PoseSequence _poseSequence;

    public PoseSequence PoseSequence
    {
        get
        {
            if (!TryValidate(out _)) return null;
            if (_poseSequence != null) return _poseSequence;

            var skeletonToBake = Skeleton;
            var layout = PoseLayout.CreateFullPose(skeletonToBake, false, false);
            var baked = AnimationClipBaker.Bake(clip, skeletonToBake, FrameCount, FrameTime);
            if (baked == null) return null;

            var nativeFrameData = new NativeArray<float>(baked, Allocator.Domain);

            _poseSequence = new PoseSequence(layout, nativeFrameData);
            return _poseSequence;
        }
    }

    public PoseBuffer GetFrame(int frameIndex) => PoseSequence.GetFrame(frameIndex);

    /// <summary>Editor-helper setup for creation menus and tests: assigns the source fields
    /// directly and invalidates any cached bake.</summary>
    public void SetSource(AnimationClip inClip, Transform inRootBone)
    {
        clip = inClip;
        skeleton = new Skeleton(inRootBone);
        rootMotionBone = new SkeletonBone(skeleton, inRootBone);
        ClearRuntimeCaches();
    }

    protected void ClearRuntimeCaches()
    {
        skeleton?.Invalidate();
        _poseSequence = null;
        _hasValidatedClip = false;
    }

    protected virtual void OnValidate() => ClearRuntimeCaches();
}
}