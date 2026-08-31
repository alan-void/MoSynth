using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// The footfalls a clip's gait phase is built from, and the settings used to find them.
/// </summary>
/// <remarks>
/// Phase is stored as its <em>anchors</em> rather than as a baked curve. A few hundred markers say
/// the same thing as a few thousand floats, they are the actual input to
/// <see cref="GaitPhase.Evaluate"/>, and — the point — they are correctable. Both of the defects
/// measured on the shipped clips are marker-level: contacts missed where the character moves fast,
/// and phase invented over a standing intro that has no anchors at all.
/// <para>
/// The detection settings live here, per clip, rather than on the database config. A single
/// threshold cannot serve two clips at different speeds: <c>walk1_subject5</c> travels at 1.27 m/s
/// against <c>walk1_subject1</c>'s 0.675 m/s, and at the shared 0.15 m/s threshold its feet are
/// detected as planted only 22% of the time against the other's 51%.
/// </para>
/// </remarks>
[Serializable]
public sealed class GaitPhaseComponent : AnimationClipComponent
{
    [Tooltip("Frames a foot was planted on, in this clip's sliced frame numbering. Detected, then " +
             "corrected by hand where the detection was wrong.")]
    public List<GaitPhase.Footfall> footfalls = new();

    [Header("Detection")]
    [Tooltip("Bone whose speed decides left foot contact. Empty picks by name (LeftToe, then LeftFoot).")]
    public string leftContactBoneName;

    [Tooltip("Bone whose speed decides right foot contact. Empty picks by name (RightToe, then RightFoot).")]
    public string rightContactBoneName;

    [Tooltip("Foot speed below which a toe counts as planted, in m/s, measured in the character's " +
             "own frame. Raise it for a faster clip.")]
    [Min(0f)]
    public float contactVelocityThreshold = 0.15f;

    [Tooltip("Half-width of the median filter over the raw contact flags, in frames. Removes " +
             "chatter; too large erases a short stance.")]
    [Min(0)]
    public int smoothingRadius = 6;

    public override string Describe()
    {
        if (footfalls == null || footfalls.Count == 0) return "no footfalls detected";

        var repeats = GaitPhase.CountRepeatedFeet(footfalls);
        var summary = $"{footfalls.Count} footfalls";
        return repeats == 0 ? summary : $"{summary}, {repeats} missed contacts";
    }

    /// <summary>Drops anchors the clip's frame range no longer contains.</summary>
    public override void OnValidate(AnnotatedAnimationClip clip)
    {
        if (footfalls == null || clip == null) return;

        var frameCount = clip.FrameCount;
        footfalls.RemoveAll(footfall => footfall.frame < 0 || footfall.frame >= frameCount);
    }

    /// <summary>Phase and rate per frame of the clip's slice, from the current footfalls.</summary>
    public void Evaluate(AnnotatedAnimationClip clip, out float[] phase, out float[] phaseRate)
    {
        var frameCount = Mathf.Max(0, clip != null ? clip.FrameCount : 0);
        phase = new float[frameCount];
        phaseRate = new float[frameCount];

        if (frameCount > 0) GaitPhase.Evaluate(footfalls, clip.FrameTime, phase, phaseRate);
    }

    /// <summary>
    /// Replaces <see cref="footfalls"/> with the ones detected in the clip, and reports the contact
    /// flags they came from so a caller can show what was actually measured.
    /// </summary>
    /// <param name="clip">The clip to read. Only its <c>[startFrame, endFrame)</c> slice is used.</param>
    /// <param name="contacts">
    /// Per frame, whether each foot was planted: index <c>frame * 2</c> is left, <c>+ 1</c> right.
    /// Smoothed, i.e. what the footfalls were actually detected from.
    /// </param>
    /// <param name="error">Why detection could not run.</param>
    public bool TryDetect(AnnotatedAnimationClip clip, out bool[] contacts, out string error)
    {
        contacts = Array.Empty<bool>();
        error = null;

        if (clip == null)
        {
            error = "No clip.";
            return false;
        }

        var skeleton = clip.Skeleton;
        if (skeleton == null)
        {
            error = "The clip has no resolvable skeleton.";
            return false;
        }

        if (!TryResolveContactBone(skeleton, leftContactBoneName, left: true, out var leftBone,
                out error) ||
            !TryResolveContactBone(skeleton, rightContactBoneName, left: false, out var rightBone,
                out error))
        {
            return false;
        }

        var frameCount = clip.FrameCount;
        if (frameCount < 2)
        {
            error = $"The clip's frame range holds {frameCount} frames; at least 2 are needed.";
            return false;
        }

        contacts = DetectContacts(clip, skeleton, leftBone, rightBone, frameCount);
        footfalls = GaitPhase.FootfallsFromContacts(contacts, frameCount);
        return true;
    }

    /// <summary>The bone whose speed decides a foot's contact, by name or by convention.</summary>
    private static bool TryResolveContactBone(Skeleton skeleton, string boneName, bool left,
        out int index, out string error)
    {
        error = null;

        if (!string.IsNullOrEmpty(boneName))
        {
            index = skeleton.IndexOfName(boneName);
            if (index >= 0) return true;

            error = $"The clip's rig has no bone named '{boneName}'.";
            return false;
        }

        if (BoneNameConventions.TryFindContactBone(skeleton, left, out index)) return true;

        // Falling back to bone 0 would silently threshold the root's speed and produce a phase that
        // is confidently wrong, which is worse than refusing.
        error = $"No {(left ? "left" : "right")} contact bone found by name. Name one explicitly.";
        return false;
    }

    /// <summary>
    /// Whether each foot is planted on each frame, from how far it travels in the character's own
    /// frame.
    /// </summary>
    /// <remarks>
    /// A clip's baked poses carry positions and rotations only — no velocity channels — so this
    /// differences consecutive character-space positions rather than composing the per-bone velocity
    /// channels the way <c>PoseExtractor.ExtractPoseContacts</c> does at bake time.
    /// <para>
    /// <b>The two do not agree.</b> On the shipped clips, at the same 0.15 m/s threshold, this
    /// measure reports a foot planted far more often than the bake does — 0.40/0.36 against
    /// 0.22/0.17 on <c>walk1_subject5</c>, and 0.57/0.56 against 0.51/0.49 on
    /// <c>walk1_subject1</c> — and finds half as many missed contacts. Differencing the composed
    /// position is the more direct measurement of whether a toe moved, so the gap is a reason to
    /// distrust the bake-time contacts rather than these; why the composition inflates the speed
    /// is not yet established.
    /// </para>
    /// </remarks>
    private bool[] DetectContacts(AnnotatedAnimationClip clip, Skeleton skeleton, int leftBone,
        int rightBone, int frameCount)
    {
        var skeletonData = skeleton.GetSkeletonData();

        var leftPositions = new float3[frameCount];
        var rightPositions = new float3[frameCount];
        for (var frame = 0; frame < frameCount; frame++)
        {
            var pose = clip.GetFrame(frame);
            leftPositions[frame] = skeletonData.CharacterSpacePosition(pose, leftBone);
            rightPositions[frame] = skeletonData.CharacterSpacePosition(pose, rightBone);
        }

        var frameTime = clip.FrameTime;
        var contacts = new bool[frameCount * 2];
        for (var frame = 0; frame < frameCount; frame++)
        {
            // The last frame has no successor to difference against, so it inherits the one before
            // it rather than being reported as a sudden plant.
            var from = math.min(frame, frameCount - 2);
            contacts[frame * 2] = IsPlanted(leftPositions[from], leftPositions[from + 1], frameTime);
            contacts[frame * 2 + 1] = IsPlanted(rightPositions[from], rightPositions[from + 1], frameTime);
        }

        GaitPhase.SmoothContacts(contacts, frameCount, smoothingRadius);
        return contacts;
    }

    private bool IsPlanted(float3 from, float3 to, float frameTime) =>
        frameTime > 0f && math.length(to - from) / frameTime < contactVelocityThreshold;
}
}
