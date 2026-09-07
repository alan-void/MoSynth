using System;
using System.Collections.Generic;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// The footfalls a clip's gait phase is built from, and the settings used to find them.
/// </summary>
/// <remarks>
/// Phase is stored as its <em>anchors</em> rather than as a baked curve. A few hundred markers say
/// the same thing as a few thousand floats, they are the actual input to
/// <see cref="GaitPhase.Evaluate"/>, and — the point — they are correctable. The defect that
/// remains marker-level is contacts missed where the character moves fast; a standing intro with no
/// anchors at all is answered by the standing rule instead, since no marker can describe it.
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
    [Tooltip("Frames a foot was planted on, numbered against the whole clip. Detected, then " +
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
    public float contactVelocityThreshold = GaitMeasure.DefaultContactVelocityThreshold;

    [Tooltip("Half-width of the median filter over the raw contact flags, in frames. Removes " +
             "chatter; too large erases a short stance.")]
    [Min(0)]
    public int smoothingRadius = GaitMeasure.DefaultSmoothingRadius;

    [Header("Standing")]
    [Tooltip("Travel speed below which the character counts as standing, in m/s. A stretch with " +
             "no footfalls is a stand only if it is also this slow; otherwise it is a missed " +
             "contact and stays unusable.")]
    [Min(0f)]
    public float standingSpeed = 0.1f;

    [Tooltip("Seconds per cycle a standing stretch sweeps its phase at. Zero holds the phase " +
             "still instead, which drops those frames from training.")]
    [Min(0f)]
    public float standingPeriod = 1f;

    public override string Describe()
    {
        if (footfalls == null || footfalls.Count == 0) return "no footfalls detected";

        var repeats = GaitPhase.CountRepeatedFeet(footfalls);
        var summary = $"{footfalls.Count} footfalls";
        return repeats == 0 ? summary : $"{summary}, {repeats} missed contacts";
    }

    /// <summary>Phase and rate for every frame of the clip, from the current footfalls.</summary>
    /// <remarks>
    /// Covers the whole clip rather than the slice, because the anchors do: an anchor just past the
    /// trim still tells the frames before it what their cycle is.
    /// </remarks>
    public void Evaluate(AnnotatedAnimationClip clip, out float[] phase, out float[] phaseRate)
    {
        var frameCount = Mathf.Max(0, clip != null ? ((SkeletonAnimation)clip).FrameCount : 0);
        phase = new float[frameCount];
        phaseRate = new float[frameCount];
        if (frameCount == 0) return;

        // Without a skeleton there is nothing to measure travel against, so the standing rule stays
        // off and every anchorless stretch is held, which is the safe reading.
        var skeleton = clip.Skeleton;
        var speed = skeleton != null ? GaitMeasure.GroundSpeed(clip, skeleton, 0, frameCount) : null;

        GaitPhase.Evaluate(footfalls, clip.FrameTime, speed, standingSpeed, standingPeriod,
            phase, phaseRate);
    }

    /// <summary>
    /// Whether each foot is planted on each frame of the clip's <c>[startFrame, endFrame)</c> slice,
    /// by this component's settings. Index <c>frame * 2</c> is left, <c>+ 1</c> right, slice-local.
    /// </summary>
    /// <remarks>
    /// The pose-database bake calls this too, so the contact flags a database records are the ones
    /// the clip editor drew when the footfalls were corrected.
    /// </remarks>
    public bool TryDetectSliceContacts(AnnotatedAnimationClip clip, out bool[] contacts, out string error)
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

        contacts = GaitMeasure.Contacts(clip, skeleton, leftBone, rightBone, clip.startFrame,
            frameCount, contactVelocityThreshold, smoothingRadius);
        return true;
    }

    /// <summary>
    /// Replaces <see cref="footfalls"/> with the ones detected in the clip, and reports the contact
    /// flags they came from so a caller can show what was actually measured.
    /// </summary>
    /// <param name="clip">The clip to read. Only its <c>[startFrame, endFrame)</c> slice is used.</param>
    /// <param name="contacts">
    /// Per frame of the <em>whole clip</em>, whether each foot was planted: index <c>frame * 2</c>
    /// is left, <c>+ 1</c> right, false outside the slice. Smoothed, i.e. what the footfalls were
    /// actually detected from. Clip-wide so that nothing downstream has to convert between two
    /// frame spaces.
    /// </param>
    /// <param name="error">Why detection could not run.</param>
    public bool TryDetect(AnnotatedAnimationClip clip, out bool[] contacts, out string error)
    {
        contacts = Array.Empty<bool>();
        if (!TryDetectSliceContacts(clip, out var sliceContacts, out error)) return false;

        var frameCount = clip.FrameCount;
        var detected = GaitPhase.FootfallsFromContacts(sliceContacts, frameCount);

        // Detection reads only the slice - that is what a trim is for - but reports in clip frames,
        // which is the space the anchors are stored and drawn in.
        var startFrame = clip.startFrame;
        footfalls = new List<GaitPhase.Footfall>(detected.Count);
        foreach (var footfall in detected)
        {
            footfalls.Add(new GaitPhase.Footfall(footfall.frame + startFrame, footfall.foot));
        }

        contacts = new bool[((SkeletonAnimation)clip).FrameCount * 2];
        Array.Copy(sliceContacts, 0, contacts, startFrame * 2, sliceContacts.Length);
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

}
}
