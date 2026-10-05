using UnityEngine;
using Unity.Collections;

namespace AnimationTools
{
/// <summary>
/// Extracts full poses, their rates, bone contacts and gait phase from an annotated clip into a
/// <see cref="PoseSet"/>.
/// </summary>
public static class PoseExtractor
{
    /// <summary>
    /// Appends the clip's poses to <paramref name="poseSet"/> without clearing it. False when the
    /// clip's skeleton does not match the pose set's and nothing was added.
    /// </summary>
    /// <param name="mirror">
    /// When given, the clip is appended as its left-to-right mirror image: poses, contacts and phase.
    /// </param>
    public static bool Extract(AnnotatedAnimationClip animationClip, PoseSet poseSet, IPoseSetSource source,
        PoseMirror mirror = null)
    {
        var clipSkeleton = animationClip.Skeleton;
        if (clipSkeleton == null)
        {
            Debug.LogError($"Clip \"{animationClip.name}\" has no resolvable skeleton; check its rig and root bone.");
            return false;
        }

        // The pose skeleton and the clip's own skeleton are the same bone tree; nothing is
        // prepended, so a mismatch means the clip belongs to a different rig.
        var poseSkeleton = poseSet.Skeleton;
        if (!Skeleton.StructurallyEqual(poseSkeleton, clipSkeleton))
        {
            Debug.LogError($"Skeleton of clip \"{animationClip.name}\" ({clipSkeleton.BoneCount} bones) " +
                $"is not structurally equal to the pose set's skeleton ({poseSkeleton?.BoneCount ?? 0} bones).");
            return false;
        }

        var frameCount = animationClip.FrameCount;

        var frames = poseSet.BeginClip(frameCount - 1, animationClip.FrameTime);

        for (var i = 0; i < frameCount - 1; i++)
        {
            ExtractPose(frames[i], animationClip, i, mirror);
        }

        for (var i = 0; i < frameCount - 2; i++)
        {
            ExtractPoseVelocities(frames[i], frames[i + 1], animationClip);
        }

        var lastPose = PoseBuffer.Allocate(poseSet.PoseLayout, Allocator.Temp);
        try
        {
            ExtractPose(lastPose, animationClip, frameCount - 1, mirror);
            ExtractPoseVelocities(frames[frames.Count - 1], lastPose, animationClip);
        }
        finally
        {
            lastPose.Dispose();
        }

        WriteContacts(animationClip, poseSet, frames, source, mirror);
        WritePhase(animationClip, poseSet, frames, mirror != null);

        return true;
    }

    /// <summary>
    /// Writes one contact flag per slot of the pose set's contact list into the database frames the
    /// clip just filled, each measured on its own bone by <see cref="GaitMeasure"/>.
    /// </summary>
    /// <remarks>
    /// The threshold and smoothing come from the clip's <see cref="GaitPhaseComponent"/> when it has
    /// one, so a clip's database contacts follow the settings its footfalls were corrected under;
    /// otherwise from the config and the default radius.
    /// </remarks>
    private static void WriteContacts(AnnotatedAnimationClip animationClip, PoseSet poseSet,
        PoseSet.PoseFrameRange frames, IPoseSetSource source, PoseMirror mirror)
    {
        var contacts = poseSet.ContactHandles;
        if (contacts.Count == 0) return;

        var gait = animationClip.GetComponent<GaitPhaseComponent>();
        var threshold = gait != null ? gait.contactVelocityThreshold : source.ContactVelocityThreshold;
        var smoothingRadius = gait != null ? gait.smoothingRadius : GaitMeasure.DefaultSmoothingRadius;

        var skeleton = animationClip.Skeleton;
        var measured = new bool[contacts.Count][];
        for (var slot = 0; slot < contacts.Count; slot++)
        {
            measured[slot] = GaitMeasure.Contacts(animationClip, skeleton, contacts.GetBoneIndex(slot),
                animationClip.startFrame, animationClip.FrameCount, threshold, smoothingRadius);
        }

        var sourceSlots = MirrorSourceSlots(contacts, mirror);
        for (var slot = 0; slot < contacts.Count; slot++)
        {
            if (sourceSlots[slot] < 0)
            {
                Debug.LogError($"Clip \"{animationClip.name}\": the mirrored copy has no contact slot for the " +
                               $"counterpart of \"{poseSet.ContactBoneNames[slot]}\"; its contacts are left unset.");
                continue;
            }

            var flags = measured[sourceSlots[slot]];
            for (var i = 0; i < frames.Count && i < flags.Length; i++)
            {
                frames[i].SetBool(contacts[slot], flags[i]);
            }
        }
    }

    /// <summary>
    /// For each contact slot, the slot whose measurement it takes: itself for an unmirrored clip, and
    /// the slot of its bone's counterpart for a mirrored one, since the mirror image plants each bone
    /// where the original planted its counterpart. -1 when the counterpart has no slot.
    /// </summary>
    public static int[] MirrorSourceSlots(ContactHandles contacts, PoseMirror mirror)
    {
        var sourceSlots = new int[contacts.Count];
        for (var slot = 0; slot < sourceSlots.Length; slot++)
        {
            if (mirror == null)
            {
                sourceSlots[slot] = slot;
                continue;
            }

            sourceSlots[slot] = contacts.TryGetSlot(mirror.Counterpart(contacts.GetBoneIndex(slot)), out var source)
                ? source
                : -1;
        }

        return sourceSlots;
    }

    /// <summary>
    /// Writes the clip's gait phase from its <see cref="GaitPhaseComponent"/>. A clip with none keeps
    /// a phase of zero at a rate of zero: the "no measurable cycle" sentinel that keeps those frames
    /// out of training.
    /// </summary>
    private static void WritePhase(AnnotatedAnimationClip animationClip, PoseSet poseSet,
        PoseSet.PoseFrameRange frames, bool mirrored)
    {
        var gait = animationClip.GetComponent<GaitPhaseComponent>();
        if (gait == null) return;

        // Anchors are numbered against the whole clip, so database frame i of this clip reads clip
        // frame startFrame + i.
        gait.Evaluate(animationClip, mirrored, out var phase, out var phaseRate);
        for (var i = 0; i < frames.Count; i++)
        {
            var clipFrame = animationClip.startFrame + i;
            if (clipFrame >= phase.Length) break;
            poseSet.SetPhase(frames.Start + i, phase[clipFrame], phaseRate[clipFrame]);
        }
    }

    /// <summary>
    /// A pose is stored exactly as the clip bakes it: bone 0's world position and rotation, rest
    /// offsets and parent-local rotations below it. The character frame is derived from the stored
    /// pose on demand — see <see cref="SimulationFrame"/> — so nothing is reparented here.
    /// </summary>
    private static void ExtractPose(PoseBuffer pose, AnnotatedAnimationClip animationClip, int frameIndex,
        PoseMirror mirror)
    {
        var frame = animationClip.GetFrame(frameIndex);
        var framePositions = frame.Positions;
        var frameRotations = frame.Rotations;
        var posePositions = pose.Positions;
        var poseRotations = pose.Rotations;

        for (var i = 0; i < posePositions.Length; i++)
        {
            posePositions[i] = framePositions[i];
            poseRotations[i] = frameRotations[i];
        }

        mirror?.Mirror(pose, pose);
    }

    private static void ExtractPoseVelocities(PoseBuffer pose, PoseBuffer nextPose, AnnotatedAnimationClip animationClip)
    {
        var positions = pose.Positions;
        var rotations = pose.Rotations;
        var velocities = pose.Velocities;
        var angularVelocities = pose.AngularVelocities;
        var nextPositions = nextPose.Positions;
        var nextRotations = nextPose.Rotations;

        for (var jointIdx = 0; jointIdx < positions.Length; jointIdx++)
        {
            var nextPos = nextPositions[jointIdx];
            var pos = positions[jointIdx];
            velocities[jointIdx] = (nextPos - pos) / animationClip.FrameTime;

            var nextRot = nextRotations[jointIdx];
            var rot = rotations[jointIdx];
            angularVelocities[jointIdx] =
                MathExtensions.AngularVelocity(rot, nextRot, animationClip.FrameTime);
        }
    }
}
}
