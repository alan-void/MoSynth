using UnityEngine;
using Unity.Collections;

namespace AnimationTools
{
/// <summary>
/// Extracts full poses, their rates, foot contacts and gait phase from an annotated clip into a
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

        WriteContactsAndPhase(animationClip, poseSet, frames, source, mirror != null);

        return true;
    }

    /// <summary>
    /// Writes the clip's foot contacts and gait phase into the database frames it just filled.
    /// </summary>
    /// <remarks>
    /// Both come from the clip's <see cref="GaitPhaseComponent"/> when it has one, measured by
    /// <see cref="GaitMeasure"/> — so what a database records is what the clip editor drew when the
    /// footfalls were corrected. A clip with no component still gets contacts, measured the same way
    /// against the config's threshold, and a phase of zero at a rate of zero: the "no measurable
    /// cycle" sentinel that keeps those frames out of training.
    /// </remarks>
    private static void WriteContactsAndPhase(AnnotatedAnimationClip animationClip, PoseSet poseSet,
        PoseSet.PoseFrameRange frames, IPoseSetSource source, bool mirrored)
    {
        var gait = animationClip.GetComponent<GaitPhaseComponent>();

        bool[] contacts;
        if (gait == null)
        {
            contacts = ConfiguredContacts(animationClip, source);
        }
        else if (!gait.TryDetectSliceContacts(animationClip, out contacts, out var error))
        {
            Debug.LogWarning($"Clip \"{animationClip.name}\": {error} No foot contacts written.");
            contacts = null;
        }

        // A mirrored clip plants its left foot where the original planted its right.
        var leftOffset = mirrored ? 1 : 0;
        var rightOffset = 1 - leftOffset;
        for (var i = 0; i < frames.Count; i++)
        {
            var frame = frames[i];
            frame.SetBool(poseSet.LeftFootContactHandle, contacts != null && contacts[i * 2 + leftOffset]);
            frame.SetBool(poseSet.RightFootContactHandle, contacts != null && contacts[i * 2 + rightOffset]);
        }

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

    /// <summary>Contacts for a clip with no <see cref="GaitPhaseComponent"/> to take settings from.</summary>
    private static bool[] ConfiguredContacts(AnnotatedAnimationClip animationClip, IPoseSetSource source)
    {
        var skeleton = animationClip.Skeleton;
        return GaitMeasure.Contacts(animationClip, skeleton,
            ResolveContactBoneIndex(skeleton, source.LeftContactBoneName, true),
            ResolveContactBoneIndex(skeleton, source.RightContactBoneName, false),
            animationClip.startFrame, animationClip.FrameCount,
            source.ContactVelocityThreshold, GaitMeasure.DefaultSmoothingRadius);
    }

    /// <summary>
    /// Resolves the contact-detection bone for one side: an explicit <paramref name="boneName"/>
    /// wins if configured, falling back to <see cref="BoneNameConventions.TryFindContactBone"/>
    /// (and finally index 0) when unset or unresolvable.
    /// </summary>
    private static int ResolveContactBoneIndex(Skeleton skeleton, string boneName, bool left)
    {
        if (!string.IsNullOrEmpty(boneName))
        {
            if (skeleton.TryFindByName(boneName, out var index)) return index;
            Debug.LogError($"Configured contact bone \"{boneName}\" not found in BVHAnimation; falling back to the name heuristic.");
        }

        if (BoneNameConventions.TryFindContactBone(skeleton, left, out var heuristicIndex)) return heuristicIndex;

        Debug.LogError($"{(left ? "Left" : "Right")}Toes not found in BVHAnimation");
        return 0;
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
