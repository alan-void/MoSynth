using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using System;
using AnimationTools;

namespace AnimationTools
{
/// <summary>
/// Extracts full pose for Motion Matching from BVHAnimation
/// </summary>
public static class PoseExtractor
{
    /// <summary>
    /// Extract the poses from bvhAnimation and store it in poseSet
    /// poseSet is not cleared, it will add bvhAnimation the the existing poses
    /// Returns true if the bvhAnimation was added to the poseSet, false otherwise
    /// </summary>
    public static bool Extract(AnnotatedAnimationClip animationClip, PoseSet poseSet, IPoseSetSource source)
    {
        var clipSkeleton = animationClip.Skeleton;
        if (clipSkeleton == null)
        {
            Debug.LogError($"Clip \"{animationClip.name}\" has no resolvable skeleton; check its rig and root bone.");
            return false;
        }

        // The pose skeleton carries a SimulationBone at index 0 that the clip's own skeleton
        // does not; everything from index 1 on must still line up bone-for-bone with the clip.
        var poseSkeleton = poseSet.Skeleton;
        if (poseSkeleton == null || !poseSkeleton.MatchesFrom(1, clipSkeleton))
        {
            Debug.LogError($"Skeleton of clip \"{animationClip.name}\" ({clipSkeleton.BoneCount} bones) " +
                $"does not match the pose set's skeleton from index 1 ({poseSkeleton?.BoneCount ?? 0} bones).");
            return false;
        }

        // Set Poses
        var nFrames = animationClip.FrameCount;
        var nBvhJoints = animationClip.Skeleton.BoneCount;

        var leftToesBoneIndex = ResolveContactBoneIndex(animationClip.Skeleton, source.LeftContactBoneName, true);
        var leftToesIndex = leftToesBoneIndex + 1; // +1 for SimulationBone
        var rightToesBoneIndex = ResolveContactBoneIndex(animationClip.Skeleton, source.RightContactBoneName, false);
        var rightToesIndex = rightToesBoneIndex + 1; // +1 for SimulationBone

        // Skeleton bone 0 is the hips. Which of its local axes points along the character's facing is
        // a property of the rig's bone roll, so read it from the rest pose instead of assuming an axis.
        var hipsForwardLocal = animationClip.Skeleton.RestLocalAxis(0, math.forward());

        // Rest local positions, hoisted once so the per-frame loop below never touches a
        // managed Transform property.
        var restLocalPositions = new float3[nBvhJoints];
        for (var i = 0; i < nBvhJoints; i++)
        {
            restLocalPositions[i] = animationClip.Skeleton.GetBone(i).RestLocalPosition;
        }

        var frames = poseSet.BeginClip(nFrames - 1, animationClip.FrameTime);

        for (var i = 0; i < nFrames - 1; i++)
        {
            ExtractPose(frames[i], animationClip, i, hipsForwardLocal, restLocalPositions);
        }

        for (var i = 0; i < nFrames - 2; i++)
        {
            ExtractPoseVelocities(frames[i], frames[i + 1], animationClip);
        }

        var lastPose = PoseBuffer.Allocate(poseSet.PoseLayout, Allocator.Temp);
        try
        {
            ExtractPose(lastPose, animationClip, nFrames - 1, hipsForwardLocal, restLocalPositions);
            ExtractPoseVelocities(frames[frames.Count - 1], lastPose, animationClip);
        }
        finally
        {
            lastPose.Dispose();
        }

        var poseSkeletonData = poseSkeleton.GetSkeletonData();
        for (var i = 0; i < nFrames - 1; i++)
        {
            // Note: this requires velocities to be pre-calculated
            ExtractPoseContacts(frames[i],
                poseSkeletonData,
                leftToesIndex,
                rightToesIndex,
                source.ContactVelocityThreshold,
                poseSet.LeftFootContactHandle,
                poseSet.RightFootContactHandle);
        }

        SmoothContacts(frames, poseSet.LeftFootContactHandle, poseSet.RightFootContactHandle);

        return true;
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
        return 0; // legacy TryFind left a default joint (index 0) on failure
    }

    private static void SmoothContacts(PoseSet.PoseFrameRange frames, ChannelHandle leftHandle, ChannelHandle rightHandle)
    {
        const int windowsRadius = 6;
        // Median filter to remove small regions where contact is either active or inactive
        var count = frames.Count;
        var leftFootContact = new bool[count];
        var rightFootContact = new bool[count];
        for (var i = 0; i < count; i++)
        {
            var frame = frames[i];
            leftFootContact[i] = frame.GetBool(leftHandle);
            rightFootContact[i] = frame.GetBool(rightHandle);
        }

        // Median Filter
        Span<bool> leftFootContactWindow = stackalloc bool[windowsRadius * 2 + 1];
        Span<bool> rightFootContactWindow = stackalloc bool[windowsRadius * 2 + 1];
        for (var i = 0; i < count; i++)
        {
            var windowIndex = 0;
            for (var j = -windowsRadius; j <= windowsRadius; j++)
            {
                var index = i + j;
                if (index < 0)
                {
                    leftFootContactWindow[windowIndex] = leftFootContact[0];
                    rightFootContactWindow[windowIndex] = rightFootContact[0];
                }
                else if (index >= count)
                {
                    leftFootContactWindow[windowIndex] = leftFootContact[count - 1];
                    rightFootContactWindow[windowIndex] = rightFootContact[count - 1];
                }
                else
                {
                    leftFootContactWindow[windowIndex] = leftFootContact[index];
                    rightFootContactWindow[windowIndex] = rightFootContact[index];
                }

                windowIndex += 1;
            }

            // Sort
            var lastFalseIndex = 0;
            for (var j = 0; j < windowsRadius * 2 + 1; j++)
            {
                if (!leftFootContactWindow[j])
                {
                    var aux = leftFootContactWindow[lastFalseIndex];
                    leftFootContactWindow[lastFalseIndex] = false;
                    leftFootContactWindow[j] = aux;
                    lastFalseIndex += 1;
                }
            }

            lastFalseIndex = 0;
            for (var j = 0; j < windowsRadius * 2 + 1; j++)
            {
                if (!rightFootContactWindow[j])
                {
                    var aux = rightFootContactWindow[lastFalseIndex];
                    rightFootContactWindow[lastFalseIndex] = false;
                    rightFootContactWindow[j] = aux;
                    lastFalseIndex += 1;
                }
            }

            // Find median
            var medianIndex = windowsRadius;
            var frame = frames[i];
            frame.SetBool(leftHandle, leftFootContactWindow[medianIndex]);
            frame.SetBool(rightHandle, rightFootContactWindow[medianIndex]);
        }
    }

    private static void ExtractPose(PoseBuffer pose, AnnotatedAnimationClip animationClip, int frameIndex,
        float3 hipsForwardLocal, float3[] restLocalPositions)
    {
        var frame = animationClip.GetFrame(frameIndex);
        var rotations = frame.Rotations;
        var posePositions = pose.Positions;
        var poseRotations = pose.Rotations;

        // Joints
        for (var i = 1; i < posePositions.Length; i++)
        {
            // rest offsets come from the skeleton, not the frame's positions section (slot 0 there is root motion)
            posePositions[i] = restLocalPositions[i - 1];
            poseRotations[i] = rotations[i - 1];
        }

        // SimulationBone
        // position and direction are hips projected on the ground
        Vector3 frameRootMotion = frame.Positions[0];
        var sbPos = new Vector3(frameRootMotion.x, 0.0f, frameRootMotion.z);
        var hipsForwardDir = (Quaternion)rotations[0] * (Vector3)hipsForwardLocal;
        hipsForwardDir.y = 0;
        hipsForwardDir = hipsForwardDir.normalized;
        var sbRot = Quaternion.LookRotation(hipsForwardDir, Vector3.up);
        posePositions[0] = sbPos;
        poseRotations[0] = sbRot;

        // make first joint (hips) position and direction relative to the simulation bone
        var inverseSbRot = math.inverse(sbRot);
        posePositions[1] = math.mul(inverseSbRot, (float3)(frameRootMotion - sbPos));
        poseRotations[1] = math.mul(inverseSbRot, (quaternion)rotations[0]);
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

        // root motion
        var vel = (nextPositions[0] - positions[0]) / animationClip.FrameTime;

        // transform the root velocity so that it is
        // with respect to the root space of the current pose
        vel = math.mul(math.inverse(rotations[0]), vel);
        velocities[0] = vel;
    }

    private static void ExtractPoseContacts(PoseBuffer pose, in SkeletonData skeleton, int leftToesIndex,
        int rightToesIndex, float contactVelocityThreshold, ChannelHandle leftHandle, ChannelHandle rightHandle)
    {
        // Contact with the ground when the joint is below a velocity threshold
        // TODO: Consider distance from the ground/contact when the joint is below a velocity threshold
        var leftToeVel = PoseFK.CharacterVelocity(pose, skeleton, leftToesIndex);
        var rightToeVel = PoseFK.CharacterVelocity(pose, skeleton, rightToesIndex);

        pose.SetBool(leftHandle, math.length(leftToeVel) < contactVelocityThreshold);
        pose.SetBool(rightHandle, math.length(rightToeVel) < contactVelocityThreshold);
    }
}
}
