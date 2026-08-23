using System;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Forward kinematics over a full-pose <see cref="PoseBuffer"/> (one Position + one
/// Rotation channel per bone, element index == bone index — see
/// <see cref="PoseLayout.CreateFullPose"/>). "Character space" is the frame of the root
/// bone's parent, i.e. the pose as-authored, with no world placement applied.
/// </summary>
/// <remarks>
/// Plain static methods over structs/NativeArrays so they stay Burst-compatible; no jobs
/// are scheduled here, callers wrap these in jobs as needed.
/// </remarks>
public static class PoseFK
{
    /// <summary>
    /// Computes character-space position and rotation for every bone in one forward pass,
    /// exploiting the DFS invariant <c>parentIndex &lt; index</c> so each bone's parent is
    /// already resolved by the time it's processed.
    /// </summary>
    public static void LocalToCharacter(in PoseBuffer pose, in SkeletonData skeleton,
        NativeArray<float3> outPositions, NativeArray<quaternion> outRotations)
    {
        CheckFullPoseLayout(pose.Layout, skeleton.BoneCount);
        Debug.Assert(outPositions.Length == skeleton.BoneCount && outRotations.Length == skeleton.BoneCount,
            "Output arrays must have one element per bone.");

        var localPositions = pose.Positions;
        var localRotations = pose.Rotations;

        outPositions[0] = localPositions[0];
        outRotations[0] = localRotations[0];

        for (var i = 1; i < skeleton.BoneCount; i++)
        {
            var parent = skeleton.ParentIndices[i];
            outRotations[i] = math.mul(outRotations[parent], localRotations[i]);
            outPositions[i] = outPositions[parent] + math.rotate(outRotations[parent], localPositions[i]);
        }
    }

    /// <summary>Character-space position of one bone, found by walking up its parent chain.</summary>
    public static float3 CharacterPosition(in PoseBuffer pose, in SkeletonData skeleton, int boneIndex)
    {
        CheckFullPoseLayout(pose.Layout, skeleton.BoneCount);

        var localPositions = pose.Positions;
        var localRotations = pose.Rotations;

        var position = float3.zero;

        while (boneIndex != 0)
        {
            position = localPositions[boneIndex] + math.rotate(localRotations[boneIndex], position);
            boneIndex = skeleton.ParentIndices[boneIndex];
        }

        return localPositions[0] + math.rotate(localRotations[0], position);
    }

    /// <summary>Character-space rotation of one bone, found by walking up its parent chain.</summary>
    public static quaternion CharacterRotation(in PoseBuffer pose, in SkeletonData skeleton, int boneIndex)
    {
        CheckFullPoseLayout(pose.Layout, skeleton.BoneCount);

        var localRotations = pose.Rotations;
        var rotation = quaternion.identity;

        while (boneIndex != 0)
        {
            rotation = math.mul(localRotations[boneIndex], rotation);
            boneIndex = skeleton.ParentIndices[boneIndex];
        }

        return math.mul(localRotations[0], rotation);
    }

    /// <summary>
    /// Character-space linear velocity of one bone. Ports the spatial-velocity transfer
    /// from <c>MotionMatching.Skeleton.GetWorldSpaceVelocity</c>: at each ancestor,
    /// <c>V_child = V_parent + cross(W_parent, rotate(R_parent, localPos)) + rotate(R_parent, v_local)</c>,
    /// accumulated from the target bone up to the root.
    /// </summary>
    /// <remarks>
    /// Legacy quirk, deliberately preserved: bone-0 (root) velocity is added WITHOUT rotating
    /// by the root rotation, exactly matching <c>Skeleton.GetWorldSpaceVelocity</c> (Skeleton.cs:170)
    /// even though the motion-matching pipeline stores that value in root-local space
    /// (<c>PoseExtractor.cs:254-259</c> rotates it). Fidelity to the legacy contact extraction
    /// takes precedence; revisit if the legacy convention is ever fixed.
    /// </remarks>
    public static float3 CharacterVelocity(in PoseBuffer pose, in SkeletonData skeleton, int boneIndex)
    {
        CheckFullPoseLayout(pose.Layout, skeleton.BoneCount);
        Debug.Assert(pose.Layout.VelocityCount == skeleton.BoneCount && pose.Layout.AngularVelocityCount == skeleton.BoneCount,
            "CharacterVelocity requires one Velocity and one AngularVelocity channel per bone.");

        var localPositions = pose.Positions;
        var localRotations = pose.Rotations;
        var localVelocities = pose.Velocities;
        var localAngularVelocities = pose.AngularVelocities;

        var posAcc = float3.zero;
        var linVelAcc = float3.zero;

        while (boneIndex != 0)
        {
            var p = localPositions[boneIndex];
            var q = localRotations[boneIndex];
            var v = localVelocities[boneIndex];
            var w = localAngularVelocities[boneIndex];

            var rotatedPosAcc = math.rotate(q, posAcc);

            linVelAcc = v + math.cross(w, rotatedPosAcc) + math.rotate(q, linVelAcc);
            posAcc = p + rotatedPosAcc;

            boneIndex = skeleton.ParentIndices[boneIndex];
        }

        var rootQ = localRotations[0];
        var rootV = localVelocities[0];
        var rootW = localAngularVelocities[0];

        var rootRotatedPosAcc = math.rotate(rootQ, posAcc);
        linVelAcc = rootV + math.cross(rootW, rootRotatedPosAcc) + math.rotate(rootQ, linVelAcc);

        return linVelAcc;
    }

    /// <summary>
    /// A pose and a skeleton that disagree on bone count is a caller bug, not bad data: every loop
    /// in this class is bounded by the skeleton but indexes the pose, so the mismatch surfaces as an
    /// <see cref="IndexOutOfRangeException"/> from deep inside <see cref="NativeSlice{T}"/> with
    /// nothing in it to say which two things disagreed. Hence a throw rather than an assert, which
    /// would log and then run off the end anyway.
    /// </summary>
    /// <remarks>
    /// The interpolated message costs the Burst-compatibility claimed above: Burst cannot compile
    /// string formatting. Nothing Burst-compiles PoseFK today — the project's [BurstCompile] code
    /// all lives under Assets/MotionMatching/Runtime/Core/Burst/ and none of it calls in here — so
    /// this is a real but unexercised tradeoff. Revisit if these ever move into a job.
    /// </remarks>
    private static void CheckFullPoseLayout(in PoseLayoutData layout, int boneCount)
    {
        if (layout.PositionCount != boneCount || layout.RotationCount != boneCount)
        {
            throw new ArgumentException(
                $"Pose has {layout.PositionCount} Position and {layout.RotationCount} Rotation " +
                $"channels but the skeleton has {boneCount} bones; they describe different rigs.",
                "pose");
        }

        Debug.Assert(layout.RotationStride == 4, "PoseFK requires Quaternion rotations.");
        Debug.Assert(layout.ScaleCount == 0, "PoseFK v1 assumes unit scale — Scale channels are not supported.");
    }
}
}
