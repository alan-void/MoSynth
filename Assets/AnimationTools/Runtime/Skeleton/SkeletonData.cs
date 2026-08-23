using System;
using System.Diagnostics.Contracts;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Unmanaged, Burst-compatible mirror of a <see cref="AnimationTools.Skeleton"/>'s bone hierarchy.
/// Arrays are indexed in the same depth-first order as the source skeleton, so
/// <c>ParentIndices[i] &lt; i</c> for every bone and index 0 is always the root.
/// Obtained via <see cref="AnimationTools.Skeleton.GetSkeletonData"/>; do not Dispose it, the
/// skeleton owns the backing arrays.
/// </summary>
public struct SkeletonData
{
    public NativeArray<int> ParentIndices;
    public NativeArray<float3> RestLocalPositions;
    public NativeArray<quaternion> RestLocalRotations;
    public int BoneCount;

    public bool IsCreated => ParentIndices.IsCreated;


    /// <summary>
    /// Computes character-space position and rotation for every bone in one forward pass,
    /// exploiting the DFS invariant <c>parentIndex &lt; index</c> so each bone's parent is
    /// already resolved by the time it's processed.
    /// </summary>
    public void LocalSpaceToCharacterSpace(PoseBuffer pose,
        NativeArray<float3> outPositions, NativeArray<quaternion> outRotations)
    {
        CheckFullPoseLayout(pose.Layout, BoneCount);
        Debug.Assert(outPositions.Length == BoneCount && outRotations.Length == BoneCount,
            "Output arrays must have one element per bone.");

        var localPositions = pose.Positions;
        var localRotations = pose.Rotations;

        outPositions[0] = localPositions[0];
        outRotations[0] = localRotations[0];

        for (var i = 1; i < BoneCount; i++)
        {
            var parent = ParentIndices[i];
            outRotations[i] = math.mul(outRotations[parent], localRotations[i]);
            outPositions[i] = outPositions[parent] + math.rotate(outRotations[parent], localPositions[i]);
        }
    }

    /// <summary>Character-space position of one bone, found by walking up its parent chain.</summary>
    [Pure]
    public float3 CharacterSpacePosition(PoseBuffer pose, int boneIndex)
    {
        CheckFullPoseLayout(pose.Layout, BoneCount);

        var localPositions = pose.Positions;
        var localRotations = pose.Rotations;

        var position = float3.zero;

        while (boneIndex != 0)
        {
            position = localPositions[boneIndex] + math.rotate(localRotations[boneIndex], position);
            boneIndex = ParentIndices[boneIndex];
        }

        return localPositions[0] + math.rotate(localRotations[0], position);
    }

    /// <summary>Character-space rotation of one bone, found by walking up its parent chain.</summary>
    [Pure]
    public quaternion CharacterSpaceRotation(PoseBuffer pose, int boneIndex)
    {
        CheckFullPoseLayout(pose.Layout, BoneCount);

        var localRotations = pose.Rotations;
        var rotation = quaternion.identity;

        while (boneIndex != 0)
        {
            rotation = math.mul(localRotations[boneIndex], rotation);
            boneIndex = ParentIndices[boneIndex];
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
    /// Bone 0's velocity is stored in the same space as its position and rotation — character
    /// space, the pose as-authored — because extraction takes plain finite differences of those
    /// channels. It is therefore added here unrotated, while every bone below it contributes
    /// through its parent's rotation.
    /// </remarks>
    [Pure]
    public float3 CharacterSpaceVelocity(in PoseBuffer pose, int boneIndex)
    {
        CheckFullPoseLayout(pose.Layout, BoneCount);
        Debug.Assert(pose.Layout.VelocityCount == BoneCount && pose.Layout.AngularVelocityCount == BoneCount,
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

            boneIndex = ParentIndices[boneIndex];
        }

        var rootQ = localRotations[0];
        var rootV = localVelocities[0];
        var rootW = localAngularVelocities[0];

        var rootRotatedPosAcc = math.rotate(rootQ, posAcc);
        linVelAcc = rootV + math.cross(rootW, rootRotatedPosAcc) + math.rotate(rootQ, linVelAcc);

        return linVelAcc;
    }

    /// <summary>
    /// Character-space angular velocity of one bone: every ancestor's local rate carried into
    /// character space by the rotations above it, summed along the parent chain. Bone 0's own
    /// channel is already character-space, so a query for bone 0 returns it unchanged.
    /// </summary>
    [Pure]
    public float3 CharacterSpaceAngularVelocity(in PoseBuffer pose, int boneIndex)
    {
        CheckFullPoseLayout(pose.Layout, BoneCount);
        Debug.Assert(pose.Layout.AngularVelocityCount == BoneCount,
            "CharacterAngularVelocity requires one AngularVelocity channel per bone.");

        var localRotations = pose.Rotations;
        var localAngularVelocities = pose.AngularVelocities;

        var angVelAcc = float3.zero;

        while (boneIndex != 0)
        {
            angVelAcc = localAngularVelocities[boneIndex] + math.rotate(localRotations[boneIndex], angVelAcc);
            boneIndex = ParentIndices[boneIndex];
        }

        return localAngularVelocities[0] + math.rotate(localRotations[0], angVelAcc);
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