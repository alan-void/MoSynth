using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Every bone of a pose expressed in the character frame that pose implies — where a network sees
/// a pose, as opposed to how one is stored.
/// </summary>
/// <remarks>
/// A pose is stored in its clip's own space: bone 0 carries a world position and rotation, and the
/// bones below it carry rest offsets and parent-local rotations. A model must not see any of that.
/// Where the character stands and which way it faces are exactly what a locomotion model has to be
/// invariant to, so both are removed by measuring the pose inside its own ground-projected,
/// yaw-only frame — see <see cref="SimulationFrame"/>.
/// <para>
/// This is the C# counterpart of what <c>Python/training/training_data.py</c> produces, and it exists so
/// that a stage running a trained model feeds it the same quantities the model was trained on.
/// That agreement is the failure this class is here to prevent: a mismatch in frame, units or
/// rate convention does not throw, it just makes the network wrong.
/// </para>
/// <para>
/// The rates are the one place the two sides are not identical by construction. Python differences
/// consecutive frame-local poses of a stored database; this composes the instantaneous rate implied
/// by the pose's own velocity channels. Since those channels are themselves finite differences over
/// the same timestep, the two agree to first order, and exactly for motion that is rigid within the
/// frame.
/// </para>
/// </remarks>
public static class CharacterSpacePose
{
    /// <summary>
    /// Fills the output arrays with every bone's state in the character frame of
    /// <paramref name="pose"/>.
    /// </summary>
    /// <param name="pose">A full pose over <paramref name="skeleton"/>.</param>
    /// <param name="skeleton">The bone hierarchy, in the pose's own depth-first order.</param>
    /// <param name="def">Which bone the character frame is read from, and its forward axis.</param>
    /// <param name="deltaTime">
    /// The timestep the pose's velocity channels were differenced over. Only read when rates are
    /// requested, and only to turn the frame's own angular velocity into a yaw rate.
    /// </param>
    /// <param name="positions">One frame-local position per bone. Required.</param>
    /// <param name="rotations">One frame-local rotation per bone. Required.</param>
    /// <param name="velocities">
    /// One frame-local linear rate per bone, in metres per second, or an uncreated array to skip
    /// the rate pass entirely. This is motion <em>relative to the moving frame</em>: a character
    /// carried along rigidly by its own frame has none.
    /// </param>
    /// <param name="angularVelocities">
    /// One frame-local angular rate per bone, as a rotation vector in radians per second. Uncreated
    /// unless <paramref name="velocities"/> is too.
    /// </param>
    public static void Extract(PoseBuffer pose, in SkeletonData skeleton, in SimulationFrameDef def,
        float deltaTime, NativeArray<float3> positions, NativeArray<quaternion> rotations,
        NativeArray<float3> velocities = default, NativeArray<float3> angularVelocities = default)
    {
        var boneCount = skeleton.BoneCount;
        Debug.Assert(positions.Length == boneCount && rotations.Length == boneCount,
            "Position and rotation arrays must have one element per bone.");

        var wantRates = velocities.IsCreated && angularVelocities.IsCreated;
        Debug.Assert(velocities.IsCreated == angularVelocities.IsCreated,
            "Linear and angular rates are extracted together or not at all.");
        Debug.Assert(!wantRates ||
                     (velocities.Length == boneCount && angularVelocities.Length == boneCount),
            "Rate arrays must have one element per bone.");

        // The outputs hold world values through the first pass and are rewritten in place by the
        // second, which is what keeps this allocation-free.
        skeleton.LocalSpaceToCharacterSpace(pose, positions, rotations);
        if (wantRates) WorldRates(pose, skeleton, rotations, velocities, angularVelocities);

        SimulationFrame.Compute(pose, skeleton, def, out var framePosition, out var frameRotation);

        var frameLinearVelocity = float3.zero;
        var frameYawRate = 0f;
        if (wantRates)
        {
            SimulationFrame.ComputeVelocity(pose, skeleton, def, deltaTime,
                out frameLinearVelocity, out frameYawRate);
        }

        for (var i = 0; i < boneCount; i++)
        {
            var worldPosition = positions[i];

            if (wantRates)
            {
                // The same split the root gets: what the frame already carries, and what is left.
                SimulationFrame.DecomposeRootVelocity(framePosition, frameRotation, frameLinearVelocity,
                    frameYawRate, worldPosition, velocities[i], angularVelocities[i],
                    out var linear, out var angular);
                velocities[i] = linear;
                angularVelocities[i] = angular;
            }

            positions[i] = SimulationFrame.ToFrameLocal(worldPosition, framePosition, frameRotation);
            rotations[i] = SimulationFrame.ToFrameLocal(rotations[i], frameRotation);
        }
    }

    /// <summary>
    /// Writes a pose given in a character frame back into <paramref name="pose"/> — the inverse of
    /// <see cref="Extract"/>.
    /// </summary>
    /// <remarks>
    /// This is how a stage running a learned model returns its prediction. The model works in the
    /// character frame and knows nothing about the storage convention, so the conversion has to
    /// live in one named place rather than being open-coded per stage: two stages writing a pose
    /// slightly differently is the same silent-disagreement failure <see cref="Extract"/> exists to
    /// prevent, only on the way out.
    /// <para>
    /// Bones below the root keep their rest offsets, because only their rotations are written. A
    /// predicted pose therefore cannot stretch a bone, which a prediction made directly in
    /// positions can and does.
    /// </para>
    /// <para>
    /// The frame is a parameter rather than something read back off the pose: the caller is
    /// deciding where the character has moved to, and the frame's own velocity written into bone 0
    /// is what <see cref="MotionSynthesisComponent"/> reads to advance the transform.
    /// </para>
    /// </remarks>
    /// <param name="pose">Destination. Must use the full pose layout over <paramref name="skeleton"/>.</param>
    /// <param name="skeleton">The bone hierarchy, in the pose's own depth-first order.</param>
    /// <param name="framePosition">Where the character frame sits, in world space.</param>
    /// <param name="frameRotation">The frame's ground-projected, yaw-only rotation.</param>
    /// <param name="frameLinearVelocity">The frame's own travel, in frame space, metres per second.</param>
    /// <param name="frameYawRate">The frame's own turn, radians per second.</param>
    /// <param name="positions">
    /// In: one frame-local position per bone. Out: the same array holding world positions — it is
    /// rewritten in place, as <see cref="Extract"/>'s outputs are, so neither direction allocates.
    /// </param>
    /// <param name="rotations">As <paramref name="positions"/>, for rotations.</param>
    /// <param name="velocities">
    /// One frame-local linear rate per bone, or an uncreated array to leave the pose's rate
    /// channels alone. Rewritten in place like the others.
    /// </param>
    /// <param name="angularVelocities">
    /// As <paramref name="velocities"/>. Uncreated unless <paramref name="velocities"/> is too.
    /// </param>
    public static void Apply(PoseBuffer pose, in SkeletonData skeleton,
        float3 framePosition, quaternion frameRotation, float3 frameLinearVelocity,
        float frameYawRate, NativeArray<float3> positions, NativeArray<quaternion> rotations,
        NativeArray<float3> velocities = default, NativeArray<float3> angularVelocities = default)
    {
        var boneCount = skeleton.BoneCount;
        Debug.Assert(positions.Length == boneCount && rotations.Length == boneCount,
            "Position and rotation arrays must have one element per bone.");

        var wantRates = velocities.IsCreated && angularVelocities.IsCreated;
        Debug.Assert(velocities.IsCreated == angularVelocities.IsCreated,
            "Linear and angular rates are applied together or not at all.");
        Debug.Assert(!wantRates ||
                     (velocities.Length == boneCount && angularVelocities.Length == boneCount),
            "Rate arrays must have one element per bone.");

        for (var i = 0; i < boneCount; i++)
        {
            var worldPosition = SimulationFrame.FromFrameLocal(positions[i], framePosition, frameRotation);
            positions[i] = worldPosition;
            rotations[i] = SimulationFrame.FromFrameLocal(rotations[i], frameRotation);

            if (!wantRates) continue;

            // The exact inverse of the split Extract takes: give the frame's own motion back.
            SimulationFrame.RecomposeRootVelocity(framePosition, frameRotation, frameLinearVelocity,
                frameYawRate, worldPosition, velocities[i], angularVelocities[i],
                out var linear, out var angular);
            velocities[i] = linear;
            angularVelocities[i] = angular;
        }

        WriteLocalPose(pose, skeleton, positions, rotations);
        if (wantRates) WriteLocalRates(pose, skeleton, rotations, velocities, angularVelocities);
    }

    /// <summary>
    /// Turns world positions and rotations into what a pose stores: bone 0 in world space, every
    /// other bone its rest offset and a parent-local rotation.
    /// </summary>
    private static void WriteLocalPose(PoseBuffer pose, in SkeletonData skeleton,
        NativeArray<float3> worldPositions, NativeArray<quaternion> worldRotations)
    {
        var localPositions = pose.Positions;
        var localRotations = pose.Rotations;

        localPositions[0] = worldPositions[0];
        localRotations[0] = worldRotations[0];

        for (var i = 1; i < skeleton.BoneCount; i++)
        {
            var parent = skeleton.ParentIndices[i];
            localRotations[i] = math.mul(math.inverse(worldRotations[parent]), worldRotations[i]);
            localPositions[i] = skeleton.RestLocalPositions[i];
        }
    }

    /// <summary>
    /// The backward pass matching <see cref="WorldRates"/>: undoes the rigid-body transfer bone by
    /// bone, each one's parent already resolved because bones are stored depth-first.
    /// </summary>
    private static void WriteLocalRates(PoseBuffer pose, in SkeletonData skeleton,
        NativeArray<quaternion> worldRotations, NativeArray<float3> velocities,
        NativeArray<float3> angularVelocities)
    {
        var localVelocities = pose.Velocities;
        var localAngularVelocities = pose.AngularVelocities;

        // Bone 0's channels are world, which is what Extract reads them as.
        localVelocities[0] = velocities[0];
        localAngularVelocities[0] = angularVelocities[0];

        for (var i = 1; i < skeleton.BoneCount; i++)
        {
            var parent = skeleton.ParentIndices[i];
            var parentRotation = worldRotations[parent];
            var inverseParentRotation = math.inverse(parentRotation);
            var offset = math.rotate(parentRotation, skeleton.RestLocalPositions[i]);

            localAngularVelocities[i] = math.rotate(inverseParentRotation,
                angularVelocities[i] - angularVelocities[parent]);
            localVelocities[i] = math.rotate(inverseParentRotation,
                velocities[i] - velocities[parent] - math.cross(angularVelocities[parent], offset));
        }
    }

    /// <summary>
    /// World linear and angular rate of every bone, in one forward pass over the hierarchy.
    /// </summary>
    /// <remarks>
    /// The rigid-body transfer, with each bone's parent already resolved because bones are stored
    /// depth-first: <c>w(i) = w(p) + R(p)·wl(i)</c> and
    /// <c>v(i) = v(p) + w(p)×(R(p)·l(i)) + R(p)·vl(i)</c>. Equivalent to
    /// <see cref="SkeletonData.CharacterSpaceVelocity"/> walking each bone's chain separately, but
    /// without redoing the shared prefix once per leaf.
    /// </remarks>
    private static void WorldRates(PoseBuffer pose, in SkeletonData skeleton,
        NativeArray<quaternion> worldRotations, NativeArray<float3> velocities,
        NativeArray<float3> angularVelocities)
    {
        var localPositions = pose.Positions;
        var localVelocities = pose.Velocities;
        var localAngularVelocities = pose.AngularVelocities;

        // Bone 0's channels are already world: extraction takes plain finite differences of its
        // world position and rotation.
        velocities[0] = localVelocities[0];
        angularVelocities[0] = localAngularVelocities[0];

        for (var i = 1; i < skeleton.BoneCount; i++)
        {
            var parent = skeleton.ParentIndices[i];
            var parentRotation = worldRotations[parent];
            var offset = math.rotate(parentRotation, localPositions[i]);

            angularVelocities[i] = angularVelocities[parent] +
                                   math.rotate(parentRotation, localAngularVelocities[i]);
            velocities[i] = velocities[parent] +
                            math.cross(angularVelocities[parent], offset) +
                            math.rotate(parentRotation, localVelocities[i]);
        }
    }
}
}
