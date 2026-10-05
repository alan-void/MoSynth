using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Plants contact bones in the world while their contact flag holds, bending the two bones above
/// each one to keep it there, and eases them back onto the animation when the contact ends.
/// </summary>
/// <remarks>
/// Place it after <c>RootFollowStage</c>, which itself runs after <c>Inertialization</c>: it asks the
/// component where this pose will be rendered, and a later change to bone 0's rates would move the
/// character out from under the planted bones. Only rotations are rewritten; the next tick re-reads
/// the rig, so the velocity channels are left alone. See
/// <c>openwiki/animation-tools/root-following.md</c>.
/// </remarks>
[Serializable]
public class ContactLockStage : MoSynthStage
{
    [Serializable]
    public class LockedBone
    {
        [Tooltip("The bone to plant. Must be one of the character's contact bones.")]
        public SkeletonBone lockedBone = new();

        [Tooltip("End of the two-bone chain bent to plant the locked bone; its parent and grandparent " +
                 "rotate. Empty is the locked bone's parent, so a toe bends the leg at hip and knee.")]
        public SkeletonBone ikEndBone = new();
    }

    [Tooltip("Bones to plant, each with the chain that holds it. Empty plants every contact bone of " +
             "the character.")]
    public List<LockedBone> lockedBones = new();

    [Tooltip("How far, in metres, the animation may pull a planted bone from where it touched down " +
             "before the lock lets go instead of over-stretching the leg. 0 never lets go early.")]
    [Min(0f)] public float maxLockDistance = 0.3f;

    [Tooltip("Time to ease half the way back onto the animation once a lock lets go.")]
    [Min(0f)] public float releaseHalfLife = 0.1f;

    /// <summary>
    /// One planted bone resolved against the skeleton: its contact slot, its lock state, and the
    /// chain root → mid → ikEnd that bends to hold it.
    /// </summary>
    private sealed class Chain
    {
        public int Slot;
        public int Locked;
        public int IkEnd;
        public int Mid;
        public int Root;
        public ContactLock Lock;
    }

    private MotionSynthesisComponent _owner;
    private readonly List<Chain> _chains = new();
    private NativeArray<float3> _characterPositions;
    private NativeArray<quaternion> _characterRotations;

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        _owner = motionSynthesisComponent;
        _chains.Clear();
        DisposeScratch();

        var skeleton = _owner.Skeleton;
        var contactHandles = _owner.ContactHandles;

        if (lockedBones.Count == 0)
        {
            if (contactHandles.Count == 0)
            {
                Debug.LogWarning($"{nameof(ContactLockStage)} on \"{_owner.name}\": the character has no " +
                                 "contact bones, so there is nothing to plant.", _owner);
            }

            for (var slot = 0; slot < contactHandles.Count; slot++)
            {
                TryAddChain(contactHandles.GetBoneIndex(slot), -1);
            }
        }
        else
        {
            foreach (var entry in lockedBones)
            {
                if (entry?.lockedBone == null || !entry.lockedBone.IsSet)
                {
                    LogEntryError("an entry has no locked bone");
                    continue;
                }

                var lockedIndex = entry.lockedBone.ResolveIndex(skeleton);
                if (lockedIndex < 0)
                {
                    LogEntryError($"locked bone \"{entry.lockedBone.Name}\" is not in the skeleton");
                    continue;
                }

                var ikEndIndex = -1;
                if (entry.ikEndBone != null && entry.ikEndBone.IsSet)
                {
                    ikEndIndex = entry.ikEndBone.ResolveIndex(skeleton);
                    if (ikEndIndex < 0)
                    {
                        LogEntryError($"IK end bone \"{entry.ikEndBone.Name}\" is not in the skeleton");
                        continue;
                    }
                }

                TryAddChain(lockedIndex, ikEndIndex);
            }
        }

        if (_chains.Count == 0) return;

        var boneCount = _owner.SkeletonData.BoneCount;
        _characterPositions = new NativeArray<float3>(boneCount, Allocator.Persistent);
        _characterRotations = new NativeArray<quaternion>(boneCount, Allocator.Persistent);
    }

    /// <summary>
    /// Builds the chain that plants <paramref name="lockedIndex"/>, or logs why it cannot and adds
    /// nothing. An <paramref name="ikEndIndex"/> of -1 means the locked bone's parent.
    /// </summary>
    private void TryAddChain(int lockedIndex, int ikEndIndex)
    {
        var skeleton = _owner.Skeleton;
        var lockedName = skeleton.GetBone(lockedIndex).Name;

        if (!_owner.ContactHandles.TryGetSlot(lockedIndex, out var slot))
        {
            LogEntryError($"\"{lockedName}\" is not one of the character's contact bones, so it has no " +
                          "contact flag to lock on");
            return;
        }

        if (_chains.Exists(chain => chain.Locked == lockedIndex))
        {
            LogEntryError($"\"{lockedName}\" is listed more than once");
            return;
        }

        if (ikEndIndex < 0) ikEndIndex = skeleton.GetParentIndex(lockedIndex);
        if (ikEndIndex < 0 || !IsSelfOrAncestor(skeleton, ikEndIndex, lockedIndex))
        {
            LogEntryError($"the IK end bone for \"{lockedName}\" must be \"{lockedName}\" or one of its ancestors");
            return;
        }

        var mid = skeleton.GetParentIndex(ikEndIndex);
        var root = mid >= 0 ? skeleton.GetParentIndex(mid) : -1;
        // Rotating bone 0 would move the character frame the lock is measured against.
        if (root < 1)
        {
            LogEntryError($"the chain above \"{skeleton.GetBone(ikEndIndex).Name}\" reaches the skeleton " +
                          "root; it needs two bones to bend that are not bone 0");
            return;
        }

        _chains.Add(new Chain { Slot = slot, Locked = lockedIndex, IkEnd = ikEndIndex, Mid = mid, Root = root });
    }

    private static bool IsSelfOrAncestor(Skeleton skeleton, int candidate, int bone)
    {
        for (var index = bone; index >= 0; index = skeleton.GetParentIndex(index))
        {
            if (index == candidate) return true;
        }
        return false;
    }

    private void LogEntryError(string problem)
    {
        Debug.LogError($"{nameof(ContactLockStage)} on \"{_owner.name}\": {problem}. Skipping that bone.", _owner);
    }

    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        if (_chains.Count == 0 || deltaTime <= 0f) return true;

        var skeletonData = _owner.SkeletonData;
        var parentIndices = skeletonData.ParentIndices;
        var contactHandles = _owner.ContactHandles;
        var rotations = pose.Rotations;

        SimulationFrame.Compute(pose, skeletonData, _owner.SimulationFrame, out var framePosition,
            out var frameRotation);
        _owner.ComputeAppliedFrame(pose, deltaTime, out var renderPosition, out var renderRotation);
        var bendDirection = math.mul(frameRotation, math.forward());

        skeletonData.LocalSpaceToCharacterSpace(pose, _characterPositions, _characterRotations);

        for (var i = 0; i < _chains.Count; i++)
        {
            var chain = _chains[i];
            var contact = pose.GetBool(contactHandles[chain.Slot]);
            var animatedLocked = _characterPositions[chain.Locked];

            var worldAnimated = PoseToWorld(animatedLocked, framePosition, frameRotation, renderPosition,
                renderRotation);
            var worldTarget = chain.Lock.Step(contact, worldAnimated, maxLockDistance, releaseHalfLife, deltaTime);
            var target = WorldToPose(worldTarget, framePosition, frameRotation, renderPosition, renderRotation);

            if (math.distancesq(target, animatedLocked) < 1e-10f) continue;

            // The ikEnd → locked segment keeps its animated rotation, so a heel can still roll over a
            // planted toe.
            var animatedIkEnd = _characterPositions[chain.IkEnd];
            var ikEndTarget = target + (animatedIkEnd - animatedLocked);

            TwoBoneIK.Solve(_characterPositions[chain.Root], _characterPositions[chain.Mid], animatedIkEnd,
                _characterRotations[chain.Root], _characterRotations[chain.Mid], ikEndTarget, bendDirection,
                out var rootRotation, out var midRotation);

            rotations[chain.Root] = math.mul(math.inverse(_characterRotations[parentIndices[chain.Root]]),
                rootRotation);
            rotations[chain.Mid] = math.mul(math.inverse(rootRotation), midRotation);
            rotations[chain.IkEnd] = math.mul(math.inverse(midRotation), _characterRotations[chain.IkEnd]);

            // A later chain may hang off bones this one just rotated.
            if (i < _chains.Count - 1)
            {
                skeletonData.LocalSpaceToCharacterSpace(pose, _characterPositions, _characterRotations);
            }
        }

        return true;
    }

    /// <summary>
    /// Pose space is the clip space bone 0 is stored in; the component renders the pose's own frame
    /// at the applied frame, so a point moves between the two by swapping one frame for the other.
    /// </summary>
    private static float3 PoseToWorld(float3 posePosition, float3 framePosition, quaternion frameRotation,
        float3 renderPosition, quaternion renderRotation) =>
        SimulationFrame.FromFrameLocal(SimulationFrame.ToFrameLocal(posePosition, framePosition, frameRotation),
            renderPosition, renderRotation);

    private static float3 WorldToPose(float3 worldPosition, float3 framePosition, quaternion frameRotation,
        float3 renderPosition, quaternion renderRotation) =>
        SimulationFrame.FromFrameLocal(SimulationFrame.ToFrameLocal(worldPosition, renderPosition, renderRotation),
            framePosition, frameRotation);

    public override void OnDestroy() => DisposeScratch();

    private void DisposeScratch()
    {
        if (_characterPositions.IsCreated) _characterPositions.Dispose();
        if (_characterRotations.IsCreated) _characterRotations.Dispose();
    }
}
}
