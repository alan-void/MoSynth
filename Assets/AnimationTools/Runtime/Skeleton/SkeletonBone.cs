using System;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// One bone of a <see cref="AnimationTools.Skeleton"/>: the rig Transform, plus the skeleton it
/// belongs to. Serializable, so it is also the type a component or asset uses to name a single bone
/// (a foot-contact bone, a feature's bone, a root motion bone).
/// </summary>
/// <remarks>
/// The skeleton is stored by value, not by reference — it is only a root Transform, so a bone costs
/// two object references. Two <see cref="SkeletonBone"/> fields on the same asset therefore hold
/// distinct <see cref="AnimationTools.Skeleton"/> instances over the same root; the derived bone
/// list is cached per root rather than per instance so they still share one walk.
/// <para>
/// Rest values are read live off the Transform, which is only the rest pose while the rig is an
/// asset — see the remarks on <see cref="AnimationTools.Skeleton"/>.
/// </para>
/// </remarks>
[Serializable]
public sealed class SkeletonBone
{
    [SerializeField] private Skeleton skeleton;
    [SerializeField] private Transform bone;

    public SkeletonBone()
    {
    }

    public SkeletonBone(Skeleton skeleton, Transform bone)
    {
        this.skeleton = skeleton;
        this.bone = bone;
    }

    /// <summary>The skeleton this bone is part of; null when the reference was never picked.</summary>
    public Skeleton Skeleton => skeleton;

    public Transform Transform => bone;

    public string Name => bone != null ? bone.name : null;

    public bool IsSet => bone != null;

    /// <summary>Index within <see cref="Skeleton"/>; -1 when either side is unset or unrelated.</summary>
    public int Index => skeleton?.IndexOf(bone) ?? -1;

    /// <summary>-1 for bone 0, and for an unresolved bone.</summary>
    public int ParentIndex
    {
        get
        {
            var index = Index;
            return index < 0 ? -1 : skeleton.GetParentIndex(index);
        }
    }

    /// <summary>Null for bone 0, and for an unresolved bone.</summary>
    public SkeletonBone Parent
    {
        get
        {
            var index = Index;
            return index < 0 ? null : skeleton.GetParent(index);
        }
    }

    public float3 RestLocalPosition => bone.localPosition;

    public quaternion RestLocalRotation => bone.localRotation;

    /// <summary>
    /// Resolves this reference against a different skeleton — a bone picked on one rig looked up in
    /// another that is structurally the same, e.g. a contact bone chosen on the pose skeleton and
    /// resolved against an individual clip's. Transform identity first, then an exact name match.
    /// Returns -1 when the skeleton is null, this reference is unset, or no bone matches.
    /// </summary>
    public int ResolveIndex(Skeleton other)
    {
        if (other == null || !IsSet) return -1;

        var index = other.IndexOf(bone);
        if (index >= 0) return index;

        return other.IndexOfName(bone.name);
    }
}
}
