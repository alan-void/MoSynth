using System;

namespace AnimationTools
{
/// <summary>
/// A piece of annotation attached to an <see cref="AnnotatedAnimationClip"/> — something true about
/// a clip that the clip cannot derive from its own curves.
/// </summary>
/// <remarks>
/// Components are inspector-authored as a <c>[SerializeReference]</c> list, so a concrete component
/// must be <c>[Serializable]</c>, have a parameterless constructor, and expose its data as public
/// mutable fields. It must not be a <see cref="UnityEngine.Object"/>.
/// <para>
/// Serialized identity is the (class, namespace, assembly) triple: renaming the type or moving it to
/// another namespace or assembly orphans every authored instance. <c>[FormerlySerializedAs]</c> does
/// not help; <c>[MovedFrom]</c> does. See openwiki/animation-tools/animation-sources.md.
/// </para>
/// <para>Subclasses in any assembly referencing this one are offered without registration.</para>
/// </remarks>
[Serializable]
public abstract class AnimationClipComponent
{
    /// <summary>Whether consumers should act on this component. A disabled one keeps its data.</summary>
    public bool isEnabled = true;

    /// <summary>One line naming what this component currently holds, for inspector summaries.</summary>
    public abstract string Describe();

    /// <summary>
    /// Called from the owning clip's <c>OnValidate</c>, after the clip has clamped its own frame
    /// range. Clamp anything that indexes into the clip here — the range may just have moved.
    /// </summary>
    public virtual void OnValidate(AnnotatedAnimationClip clip)
    {
    }
}
}
