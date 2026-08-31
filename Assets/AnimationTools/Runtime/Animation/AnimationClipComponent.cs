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
/// A component's serialized identity is its <em>(class, namespace, assembly)</em> triple. Renaming
/// the type, moving its namespace, or moving its file into a different assembly orphans every
/// instance already authored on an asset, and the data is dropped the next time that asset is
/// saved. <c>[FormerlySerializedAs]</c> does not help — it renames fields, not managed references.
/// Name a component once, deliberately.
/// </para>
/// <para>
/// Subclasses may live in any assembly that references this one; the type dropdown finds them
/// without registration. See the wiki's synthesis-pipeline page for the same seam used by
/// <see cref="MoSynthStage"/> and <c>BenchmarkOverride</c>.
/// </para>
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
