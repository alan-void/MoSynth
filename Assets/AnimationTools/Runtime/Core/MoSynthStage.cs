using System;

namespace AnimationTools
{
/// <summary>
/// One step of a <see cref="MotionSynthesisComponent"/>'s pipeline, and the seam every synthesis
/// method plugs into: motion matching, a neural motion field, root-motion correction, a visualizer.
/// A stage rewrites the pipeline pose in place; the component owns the pose, the skeleton and the
/// frame loop.
/// </summary>
/// <remarks>
/// Stages are inspector-authored as a <c>[SerializeReference]</c> list, so a concrete stage must be
/// <c>[Serializable]</c> with a parameterless constructor. It is a plain class, not a MonoBehaviour:
/// the callbacks below are all it gets. They run in declaration order — GetSkeleton on every stage,
/// then Init on every stage, then Apply each tick, then OnDestroy.
/// </remarks>
[Serializable]
public abstract class MoSynthStage
{
    /// <summary>
    /// When false the component skips <see cref="Apply"/>, but still runs the other three callbacks.
    /// So disabling a stage cannot change the skeleton or the pose layout mid-run.
    /// </summary>
    public bool isEnabled = true;

    /// <summary>
    /// Contributes to the skeleton the whole pipeline runs on, before any <see cref="Init"/>.
    /// Return <paramref name="inSkeleton"/> unchanged unless this stage is the one that
    /// <em>sources</em> the skeleton, as a motion matching stage does from its database.
    /// </summary>
    /// <param name="inSkeleton">What the previous stage returned; null if no earlier stage supplied one.</param>
    public virtual Skeleton GetSkeleton(Skeleton inSkeleton)
    {
        return inSkeleton;
    }

    /// <summary>
    /// One-time setup. By now the skeleton, the pose layout and the scene rig binding are all
    /// settled, so everything on the component is usable.
    /// </summary>
    public abstract void Init(MotionSynthesisComponent motionSynthesisComponent);

    /// <summary>
    /// Editor-only inspector validation. Runs on every repaint and outside play mode, so keep it
    /// cheap and do not assume <see cref="Init"/> has run.
    /// </summary>
    public virtual void OnValidate()
    {
    }

    /// <summary>
    /// Transforms the pipeline pose, once per synthesis tick.
    /// </summary>
    /// <param name="pose">
    /// The pose so far. <see cref="PoseBuffer"/> is a view struct, so writing through this parameter
    /// is how a stage produces output — there is nothing to return.
    /// </param>
    /// <param name="deltaTime">Synthesis timestep, which is not necessarily Time.deltaTime.</param>
    /// <returns>
    /// True to continue the pipeline. False means "this pose is final": the component still applies
    /// it, so returning false is not the same as discarding the tick.
    /// </returns>
    /// <remarks>
    /// A stage that replaces the pose discontinuously must set
    /// <see cref="MotionSynthesisComponent.PoseDiscontinuity"/>, which is how downstream blending
    /// stages know to re-anchor.
    /// </remarks>
    public abstract bool Apply(PoseBuffer pose, float deltaTime);

    /// <summary>Teardown. Release native collections and unmanaged state here — nothing else will.</summary>
    public virtual void OnDestroy()
    {
    }
}
}
