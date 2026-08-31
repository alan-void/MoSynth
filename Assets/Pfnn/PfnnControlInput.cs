using System.Linq;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace Pfnn
{
/// <summary>
/// Base for components that steer a <see cref="PfnnStage"/> by saying where the character should be
/// going over the next second.
/// </summary>
/// <remarks>
/// A PFNN wants a trajectory, not a heading, which is why this is not shaped like
/// <c>MotionFieldControlInput</c>. It is also not a <c>MotionMatchingControlInput</c>: every one of
/// those resolves its horizons out of a <c>MotionMatchingData</c> in <c>Start</c>, so none of them
/// works on a character that has no motion matching stage.
/// <para>
/// Only the <em>future</em> half of the window comes from here. The past half is where the
/// character actually went, which the stage keeps itself — feeding the desired path backwards would
/// tell the network the character had already been following it, which during training was only
/// ever true.
/// </para>
/// </remarks>
public abstract class PfnnControlInput : MonoBehaviour, IMotionSynthesisControlInput
{
    [Tooltip("Character to steer. Found on this object or a parent when left empty.")]
    [SerializeField]
    protected MotionSynthesisComponent synthesisComponent;

    public MotionSynthesisComponent Synthesizer => synthesisComponent;

    private PfnnStage _stage;
    private bool _warnedNoStage;

    /// <summary>World position of the character's simulation frame.</summary>
    protected Vector3 RootPosition => synthesisComponent.transform.position;

    /// <summary>The stage this input drives, or null until the synthesis component has woken.</summary>
    protected PfnnStage Stage
    {
        get
        {
            // Stages are populated in the synthesis component's Awake; resolve lazily so this
            // component works regardless of Awake ordering.
            if (_stage != null) return _stage;

            _stage = synthesisComponent.stages?.OfType<PfnnStage>().FirstOrDefault();
            if (_stage == null && !_warnedNoStage)
            {
                Debug.LogWarning($"[PFNN] {GetType().Name} on '{name}' found no PfnnStage on the " +
                                 "synthesis component.", this);
                _warnedNoStage = true;
            }

            return _stage;
        }
    }

    protected virtual void Awake()
    {
        if (synthesisComponent == null)
        {
            synthesisComponent = GetComponentInParent<MotionSynthesisComponent>();
        }

        if (synthesisComponent == null)
        {
            Debug.LogError($"[PFNN] {GetType().Name} on '{name}' has no MotionSynthesisComponent " +
                           "assigned or in its parents.", this);
            enabled = false;
        }
    }

    private void Update()
    {
        if (Stage == null) return;
        Stage.ControlInput = this;
        OnUpdate();
    }

    /// <summary>Refresh whatever <see cref="TryGetFutureSample"/> reads, once per frame.</summary>
    protected virtual void OnUpdate()
    {
    }

    /// <summary>
    /// Where the character should be, and which way it should face, this many synthesis frames from
    /// now.
    /// </summary>
    /// <remarks>
    /// Both are in world space on the ground plane; the stage puts them into the character frame,
    /// so an implementation never has to know which frame the network is being queried in.
    /// Returning false leaves the sample at the character's current position and facing, which is
    /// what a controller with nothing to say should produce — a character standing still.
    /// </remarks>
    /// <param name="frameOffset">Synthesis frames ahead. Always positive.</param>
    public abstract bool TryGetFutureSample(int frameOffset, out float2 position, out float2 direction);
}
}
