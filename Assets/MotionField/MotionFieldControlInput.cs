using System.Linq;
using AnimationTools;
using UnityEngine;

namespace MotionField
{
/// <summary>
/// Base for components that steer a MotionFieldStage. Subclasses supply a desired world-space
/// heading each frame; the base pushes it into the stage before the synthesis tick so the goal
/// angle is measured against the root pose the stage is about to step from.
/// </summary>
public abstract class MotionFieldControlInput : MonoBehaviour
{
    [Tooltip("Character to steer. Found on this object or a parent when left empty.")]
    [SerializeField]
    protected MotionSynthesisComponent synthesisComponent;

    public MotionSynthesisComponent Synthesizer => synthesisComponent;

    private MotionFieldStage _stage;
    private bool _warnedNoStage;

    /// <summary>World position of the character's simulation frame.</summary>
    protected Vector3 RootPosition => synthesisComponent.transform.position;

    protected virtual void Awake()
    {
        if (synthesisComponent == null)
        {
            synthesisComponent = GetComponentInParent<MotionSynthesisComponent>();
        }

        if (synthesisComponent == null)
        {
            Debug.LogError($"[MotionField] {GetType().Name} on '{name}' has no " +
                           "MotionSynthesisComponent assigned or in its parents.", this);
            enabled = false;
        }
    }

    private void Update()
    {
        // Stages are populated in the synthesis component's Awake; resolve lazily so this
        // component works regardless of Awake ordering.
        if (_stage == null)
        {
            _stage = synthesisComponent.stages?.OfType<MotionFieldStage>().FirstOrDefault();
            if (_stage == null)
            {
                if (!_warnedNoStage)
                {
                    Debug.LogWarning($"[MotionField] {GetType().Name} on '{name}' found no " +
                                     "MotionFieldStage on the synthesis component.", this);
                    _warnedNoStage = true;
                }
                return;
            }
        }

        // A near-zero vector keeps the stage's last heading while still refreshing Theta
        // against the root's current facing, so an idle stick does not let Theta go stale.
        _stage.SetDesiredDirection(GetDesiredWorldDirection());
    }

    /// <summary>
    /// The heading the character should turn toward, in world space, evaluated once per frame.
    /// Return a near-zero vector to keep the previous heading.
    /// </summary>
    protected abstract Vector3 GetDesiredWorldDirection();

#if UNITY_EDITOR
    protected virtual void OnDrawGizmos()
    {
        if (!Application.isPlaying || _stage == null || synthesisComponent == null) return;
        var origin = RootPosition + Vector3.up * 0.05f;
        Gizmos.color = new Color(1.0f, 0.3f, 0.1f, 1.0f); // matches MM current-dir gizmo style
        Gizmos.DrawSphere(origin, 0.05f);
        Gizmos.DrawLine(origin, origin + _stage.DesiredWorldDirection * 0.5f);
    }
#endif
}
}
