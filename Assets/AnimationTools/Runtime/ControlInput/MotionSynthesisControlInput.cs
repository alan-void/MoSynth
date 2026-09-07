using UnityEngine;
using UnityEngine.Serialization;

namespace AnimationTools
{
/// <summary>
/// Base for the components that steer a character. It owns the link to the
/// <see cref="MotionSynthesisComponent"/> being driven, claims that character while it is enabled,
/// and — when the subclass is stick-driven — subscribes it to the player's movement input.
/// </summary>
/// <remarks>
/// The link is stored here and nowhere else. A stage that needs the input asks the component it was
/// initialised with, rather than carrying a second reference that has to be paired with this one by
/// hand. Two inputs cannot claim one character: the second to enable warns and disables itself, so
/// switching between them means deactivating one.
/// </remarks>
public abstract class MotionSynthesisControlInput : MonoBehaviour, IMotionSynthesisControlInput
{
    [Tooltip("Character to steer. Found on this object or a parent when left empty.")]
    [FormerlySerializedAs("motionSynthesizer")]
    [FormerlySerializedAs("synthesisComponent")]
    [SerializeField]
    protected MotionSynthesisComponent synthesizer;

    public MotionSynthesisComponent Synthesizer => synthesizer;

    /// <summary>
    /// Whether this input holds the character. False once it has been refused and disabled itself,
    /// which happens inside <see cref="OnEnable"/> — an override that acquires anything of its own
    /// should check this after calling base and return, or it will set up state nothing will use.
    /// </summary>
    protected bool IsBound { get; private set; }

    protected virtual void Awake()
    {
        if (synthesizer == null) synthesizer = GetComponentInParent<MotionSynthesisComponent>();

        if (synthesizer == null)
        {
            Debug.LogError($"{GetType().Name} on '{name}' has no MotionSynthesisComponent assigned " +
                           "or in its parents.", this);
            enabled = false;
        }
    }

    protected virtual void OnEnable()
    {
        IsBound = synthesizer != null && synthesizer.TryBindControlInput(this);
        if (!IsBound)
        {
            enabled = false;
            return;
        }

        if (this is IMotionSynthesisDirectionControlInput direction)
        {
            UserInput.MoveChanged += direction.SetMovementDirection;
            // An input that enables mid-session starts from the stick as it is now, not from zero.
            direction.SetMovementDirection(UserInput.Move);
        }
    }

    protected virtual void OnDisable()
    {
        IsBound = false;

        if (this is IMotionSynthesisDirectionControlInput direction)
        {
            UserInput.MoveChanged -= direction.SetMovementDirection;
        }

        if (synthesizer != null) synthesizer.UnbindControlInput(this);
    }
}
}
