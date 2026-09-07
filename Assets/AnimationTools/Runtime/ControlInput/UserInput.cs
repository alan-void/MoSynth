using System;
using UnityEngine;
using UnityEngine.Events;

namespace AnimationTools
{
/// <summary>
/// The player's steering input, published as a static channel every control input listens on.
/// </summary>
/// <remarks>
/// Static rather than a reference each listener has to be handed, because a control input arrives
/// with its character — dropped in as a prefab, or spawned by the benchmark sweep — and has no way
/// to reach a scene object it was never wired to. Anything that is not a control input can still be
/// driven the authored way, through <see cref="inputActionsPlayerMove"/>.
/// </remarks>
public class UserInput : MonoBehaviour
{
    public static UserInput I { get; private set; }
    InputActions _inputActions;

    /// <summary>The last movement direction the player asked for; zero before anything is pressed.</summary>
    public static Vector2 Move { get; private set; }

    /// <summary>
    /// Raised on both edges of the move action. <see cref="MotionSynthesisControlInput"/> subscribes
    /// itself while it is enabled, so a steerable character needs no wiring at all.
    /// </summary>
    public static event Action<Vector2> MoveChanged;

    [SerializeField]
    UnityEvent<Vector2> inputActionsPlayerMove;

    // Statics outlive a play session when Enter Play Mode Options is set to skip the domain reload,
    // which would leave the event holding destroyed subscribers.
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetStatics()
    {
        I = null;
        Move = Vector2.zero;
        MoveChanged = null;
    }

    private void Awake()
    {
        if (I != null && I != this)
        {
            Destroy(this);
            return;
        }

        I = this;

        _inputActions = new InputActions();
        _inputActions.Player.Enable();
    }

    private void Start()
    {
        // Both edges: performed alone latches the last direction when the keys are released, which
        // every control input reads as a stick still being held.
        _inputActions.Player.Move.performed += ctx => Publish(ctx.ReadValue<Vector2>());
        _inputActions.Player.Move.canceled += ctx => Publish(ctx.ReadValue<Vector2>());
    }

    private void Publish(Vector2 movementDirection)
    {
        Move = movementDirection;
        MoveChanged?.Invoke(movementDirection);
        inputActionsPlayerMove?.Invoke(movementDirection);
    }
}
}
