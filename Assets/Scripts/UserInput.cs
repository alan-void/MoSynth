using System;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Serialization;

namespace AnimationTools
{
public class UserInput : MonoBehaviour
{
    public static UserInput I { get; private set; }
    InputActions _inputActions;
    
    [SerializeField]
    UnityEvent<Vector2> inputActionsPlayerMove;
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
        _inputActions.Player.Move.performed += ctx => inputActionsPlayerMove?.Invoke(ctx.ReadValue<Vector2>());
        _inputActions.Player.Move.canceled += ctx => inputActionsPlayerMove?.Invoke(ctx.ReadValue<Vector2>());
    }
}
}