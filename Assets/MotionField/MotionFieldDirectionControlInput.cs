using AnimationTools;
using UnityEngine;

namespace MotionField
{
/// <summary>
/// Steers the motion field from a 2D movement input (gamepad stick or WASD), interpreted
/// relative to a reference transform's facing. The base subscribes it to the player's move
/// action; nothing has to be wired in the scene.
/// </summary>
public class MotionFieldDirectionControlInput : MotionFieldControlInput, IMotionSynthesisDirectionControlInput
{
    [Tooltip("Input is taken relative to this transform's facing. Falls back to the main camera.")]
    [SerializeField]
    private Transform inputReference;

    [Tooltip("Speed requested at full stick deflection, in m/s. A partial deflection asks for " +
             "proportionally less, and a released stick asks the character to stand.")]
    [SerializeField]
    [Min(0f)]
    private float maxSpeed = SplinePursuit.DefaultTargetSpeed;

    private Vector2 _inputMovement;

    /// <summary>Latest 2D movement input; x is strafe, y is forward.</summary>
    public void SetMovementDirection(Vector2 movementDirection)
    {
        _inputMovement = movementDirection;
    }

    protected override void GetDesiredMotion(out Vector3 worldDirection, out float speed)
    {
        speed = maxSpeed * Mathf.Clamp01(_inputMovement.magnitude);

        // A released stick keeps the last heading: standing still asks nothing of the direction.
        if (_inputMovement.sqrMagnitude < 1e-4f)
        {
            worldDirection = Vector3.zero;
            return;
        }

        var reference = inputReference;
        if (reference == null && Camera.main != null) reference = Camera.main.transform;
        if (reference == null)
        {
            worldDirection = new Vector3(_inputMovement.x, 0f, _inputMovement.y);
            return;
        }

        var right = reference.right;
        var ahead = reference.forward;
        right.y = 0f;
        ahead.y = 0f;
        worldDirection = right.normalized * _inputMovement.x + ahead.normalized * _inputMovement.y;
    }
}
}
