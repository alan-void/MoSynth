using UnityEngine;

namespace MotionMatching.Testing
{
/// <summary>
/// Test harness: holds the stick fully forward every frame. Drives a character without a player, for
/// eyeballing a database's straight-line locomotion.
/// </summary>
public class MoveForwardInput : MonoBehaviour
{
    public DirectionControlInput MMController;
    
    private void Update()
    {
        MMController.SetMovementDirection(new Vector2(0, 1));
    }
    
}
}