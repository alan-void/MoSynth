using UnityEngine;

/// <summary>
/// Holds a text label above a moving target and turns it to face the camera, so it stays readable
/// whichever way the target is facing.
/// </summary>
public class FollowLabel : MonoBehaviour
{
    public Transform target;
    public Vector3 offset = new(0f, 1.2f, 0f);

    private void LateUpdate()
    {
        if (target == null) return;

        transform.position = target.position + offset;
        var camera = Camera.main;
        if (camera != null) transform.rotation = camera.transform.rotation;
    }
}
