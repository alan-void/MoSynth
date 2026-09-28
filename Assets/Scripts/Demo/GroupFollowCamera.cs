using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Keeps a group of characters in view: follows their centroid and pulls back as they spread out.
/// </summary>
public class GroupFollowCamera : MonoBehaviour
{
    public List<Transform> targets = new();

    [Tooltip("Direction and base distance from the group's centroid to the camera.")]
    public Vector3 offset = new(0f, 4f, -10f);

    [Tooltip("Extra distance per metre of spread, so a group that drifts apart stays in frame.")]
    public float distancePerSpread = 0.7f;

    [Tooltip("Seconds for the camera to cover half the distance to where it should be.")]
    public float halfLife = 0.3f;

    private void LateUpdate()
    {
        if (!TryGetBounds(out var bounds)) return;

        var spread = Mathf.Max(bounds.size.x, bounds.size.z);
        var desired = bounds.center + offset + offset.normalized * (spread * distancePerSpread);
        var blend = 1f - Mathf.Pow(0.5f, Time.deltaTime / Mathf.Max(halfLife, 1e-4f));

        transform.position = Vector3.Lerp(transform.position, desired, blend);
        transform.rotation = Quaternion.LookRotation(bounds.center - transform.position);
    }

    private bool TryGetBounds(out Bounds bounds)
    {
        bounds = default;
        var any = false;
        foreach (var target in targets)
        {
            if (target == null) continue;
            if (!any) bounds = new Bounds(target.position, Vector3.zero);
            else bounds.Encapsulate(target.position);
            any = true;
        }

        return any;
    }
}
