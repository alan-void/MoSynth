using System;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Thresholds deciding, from a contact bone's recorded world position alone, whether it is on the
/// ground. Height is measured above the bone's database rest height (<see cref="ContactFloor"/>),
/// so every method is judged by the same rule rather than by its own contact flags.
/// </summary>
[Serializable]
public class GroundContactSettings
{
    [Tooltip("Height above the bone's database rest height (the 5th percentile of its height over the " +
             "database frames flagged in contact), in meters, at which footskate stops counting and the bone stops being " +
             "in contact. Slip is weighted by clamp(2 - 2^(h/H), 0, 1) below it.")]
    [Min(1e-4f)]
    public float heightThreshold = 0.025f;

    [Tooltip("Horizontal speed, in m/s, below which a low bone counts as in contact. Used for " +
             "contactFraction and contactFlagAgreement only, never to gate footskate.")]
    [Min(0f)]
    public float speedThreshold = 0.15f;
}
}
