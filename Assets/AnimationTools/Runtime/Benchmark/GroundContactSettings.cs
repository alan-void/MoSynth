using System;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Thresholds deciding, from a contact bone's recorded world position alone, whether it is on the
/// ground. Used by <see cref="MotionQualityMetricsCalculator"/> so every method is judged by the
/// same rule rather than by its own contact flags.
/// </summary>
[Serializable]
public class GroundContactSettings
{
    [Tooltip("Height above the bone's contact height, in meters, at which footskate stops counting and " +
             "the bone stops being in contact. Slip is weighted by clamp(2 - 2^(h/H), 0, 1) below it.")]
    [Min(1e-4f)]
    public float heightThreshold = 0.025f;

    [Tooltip("Horizontal speed, in m/s, below which a low bone counts as in contact. Used for " +
             "contactFraction and contactFlagAgreement only, never to gate footskate.")]
    [Min(0f)]
    public float speedThreshold = 0.15f;

    [Tooltip("Quantile of each bone's recorded height taken as its contact height for the run, because " +
             "a toe joint does not sit at y = 0 when planted. Assumes flat ground.")]
    [Range(0f, 1f)]
    public float contactHeightPercentile = 0.05f;
}
}
