using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Splines;

namespace AnimationTools
{
/// <summary>Common surface over the control-input hierarchies (MotionMatching, MotionField) so
/// synthesis-agnostic tools can drive and inspect them.</summary>
public interface IMotionSynthesisControlInput
{
    MotionSynthesisComponent Synthesizer { get; }
}

/// <summary>A control input that steers a character along a spline.</summary>
public interface IMotionSynthesisSplineControlInput : IMotionSynthesisControlInput
{
    SplineContainer SplineContainer { get; set; }

    /// <summary>Desired locomotion speed in m/s; <see cref="float.NaN"/> when the input has no speed model.</summary>
    float TargetSpeed { get; }

    /// <summary>
    /// World direction a character driven by this input should be <em>facing</em> when it is placed
    /// at the start of the path. Not necessarily the direction of travel: a path that carries its own
    /// facing answers from that, so a character authored to set off strafing spawns strafing.
    /// Callers should treat a zero-length result as "no opinion" and keep the rotation they have.
    /// </summary>
    float3 GetWorldInitDirection();
}

/// <summary>A control input steered by a 2D movement direction (stick/WASD-style).</summary>
public interface IMotionSynthesisDirectionControlInput : IMotionSynthesisControlInput
{
    void SetMovementDirection(Vector2 movementDirection);
}
}
