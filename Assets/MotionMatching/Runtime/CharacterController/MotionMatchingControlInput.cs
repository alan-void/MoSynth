using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Events;

namespace MotionMatching
{
/// <summary>
/// Turns intent — a stick direction, a spline to follow, a crowd to avoid — into the trajectory
/// features that <see cref="MotionMatchingStage"/> searches the animation database with. This is
/// the "what should the character be doing" half of motion matching; the stage is the "which
/// animation frame looks most like that" half.
/// </summary>
/// <remarks>
/// Every subclass works the same way: it drives a lightweight <em>simulation object</em> (usually
/// its own Transform), then predicts where that object will be at each of the database's prediction
/// horizons. Those predictions are the trajectory. Nothing here poses the character — the character
/// follows only because the search keeps picking frames that move like the prediction.
/// <para>
/// Subclasses differ in how the desired motion is decided: from a stick
/// (<see cref="DirectionControlInput"/>), from a path (<see cref="SplineControlInput"/>,
/// <see cref="PathControlInput"/>), or from a path plus avoidance (the Crowd variants).
/// </para>
/// <para>
/// To add one: implement <see cref="OnUpdate"/> and <see cref="GetTrajectoryFeature"/>. Also
/// implement <see cref="IMotionSynthesisDirectionControlInput"/> or
/// <see cref="IMotionSynthesisSplineControlInput"/> so synthesis-agnostic tools, such as the
/// path-following metrics harness, can drive it.
/// </para>
/// </remarks>
public abstract class MotionMatchingControlInput : MonoBehaviour
{
    // TODO: Create a OnValidate() (other name because it will collide with Unity's
    //       that validates if the current MMData has the necessary trajectories requeried
    //       by the current controller (eg. simulation bone pos + dir, or HMD + L/R controllers pos + dir)

    /// <summary>The stage subscribes and searches immediately instead of waiting out its interval.</summary>
    public UnityAction OnHighInputChange;

    [SerializeReference] public MotionSynthesisComponent motionSynthesizer;

    public MotionSynthesisComponent Synthesizer => motionSynthesizer;

    /// <summary>
    /// Frame time of the database. Prediction horizons are counted in database frames, so converting
    /// one to seconds means multiplying by this, not by Time.deltaTime.
    /// </summary>
    public float DatabaseDeltaTime { get; private set; }

    private void LateUpdate()
    {
        DatabaseDeltaTime = motionSynthesizer.GetMmData().GetOrImportPoseSet().FrameTime;
        OnUpdate();
    }

    /// <summary>
    /// Call this method to notify Motion Matching that a large change in the input has been made.
    /// Therefore, an immediate Motion Matching search should be performed.
    /// </summary>
    protected void NotifyInputChangedQuickly()
    {
        OnHighInputChange?.Invoke();
    }

    /// <summary>
    /// Use this instead of Unity's Update() method. Advance the simulation object and refresh the
    /// predictions here.
    /// </summary>
    protected abstract void OnUpdate();

    /// <summary>
    /// Return the initial world position of the character controller.
    /// </summary>
    public abstract float3 GetWorldInitPosition();

    /// <summary>
    /// Return the initial world direction of the character controller.
    /// </summary>
    public abstract float3 GetWorldInitDirection();

    /// <summary>
    /// Return the current world position of the character controller.
    /// </summary>
    public abstract float3 GetPosition();

    /// <summary>
    /// Return the target speed of the character, which may be different from the current speed.
    /// </summary>
    public abstract float GetTargetSpeed();

    /// <summary>
    /// Get the prediction in character space of a trajectory feature.
    /// e.g., suppose that the feature is the projected position of the character at frames 20, 40 and 60 in the future:
    ///       then, since the projected position is 2D (2 floats), thus, output[0] and output[1] should be filled with the X and Z coordinates.
    ///       e.g., when index==1, it should return the position of the character at frame 40.
    /// </summary>
    /// <param name="index">Which prediction horizon, indexing the feature's predictionFrames.</param>
    /// <param name="character">
    /// The character's simulation frame. Predictions must be relative to it, since the database
    /// features were baked that way.
    /// </param>
    /// <param name="span">Exactly the feature's float count; fill all of it.</param>
    public abstract void GetTrajectoryFeature(TrajectoryFeatureChannel feature, int index, Transform character,
        Span<float> span);

    /// <summary>Every horizon at once. Unimplemented; the stage calls the overload above.</summary>
    public virtual float[] GetTrajectoryFeature(TrajectoryFeatureChannel feature)
    {
        throw new NotImplementedException();
    }
}
}
