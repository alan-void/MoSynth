using System;
using AnimationTools;
using Unity.Mathematics;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// Turns intent — a stick direction, a spline to follow, a crowd to avoid — into the trajectory
/// features that <see cref="MotionMatchingStage"/> searches the animation database with.
/// </summary>
/// <remarks>
/// A subclass drives a lightweight simulation object and predicts it at each prediction horizon;
/// nothing here poses the character. To add one, implement <see cref="OnUpdate"/> and
/// <see cref="GetTrajectoryFeature"/>, plus <see cref="IMotionSynthesisDirectionControlInput"/> or
/// <see cref="IMotionSynthesisSplineControlInput"/> so synthesis-agnostic tools can drive it.
/// See openwiki/motion-matching/control-inputs.md.
/// </remarks>
public abstract class MotionMatchingControlInput : MotionSynthesisControlInput
{
    // TODO: validate that the current MotionMatchingData has the trajectory features this input needs.

    private bool _highInputChange;

    /// <summary>
    /// Frame time of the database. Prediction horizons are counted in database frames, so converting
    /// one to seconds means multiplying by this, not by Time.deltaTime.
    /// </summary>
    public float DatabaseDeltaTime { get; private set; }

    /// <summary>
    /// Advances the input before the synthesis tick (which runs in LateUpdate), so the stage searches
    /// with this frame's trajectory.
    /// </summary>
    private void Update()
    {
        DatabaseDeltaTime = synthesizer.GetMmData().GetOrImportPoseSet().FrameTime;
        OnUpdate();
    }

    /// <summary>
    /// Requests an immediate search, for an input change too large to wait out the search interval.
    /// </summary>
    protected void NotifyInputChangedQuickly()
    {
        _highInputChange = true;
    }

    /// <summary>
    /// Whether the input has changed enough since the last call that the stage should search now
    /// instead of waiting out its interval. Reading it clears it.
    /// </summary>
    /// <remarks>
    /// Latched rather than an event, because an input can bind after the stage initialises and so
    /// there is no reliable moment for the stage to subscribe.
    /// </remarks>
    public bool ConsumeHighInputChange()
    {
        var changed = _highInputChange;
        _highInputChange = false;
        return changed;
    }

    /// <summary>
    /// Use this instead of Unity's Update(): advance the simulation object and refresh the predictions.
    /// </summary>
    protected abstract void OnUpdate();

    /// <summary>The world position the character should start at.</summary>
    public abstract float3 GetWorldInitPosition();

    /// <summary>The world direction the character should start facing.</summary>
    public abstract float3 GetWorldInitDirection();

    /// <summary>The current world position of the simulation object.</summary>
    public abstract float3 GetPosition();

    /// <summary>The speed the character is asked to move at, which may differ from its current speed.</summary>
    public abstract float GetTargetSpeed();

    /// <summary>
    /// One horizon of a simulation-frame trajectory feature, in character space. Only called for
    /// channels with <c>simulationBone</c> set; bone channels go through
    /// <see cref="GetBoneTrajectoryFeature"/>.
    /// </summary>
    /// <param name="index">Which prediction horizon, indexing the feature's predictionFrames.</param>
    /// <param name="character">
    /// The character's simulation frame. Predictions must be relative to it, since the database
    /// features were baked that way.
    /// </param>
    /// <param name="span">Exactly the feature's float count; fill all of it.</param>
    public abstract void GetTrajectoryFeature(TrajectoryFeatureChannel feature, int index, Transform character,
        Span<float> span);

    /// <summary>
    /// One horizon of a bone (non-simulation-frame) trajectory channel, in character space.
    /// Return true after filling <paramref name="output"/> to switch the channel on for this search;
    /// return false (the default) to switch it off, and the stage zeroes its weights.
    /// </summary>
    public virtual bool GetBoneTrajectoryFeature(TrajectoryFeatureChannel feature, int index, Transform character,
        Span<float> output)
    {
        return false;
    }

    /// <summary>Every horizon at once. Unimplemented; the stage calls the overload above.</summary>
    public virtual float[] GetTrajectoryFeature(TrajectoryFeatureChannel feature)
    {
        throw new NotImplementedException();
    }

    // --- Shared plumbing -----------------------------------------------------------------------

    /// <summary>
    /// The database's position and direction trajectory channels, resolved by name, plus the
    /// horizons they share.
    /// </summary>
    protected readonly struct TrajectoryFeaturePair
    {
        public readonly int PositionIndex;
        public readonly int DirectionIndex;

        /// <summary>Horizons in database frames, ascending, negative meaning the past.</summary>
        public readonly int[] PredictionFrames;

        public TrajectoryFeaturePair(int positionIndex, int directionIndex, int[] predictionFrames)
        {
            PositionIndex = positionIndex;
            DirectionIndex = directionIndex;
            PredictionFrames = predictionFrames;
        }

        public int PredictionCount => PredictionFrames.Length;
    }

    /// <summary>
    /// Finds the two named trajectory channels in the database this input drives. Both must predict
    /// at the same horizons, since an input answers a position and a facing for each one.
    /// </summary>
    protected TrajectoryFeaturePair ResolveTrajectoryFeaturePair(string positionFeatureName,
        string directionFeatureName)
    {
        var trajectoryFeatures = synthesizer.GetMmData().trajectoryFeatures;
        var positionIndex = -1;
        var directionIndex = -1;
        for (var i = 0; i < trajectoryFeatures.Count; i++)
        {
            if (trajectoryFeatures[i].name == positionFeatureName) positionIndex = i;
            if (trajectoryFeatures[i].name == directionFeatureName) directionIndex = i;
        }

        Debug.Assert(positionIndex != -1, $"Trajectory feature \"{positionFeatureName}\" not found");
        Debug.Assert(directionIndex != -1, $"Trajectory feature \"{directionFeatureName}\" not found");

        var positionFrames = trajectoryFeatures[positionIndex].predictionFrames;
        var directionFrames = trajectoryFeatures[directionIndex].predictionFrames;
        Debug.Assert(positionFrames.Length == directionFrames.Length,
            $"\"{positionFeatureName}\" and \"{directionFeatureName}\" must predict at the same horizons");
        for (var i = 0; i < positionFrames.Length && i < directionFrames.Length; i++)
        {
            Debug.Assert(positionFrames[i] == directionFrames[i],
                $"\"{positionFeatureName}\" and \"{directionFeatureName}\" must predict at the same horizons");
        }

        return new TrajectoryFeaturePair(positionIndex, directionIndex, positionFrames);
    }

    /// <summary>Writes a world ground-plane point into a feature span, in the character's frame.</summary>
    protected static void WritePlanarPosition(Transform character, float2 world, Span<float> output)
    {
        var local = character.InverseTransformPoint(new Vector3(world.x, 0f, world.y));
        output[0] = local.x;
        output[1] = local.z;
    }

    /// <summary>Writes a world ground-plane direction into a feature span, in the character's frame.</summary>
    protected static void WritePlanarDirection(Transform character, float2 world, Span<float> output)
    {
        var local = character.InverseTransformDirection(new Vector3(world.x, 0f, world.y));
        output[0] = local.x;
        output[1] = local.z;
    }

    /// <summary>A Transform's world position, flattened onto the ground plane.</summary>
    protected static float2 PlanarPosition(Transform t) => new(t.position.x, t.position.z);

    /// <summary>A Transform's forward, flattened onto the ground plane and normalized.</summary>
    protected static float2 PlanarForward(Transform t) =>
        math.normalizesafe(new float2(t.forward.x, t.forward.z), new float2(0f, 1f));
}
}
