using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

namespace AnimationTools
{
/// <summary>
/// Drives one character by running a pipeline of <see cref="MoSynthStage"/>s over a pose buffer
/// each tick, then writing the result onto the character's Transforms. The component owns the
/// skeleton, the pose layout and the frame loop; the stages own the synthesis.
/// </summary>
/// <remarks>
/// Awake settles things in order, each step depending on the last: bind the serialized skeleton to
/// the scene rig, derive the simulation frame definition from it, build the pose layout, then Init
/// the stages.
/// <para>
/// Each tick: read the rig into <see cref="CurrentPose"/>, copy to a scratch buffer, run the
/// enabled stages over it until one returns false, apply the result, raise
/// <see cref="OnPoseApplied"/>.
/// </para>
/// <para>
/// A pose is stored in its own clip space: bone 0 is the rig's real root carrying world position
/// and rotation, bones 1.. carry rest offsets and parent-local rotations. This component's own
/// Transform is the character frame — the pose is written under it frame-locally, and it is the
/// Transform that root motion advances, so it is also the capsule anchor a controller would drive.
/// </para>
/// </remarks>
public class MotionSynthesisComponent : MonoBehaviour, ISkeletonProvider
{
    /// <summary>
    /// The pose as read off the character's Transforms at the start of the current tick, i.e. the
    /// input to the pipeline before any stage has run.
    /// </summary>
    [NonSerialized] public PoseBuffer CurrentPose;

    /// <summary>Layout of <see cref="CurrentPose"/> and the pose passed to every stage's Apply.</summary>
    public PoseLayout PoseLayout { get; private set; }

    /// <summary>Foot-contact Bool channels of the pipeline pose.</summary>
    public ChannelHandle LeftFootContactHandle { get; private set; }
    public ChannelHandle RightFootContactHandle { get; private set; }

    // Reused every LateUpdate as the mutable pose the stage chain runs on.
    private PoseBuffer _scratchPose;

    // An asset rig at its rest pose, deliberately not characterRig: a skeleton over a live scene
    // rig would report whatever pose it's currently animated to as its rest pose, which silently
    // corrupts FK. The drawer refuses scene objects for exactly that reason.
    [SerializeField]
    [Tooltip("The rig the pipeline runs on. Assign the root bone from the FBX *asset*, not from a " +
             "rig in the scene.")]
    private Skeleton skeleton = new();

    public Skeleton Skeleton => skeleton;

    /// <summary>Burst-compatible mirror of <see cref="Skeleton"/>, valid from Awake onwards.</summary>
    public SkeletonData SkeletonData => skeleton.GetSkeletonData();

    /// <summary>
    /// Which bone the character frame is read from, and that bone's forward axis. Derived from
    /// <see cref="Skeleton"/>: the root bone and its rest forward axis.
    /// </summary>
    public SimulationFrameDef SimulationFrame { get; private set; }

    [SerializeField]
    [Tooltip("The rig this component drives; bones are bound to the pipeline skeleton by name, " +
             "starting at the skeleton root. Every Transform between the root bone and this " +
             "component must be identity, since the pose is written under this Transform.")]
    private SkeletonBoneOverrides characterRig = new();

    public SkeletonBoneOverrides CharacterRig => characterRig;

    /// <summary>
    /// The scene Transforms this component drives, indexed by skeleton bone index. These are what
    /// renders the character.
    /// </summary>
    [NonSerialized] public Transform[] SkeletonTransforms;

    [Tooltip(
        "The frame rate of the animation synthesis. This is used to calculate the time step of the motion synthesis." +
        "Set to 0 for uncapped.")]
    public float synthesisFrameRate = 30f;

    /// <summary>The pipeline, run in order every tick. See <see cref="MoSynthStage"/>.</summary>
    [SerializeReference] [SubclassSelector]
    public List<MoSynthStage> stages = new();

    [Tooltip("Whether to animate the root position by Motion Matching or not.")]
    // maybe change this to 'root motion'?
    public bool rootPositionsMask = true;

    /// <summary>
    /// True when an upstream stage replaced the pose discontinuously this tick (e.g. a motion
    /// matching search jumped to a new frame). Set by the stage that caused the jump; cleared at
    /// the start of every synthesis tick. Downstream blending stages read it to re-anchor.
    /// </summary>
    public bool PoseDiscontinuity { get; set; }

    /// <summary>
    /// When set, every tick times each stage's <see cref="MoSynthStage.Apply"/> into
    /// <see cref="StageApplyTicks"/>. Off by default: it costs a timestamp pair per stage and
    /// nothing in normal play reads the result.
    /// </summary>
    [NonSerialized] public bool MeasureStageCost;

    /// <summary>
    /// How long each stage's <see cref="MoSynthStage.Apply"/> took on the last tick, in
    /// <see cref="Stopwatch"/> ticks, indexed like <see cref="stages"/>. Written only while
    /// <see cref="MeasureStageCost"/> is set; a stage that was disabled, or that the pipeline never
    /// reached because an earlier stage returned false, holds 0 for that tick.
    /// </summary>
    [NonSerialized] public long[] StageApplyTicks;

    /// <summary>
    /// Fired at the end of every synthesis tick, after the post-stage pose has been applied to the
    /// skeleton transforms. The <see cref="PoseBuffer"/> is a view over the component's scratch pose:
    /// read it synchronously, do not hold the reference past the next tick, do not write to it.
    /// Not fired on Unity frames the frame-rate limiter skips.
    /// </summary>
    public event Action<PoseBuffer, float> OnPoseApplied;

    /// <summary>
    /// Time.DeltaTime if frame rate is not restricted. 1/animationFrameRate if restricted.
    /// </summary>
    private float _animationDeltaTime;

    /// <summary>Countdown to the next synthesis tick while the frame rate is capped.</summary>
    private float _timeTillNextAnimationUpdate;

    bool IsFrameRateRestricted => synthesisFrameRate > 1e-5;

    private void Awake()
    {
        if (IsFrameRateRestricted)
        {
            _animationDeltaTime = 1.0f / synthesisFrameRate;
        }

        stages.RemoveAll(stage => stage == null);
        StageApplyTicks = new long[stages.Count];

        if (skeleton?.Root == null)
        {
            Debug.LogError($"MotionSynthesisComponent \"{name}\": no skeleton assigned. Assign the " +
                           "rig's root bone (from the FBX asset) to the Skeleton field. Disabling " +
                           "the component.");
            enabled = false;
            return;
        }

        if (characterRig == null) characterRig = new SkeletonBoneOverrides();
        if (!characterRig.IsSet)
        {
            Debug.LogWarning($"MotionSynthesisComponent \"{name}\": characterRig is unset; searching under this component's own transform.");
            characterRig.SetRoot(transform);
        }

        SkeletonTransforms = characterRig.Bind(skeleton);

        var missingBoneCount = 0;
        for (var i = 0; i < SkeletonTransforms.Length; i++)
        {
            if (SkeletonTransforms[i] == null) missingBoneCount++;
        }

        if (missingBoneCount > 0)
        {
            Debug.LogError($"MotionSynthesisComponent \"{name}\": {missingBoneCount} bone(s) could not be bound to characterRig (see errors above). Disabling the component.");
            enabled = false;
            return;
        }

        SimulationFrame = SimulationFrameDef.Default(skeleton);

        InitCurrentPose();

        foreach (var stage in stages)
        {
            stage.Init(this);
        }
    }

    private void LateUpdate()
    {
        if (!TryBeginSynthesisTick()) return;

        ConstructCurrentPoseFromSkeletonTransforms();

        PoseDiscontinuity = false;
        _scratchPose.CopyFrom(CurrentPose);
        var pose = _scratchPose;
        if (MeasureStageCost) Array.Clear(StageApplyTicks, 0, StageApplyTicks.Length);
        for (var i = 0; i < stages.Count; i++)
        {
            var stage = stages[i];
            if (!stage.isEnabled) continue;

            var startTimestamp = MeasureStageCost ? Stopwatch.GetTimestamp() : 0L;
            // A stage returning false means "this pose is final", so the result is still applied.
            var advance = stage.Apply(pose, _animationDeltaTime);
            if (MeasureStageCost) StageApplyTicks[i] = Stopwatch.GetTimestamp() - startTimestamp;
            if (!advance) break;
        }

        ApplyPoseToSkeletonTransforms(pose);

        OnPoseApplied?.Invoke(pose, _animationDeltaTime);
    }

    /// <summary>
    /// Whether this Unity frame carries a synthesis tick, and if so its timestep. Capped, frames are
    /// skipped so synthesis advances at <see cref="synthesisFrameRate"/> whatever the render rate.
    /// </summary>
    private bool TryBeginSynthesisTick()
    {
        if (!IsFrameRateRestricted)
        {
            _animationDeltaTime = Time.deltaTime;
            return true;
        }

        _timeTillNextAnimationUpdate -= Time.deltaTime;
        if (_timeTillNextAnimationUpdate > 0f) return false;

        _animationDeltaTime = 1f / synthesisFrameRate;
        _timeTillNextAnimationUpdate += _animationDeltaTime;
        return true;
    }

    /// <summary>Builds the pose layout, allocates the buffers, and seeds them from the rig.</summary>
    void InitCurrentPose()
    {
        PoseLayout = PoseLayoutBuilder.Build(skeleton, out var contacts);
        LeftFootContactHandle = contacts.Left;
        RightFootContactHandle = contacts.Right;

        CurrentPose = PoseBuffer.Allocate(PoseLayout, Allocator.Persistent);
        _scratchPose = PoseBuffer.Allocate(PoseLayout, Allocator.Persistent);

        // Velocities, angular velocities and contacts are already zeroed by Allocate.
        RigPoseReader.Seed(CurrentPose, SkeletonTransforms);
    }

    /// <summary>
    /// Refreshes <see cref="CurrentPose"/> from the Transforms, so each tick starts from where the
    /// rig actually is rather than from what the pipeline last produced.
    /// </summary>
    /// <remarks>
    /// Since <see cref="ApplyPoseToSkeletonTransforms"/> writes bone 0 frame-locally under this
    /// component's Transform, the frame derived from the pose built here is that Transform — which
    /// holds only while everything between bone 0's Transform and this one is identity.
    /// </remarks>
    void ConstructCurrentPoseFromSkeletonTransforms()
    {
        RigPoseReader.Read(CurrentPose, SkeletonTransforms, _animationDeltaTime);
    }

    /// <summary>
    /// Writes the finished pose onto the Transforms, re-anchored under this component's Transform:
    /// bone 0 is placed relative to the frame the pose itself implies, so a pose from anywhere in a
    /// database lands on the character. The Transform is then <em>advanced</em> by the frame's own
    /// velocity, so the character accumulates movement rather than being teleported each tick.
    /// </summary>
    private void ApplyPoseToSkeletonTransforms(PoseBuffer pose)
    {
        var positions = pose.Positions;
        var rotations = pose.Rotations;

        // Qualified because the SimulationFrame property below shadows the type of the same name.
        AnimationTools.SimulationFrame.Compute(pose, SkeletonData, SimulationFrame,
            out var framePos, out var frameRot);

        SkeletonTransforms[0].localPosition =
            AnimationTools.SimulationFrame.ToFrameLocal(positions[0], framePos, frameRot);
        SkeletonTransforms[0].localRotation =
            AnimationTools.SimulationFrame.ToFrameLocal(rotations[0], frameRot);

        for (var i = 1; i < skeleton.BoneCount; i++)
        {
            SkeletonTransforms[i].localRotation = rotations[i];
        }

        if (rootPositionsMask)
        {
            AnimationTools.SimulationFrame.ComputeVelocity(pose, SkeletonData, SimulationFrame,
                _animationDeltaTime, out var linearVelocity, out var yawRate);

            transform.position += transform.rotation * (Vector3)linearVelocity * _animationDeltaTime;
            var deltaRotation = MathExtensions.QuaternionFromScaledAngleAxis(
                new float3(0f, yawRate * _animationDeltaTime, 0f));
            transform.rotation = deltaRotation * transform.rotation;
        }

        // TODO: inertialized hips blending across a rootPositionsMask change, and toes-floor
        // penetration correction, both used to happen here. They were dropped when the pipeline
        // moved to stages; the intended home for each is a MoSynthStage running after the pose is
        // produced, rather than another special case inside the orchestrator.
    }

    private void OnValidate()
    {
        foreach (var stage in stages)
        {
            stage?.OnValidate();
        }
    }

    private void OnDestroy()
    {
        foreach (var stage in stages)
        {
            stage?.OnDestroy();
        }

        if (CurrentPose.IsCreated) CurrentPose.Dispose();
        if (_scratchPose.IsCreated) _scratchPose.Dispose();
    }

    // --- Unimplemented: pose adjustment and feature read-back ---------------------------------
    //
    // These all throw, but the crowd and collision control inputs (and Obstacle) still call them and
    // will throw the moment those paths run. Kept explicit because it is the contract they were
    // written against:
    //
    //  * Root*                 -- the synthesized root's motion state in world space.
    //  * Set*Adjustment        -- nudge the root off what the database produced, for collision
    //                             response and crowd steering, blended in rather than jumped.
    //  * Get*Feature           -- read back the trajectory being predicted, to steer against it.
    //
    // Root motion is this component's own Transform and the trajectory lives in the motion matching
    // stage's query vector, so finishing this means routing to those, not adding state here.

    public float3 RootVelocity { get; protected set; }
    public float3 RootAngularVelocity { get; protected set; }
    public float3 RootPosition { get; protected set; }
    public quaternion RootRotation { get; protected set; }

    public void SetRotAdjustment(quaternion adjustmentRotation)
    {
        throw new NotImplementedException();
    }

    public void SetPosAdjustment(float3 adjustmentPosition)
    {
        throw new NotImplementedException();
    }

    public float3 GetMainPositionFeature(int trajectoryIndex)
    {
        throw new NotImplementedException();
    }

    public float4 GetEnvironmentFeature(string featureName, int trajectoryIndex)
    {
        throw new NotImplementedException();
    }
}
}
