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
/// Awake settles things in order, each step depending on the last: ask the stages for a skeleton,
/// bind it to the scene rig, build the pose layout, then Init the stages.
/// <para>
/// Each tick: read the rig into <see cref="CurrentPose"/>, copy to a scratch buffer, run the
/// enabled stages over it until one returns false, apply the result, raise
/// <see cref="OnPoseApplied"/>.
/// </para>
/// </remarks>
public class MotionSynthesisComponent : MonoBehaviour, ISkeletonProvider
{
    /// <summary>
    /// Bone 0: the character's ground-plane proxy, not a bone of the art rig. Bound to this
    /// component's own Transform, and the frame that root motion and character-space features are
    /// relative to.
    /// </summary>
    public const int SimulationBoneIndex = 0;

    /// <summary>
    /// Bone 1: the rig's root joint. Besides the simulation bone, the only bone whose
    /// <em>position</em> is animated — the rest contribute rotation only, keeping bone lengths fixed.
    /// </summary>
    public const int HipsBoneIndex = 1;

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

    // Built from the stages (an asset rig at its rest pose), deliberately not from characterRig:
    // a skeleton over a live scene rig would report whatever pose it's currently animated to as
    // its rest pose, which silently corrupts FK.
    private Skeleton _skeleton;
    public Skeleton Skeleton => _skeleton;

    [SerializeField]
    [Tooltip("The rig this component drives; bones are bound to the pipeline skeleton by name.")]
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
        _skeleton = null;

        if (IsFrameRateRestricted)
        {
            _animationDeltaTime = 1.0f / synthesisFrameRate;
        }

        stages.RemoveAll(stage => stage == null);
        StageApplyTicks = new long[stages.Count];
        foreach (var stage in stages)
        {
            _skeleton = stage.GetSkeleton(_skeleton);
        }

        if (_skeleton == null)
        {
            Debug.LogError($"MotionSynthesisComponent \"{name}\": no stage provided a skeleton " +
                           "(e.g. a MotionMatchingStage). Disabling the component.");
            enabled = false;
            return;
        }

        if (characterRig == null) characterRig = new SkeletonBoneOverrides();
        if (!characterRig.IsSet)
        {
            Debug.LogWarning($"MotionSynthesisComponent \"{name}\": characterRig is unset; searching under this component's own transform.");
            characterRig.SetRoot(transform);
        }

        // The simulation bone is this component's own Transform rather than anything in the art
        // rig, so it is supplied directly instead of being resolved by name against the rig.
        SkeletonTransforms = characterRig.Bind(_skeleton, indexZeroOverride: transform);

        var missingBoneCount = 0;
        for (var i = SimulationBoneIndex + 1; i < SkeletonTransforms.Length; i++)
        {
            if (SkeletonTransforms[i] == null) missingBoneCount++;
        }

        if (missingBoneCount > 0)
        {
            Debug.LogError($"MotionSynthesisComponent \"{name}\": {missingBoneCount} bone(s) could not be bound to characterRig (see errors above). Disabling the component.");
            enabled = false;
            return;
        }

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
        PoseLayout = PoseLayoutBuilder.Build(_skeleton, out var contacts);
        LeftFootContactHandle = contacts.Left;
        RightFootContactHandle = contacts.Right;

        CurrentPose = PoseBuffer.Allocate(PoseLayout, Allocator.Persistent);
        _scratchPose = PoseBuffer.Allocate(PoseLayout, Allocator.Persistent);

        var positions = CurrentPose.Positions;
        var rotations = CurrentPose.Rotations;
        for (var i = 0; i < SkeletonTransforms.Length; i++)
        {
            rotations[i] = SkeletonTransforms[i].localRotation;
            positions[i] = SkeletonTransforms[i].localPosition;
        }
        // Velocities, angular velocities and contacts are already zeroed by Allocate.
    }

    /// <summary>
    /// Refreshes <see cref="CurrentPose"/> from the Transforms, so each tick starts from where the
    /// rig actually is rather than from what the pipeline last produced.
    /// </summary>
    /// <remarks>
    /// The velocity pass must run first: it differences against the previous tick's values, still
    /// held in the buffer, so it has to read them before the pose pass overwrites them. Note it
    /// writes per-tick deltas, not the per-second rates the velocity channels normally carry —
    /// tolerable only because any stage that replaces the pose overwrites them first.
    /// </remarks>
    void ConstructCurrentPoseFromSkeletonTransforms()
    {
        var positions = CurrentPose.Positions;
        var rotations = CurrentPose.Rotations;
        var velocities = CurrentPose.Velocities;
        var angularVelocities = CurrentPose.AngularVelocities;

        for (var i = 0; i < angularVelocities.Length; i++)
        {
            var inverseLocalRotation = Quaternion.Inverse(rotations[i]);
            angularVelocities[i] =
                (SkeletonTransforms[i].localRotation * inverseLocalRotation).eulerAngles;
            velocities[i] =
                (float3)SkeletonTransforms[i].localPosition - positions[i];
        }

        // Only the simulation bone and the hips carry position; every other bone is rotation only.
        positions[SimulationBoneIndex] = SkeletonTransforms[SimulationBoneIndex].localPosition;
        positions[HipsBoneIndex] = SkeletonTransforms[HipsBoneIndex].localPosition;

        for (var i = 0; i < SkeletonTransforms.Length; i++)
        {
            rotations[i] = SkeletonTransforms[i].localRotation;
        }

        // Foot contacts are not recoverable from the Transforms alone, so whichever stage owns them
        // writes them into the pipeline pose instead of them being seeded here.
    }

    /// <summary>
    /// Writes the finished pose onto the Transforms. Rotations and the hips position are absolute;
    /// the simulation bone is instead <em>advanced</em> by the root-motion velocities, so the
    /// character accumulates movement rather than being teleported each tick.
    /// </summary>
    private void ApplyPoseToSkeletonTransforms(PoseBuffer pose)
    {
        var positions = pose.Positions;
        var rotations = pose.Rotations;

        for (var i = SimulationBoneIndex + 1; i < _skeleton.BoneCount; i++)
        {
            SkeletonTransforms[i].localRotation = rotations[i];
        }

        SkeletonTransforms[HipsBoneIndex].localPosition = positions[HipsBoneIndex];

        if (rootPositionsMask)
        {
            var simulationBone = SkeletonTransforms[SimulationBoneIndex];
            simulationBone.localPosition +=
                simulationBone.localRotation * pose.Velocities[SimulationBoneIndex] * _animationDeltaTime;
            var angularVelocity = pose.AngularVelocities[SimulationBoneIndex];
            var deltaRotation = MathExtensions.QuaternionFromScaledAngleAxis(angularVelocity * _animationDeltaTime);
            simulationBone.localRotation = deltaRotation * simulationBone.localRotation;
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
    // Root motion now lives in RootMotionCorrectionStage and the trajectory in the motion matching
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
