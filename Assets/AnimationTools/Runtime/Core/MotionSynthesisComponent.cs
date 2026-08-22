using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools
{
public class MotionSynthesisComponent : MonoBehaviour, ISkeletonProvider
{
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
    /// The transforms of the character controlled by this <see cref="MotionSynthesisComponent"/>.
    /// These are the transforms that will be used to render the character.
    /// </summary>
    [NonSerialized] public Transform[] SkeletonTransforms;

    [Tooltip(
        "The frame rate of the animation synthesis. This is used to calculate the time step of the motion synthesis." +
        "Set to 0 for uncapped.")]
    public float synthesisFrameRate = 30f;

    [SerializeReference] [SubclassSelector]
    public List<MoSynthStage> stages = new();

    [Tooltip("Whether to animate the root position by Motion Matching or not.")]
    // maybe change this to 'root motion'?
    public bool rootPositionsMask = true;

    public float3 RootVelocity { get; protected set; }
    public float3 RootAngularVelocity { get; protected set; }
    public float3 RootPosition { get; protected set; }
    public quaternion RootRotation { get; protected set; }

    /// <summary>
    /// True when an upstream stage replaced the pose discontinuously this tick (e.g. a motion
    /// matching search jumped to a new frame). Set by the stage that caused the jump; cleared at
    /// the start of every synthesis tick. Downstream blending stages read it to re-anchor.
    /// </summary>
    public bool PoseDiscontinuity { get; set; }

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

    bool IsFrameRateRestricted => synthesisFrameRate > 1e-5;

    private void Awake()
    {
        _skeleton = null;

        if (IsFrameRateRestricted)
        {
            _animationDeltaTime = 1.0f / synthesisFrameRate;
        }

        stages.RemoveAll(stage => stage == null);
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

        SkeletonTransforms = characterRig.Bind(_skeleton, indexZeroOverride: transform);

        var missingBoneCount = 0;
        for (var i = 1; i < SkeletonTransforms.Length; i++)
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

    private float _timeTillNextAnimationUpdate = 0f;

    private void LateUpdate()
    {
        if (!IsFrameRateRestricted)
        {
            _animationDeltaTime = Time.deltaTime;
        }
        else
        {
            _timeTillNextAnimationUpdate -= Time.deltaTime;
            if (_timeTillNextAnimationUpdate <= 0f)
            {
                _animationDeltaTime = 1f / synthesisFrameRate;
                _timeTillNextAnimationUpdate += _animationDeltaTime;
            }
            else
            {
                return;
            }
        }

        ConstructCurrentPoseFromSkeletonTransforms();

        PoseDiscontinuity = false;
        _scratchPose.CopyFrom(CurrentPose);
        var pose = _scratchPose;
        foreach (var stage in stages)
        {
            if (stage.isEnabled)
            {
                if (!stage.Apply(pose, _animationDeltaTime))
                {
                    break;
                }
            }
        }

        ApplyPoseToSkeletonTransforms(pose);

        OnPoseApplied?.Invoke(pose, _animationDeltaTime);
    }

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

    void ConstructCurrentPoseFromSkeletonTransforms()
    {
        // // Simulation Bone
        // float3 pos = skeletonTransforms[0].position;
        // var rot = skeletonTransforms[0].rotation;
        //
        // // world space to local space
        // var localSpacePos = math.mul(math.inverse(_mmTransformOriginRot), (pos - _mmTransformOriginPos));
        // var localSpaceRot = math.mul(math.inverse(_mmTransformOriginRot), rot);
        //
        // // local space to animation space
        // pose.JointLocalRotations[0] = math.mul(_animationSpaceOriginRot, localSpaceRot);
        //
        // pose.JointLocalPositions[0] = math.mul(_inverseAnimationSpaceOriginRot, localSpacePos) + _animationSpaceOriginPos;

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

        positions[0] = SkeletonTransforms[0].localPosition;
        rotations[0] = SkeletonTransforms[0].localRotation;

        for (var i = 1; i < SkeletonTransforms.Length; i++)
        {
            rotations[i] = SkeletonTransforms[i].localRotation;
        }

        // hip
        positions[1] = SkeletonTransforms[1].localPosition;


        // CurrentPose.LeftFootContact = ?;
        // CurrentPose.RightFootContact = ?;
    }

    private void ApplyPoseToSkeletonTransforms(PoseBuffer pose)
    {
        var positions = pose.Positions;
        var rotations = pose.Rotations;

        // Motion
        // if (rootPositionsMask)
        // {
        //     // Motion Matching Root Motion + Floor Height
        //     Vector3 simulationBone = pose.JointLocalPositions[0];
        //     // simulationBone.y = _floorHeight;
        //     transform.position = simulationBone;
        // }

        for (var i = 1; i < _skeleton.BoneCount; i++)
        {
            SkeletonTransforms[i].localRotation = rotations[i];
        }

        // hips
        SkeletonTransforms[1].localPosition = positions[1];

        // root
        if (rootPositionsMask)
        {
            var rootTransform = SkeletonTransforms[0];
            rootTransform.localPosition +=
                rootTransform.localRotation * pose.Velocities[0] * _animationDeltaTime;
            var angVel = pose.AngularVelocities[0];
            var rootRotation = MathExtensions.QuaternionFromScaledAngleAxis(angVel * _animationDeltaTime);
            rootTransform.localRotation = rootRotation * rootTransform.localRotation;
        }

        // if (blendPoses && _previousHipsPositionMask != rootPositionsMask)
        // {
        //     // Position Transition
        //     float3 offsetAngVel = float3.zero;
        //     Inertialization.InertializeJointTransition(_previousHipsPosition, float3.zero,
        //         targetHipsPosition, float3.zero,
        //         ref _offsetHipsPosition, ref offsetAngVel);
        // }
        //
        // if (blendPoses)
        // {
        //     float3 offsetAngVel = float3.zero;
        //     Inertialization.InertializeJointUpdate(targetHipsPosition, float3.zero,
        //         blendHalfLife, Time.deltaTime,
        //         ref _offsetHipsPosition, ref offsetAngVel,
        //         out float3 inertializedHipsPosition, out _);
        //     _targetBones[0].position = inertializedHipsPosition;
        // }
        // else
        // {
        //     _targetBones[0].position = targetHipsPosition;
        // }

        // // Toes-Floor Penetration
        // if (avoidToesFloorPenetration)
        // {
        //     const int leftToesIndex = 17;
        //     const int rightToesIndex = 21;
        //     float soleHeightOffset = Mathf.Min(_targetBones[leftToesIndex].TransformPoint(toesSoleOffset).y,
        //         _targetBones[rightToesIndex].TransformPoint(toesSoleOffset).y);
        //     soleHeightOffset = soleHeightOffset < _floorHeight ? -soleHeightOffset : 0.0f;
        //
        //     const float movingAverageFactor = 0.99f;
        //     _toesPenetrationMovingCorrection = _toesPenetrationMovingCorrection * movingAverageFactor +
        //                                        (soleHeightOffset + _floorHeight) * (1.0f - movingAverageFactor);
        //
        //     Vector3 hipsPos = _targetBones[0].position;
        //     hipsPos.y += _toesPenetrationMovingCorrection;
        //     _targetBones[0].position = hipsPos;
        // }
        //
        // // Update State
        // UpdatePreviousInertialization();
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