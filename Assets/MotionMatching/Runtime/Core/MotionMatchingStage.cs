using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using AnimationTools;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Serialization;
using System.Linq;

namespace MotionMatching
{
[Serializable]
public class MotionMatchingStage : MoSynthStage
{
    private MotionSynthesisComponent _owner;
    
    [FormerlySerializedAs("characterController")] public MotionMatchingControlInput controlInput;
    
    
    public MotionMatchingData mmData;
    private PoseSet _poseSet;
    
    [SerializeReference] [SubclassSelector]
    public MotionMatchingSearch mmSearch = new BvhMotionMatchingSearch();
    
    /// <summary>
    /// The interval in seconds between two Motion Matching searches when there are no sudden input changes.
    /// </summary>
    public float searchInterval = 10.0f / 60.0f;
    
    /// <summary>
    /// The time left until the next search.
    /// </summary>
    private float _searchTimeLeft;

    [Tooltip("How important is the trajectory (future positions + future directions)")]
    [Range(0.0f, 1.0f)]
    public float responsiveness = 1.0f;

    [Tooltip("How important is the current pose")] [Range(0.0f, 1.0f)]
    public float quality = 1.0f;

    [SerializeField]
    private int minFrameSwitchDistance = 20;

    
    // TODO: editor inspector for feature weights
    [SerializeField]
    private List<float> featureWeights = new();
    NativeArray<float> _featureWeights;
    public NativeArray<float> FeatureWeights => _featureWeights;

    private NativeArray<float> _queryFeatureVector;
    public NativeArray<float> QueryFeatureVector => _queryFeatureVector;

    
    /// <summary>
    /// Current frame index in the pose/feature set
    /// </summary>
    public int CurrentFrame { get; private set; }

    private NativeArray<bool> _tagMask;

    /// <summary>
    /// Current frame index as float to keep track of variable frame rate
    /// </summary>
    private float _currentFrameTime;

    private float _databaseFrameRate;


    // Contact TODO: this frame? prev frame ?
    public bool IsLeftFootContact { get; private set; }
    public bool IsRightFootContact { get; private set; }

    public override void Init(MotionSynthesisComponent motionSynthesisComponent)
    {
        _owner = motionSynthesisComponent;
        _poseSet = mmData.GetOrImportPoseSet();
        var featureSet = mmData.GetOrImportFeatureSet();

        if (_poseSet == null || featureSet == null)
        {
            mmData.TryValidate(out var error);
            Debug.LogError($"MotionMatchingStage: MotionMatchingData \"{mmData.name}\" is not usable — {error}");
            return;
        }

        Assert.IsTrue(controlInput, "mmCharacterController not set");
        // Force search on significant input change
        controlInput.OnHighInputChange += () => { _searchTimeLeft = 0; };
        
        Assert.IsTrue(
            motionSynthesisComponent.SkeletonTransforms.Length == _poseSet.Skeleton.BoneCount,
            "Number of Skeleton transforms does not match skeleton bones " +
            "in MotionMatchingData.");
        
        _databaseFrameRate = 1f / _poseSet.FrameTime;
        
        _featureWeights = new NativeArray<float>(featureSet.FeatureSize, Allocator.Domain);
        // copy serialized weights
        for (int i = 0; i < math.min(featureWeights.Count, _featureWeights.Length); i++)
        {
            _featureWeights[i] = featureWeights[i];
        }
        _queryFeatureVector = new NativeArray<float>(featureSet.FeatureSize, Allocator.Domain);
        
        _tagMask = new NativeArray<bool>(featureSet.NumberFeatureVectors, Allocator.Domain);
        for (var i = 0; i < _tagMask.Length; i++)
        {
            _tagMask[i] = true;
        }
        
        // Search first Frame valid (to start with a valid pose)
        for (var i = 0; i < featureSet.NumberFeatureVectors; i++)
        {
            if (!featureSet.IsValidFeature(i)) continue;
            CurrentFrame = i;
            break;
        }
        
        mmSearch.Initialize(featureSet, _tagMask, _featureWeights);
    }

    public override Skeleton GetSkeleton(Skeleton inSkeleton)
    {
        _poseSet = mmData.GetOrImportPoseSet();
        // Null when mmData is misconfigured; MotionSynthesisComponent reports "no stage provided
        // a skeleton" and disables itself, and the asset's own inspector says what is wrong.
        return _poseSet?.Skeleton;
    }
    
    public override bool Apply(PoseBuffer pose, float deltaTime)
    {
        // _searchTimeLeft -= deltaTime;
        if (_searchTimeLeft <= 0)
        {
            FillQueryVector();
            
            var currentDistance = float.MaxValue;
            var featureSet = mmData.FeatureSet;
            var isCurrentFrameValid = featureSet.IsValidFeature(CurrentFrame) && _tagMask[CurrentFrame];
            if(isCurrentFrameValid)
            {
                var currentFeatureVector = featureSet.GetFeatureVector(CurrentFrame);
                currentDistance = SqrDistance(_queryFeatureVector, currentFeatureVector, _featureWeights);
            }

            var bestFrame = mmSearch.FindBestFrame(_queryFeatureVector, currentDistance);
            
            if(isCurrentFrameValid && bestFrame == -1) bestFrame = CurrentFrame;
            Debug.Assert(bestFrame != -1, "Motion Matching is not able to find any valid pose. Maybe the motion database is empty or the query tag used produces an empty set of poses?");
            
            if(math.abs(CurrentFrame - bestFrame) > minFrameSwitchDistance)
            {
                CurrentFrame = bestFrame;
                _owner.PoseDiscontinuity = true;
            }
            
            _searchTimeLeft = searchInterval;
        }
        else
        {
            _searchTimeLeft -= deltaTime;
        }
        
        // Advance frames with time
        var preAdvanceFrame = CurrentFrame;
        _currentFrameTime = CurrentFrame + math.frac(_currentFrameTime);
        _currentFrameTime += deltaTime * _databaseFrameRate;
        CurrentFrame = (int)math.floor(_currentFrameTime);

        // Running off the end of a clip into the next one is a pose jump, same as a search switch.
        if (CurrentFrame != preAdvanceFrame &&
            _poseSet.GetAnimationClipIndex(CurrentFrame) != _poseSet.GetAnimationClipIndex(preAdvanceFrame))
        {
            _owner.PoseDiscontinuity = true;
        }
        
        pose.CopyFrom(_poseSet.GetPoseBuffer(CurrentFrame));
        
        // pose.jointLocalPositions[0] = math.transform(_animToWorld, pose.jointLocalPositions[0]);
        // pose.jointLocalRotations[0] = math.mul(new quaternion(_animToWorld), pose.jointLocalRotations[0]);
        return true;
    }
    
    [Pure]
    public static float SqrDistance(ReadOnlySpan<float> featureVectorA, ReadOnlySpan<float> featureVectorB, ReadOnlySpan<float> featureWeights)
    {
        var sqrDistance = 0.0f;
        for (int i = 0; i < featureVectorA.Length; i++)
        {
            var diff = featureVectorA[i] - featureVectorB[i];
            sqrDistance += diff * diff * featureWeights[i];
        }
        return sqrDistance;
    }
    
    public void FillQueryVector()
    {
        var simulationBone = _owner.SkeletonTransforms[0];
        var queryFeatureSpan = _queryFeatureVector.AsSpan();
        var featureSet = mmData.GetOrImportFeatureSet();

        // Trajectory features
        for (var i = 0; i < mmData.trajectoryFeatures.Count; i++)
        {
            var featureDef = mmData.trajectoryFeatures[i];
            var featureSize = featureSet.GetTrajectoryFeatureFloatCount(i);
            for (var p = 0; p < featureSet.GetPredictionCount(i); ++p)
            {
                var feature = queryFeatureSpan.Slice(featureSet.GetTrajectoryFeatureOffset(i, p), featureSize);
                controlInput.GetTrajectoryFeature(featureDef, p, simulationBone, feature);
            }
        }

        featureSet.NormalizeTrajectory(queryFeatureSpan);

        // TODO:
        // The currentPose of the character could be quite different from the
        // one poses stored in mmData due to retargeting.
        // We can use the currentPose if we implement a backpropagation
        // that can inverse the retargeting.

        featureSet.GetPoseFeatures(queryFeatureSpan.Slice(featureSet.PoseOffset, featureSet.PoseFloatCount),
            CurrentFrame);
    }

    // TODO call from editor
    public void UpdateFeatureWeights()
    {
        var featureSet = mmData.GetOrImportFeatureSet();
        var definitionCount = mmData.trajectoryFeatures.Count + mmData.poseFeatures.Count;

        // One weight per feature definition is read from the head of the same array this then
        // fills in per float, so the source values have to be taken before the first write.
        var definitionWeights = new float[definitionCount];
        for (var i = 0; i < definitionCount && i < _featureWeights.Length; i++)
        {
            definitionWeights[i] = _featureWeights[i];
        }

        for (var i = 0; i < mmData.trajectoryFeatures.Count; i++)
        {
            var featureSize = featureSet.GetTrajectoryFeatureFloatCount(i);
            var weight = definitionWeights[i] * responsiveness;
            for (var p = 0; p < featureSet.GetPredictionCount(i); ++p)
            {
                var offset = featureSet.GetTrajectoryFeatureOffset(i, p);
                for (var f = 0; f < featureSize; f++)
                {
                    _featureWeights[offset + f] = weight;
                }
            }
        }

        for (var i = 0; i < mmData.poseFeatures.Count; i++)
        {
            var weight = definitionWeights[mmData.trajectoryFeatures.Count + i] * quality;
            var offset = featureSet.PoseOffset + i * FeatureSet.FloatsPerPoseFeature;
            for (var f = 0; f < FeatureSet.FloatsPerPoseFeature; f++)
            {
                _featureWeights[offset + f] = weight;
            }
        }
    }

    public override void OnValidate()
    {
        if(mmData == null) return;

        // Null while the asset is misconfigured; its own inspector reports why, and OnValidate
        // runs every repaint, so this must stay silent.
        var featureSet = mmData.GetOrImportFeatureSet();
        if (featureSet == null) return;

        var featureSize = featureSet.FeatureSize;
        if(featureWeights.Count < featureSize)
        {
            for (var i = featureWeights.Count; i < featureSize; i++)
            {
                featureWeights.Add(1.0f);
            }
        }
        else if(featureWeights.Count > featureSize)
        {
            featureWeights.RemoveRange(featureSize, featureWeights.Count - featureSize);
        }

        if (_featureWeights.Length != featureSize)
        {
            _featureWeights = new NativeArray<float>(featureSize, Allocator.Domain);
        }

        for (int i = 0; i < featureWeights.Count; i++)
        {
            _featureWeights[i] = featureWeights[i];
        }
    }

    public MotionMatchingData MmData => mmData;
    public float DatabaseFrameTime => mmData.GetOrImportPoseSet().FrameTime;
    public float3 RootVelocity { get; }
    public float3 RootAngularVelocity { get; }
    public float3 RootPosition { get; }
    public quaternion RootRotation { get; }
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