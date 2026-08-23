using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using AnimationTools;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Serialization;

namespace MotionMatching
{
/// <summary>
/// Synthesis by search: every tick this stage plays the next frame of an animation database, and
/// periodically asks "is there a better frame to be playing?" — comparing what the control input
/// wants (the query vector) against every frame's precomputed feature vector, and jumping if the
/// answer is a clear yes.
/// </summary>
/// <remarks>
/// Three replaceable parts: <see cref="mmData"/> (the database), <see cref="controlInput"/> (what
/// the character is being asked to do) and <see cref="mmSearch"/> (how the database is searched).
/// <para>
/// Playback and search run on separate clocks — playback advances every tick, search at most once
/// per <see cref="searchInterval"/>. This stage also sources the pipeline skeleton, since the
/// skeleton comes from the database rather than the scene.
/// </para>
/// </remarks>
[Serializable]
public class MotionMatchingStage : MoSynthStage
{
    private MotionSynthesisComponent _owner;

    /// <summary>Supplies the query vector: what the character is being asked to do next.</summary>
    [FormerlySerializedAs("characterController")] public MotionMatchingControlInput controlInput;

    /// <summary>The animation database and its feature configuration.</summary>
    public MotionMatchingData mmData;

    private PoseSet _poseSet;

    /// <summary>
    /// How the database is searched. Swappable strategy — see <see cref="MotionMatchingSearch"/>.
    /// </summary>
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

    /// <summary>
    /// A winning frame closer than this to the one playing is ignored. Nearby frames look almost
    /// identical, so jumping to one costs a discontinuity and buys nothing.
    /// </summary>
    [SerializeField]
    private int minFrameSwitchDistance = 20;

    // TODO: editor inspector for feature weights
    /// <summary>
    /// Authored importance, one entry per feature definition. Expanded into the per-float weights
    /// the search uses by <see cref="UpdateFeatureWeights"/>.
    /// </summary>
    [SerializeField]
    private List<float> featureWeights = new();

    /// <summary>Per-float weights handed to the search; length is the feature vector size.</summary>
    NativeArray<float> _featureWeights;

    public NativeArray<float> FeatureWeights => _featureWeights;

    /// <summary>
    /// The desired next state, laid out exactly like a database feature vector so the two compare
    /// directly. Rebuilt each search by <see cref="FillQueryVector"/>.
    /// </summary>
    private NativeArray<float> _queryFeatureVector;

    public NativeArray<float> QueryFeatureVector => _queryFeatureVector;

    /// <summary>
    /// Current frame index in the pose/feature set
    /// </summary>
    public int CurrentFrame { get; private set; }

    /// <summary>
    /// Narrows which frames the search may return. Currently all-true — the hook for tag queries
    /// ("only walking frames") exists, but nothing populates it yet.
    /// </summary>
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
        if (_searchTimeLeft <= 0)
        {
            SearchForBetterFrame();
            _searchTimeLeft = searchInterval;
        }
        else
        {
            _searchTimeLeft -= deltaTime;
        }

        AdvancePlayback(deltaTime);

        pose.CopyFrom(_poseSet.GetPoseBuffer(CurrentFrame));
        return true;
    }

    /// <summary>Rebuilds the query vector and jumps to a better frame, if one is worth the jump.</summary>
    private void SearchForBetterFrame()
    {
        FillQueryVector();

        // Score the frame already playing, so the search only reports something better.
        var currentDistance = float.MaxValue;
        var featureSet = mmData.FeatureSet;
        var isCurrentFrameValid = featureSet.IsValidFeature(CurrentFrame) && _tagMask[CurrentFrame];
        if (isCurrentFrameValid)
        {
            var currentFeatureVector = featureSet.GetFeatureVector(CurrentFrame);
            currentDistance = SqrDistance(_queryFeatureVector, currentFeatureVector, _featureWeights);
        }

        var bestFrame = mmSearch.FindBestFrame(_queryFeatureVector, currentDistance);

        if (isCurrentFrameValid && bestFrame == -1) bestFrame = CurrentFrame;
        Debug.Assert(bestFrame != -1, "Motion Matching is not able to find any valid pose. Maybe the motion database is empty or the query tag used produces an empty set of poses?");

        if (math.abs(CurrentFrame - bestFrame) > minFrameSwitchDistance)
        {
            CurrentFrame = bestFrame;
            _owner.PoseDiscontinuity = true;
        }
    }

    /// <summary>
    /// Advances the playhead by real time, converted to database frames. Fractional frame time
    /// carries across ticks, so playback stays at the right speed at any synthesis rate.
    /// </summary>
    private void AdvancePlayback(float deltaTime)
    {
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
    }

    /// <summary>
    /// The metric every search must agree on: weighted squared distance. Squared rather than true
    /// distance because only the ordering matters, and it lets a search abandon a candidate early.
    /// </summary>
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

    /// <summary>
    /// Builds the vector to search for: the trajectory the control input wants, then the pose
    /// features to hold on to. Offsets come from the <see cref="FeatureSet"/> so the query matches
    /// the database layout.
    /// </summary>
    public void FillQueryVector()
    {
        var simulationBone = _owner.SkeletonTransforms[MotionSynthesisComponent.SimulationBoneIndex];
        var queryFeatureSpan = _queryFeatureVector.AsSpan();
        var featureSet = mmData.GetOrImportFeatureSet();

        // One slice per prediction horizon.
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

        // The database's trajectory floats are normalized, so the query's must be too.
        featureSet.NormalizeTrajectory(queryFeatureSpan);

        // Pose features come from the frame playing, not the character, which keeps the query in
        // the database's own pose space.
        // TODO:
        // The currentPose of the character could be quite different from the
        // one poses stored in mmData due to retargeting.
        // We can use the currentPose if we implement a backpropagation
        // that can inverse the retargeting.
        featureSet.GetPoseFeatures(queryFeatureSpan.Slice(featureSet.PoseOffset, featureSet.PoseFloatCount),
            CurrentFrame);
    }

    /// <summary>
    /// Expands <see cref="featureWeights"/> into per-float weights, scaling trajectory features by
    /// <see cref="responsiveness"/> and pose features by <see cref="quality"/>.
    /// </summary>
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

    /// <summary>Keeps the weight list as long as the feature vector, padding new entries with 1.</summary>
    public override void OnValidate()
    {
        if (mmData == null) return;

        // Null while the asset is misconfigured; its own inspector reports why, and OnValidate
        // runs every repaint, so this must stay silent.
        var featureSet = mmData.GetOrImportFeatureSet();
        if (featureSet == null) return;

        var featureSize = featureSet.FeatureSize;
        if (featureWeights.Count < featureSize)
        {
            for (var i = featureWeights.Count; i < featureSize; i++)
            {
                featureWeights.Add(1.0f);
            }
        }
        else if (featureWeights.Count > featureSize)
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

    // Unimplemented, for the same reason as the matching surface on MotionSynthesisComponent --
    // see the comment there.

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
