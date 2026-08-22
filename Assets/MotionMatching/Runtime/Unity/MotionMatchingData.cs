using System;
using System.Collections.Generic;
using AnimationTools;
using UnityEngine;
using Unity.Mathematics;
using System.IO;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace MotionMatching
{
/// <summary>
/// Defines all data used for Motion Matching in one avatar
/// Contains animation clips, feature definitions, and other data
/// </summary>
[CreateAssetMenu(fileName = "MotionMatchingData", menuName = "MotionMatching/MotionMatchingData")]
public class MotionMatchingData : ScriptableObject, IPoseSetSource
{
    [SerializeField]
    public List<AnnotatedAnimationClip> animationClips = new();

    [Tooltip("The pose skeleton: root must be the rig's identity armature node. Index 0 of this " +
             "skeleton IS the SimulationBone; index 1 is the first real bone (Hips).")]
    [SerializeField] private Skeleton skeleton = new();

    // TODO: Implement Savitzky-Golay filter or similar low-pass filter in Unity (before I was using Python implementation)
    //public bool SmoothSimulationBone; // Smooth the simulation bone (articial root added during pose extraction) using Savitzky-Golay filter
    public float
        contactVelocityThreshold =
            0.15f; // Minimum velocity of the foot to be considered in movement and not in contact with the ground

    [SerializeField]
    [Tooltip("Bone whose velocity drives foot-contact detection; leave unset to pick by name (LeftToe/RightToe).")]
    private SkeletonBone leftContactBone = new();

    [SerializeField]
    [Tooltip("Bone whose velocity drives foot-contact detection; leave unset to pick by name (LeftToe/RightToe).")]
    private SkeletonBone rightContactBone = new();

    public List<TrajectoryFeatureChannel> trajectoryFeatures = new();

    public List<PoseFeatureChannel> poseFeatures = new();

    private PoseSet _poseSet;

    public FeatureSet FeatureSet
    {
        get => _featureSet;
        private set => _featureSet = value;
    }

    // Information extracted from the rig's rest pose
    [SerializeField]
    private float3[] jointsLocalForward; // Local forward vector of each joint

    private FeatureSet _featureSet;
    public bool JointsLocalForwardError => jointsLocalForward == null;

    /// <summary>The pose skeleton. Index 0 is the SimulationBone; index 1 is the first real bone.</summary>
    public Skeleton Skeleton => skeleton;

    // IPoseSetSource --- the subset of this asset the pose-database pipeline actually reads.
    // Exposed as properties so a consumer that only wants a pose database (MotionField's config)
    // can supply one without also being a full Motion Matching asset.
    public List<AnnotatedAnimationClip> AnimationClips => animationClips;
    public float ContactVelocityThreshold => contactVelocityThreshold;
    public string LeftContactBoneName => leftContactBone?.Name;
    public string RightContactBoneName => rightContactBone?.Name;

    /// <summary>
    /// Frames of lookahead the longest trajectory feature needs. Poses closer than this to the end
    /// of a clip cannot be used for prediction.
    /// </summary>
    public int MaximumFramesPrediction
    {
        get
        {
            int maximum = 0;
            foreach (var t in trajectoryFeatures)
            {
                if (t.predictionFrames.Length > 0 && t.predictionFrames[^1] > maximum)
                {
                    maximum = t.predictionFrames[^1];
                }
            }

            return maximum;
        }
    }

    /// <summary>
    /// Checks that this asset can produce a pose database: a skeleton, at least one clip, and every
    /// clip's skeleton lining up with this one from bone 1 onward (bone 0 here is the
    /// SimulationBone, which no clip has). Returns false with a message suitable for an Inspector
    /// HelpBox. Never logs — inspectors call it every repaint.
    /// </summary>
    public bool TryValidate(out string error)
    {
        if (skeleton == null || !skeleton.IsSet)
        {
            error = "No skeleton assigned. Drop the rig's armature node here — it becomes the " +
                    "SimulationBone at index 0, with the first real bone at index 1.";
            return false;
        }

        if (animationClips == null || animationClips.Count == 0)
        {
            error = "Assign at least one animation clip.";
            return false;
        }

        for (var i = 0; i < animationClips.Count; i++)
        {
            var clip = animationClips[i];
            if (clip == null)
            {
                error = $"Clip {i} is empty.";
                return false;
            }

            if (!clip.TryValidate(out var clipError))
            {
                error = $"Clip \"{clip.name}\": {clipError}";
                return false;
            }

            if (!skeleton.MatchesFrom(1, clip.Skeleton))
            {
                error = $"Clip \"{clip.name}\" has {clip.Skeleton.BoneCount} bones, which do not match this " +
                        $"asset's skeleton from bone 1 onward ({skeleton.BoneCount - 1} bones).";
                return false;
            }
        }

        error = null;
        return true;
    }

    /// <summary>
    /// Null when <see cref="TryValidate"/> fails. Reached from OnValidate every repaint, so it
    /// reports through the inspector rather than the console.
    /// </summary>
    public PoseSet GetOrImportPoseSet()
    {
        if (_poseSet == null)
        {
            if (!TryValidate(out _)) return null;

            PROFILE.BEGIN_SAMPLE_PROFILING("Pose Import");
            PoseSerializer serializer = new PoseSerializer();
            if (!serializer.Deserialize(GetAssetPath(), name, skeleton, out PoseSet poseSet))
            {
                Debug.LogWarning("Failed to read pose set. Creating it in runtime instead.");
                ImportPoseSet();
#if UNITY_EDITOR
                if (_poseSet != null)
                {
                    PROFILE.BEGIN_SAMPLE_PROFILING("Pose Serialize");
                    PoseSerializer poseSerializer = new PoseSerializer();
                    poseSerializer.Serialize(_poseSet, GetAssetPath(), this.name);
                    PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Pose Serialize");
                }
#endif
            }
            else
            {
                _poseSet = poseSet;
            }

            PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Pose Import");
        }

        return _poseSet;
    }

    public void ImportPoseSet()
    {
        if (!TryValidate(out var error))
        {
            Debug.LogError($"MotionMatchingData \"{name}\": {error}");
            _poseSet = null;
            return;
        }

        _poseSet = new PoseSet();
        _poseSet.SetSkeleton(skeleton);
        for (int i = 0; i < animationClips.Count; i++)
        {
            var clip = animationClips[i];
            if (clip == null || clip.Skeleton == null || !skeleton.MatchesFrom(1, clip.Skeleton))
            {
                Debug.LogError($"[MotionMatchingData] Clip {i} (\"{(clip != null ? clip.name : "null")}\")'s " +
                                "skeleton does not match this asset's skeleton from bone 1 (Hips) onward; skipping.");
                continue;
            }

            // Extract poses
            if (!PoseExtractor.Extract(clip, _poseSet, this))
            {
                Debug.LogWarning("[FeatureDebug] Failed to extract poseSet from AnimationDat. Animation Index: " + i);
            }
        }

        _poseSet.ConvertTagsToNativeArrays();
        Debug.Log("Number of poses: " + _poseSet.NumberPoses);
    }

    /// <summary>
    /// Null when <see cref="TryValidate"/> fails, for the same reason as
    /// <see cref="GetOrImportPoseSet"/>.
    /// </summary>
    public FeatureSet GetOrImportFeatureSet()
    {
        if (FeatureSet == null)
        {
            if (!TryValidate(out _)) return null;

            PROFILE.BEGIN_SAMPLE_PROFILING("Feature Import");
            FeatureSerializer serializer = new FeatureSerializer();
            if (!serializer.Deserialize(GetAssetPath(), name, this, out FeatureSet featureSet))
            {
                Debug.LogWarning("Failed to read feature set. Creating it in runtime instead.");
                ImportFeatureSet();
#if UNITY_EDITOR
                PROFILE.BEGIN_SAMPLE_PROFILING("Feature Serialize");
                FeatureSerializer featureSerializer = new FeatureSerializer();
                featureSerializer.Serialize(FeatureSet, this, GetAssetPath(), this.name);
                PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Feature Serialize");
#endif
            }
            else
            {
                FeatureSet = featureSet;
            }

            PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Feature Import");
        }

        return FeatureSet;
    }

    public void ImportFeatureSet()
    {
        _poseSet = GetOrImportPoseSet();
        if (_poseSet == null)
        {
            TryValidate(out var error);
            Debug.LogError($"MotionMatchingData \"{name}\": {error}");
            return;
        }

        FeatureSet = new FeatureSet(this);
        FeatureSet.Extract(_poseSet, this);
        FeatureSet.NormalizeFeatures();
    }

    /// <summary>
    /// Per-joint local forward axis, derived from the rig's rest pose. The exported FBX rig is a
    /// T-Pose facing Unity forward, so the character's forward is +Z and its right is +X; each joint
    /// stores whichever of those axes it should be considered to face, pulled into its own local frame.
    /// </summary>
    public void ComputeJointsLocalForward()
    {
        jointsLocalForward = new float3[skeleton.BoneCount];
        jointsLocalForward[0] = math.forward();

        for (var i = 1; i < jointsLocalForward.Length; i++)
        {
            var boneName = skeleton.GetBone(i).Name;

            // Arms point sideways in a T-Pose, so their "forward" is the character's right axis
            var worldForward = math.forward();
            if (BoneNameConventions.IsLeftArmBone(boneName))
            {
                worldForward = -math.right();
            }
            else if (BoneNameConventions.IsRightArmBone(boneName))
            {
                worldForward = math.right();
            }

            jointsLocalForward[i] =
                math.mul(math.inverse(skeleton.RestCharacterRotation(i)), worldForward);
        }
    }

    /// <summary>
    /// Returns the local forward vector of the given joint index (after adding simulation bone).
    /// Vector computed from the rig's rest pose by <see cref="ComputeJointsLocalForward"/>.
    /// </summary>
    public float3 GetLocalForward(int jointIndex)
    {
        Debug.Assert(!JointsLocalForwardError, "JointsLocalForward is not initialized");
        return jointsLocalForward[jointIndex];
    }

    public string GetAssetPath()
    {
        string path = Path.Combine(Application.streamingAssetsPath, "MMDatabases", name);
#if UNITY_EDITOR
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
#endif
        return path;
    }
}
}
