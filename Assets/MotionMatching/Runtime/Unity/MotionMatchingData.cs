using System;
using System.Collections.Generic;
using AnimationTools;
using UnityEngine;
using Unity.Mathematics;
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

    [Tooltip("The pose skeleton: root must be the rig's root bone, identical to each clip's " +
             "skeleton. Index 0 is that root bone.")]
    [SerializeField] private Skeleton skeleton = new();

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

    /// <summary>The pose skeleton: the rig's root bone, identical to each clip's skeleton.</summary>
    public Skeleton Skeleton => skeleton;

    // IPoseSetSource --- the subset of this asset the pose-database pipeline actually reads.
    // Exposed as properties so a consumer that only wants a pose database (MotionField's config)
    // can supply one without also being a full Motion Matching asset.
    public List<AnnotatedAnimationClip> AnimationClips => animationClips;
    public float ContactVelocityThreshold => contactVelocityThreshold;
    public string LeftContactBoneName => leftContactBone?.Name;
    public string RightContactBoneName => rightContactBone?.Name;

    /// <summary>
    /// Whether this asset describes a feature vector at all. False makes it a pose-only database:
    /// the generator bakes the .mmpose and skips the .mmfeatures, and <see cref="MotionMatchingStage"/>
    /// refuses it, since with no channels every frame scores identically.
    /// </summary>
    public bool HasFeatureChannels => trajectoryFeatures.Count > 0 || poseFeatures.Count > 0;

    /// <summary>
    /// Frames of lookahead the furthest-forward trajectory sample needs. Poses closer than this to
    /// the end of a clip cannot be used for prediction.
    /// </summary>
    public int MaximumFramesPrediction => FramesPrediction(ahead: true);

    /// <summary>
    /// Frames of history the furthest-back trajectory sample needs, mirroring
    /// <see cref="MaximumFramesPrediction"/> at the start of a clip.
    /// </summary>
    /// <remarks>
    /// A negative prediction frame is how a feature reaches into the past. Both the classic
    /// motion-matching past-trajectory term and a PFNN-style trajectory window (which spans roughly
    /// a second either side of the query frame) are authored that way.
    /// </remarks>
    public int MaximumFramesHistory => FramesPrediction(ahead: false);

    /// <summary>
    /// The furthest a trajectory feature reaches in one direction, as a non-negative frame count.
    /// Prediction frames are not required to be sorted, so this scans them all.
    /// </summary>
    private int FramesPrediction(bool ahead)
    {
        var furthest = 0;
        foreach (var feature in trajectoryFeatures)
        {
            if (feature?.predictionFrames == null) continue;

            foreach (var frame in feature.predictionFrames)
            {
                var distance = ahead ? frame : -frame;
                if (distance > furthest) furthest = distance;
            }
        }

        return furthest;
    }

    /// <summary>
    /// Checks that this asset can produce a pose database: a skeleton, at least one clip, and every
    /// clip's skeleton being structurally identical to this one. Returns false with a message
    /// suitable for an Inspector HelpBox. Never logs — inspectors call it every repaint.
    /// </summary>
    public bool TryValidate(out string error) => PoseSetImporter.TryValidate(this, out error);

    /// <summary>
    /// Null when <see cref="TryValidate"/> fails. Reached from OnValidate every repaint, so it
    /// reports through the inspector rather than the console.
    /// </summary>
    public PoseSet GetOrImportPoseSet()
    {
        if (_poseSet == null)
        {
            PROFILE.BEGIN_SAMPLE_PROFILING("Pose Import");
            _poseSet = PoseSetImporter.GetOrImport(this);
            PROFILE.END_AND_PRINT_SAMPLE_PROFILING("Pose Import");
        }

        return _poseSet;
    }

    public void ImportPoseSet() => _poseSet = PoseSetImporter.Import(this);

    /// <summary>
    /// Null when <see cref="TryValidate"/> fails, for the same reason as
    /// <see cref="GetOrImportPoseSet"/>, and null for a pose-only asset
    /// (<see cref="HasFeatureChannels"/>).
    /// </summary>
    public FeatureSet GetOrImportFeatureSet()
    {
        if (FeatureSet == null)
        {
            if (!TryValidate(out _)) return null;
            if (!HasFeatureChannels) return null;

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

        for (var i = 0; i < jointsLocalForward.Length; i++)
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
    /// The local forward vector of one joint, indexed like <see cref="Skeleton"/>. Computed from
    /// the rig's rest pose by <see cref="ComputeJointsLocalForward"/>.
    /// </summary>
    public float3 GetLocalForward(int jointIndex)
    {
        Debug.Assert(!JointsLocalForwardError, "JointsLocalForward is not initialized");
        return jointsLocalForward[jointIndex];
    }

    public string GetAssetPath() => PoseSetImporter.GetDatabasePath("MMDatabases", name);
}
}
