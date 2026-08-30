using System;
using System.Collections.Generic;
using System.IO;
using AnimationTools;
using UnityEngine;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace MotionField
{
/// <summary>
/// Everything a motion field needs: which animations form the database, and the hyperparameters
/// used to train and run the value function over it.
///
/// This implements <see cref="IPoseSetSource"/> so it can produce its own .mmpose database
/// without being a <see cref="MotionMatchingData"/>. It deliberately carries none of the
/// trajectory/pose feature machinery -- a motion field searches on full-body pose and velocity, so
/// the feature set a Motion Matching query needs has no meaning here.
/// </summary>
[CreateAssetMenu(fileName = "MotionFieldConfig", menuName = "MotionField/MotionFieldConfig")]
public class MotionFieldConfig : ScriptableObject, IPoseSetSource
{
    [Header("Animation Database")]
    [Tooltip("Clips forming the motion field.")]
    public List<AnnotatedAnimationClip> animationClips = new();

    [Tooltip("The pose skeleton: root must be the rig's root bone, identical to each clip's " +
             "skeleton. Index 0 is that root bone.")]
    [SerializeField] private Skeleton skeleton = new();

    [Tooltip("Foot speed below which a toe counts as planted.")]
    public float contactVelocityThreshold = 0.15f;

    [Tooltip("Bone whose velocity drives foot-contact detection; leave unset to pick by name (LeftToe/RightToe).")]
    public SkeletonBone leftContactBone = new();

    [Tooltip("Bone whose velocity drives foot-contact detection; leave unset to pick by name (LeftToe/RightToe).")]
    public SkeletonBone rightContactBone = new();

    [Header("Motion Field")]
    [Tooltip("Neighbours considered per step. Also the number of candidate actions, since each " +
             "action emphasises one neighbour.")]
    [Min(2)] public int kNeighbors = 15;

    [Tooltip("Per-step pull back toward the nearest database state. Stops the field drifting into " +
             "regions it has no data for; too high and it just replays the database.")]
    [Range(0f, 1f)] public float tugRatio = 0.1f;

    [Tooltip("Overall weight on the joint-position half of the similarity metric. Per-bone " +
             "weights multiply on top of this. The defaults put roughly three quarters of the " +
             "distance on positions, matching the reference implementation -- a velocity-dominated " +
             "metric matches states that merely move alike, pose ignored, which is how the field " +
             "wanders into idle poses mid-stride.")]
    public float posWeight = 1.0f;

    [Tooltip("Overall weight on the joint-velocity half of the similarity metric (velocities are " +
             "m/s, so raw magnitudes run several times the position half's metres -- 0.06 is what " +
             "lands the measured split near 75/25 on this data). Per-bone weights multiply on top " +
             "of this.")]
    public float velWeight = 0.06f;

    [Tooltip("Reward bonus for actions that land in moving states -- the reference " +
             "implementation's state_score * factor term. At 0 every reward is <= 0 and an idle " +
             "pose that faces the goal scores a perfect zero forever, so the policy freezes into " +
             "it. Changing this needs a retrain.")]
    [Min(0f)] public float locomotionFactor = 1.0f;

    [Tooltip("Root speed, m/s, at which a database state counts as fully moving. The locomotion " +
             "score ramps linearly from 0 below it.")]
    [Min(0.01f)] public float locomotionSpeedThreshold = 0.5f;

    /// <summary>
    /// Per-joint emphasis in the similarity metric, keyed by skeleton joint name.
    /// </summary>
    /// <remarks>
    /// Hidden from the default inspector on purpose: as a raw list this is 23 nameless elements
    /// with no indication of which bone each one is. <c>MotionFieldConfigEditor</c> draws it as the
    /// skeleton hierarchy instead. Only entries that differ from 1 need to be stored, and joints
    /// absent from the list are simply left at 1.
    /// </remarks>
    [HideInInspector] public List<BoneWeight> boneWeights = new();

    /// <summary>
    /// How much one joint counts toward the k-NN distance.
    /// </summary>
    /// <remarks>
    /// Keyed by <see cref="name"/> rather than by index because the metric sums over joints in
    /// depth-first order, which is not guaranteed to survive a skeleton being re-extracted with a
    /// different hierarchy. Resolving by name on the Python side means a moved joint takes its
    /// weight with it instead of silently applying it to whatever now sits at that index.
    /// </remarks>
    [Serializable]
    public struct BoneWeight
    {
        [Tooltip("Skeleton joint name, as it appears in this config's skeleton.")]
        public string name;

        [Tooltip("Scales this joint's position contribution to the distance.")] [Min(0f)]
        public float position;

        [Tooltip("Scales this joint's velocity contribution to the distance.")] [Min(0f)]
        public float velocity;

        public BoneWeight(string name, float position = 1f, float velocity = 1f)
        {
            this.name = name;
            this.position = position;
            this.velocity = velocity;
        }

        public bool IsNeutral => Mathf.Approximately(position, 1f) && Mathf.Approximately(velocity, 1f);
    }

    /// <summary>This joint's weight, or a neutral one when the list does not mention it.</summary>
    public BoneWeight GetBoneWeight(string jointName)
    {
        for (int i = 0; i < boneWeights.Count; i++)
        {
            if (boneWeights[i].name == jointName) return boneWeights[i];
        }

        return new BoneWeight(jointName);
    }

    /// <summary>
    /// Store a joint's weight, dropping the entry again once it is back at 1/1 so the asset does
    /// not accumulate rows that say nothing.
    /// </summary>
    public void SetBoneWeight(BoneWeight weight)
    {
        int existing = boneWeights.FindIndex(w => w.name == weight.name);

        if (weight.IsNeutral)
        {
            if (existing >= 0) boneWeights.RemoveAt(existing);
            return;
        }

        if (existing >= 0) boneWeights[existing] = weight;
        else boneWeights.Add(weight);
    }

    /// <summary>True when at least one joint is weighted away from 1.</summary>
    public bool HasBoneWeights
    {
        get
        {
            for (int i = 0; i < boneWeights.Count; i++)
            {
                if (!boneWeights[i].IsNeutral) return true;
            }

            return false;
        }
    }

    [Header("Training")]
    [Tooltip("Goal headings the value function is fitted over, spanning a full turn.")]
    [Min(3)] public int thetaCount = 17;

    [Tooltip("Maximum Bellman iterations. Training stops early once the residual settles.")]
    [Min(1)] public int epochs = 300;

    [Tooltip("Discount factor. 0.99 gives roughly a 100-frame planning horizon.")]
    [Range(0.5f, 0.9999f)] public float gamma = 0.99f;

    public enum ComputeDevice { Auto, Cuda, Cpu }

    [Tooltip("Auto prefers CUDA when torch reports it available.")]
    public ComputeDevice device = ComputeDevice.Auto;

    [Tooltip("Queries per batched k-NN chunk. Bounds VRAM; lower it if training runs out of memory.")]
    [Min(1)] public int knnChunk = 64;

    [Tooltip("Database states integrated per precompute chunk. Bounds host RAM.")]
    [Min(1)] public int stateChunk = 512;

    public enum UmapFeatures
    {
        /// <summary>Joint positions only: a pose manifold, velocity carried by the edges.</summary>
        PositionOnly,

        /// <summary>The whole similarity feature, i.e. a phase space.</summary>
        PositionAndVelocity
    }

    [Header("Debug Visualization")]
    [Tooltip("What UMAP projects. PositionOnly lays the cloud out as a pose manifold, so a state's " +
             "velocity is the edge to the next frame's point. PositionAndVelocity projects the " +
             "whole similarity feature, where one pose held at two speeds lands in two places.")]
    public UmapFeatures umapFeatures = UmapFeatures.PositionOnly;

    [Tooltip("UMAP neighbourhood size. Larger values favour the global shape of the field over " +
             "local detail; 80 matches the reference notebook.")]
    [Min(2)] public int umapNeighbors = 80;

    [Tooltip("How tightly UMAP is allowed to pack points. Lower separates clusters more sharply.")]
    [Range(0f, 1f)] public float umapMinDist = 0.1f;

    [Tooltip("Embedding dimensions. 3 draws a volume in the scene; 2 flattens it to a plane.")]
    [Range(2, 3)] public int umapComponents = 3;

    [Tooltip("UMAP is stochastic. A fixed seed keeps the picture stable between rebuilds so a " +
             "change on screen means a change in the data.")]
    public int umapSeed = 42;

    private PoseSet _poseSet;

    // --- IPoseSetSource ---------------------------------------------------------------------

    public List<AnnotatedAnimationClip> AnimationClips => animationClips;
    public float ContactVelocityThreshold => contactVelocityThreshold;
    public string LeftContactBoneName => leftContactBone?.Name;
    public string RightContactBoneName => rightContactBone?.Name;

    /// <summary>The pose skeleton: the rig's root bone, identical to each clip's skeleton.</summary>
    public Skeleton Skeleton => skeleton;

    /// <summary>No trajectory features, so every pose is usable.</summary>
    public int MaximumFramesPrediction => 0;

    /// <summary>
    /// Where the database and the trained value function live. A folder of its own rather than
    /// MMDatabases, which the Motion Matching inspector owns and regenerates.
    /// </summary>
    public string GetAssetPath() => PoseSetImporter.GetDatabasePath("MotionFields", name);

    // --- Motion field artefacts -------------------------------------------------------------

    /// <summary>The extracted pose database: one row per frame of the assigned clips.</summary>
    public string GetPoseDatabasePath() => Path.Combine(GetAssetPath(), name + ".mmpose");

    /// <summary>The trained value function, shipped with the player.</summary>
    public string GetValueFunctionPath() => Path.Combine(GetAssetPath(), name + ".mffield.npz");

    /// <summary>
    /// UMAP projection of the database, read by <see cref="MotionFieldVisualizer"/>. Debug-only, so
    /// it is safe to delete; the visualizer just draws nothing without it.
    /// </summary>
    public string GetEmbeddingPath() => Path.Combine(GetAssetPath(), name + ".mfembed.npz");

    // --- Build state ------------------------------------------------------------------------

    /// <summary>
    /// Whether the pose database on disk was extracted from the config as it now stands.
    /// </summary>
    /// <remarks>
    /// Set by the Generate Pose Database button and cleared by <c>MotionFieldConfigEditor</c> when a
    /// field is edited. The editor's change check cannot tell which field moved, so any edit clears
    /// this -- over-flagging rather than missing a change that matters.
    ///
    /// Initialised true so a config saved before this field existed -- which was, by definition,
    /// current when it was last generated -- does not demand a pointless rebuild the first time it
    /// is opened. A config that has never been generated has no <c>.mmpose</c>, and every reader
    /// takes the file's absence, not the flag, as the answer.
    /// </remarks>
    [HideInInspector] public bool hasPoseDatabase = true;

    /// <summary>
    /// Whether the value function on disk was trained on the config as it now stands. Cleared by an
    /// edit and by regenerating the database, since extraction renumbers every state the value
    /// function indexes. See <see cref="hasPoseDatabase"/> for why it starts true.
    /// </summary>
    [HideInInspector] public bool hasTrained = true;

    /// <summary>
    /// <see cref="umapFeatures"/> as the literal `motion_field_embedding.compute_embedding`
    /// expects, so the Python string lives in one place rather than at every call site.
    /// </summary>
    public string UmapFeatureModeName =>
        umapFeatures == UmapFeatures.PositionOnly ? "position" : "full";

    public string DeviceName => device switch
    {
        ComputeDevice.Cuda => "cuda",
        ComputeDevice.Cpu => "cpu",
        _ => "auto"
    };

    // --- Pose database ----------------------------------------------------------------------

    /// <summary>
    /// Checks that this config can produce a pose database: a skeleton, at least one clip, and
    /// every clip's skeleton being structurally identical to this one. Returns false with a message
    /// suitable for an Inspector HelpBox. Never logs — inspectors call it every repaint.
    /// </summary>
    public bool TryValidate(out string error) => PoseSetImporter.TryValidate(this, out error);

    /// <summary>Null when <see cref="TryValidate"/> fails; the config's inspector reports why.</summary>
    public PoseSet GetOrImportPoseSet() => _poseSet ??= PoseSetImporter.GetOrImport(this);

    /// <summary>Extract the pose database from the animation clips, in memory.</summary>
    public void ImportPoseSet() => _poseSet = PoseSetImporter.Import(this);

    /// <summary>Drop the cached pose set so the next access re-reads it from disk.</summary>
    public void InvalidatePoseSet()
    {
        _poseSet = null;
    }

    /// <summary>
    /// This config's own pose skeleton -- the skeleton the similarity metric is actually computed
    /// over, which is why the bone weight editor reads it here rather than from the source
    /// animation clips: a clip can be authored against a differently shaped rig, so its joint list
    /// would not line up. False when no skeleton has been assigned.
    /// </summary>
    public bool TryGetDatabaseSkeleton(out Skeleton result)
    {
        result = skeleton;
        return skeleton != null && skeleton.IsSet;
    }
}
}
