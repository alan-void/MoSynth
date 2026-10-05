using System.Collections.Generic;
using System.IO;
using AnimationTools;
using UnityEngine;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace Pfnn
{
/// <summary>
/// Everything a phase-functioned network is trained from and run with: the clips, the bones it
/// predicts, the shape of the network, and the training hyperparameters.
/// </summary>
/// <remarks>
/// Owns its pose database rather than borrowing a <c>MotionMatchingData</c>, so the method does not
/// depend on Motion Matching assets. See openwiki/pfnn/training-and-checkpoints.md.
/// </remarks>
[CreateAssetMenu(fileName = "PfnnConfig", menuName = "MoSynth/PFNN Config")]
public class PfnnConfig : ScriptableObject, IPoseSetSource
{
    public enum ComputeDevice
    {
        Auto,
        Cuda,
        Cpu
    }

    // --- Database ---------------------------------------------------------------------------

    [Tooltip("Clips the pose database is extracted from.")]
    public List<AnnotatedAnimationClip> animationClips = new();

    [SerializeField]
    [Tooltip("The rig, at rest, that every clip is authored against. Must be an asset rig.")]
    private Skeleton skeleton;

    [Tooltip("Foot speed below which a toe counts as planted, in m/s. Gait phase is reconstructed " +
             "from these contacts, so it is what decides the network's phase.")]
    public float contactVelocityThreshold = 0.15f;

    [Tooltip("Bones whose contact the database flags, one channel each, in this order. The network " +
             "takes and predicts one flag per bone. Empty means no contact channels.")]
    public List<SkeletonBone> contactBones = new();

    [Tooltip("Also bake every clip mirrored left-to-right, doubling the database.")]
    public bool mirrorClips;

    // --- Model ------------------------------------------------------------------------------

    [Header("Model")]
    [Tooltip("Frames either side of the current one that the trajectory window spans. 30 at 30 fps " +
             "is one second, which is the horizon the paper uses.")]
    [Min(1)] public int windowRadiusFrames = 30;

    [Tooltip("Frames between adjacent samples of that window. 5 gives 13 samples across it.")]
    [Min(1)] public int windowStrideFrames = 5;

    [Tooltip("Width of the two hidden layers. The paper uses 512; 256 is a better match for a " +
             "database of a few thousand frames.")]
    [Min(8)] public int hiddenUnits = 256;

    [Tooltip("Probability of dropping a unit during training. The paper quotes 0.7 as a retention " +
             "rate, which is 0.3 here.")]
    [Range(0f, 0.9f)] public float dropout = 0.3f;

    /// <summary>
    /// Bones the network does not predict, by name.
    /// </summary>
    /// <remarks>
    /// Sparse and keyed by name, so a joint that moves in the hierarchy keeps its setting;
    /// <c>PfnnConfigEditor</c> draws it as the skeleton hierarchy. Must stay closed under subtree:
    /// a predicted rotation needs its parent's frame, and the Python side refuses a selection that
    /// is not.
    /// </remarks>
    [HideInInspector] public List<string> excludedBones = new();

    // --- Training ---------------------------------------------------------------------------

    [Header("Training")]
    [Min(1)] public int epochs = 150;
    [Min(1)] public int batchSize = 32;
    public float learningRate = 1e-4f;

    [Tooltip("AdamW weight decay. The main defence against overfitting on a small database.")]
    public float weightDecay = 2.5e-3f;

    [Tooltip("Fraction of the frames held out to measure on. Held out as a contiguous tail, " +
             "because neighbouring frames of an animation are near-duplicates and a random split " +
             "would report a validation loss that measures nothing.")]
    [Range(0f, 0.5f)] public float validationFraction = 0.1f;

    public int seed = 42;
    public ComputeDevice device = ComputeDevice.Auto;

    private PoseSet _poseSet;

    // --- IPoseSetSource ---------------------------------------------------------------------

    public List<AnnotatedAnimationClip> AnimationClips => animationClips;
    public float ContactVelocityThreshold => contactVelocityThreshold;
    public IReadOnlyList<string> ContactBoneNames => ContactBoneSources.NamesOf(contactBones);
    public bool MirrorClips => mirrorClips;

    /// <summary>The pose skeleton: the rig's root bone, identical to each clip's skeleton.</summary>
    public Skeleton Skeleton => skeleton;

    /// <summary>Where the database and the trained checkpoint live.</summary>
    public string GetAssetPath() => PoseSetImporter.GetDatabasePath("Pfnn", name);

    // --- Artefacts --------------------------------------------------------------------------

    /// <summary>The extracted pose database: one row per frame of the assigned clips.</summary>
    public string GetPoseDatabasePath() => Path.Combine(GetAssetPath(), name + ".mmpose");

    /// <summary>The trained network, shipped with the player.</summary>
    public string GetCheckpointPath() => Path.Combine(GetAssetPath(), name + ".pfnn.npz");

    // --- Build state ------------------------------------------------------------------------

    /// <summary>
    /// Whether the pose database on disk was extracted from the config as it now stands. Set by the
    /// Generate button and cleared by any inspector edit, since the change check cannot tell which
    /// field moved. Defaults true so an asset without the field does not demand a rebuild.
    /// </summary>
    [HideInInspector] public bool hasPoseDatabase = true;

    /// <summary>
    /// Whether the checkpoint on disk was trained on the config as it now stands. Also cleared by
    /// regenerating the database, since that changes the frames the network was fitted to.
    /// </summary>
    [HideInInspector] public bool hasTrained = true;

    /// <summary><see cref="device"/> as the literal the Python side expects.</summary>
    public string DeviceName => device switch
    {
        ComputeDevice.Cuda => "cuda",
        ComputeDevice.Cpu => "cpu",
        _ => "auto"
    };

    // --- Predicted bones --------------------------------------------------------------------

    /// <summary>Whether the network predicts this bone. Bones absent from the list are predicted.</summary>
    public bool IsPredicted(string boneName) => !excludedBones.Contains(boneName);

    /// <summary>
    /// Include or exclude one bone, keeping <see cref="excludedBones"/> sparse.
    /// </summary>
    /// <remarks>
    /// Callers must apply it to a whole subtree (see <see cref="excludedBones"/>); the config does
    /// not know the hierarchy, so it cannot enforce that.
    /// </remarks>
    public void SetPredicted(string boneName, bool predicted)
    {
        if (predicted) excludedBones.Remove(boneName);
        else if (!excludedBones.Contains(boneName)) excludedBones.Add(boneName);
    }

    /// <summary>True when at least one bone is held back from the model.</summary>
    public bool HasExcludedBones => excludedBones.Count > 0;

    // --- Pose database ----------------------------------------------------------------------

    /// <summary>
    /// Checks that this config can produce a pose database: a skeleton, at least one clip, and every
    /// clip's skeleton being structurally identical to this one. Loads every clip, so never call it
    /// per repaint.
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
    /// This config's own pose skeleton — the bone list the network's selection is expressed over.
    /// False when no skeleton has been assigned.
    /// </summary>
    public bool TryGetDatabaseSkeleton(out Skeleton result)
    {
        result = skeleton;
        return skeleton != null && skeleton.IsSet;
    }
}
}
