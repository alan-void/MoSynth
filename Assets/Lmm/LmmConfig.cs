using System.Collections.Generic;
using System.IO;
using AnimationTools;
using MotionMatching;
using UnityEngine;

namespace Lmm
{
/// <summary>
/// Everything a Learned Motion Matching model is trained from and run with: which database it
/// learns, which bones it predicts, the shape of the networks, and the training hyperparameters.
/// </summary>
/// <remarks>
/// Unlike <c>PfnnConfig</c> and <c>MotionFieldConfig</c>, this owns no pose database. It points at
/// a <see cref="MotionMatchingData"/> and reads both halves of that asset's generated output — the
/// poses it learns to reconstruct and the <em>same</em> matching feature vectors the classic
/// matcher searches. That is deliberate and it is the whole basis of the comparison: a learned
/// matcher answering a differently shaped query would not be replacing the search, it would be
/// doing a different job.
/// <para>
/// Its own artefact is one file, <c>&lt;name&gt;.lmm.npz</c>, under
/// <c>StreamingAssets/Lmm/&lt;name&gt;/</c>. See the wiki's learned motion matching page for what
/// the networks do with it.
/// </para>
/// </remarks>
[CreateAssetMenu(fileName = "LmmConfig", menuName = "MoSynth/Learned Motion Matching Config")]
public class LmmConfig : ScriptableObject
{
    public enum ComputeDevice
    {
        Auto,
        Cuda,
        Cpu
    }

    // --- Database ---------------------------------------------------------------------------

    [Tooltip("The database this learns. Both its .mmpose and its .mmfeatures are read, so it must " +
             "author feature channels — a pose-only asset has no query vector to learn against.")]
    public MotionMatchingData mmData;

    /// <summary>
    /// Bones the decompressor does not predict, by name.
    /// </summary>
    /// <remarks>
    /// Hidden from the default inspector on purpose: as a raw list this is anonymous strings in
    /// whatever order they were clicked. <c>LmmConfigEditor</c> draws it as the skeleton hierarchy
    /// with a toggle per bone instead. Sparse and keyed by name, so a joint that moves in the
    /// hierarchy takes its setting with it.
    ///
    /// Excluding a bone excludes its whole subtree — see
    /// <see cref="PredictedBoneSelection"/> for why a rotation needs its parent's frame.
    /// </remarks>
    [HideInInspector] public List<string> excludedBones = new();

    // --- Search -----------------------------------------------------------------------------

    /// <summary>
    /// Authored importance, one entry per feature definition of <see cref="mmData"/>, trajectory
    /// features first. Expanded into per-float weights by <see cref="ExpandFeatureWeights"/>.
    /// </summary>
    /// <remarks>
    /// These live on the config rather than on the stage, which is the opposite of
    /// <c>MotionMatchingStage</c>, because training has to see them: the phase C projector learns
    /// to approximate a nearest-neighbour lookup under a particular metric, and one fitted against
    /// uniform weights would approximate a search nobody runs. They go into the checkpoint and the
    /// stage refuses a checkpoint whose weights no longer match.
    /// </remarks>
    [Tooltip("One weight per feature definition, trajectory features first. Training sees these, " +
             "so changing one means retraining.")]
    public List<float> featureWeights = new();

    // --- Model ------------------------------------------------------------------------------

    [Header("Model")]
    [Tooltip("Floats the pose is compressed to. 32 is what the reference implementation encodes a " +
             "rig of this size into.")]
    [Min(2)] public int latentSize = 32;

    [Tooltip("Compressor hidden width. 0 takes the reference implementation's 516.")]
    [Min(0)] public int compressorHiddenUnits;

    [Tooltip("Decompressor hidden width. 0 takes the reference implementation's 512. It is one " +
             "layer deep: decoding a latent into a pose is a smooth map and wants width, not depth.")]
    [Min(0)] public int decompressorHiddenUnits;

    // --- Training ---------------------------------------------------------------------------

    [Header("Training")]
    [Tooltip("w_vreg: how hard the latent is held to moving smoothly from frame to frame. The one " +
             "loss weight the paper declines to give a number for, and the one worth tuning — it " +
             "decides whether the phase B stepper has a trajectory to advance or noise. Too high " +
             "and the latent cannot carry enough to reconstruct from.")]
    [Min(0f)] public float latentVelocityWeight = 0.01f;

    [Tooltip("Optimiser steps. Counted in steps rather than epochs because the paper's schedule " +
             "is, and because an epoch means something different on every database size.")]
    [Min(1)] public int iterations = 150000;

    [Min(1)] public int batchSize = 256;
    public float learningRate = 1e-3f;

    [Tooltip("AdamW weight decay.")]
    public float weightDecay = 1e-3f;

    [Tooltip("Multiplied into the learning rate every 1000 steps.")]
    [Range(0.5f, 1f)] public float learningRateDecay = 0.99f;

    [Tooltip("Stop after this many seconds regardless, 0 for no limit. A run that has to fit a " +
             "budget should be cut by the clock rather than by guessing an iteration count.")]
    [Min(0f)] public float maxSeconds;

    [Tooltip("Fraction of the frame pairs held out to measure on. Held out as a contiguous tail, " +
             "because neighbouring frames of an animation are near-duplicates and a random split " +
             "would report a validation loss that measures nothing. The parameters that scored " +
             "best on it are what gets written.")]
    [Range(0f, 0.5f)] public float validationFraction = 0.1f;

    [Tooltip("Steps between held-out scores.")]
    [Min(1)] public int validationInterval = 1000;

    public int seed = 42;
    public ComputeDevice device = ComputeDevice.Auto;

    // --- Build state ------------------------------------------------------------------------

    /// <summary>
    /// Whether the checkpoint on disk was trained on the config as it now stands. Cleared by any
    /// inspector edit — the change check cannot tell which field moved, so it over-flags rather
    /// than miss one that matters.
    /// </summary>
    [HideInInspector] public bool hasTrained;

    // --- Paths ------------------------------------------------------------------------------

    /// <summary>Where the trained model lives. Its database stays under the source asset's folder.</summary>
    public string GetAssetPath() => PoseSetImporter.GetDatabasePath("Lmm", name);

    /// <summary>The trained networks and baked latents, shipped with the player.</summary>
    public string GetCheckpointPath() => Path.Combine(GetAssetPath(), name + ".lmm.npz");

    /// <summary>
    /// <see cref="device"/> as the literal the Python side expects, so the string lives in one
    /// place rather than at every call site.
    /// </summary>
    public string DeviceName => device switch
    {
        ComputeDevice.Cuda => "cuda",
        ComputeDevice.Cpu => "cpu",
        _ => "auto"
    };

    // --- Validation -------------------------------------------------------------------------

    /// <summary>How many authored weights this config should carry: one per feature definition.</summary>
    public int FeatureDefinitionCount =>
        mmData == null ? 0 : mmData.trajectoryFeatures.Count + mmData.poseFeatures.Count;

    /// <summary>
    /// Checks that this config can be trained: a database that carries both a pose set and a
    /// feature set. Never logs — inspectors call it every repaint.
    /// </summary>
    public bool TryValidate(out string error)
    {
        if (mmData == null)
        {
            error = "no MotionMatchingData assigned — this config learns someone else's database " +
                    "rather than owning one.";
            return false;
        }

        if (!mmData.HasFeatureChannels)
        {
            error = $"\"{mmData.name}\" authors no feature channels, so it bakes no .mmfeatures " +
                    "and there is no query vector to learn against.";
            return false;
        }

        return mmData.TryValidate(out error);
    }

    /// <summary>
    /// Expands <see cref="featureWeights"/> into one weight per float of a feature vector, exactly
    /// as <c>MotionMatchingStage.UpdateFeatureWeights</c> does and as the Python side does when it
    /// bakes them into a checkpoint.
    /// </summary>
    /// <remarks>
    /// Three implementations of one expansion would be two too many; this is the C# one, and the
    /// stage compares its result against the checkpoint's, which is how a disagreement is caught
    /// rather than quietly trained around.
    /// </remarks>
    public void ExpandFeatureWeights(FeatureSet featureSet, float[] destination)
    {
        for (var i = 0; i < mmData.trajectoryFeatures.Count; i++)
        {
            var weight = WeightOf(i);
            var featureSize = featureSet.GetTrajectoryFeatureFloatCount(i);
            for (var p = 0; p < featureSet.GetPredictionCount(i); ++p)
            {
                var offset = featureSet.GetTrajectoryFeatureOffset(i, p);
                for (var f = 0; f < featureSize; f++) destination[offset + f] = weight;
            }
        }

        for (var i = 0; i < mmData.poseFeatures.Count; i++)
        {
            var weight = WeightOf(mmData.trajectoryFeatures.Count + i);
            var offset = featureSet.PoseOffset + i * FeatureSet.FloatsPerPoseFeature;
            for (var f = 0; f < FeatureSet.FloatsPerPoseFeature; f++) destination[offset + f] = weight;
        }
    }

    // --- Predicted bones --------------------------------------------------------------------

    /// <summary>Whether the decompressor predicts this bone. Bones absent from the list are predicted.</summary>
    public bool IsPredicted(string boneName) => !excludedBones.Contains(boneName);

    /// <summary>
    /// Include or exclude one bone, keeping <see cref="excludedBones"/> sparse.
    /// </summary>
    /// <remarks>
    /// Only ever call this with a whole subtree — see <see cref="excludedBones"/>. The editor's
    /// toggle does that; nothing here can enforce it, because a config does not know the hierarchy.
    /// </remarks>
    public void SetPredicted(string boneName, bool predicted)
    {
        if (predicted) excludedBones.Remove(boneName);
        else if (!excludedBones.Contains(boneName)) excludedBones.Add(boneName);
    }

    /// <summary>True when at least one bone is held back from the model.</summary>
    public bool HasExcludedBones => excludedBones.Count > 0;

    /// <summary>
    /// The database's pose skeleton — the bone list the selection is expressed over. False when no
    /// database has been assigned.
    /// </summary>
    public bool TryGetDatabaseSkeleton(out Skeleton result)
    {
        result = mmData == null ? null : mmData.Skeleton;
        return result != null && result.IsSet;
    }

    /// <summary>An authored weight, defaulting to 1 for a definition the list has not caught up with.</summary>
    private float WeightOf(int definitionIndex) =>
        definitionIndex < featureWeights.Count ? featureWeights[definitionIndex] : 1f;

    /// <summary>Keeps the weight list one entry long per feature definition, padding with 1.</summary>
    private void OnValidate()
    {
        var wanted = FeatureDefinitionCount;
        while (featureWeights.Count < wanted) featureWeights.Add(1f);
        if (featureWeights.Count > wanted)
        {
            featureWeights.RemoveRange(wanted, featureWeights.Count - wanted);
        }
    }
}
}
