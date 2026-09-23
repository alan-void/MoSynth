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
/// Owns no pose database: it reads a <see cref="MotionMatchingData"/>'s poses and the <em>same</em>
/// feature vectors the classic matcher searches, which is the basis of the comparison. Its own
/// artefact is <c>StreamingAssets/Lmm/&lt;name&gt;/&lt;name&gt;.lmm.npz</c>. See
/// openwiki/motion-matching/learned-motion-matching.md.
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
    /// Sparse and keyed by name, so a joint that moves in the hierarchy keeps its setting;
    /// <c>LmmConfigEditor</c> draws it as the skeleton hierarchy. Excluding a bone excludes its
    /// whole subtree — see <see cref="PredictedBoneSelection"/>.
    /// </remarks>
    [HideInInspector] public List<string> excludedBones = new();

    // --- Search -----------------------------------------------------------------------------

    /// <summary>
    /// Authored importance, one entry per feature definition of <see cref="mmData"/>, trajectory
    /// features first. Expanded into per-float weights by <see cref="ExpandFeatureWeights"/>.
    /// </summary>
    /// <remarks>
    /// On the config rather than the stage (unlike <c>MotionMatchingStage</c>) because the projector
    /// is trained under this metric. They go into the checkpoint, and the stage refuses a checkpoint
    /// whose weights no longer match.
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

    [Tooltip("Stepper hidden width. 0 takes the reference implementation's 512, two layers deep.")]
    [Min(0)] public int stepperHiddenUnits;

    [Tooltip("Projector hidden width. 0 takes the reference implementation's 512. It is four " +
             "layers deep — the deepest of the four networks, because approximating a " +
             "nearest-neighbour lookup wants depth rather than width.")]
    [Min(0)] public int projectorHiddenUnits;

    // --- Stepper ----------------------------------------------------------------------------

    [Header("Stepper")]
    [Tooltip("Fit the stepper, which advances the state between searches so the database need not " +
             "be played. Off writes an autoencoder-only checkpoint, which is all the " +
             "DecompressorOnly mode needs.")]
    public bool trainStepper = true;

    /// <summary>
    /// Database frames the stepper is unrolled over while training.
    /// </summary>
    /// <remarks>
    /// Must cover the stage's search interval, which is how long the state runs uncorrected; the
    /// default is two searches' worth at 10/60 s on a 60 Hz database. Cost is linear in it.
    /// </remarks>
    [Tooltip("Frames the stepper is unrolled over. It has to cover the stage's search interval, " +
             "which is how long the state runs uncorrected.")]
    [Min(2)] public int stepperWindow = 20;

    [Tooltip("Ceiling on optimiser steps for the stepper. Each one unrolls the whole window, so a " +
             "step costs roughly stepperWindow forward passes. Stepper Patience is what normally " +
             "stops the fit.")]
    [Min(1)] public int stepperIterations = 30000;

    /// <summary>
    /// Held-out scores without an improvement before the stepper fit stops. 0 never stops early.
    /// </summary>
    /// <remarks>
    /// This, not <see cref="stepperIterations"/>, is the stopping rule. See
    /// openwiki/motion-matching/learned-motion-matching.md.
    /// </remarks>
    [Tooltip("Held-out scores without an improvement before the stepper fit stops. This is the " +
             "real stopping rule; 0 disables it and runs to Stepper Iterations.")]
    [Min(0)] public int stepperPatience = 5;

    [Tooltip("Stop fitting the stepper after this many seconds regardless, 0 for no limit.")]
    [Min(0f)] public float stepperMaxSeconds;

    // --- Projector --------------------------------------------------------------------------

    /// <summary>
    /// Fit the projector, which replaces the search itself. Needs <see cref="trainStepper"/>.
    /// </summary>
    /// <remarks>
    /// The <c>Full</c> mode runs both networks, so turning the stepper off turns this off with it.
    /// </remarks>
    [Header("Projector")]
    [Tooltip("Fit the projector, which answers a query with a state instead of looking one up. " +
             "Needs the stepper: the Full mode runs both.")]
    public bool trainProjector = true;

    [Tooltip("Ceiling on optimiser steps for the projector. Each one searches the whole database " +
             "for the true nearest neighbour of every query in the batch, so a step costs more " +
             "than the network. Projector Patience is what normally stops the fit.")]
    [Min(1)] public int projectorIterations = 30000;

    [Tooltip("Held-out scores without an improvement before the projector fit stops. This is the " +
             "real stopping rule; 0 disables it and runs to Projector Iterations.")]
    [Min(0)] public int projectorPatience = 5;

    /// <summary>
    /// How far a training query is displaced from the database frame it was drawn from.
    /// </summary>
    /// <remarks>
    /// Each sample is displaced by a uniform fraction of this, so a batch spans queries from
    /// near-exact to far from any frame. Raising it trades accuracy near the data for behaviour
    /// further from it.
    /// </remarks>
    [Tooltip("How far training queries are displaced, in units of each feature's own spread plus " +
             "one. The projector only learns to answer a query that misses if it is shown one.")]
    [Min(0f)] public float projectorNoise = 1f;

    [Tooltip("Stop fitting the projector after this many seconds regardless, 0 for no limit.")]
    [Min(0f)] public float projectorMaxSeconds;

    // --- Training ---------------------------------------------------------------------------

    [Header("Training")]
    [Tooltip("w_vreg: how hard the latent is held to moving smoothly from frame to frame. The one " +
             "loss weight the paper declines to give a number for, and the one worth tuning — it " +
             "decides whether the stepper has a trajectory to advance or noise. Too high " +
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
    /// inspector edit, since the change check cannot tell which field moved.
    /// </summary>
    [HideInInspector] public bool hasTrained;

    // --- Paths ------------------------------------------------------------------------------

    /// <summary>Where the trained model lives. Its database stays under the source asset's folder.</summary>
    public string GetAssetPath() => PoseSetImporter.GetDatabasePath("Lmm", name);

    /// <summary>The trained networks and baked latents, shipped with the player.</summary>
    public string GetCheckpointPath() => Path.Combine(GetAssetPath(), name + ".lmm.npz");

    /// <summary><see cref="device"/> as the literal the Python side expects.</summary>
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
    /// The stage compares this result against the checkpoint's, so a disagreement is caught.
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
