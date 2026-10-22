using System;
using System.Collections.Generic;
using System.IO;
using AnimationTools;
using GameplayTags;
using UnityEngine;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace Style
{
/// <summary>
/// Everything a style transfer model is trained from and run with: the styled clips and their
/// style and gait tags, the bones it restyles, the shape of the network, and the training settings.
/// </summary>
/// <remarks>
/// Owns its pose database, as <c>PfnnConfig</c> does. Which style and gait each clip is comes from
/// its whole-clip tags and is baked beside the database as <c>&lt;name&gt;.style_labels.json</c>,
/// because the <c>.mmpose</c> clip table carries only frame ranges.
/// </remarks>
[CreateAssetMenu(fileName = "StyleTransferConfig", menuName = "MoSynth/Style Transfer Config")]
public class StyleTransferConfig : ScriptableObject, IPoseSetSource
{
    public enum ComputeDevice
    {
        Auto,
        Cuda,
        Cpu
    }

    /// <summary>One clip's labels, in the sidecar's JSON shape.</summary>
    [Serializable]
    public class ClipLabel
    {
        public string style;
        public string gait;
    }

    [Serializable]
    private class ClipLabelFile
    {
        public List<ClipLabel> clips = new();
    }

    // --- Database ---------------------------------------------------------------------------

    [Tooltip("Clips the pose database is extracted from. Each carries one whole-clip tag under " +
             "Style Root and one under Gait Root.")]
    public List<AnnotatedAnimationClip> animationClips = new();

    [SerializeField]
    [Tooltip("The rig, at rest, that every clip is authored against. Must be an asset rig.")]
    private Skeleton skeleton;

    [Tooltip("Foot speed below which a toe counts as planted, in m/s.")]
    public float contactVelocityThreshold = 0.15f;

    [Tooltip("The two feet whose contact drives the gait phase: left first, then right.")]
    public List<SkeletonBone> contactBones = new();

    [Tooltip("Must stay off: a mirrored copy would carry its source clip's labels onto a body " +
             "moving the other way, and the label sidecar has one entry per source clip.")]
    public bool mirrorClips;

    // --- Labels -----------------------------------------------------------------------------

    [Header("Labels")]
    [Tooltip("Parent of every style tag, e.g. 'style'.")]
    public GameplayTagSO styleRoot;

    [Tooltip("Parent of every gait tag, e.g. 'gait'.")]
    public GameplayTagSO gaitRoot;

    [Tooltip("The reference style every other one is paired against, e.g. style.Neutral.")]
    public GameplayTagSO neutralStyle;

    [Tooltip("Styles left out of training, such as those defined by cadence.")]
    public List<GameplayTagSO> excludedStyles = new();

    /// <summary>
    /// Bones the model does not restyle, by name. Sparse and closed under subtree, like
    /// <c>PfnnConfig.excludedBones</c>; <c>StyleTransferConfigEditor</c> draws it as the hierarchy.
    /// </summary>
    [HideInInspector] public List<string> excludedBones = new();

    // --- Model ------------------------------------------------------------------------------

    [Header("Model")]
    [Min(8)] public int hiddenUnits = 512;
    [Min(1)] public int layers = 3;

    [Tooltip("Width of the learned per-style embedding that modulates the dynamics network.")]
    [Min(1)] public int embedDim = 32;

    [Tooltip("Seconds into the past the network sees descriptors from, ascending.")]
    public List<float> historyOffsets = new() { 0.1f, 0.2f };

    [Tooltip("Width, in radians, of the gait-phase kernel that pairs a styled frame with neutral ones.")]
    public float gaitSigma = 0.35f;

    // --- Training ---------------------------------------------------------------------------

    [Header("Training")]
    [Min(1)] public int epochs = 60;
    [Min(1)] public int batchSize = 1024;
    public float learningRate = 1e-3f;
    public float weightDecay = 1e-4f;
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

    /// <summary>Where the database, its labels and the trained checkpoint live.</summary>
    public string GetAssetPath() => PoseSetImporter.GetDatabasePath("Style", name);

    // --- Artefacts --------------------------------------------------------------------------

    public string GetPoseDatabasePath() => Path.Combine(GetAssetPath(), name + ".mmpose");

    /// <summary>One <c>{style, gait}</c> entry per database clip, in database clip order.</summary>
    public string GetLabelsPath() => Path.Combine(GetAssetPath(), name + ".style_labels.json");

    public string GetCheckpointPath() => Path.Combine(GetAssetPath(), name + ".style.npz");

    // --- Build state ------------------------------------------------------------------------

    /// <summary>
    /// Whether the database and labels on disk were baked from the config as it now stands. Set by
    /// the Generate button, cleared by any inspector edit.
    /// </summary>
    [HideInInspector] public bool hasPoseDatabase = true;

    /// <summary>Whether the checkpoint on disk was trained on the config as it now stands.</summary>
    [HideInInspector] public bool hasTrained = true;

    /// <summary><see cref="device"/> as the literal the Python side expects.</summary>
    public string DeviceName => DeviceNameOf(device);

    public static string DeviceNameOf(ComputeDevice computeDevice) => computeDevice switch
    {
        ComputeDevice.Cuda => "cuda",
        ComputeDevice.Cpu => "cpu",
        _ => "auto"
    };

    /// <summary>Tag names of <see cref="excludedStyles"/>, skipping unset entries.</summary>
    public List<string> ExcludedStyleNames()
    {
        var names = new List<string>();
        foreach (var tag in excludedStyles)
        {
            if (tag != null) names.Add(tag.TagName);
        }

        return names;
    }

    // --- Validation -------------------------------------------------------------------------

    /// <summary>
    /// The checks that need no clip loaded: the pose-set settings, mirroring off, two contact bones
    /// in left-right order, and the label tags. Cheap enough for every inspector repaint.
    /// </summary>
    public bool TryValidateSettings(out string error)
    {
        if (!PoseSetImporter.TryValidateSettings(this, out error)) return false;

        if (mirrorClips)
        {
            error = "Mirror Clips must be off: the style labels have one entry per source clip.";
            return false;
        }

        var contacts = ContactBoneNames;
        if (contacts.Count != 2)
        {
            error = $"Exactly two contact bones are needed, left foot then right, for the gait " +
                    $"phase; {contacts.Count} are listed.";
            return false;
        }

        // Slot 0 is the left foot everywhere gait phase is read, so a swapped list would put every
        // footfall on the wrong side of the cycle.
        if (BoneNameConventions.SideOf(contacts[0]) == BoneNameConventions.BodySide.Right ||
            BoneNameConventions.SideOf(contacts[1]) == BoneNameConventions.BodySide.Left)
        {
            error = $"Contact bones {ContactBoneSources.Describe(contacts)} look swapped: list the " +
                    "left foot first, then the right.";
            return false;
        }

        // The stage removes the yaw from bone 0's delta, so the model must always produce one.
        if (excludedBones.Contains(skeleton.GetBone(0).Name))
        {
            error = $"The root bone '{skeleton.GetBone(0).Name}' must stay styled.";
            return false;
        }

        if (styleRoot == null || gaitRoot == null)
        {
            error = "Assign Style Root and Gait Root: they are where each clip's labels are read from.";
            return false;
        }

        if (neutralStyle == null || neutralStyle == styleRoot || !neutralStyle.IsChildOf(styleRoot))
        {
            error = $"Neutral Style must be a tag under '{styleRoot.TagFullName}'.";
            return false;
        }

        error = null;
        return true;
    }

    /// <summary>
    /// Checks that this config can produce a labelled pose database: <see cref="TryValidateSettings"/>,
    /// every clip against the skeleton, every clip labelled, and the neutral style among them.
    /// Loads every clip, so never call it per repaint.
    /// </summary>
    public bool TryValidate(out string error)
    {
        if (!TryValidateSettings(out error)) return false;
        if (!PoseSetImporter.TryValidate(this, out error)) return false;
        return TryBuildClipLabels(out _, out error);
    }

    /// <summary>
    /// One label per clip in <see cref="animationClips"/> order, which is the database's clip order
    /// while mirroring is off: the importer appends exactly one clip per source clip once
    /// <see cref="PoseSetImporter.TryValidate"/> passes.
    /// </summary>
    public bool TryBuildClipLabels(out List<ClipLabel> labels, out string error)
    {
        labels = new List<ClipLabel>(animationClips.Count);
        error = null;

        if (styleRoot == null || gaitRoot == null)
        {
            error = "Assign Style Root and Gait Root.";
            return false;
        }

        var hasNeutral = false;
        for (var i = 0; i < animationClips.Count; i++)
        {
            var clip = animationClips[i];
            if (clip == null)
            {
                error = $"Clip {i} is empty.";
                return false;
            }

            if (!TryFindLabel(clip, styleRoot, out var style, out error) ||
                !TryFindLabel(clip, gaitRoot, out var gait, out error))
            {
                error = $"Clip \"{clip.name}\": {error}";
                return false;
            }

            if (style == neutralStyle) hasNeutral = true;
            labels.Add(new ClipLabel { style = style.TagName, gait = gait.TagName });
        }

        if (!hasNeutral)
        {
            error = $"No clip is tagged {(neutralStyle != null ? neutralStyle.TagFullName : "<neutral>")}, " +
                    "and every style is paired against it.";
            return false;
        }

        return true;
    }

    /// <summary>The labels as the sidecar's JSON text.</summary>
    public static string LabelsToJson(List<ClipLabel> labels) =>
        JsonUtility.ToJson(new ClipLabelFile { clips = labels }, prettyPrint: true);

    /// <summary>
    /// The one tag below <paramref name="root"/> that is on at the clip's slice start. The slice
    /// start is where a whole-clip channel is guaranteed to be on, whatever the trim.
    /// </summary>
    private static bool TryFindLabel(AnnotatedAnimationClip clip, GameplayTagSO root,
        out GameplayTagSO label, out string error)
    {
        label = null;
        error = null;

        if (clip.TryGetComponent<AnimationTagComponent>(out var tags) && tags.isEnabled &&
            tags.channels != null)
        {
            foreach (var channel in tags.channels)
            {
                if (channel?.tag == null || channel.tag == root || !channel.tag.IsChildOf(root)) continue;
                if (!AnimationTagging.IsOn(channel.toggles, clip.startFrame)) continue;

                if (label != null)
                {
                    error = $"tagged both {label.TagFullName} and {channel.tag.TagFullName} under " +
                            $"'{root.TagFullName}'; exactly one is allowed.";
                    return false;
                }

                label = channel.tag;
            }
        }

        if (label != null) return true;

        error = $"no tag under '{root.TagFullName}' is on at its first frame.";
        return false;
    }

    // --- Styled bones -----------------------------------------------------------------------

    /// <summary>True when at least one bone is held back from the model.</summary>
    public bool HasExcludedBones => excludedBones.Count > 0;

    // --- Pose database ----------------------------------------------------------------------

    /// <summary>Null when the settings check fails; the config's inspector reports why.</summary>
    public PoseSet GetOrImportPoseSet() => _poseSet ??= PoseSetImporter.GetOrImport(this);

    /// <summary>Extract the pose database from the animation clips, in memory.</summary>
    public void ImportPoseSet() => _poseSet = PoseSetImporter.Import(this);

    /// <summary>Drop the cached pose set so the next access re-reads it from disk.</summary>
    public void InvalidatePoseSet()
    {
        _poseSet = null;
    }

    /// <summary>This config's own pose skeleton. False when none has been assigned.</summary>
    public bool TryGetDatabaseSkeleton(out Skeleton result)
    {
        result = skeleton;
        return skeleton != null && skeleton.IsSet;
    }
}
}
