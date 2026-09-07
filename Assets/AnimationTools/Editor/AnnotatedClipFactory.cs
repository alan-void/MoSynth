using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Turns imported clips plus the rig they were authored against into
/// <see cref="AnnotatedAnimationClip"/> assets. Shared by the single-selection creation menu and the
/// batch window, so both guess the root bone and refuse a humanoid rig the same way.
/// </summary>
public static class AnnotatedClipFactory
{
    /// <summary>What a batch run did, or why it could not start.</summary>
    public sealed class BatchReport
    {
        public bool Ok;
        public string Error;
        public int TotalClips;
        public int MatchedClips;
        public int Created;
        public int Updated;
        public int Detected;
        public int DetectionFailed;

        public string Summary()
        {
            if (!Ok) return Error;

            var summary = $"{Created} created, {Updated} updated ({MatchedClips} of {TotalClips} takes matched).";
            if (Detected > 0 || DetectionFailed > 0)
            {
                summary += $" Footfalls detected on {Detected}";
                summary += DetectionFailed > 0 ? $", failed on {DetectionFailed}." : ".";
            }

            return summary;
        }
    }

    /// <summary>
    /// The rig's root bone, guessed: the first descendant named "*Hips", else the far end of a
    /// single-child chain from the rig. Null when neither reads, which callers must report rather
    /// than write an asset with an empty skeleton.
    /// </summary>
    public static Transform GuessRootBone(Transform rig)
    {
        if (rig == null) return null;

        var hips = FindHipsRecursive(rig);
        if (hips != null) return hips;

        if (rig.childCount != 1) return null;

        var current = rig.GetChild(0);
        while (current.childCount == 1)
        {
            current = current.GetChild(0);
        }

        return current;
    }

    private static Transform FindHipsRecursive(Transform transform)
    {
        for (var i = 0; i < transform.childCount; i++)
        {
            var child = transform.GetChild(i);
            if (child.name.EndsWith("Hips", StringComparison.OrdinalIgnoreCase)) return child;

            var found = FindHipsRecursive(child);
            if (found != null) return found;
        }

        return null;
    }

    /// <summary>
    /// Whether the model at this path imports as a humanoid. Muscle clips produce no transform
    /// motion on a plain hierarchy, so this has to be caught before an asset is written.
    /// </summary>
    public static bool IsHumanoid(string assetPath)
    {
        var importer = AssetImporter.GetAtPath(assetPath) as ModelImporter;
        return importer != null && importer.animationType == ModelImporterAnimationType.Human;
    }

    /// <summary>Whether the model importer is still applying keyframe reduction to its clips.</summary>
    public static bool UsesKeyframeReduction(string assetPath)
    {
        var importer = AssetImporter.GetAtPath(assetPath) as ModelImporter;
        return importer != null &&
               importer.animationCompression != ModelImporterAnimationCompression.Off;
    }

    /// <summary>Every AnimationClip an imported asset carries, in import order.</summary>
    public static List<AnimationClip> LoadClips(string assetPath)
    {
        var clips = new List<AnimationClip>();
        foreach (var representation in AssetDatabase.LoadAllAssetRepresentationsAtPath(assetPath))
        {
            if (representation is AnimationClip clip) clips.Add(clip);
        }

        return clips;
    }

    /// <summary>
    /// The rig root and clips of an imported model, or false with a message suitable for a HelpBox.
    /// </summary>
    public static bool TryResolveModel(GameObject model, out Transform rootBone,
        out List<AnimationClip> clips, out string error)
    {
        rootBone = null;
        clips = new List<AnimationClip>();
        error = null;

        var assetPath = model != null ? AssetDatabase.GetAssetPath(model) : null;
        if (string.IsNullOrEmpty(assetPath))
        {
            error = "Pick an imported model asset.";
            return false;
        }

        if (IsHumanoid(assetPath))
        {
            error = "Humanoid rigs are not supported — muscle clips can't sample onto a plain " +
                    "Transform hierarchy. Set the rig to Generic.";
            return false;
        }

        rootBone = GuessRootBone(model.transform);
        if (rootBone == null)
        {
            error = $"Could not find a root bone in \"{model.name}\" — no descendant named '*Hips' " +
                    "and no single-child chain to follow.";
            return false;
        }

        clips = LoadClips(assetPath);
        if (clips.Count == 0)
        {
            error = "This asset carries no AnimationClip.";
            return false;
        }

        return true;
    }

    /// <summary>
    /// Whether a clip name is selected by a filter. An empty filter takes everything; otherwise the
    /// filter is a case-insensitive substring, which is what take names from one capture session
    /// share.
    /// </summary>
    public static bool MatchesFilter(string name, string filter)
    {
        if (string.IsNullOrWhiteSpace(filter)) return true;
        if (string.IsNullOrEmpty(name)) return false;

        return name.IndexOf(filter.Trim(), StringComparison.OrdinalIgnoreCase) >= 0;
    }

    /// <summary>The names a filter selects, in the order they were given.</summary>
    public static List<string> SelectClipNames(IEnumerable<string> names, string filter)
    {
        var selected = new List<string>();
        if (names == null) return selected;

        foreach (var name in names)
        {
            if (MatchesFilter(name, filter)) selected.Add(name);
        }

        return selected;
    }

    /// <summary>
    /// Where a clip's annotated asset goes inside a folder. Always forward-slashed and always the
    /// same for the same clip, so re-running a batch updates assets in place instead of making
    /// duplicates beside them.
    /// </summary>
    public static string AssetPathFor(string folder, string clipName)
    {
        var trimmed = (folder ?? string.Empty).Replace('\\', '/').TrimEnd('/');
        return $"{trimmed}/{SanitizeFileName(clipName)}.asset";
    }

    private static string SanitizeFileName(string name)
    {
        if (string.IsNullOrEmpty(name)) return "clip";

        var invalid = Path.GetInvalidFileNameChars();
        var chars = name.ToCharArray();
        for (var i = 0; i < chars.Length; i++)
        {
            if (Array.IndexOf(invalid, chars[i]) >= 0) chars[i] = '_';
        }

        return new string(chars);
    }

    /// <summary>Creates every folder of an "Assets/..." path that does not exist yet.</summary>
    /// <remarks>
    /// The disk is checked as well as the AssetDatabase, which can still list a folder that was
    /// deleted outside the Editor. Trusting it alone turns that into a failure to write the asset,
    /// reported as a missing path rather than a missing folder.
    /// </remarks>
    public static bool EnsureFolder(string folder)
    {
        if (string.IsNullOrWhiteSpace(folder)) return false;
        if (AssetDatabase.IsValidFolder(folder) && Directory.Exists(ToAbsolute(folder))) return true;

        var parts = folder.Replace('\\', '/').TrimEnd('/').Split('/');
        if (parts.Length < 2 || parts[0] != "Assets") return false;

        var current = "Assets";
        for (var i = 1; i < parts.Length; i++)
        {
            var next = $"{current}/{parts[i]}";
            if (!AssetDatabase.IsValidFolder(next) || !Directory.Exists(ToAbsolute(next)))
            {
                AssetDatabase.CreateFolder(current, parts[i]);
            }

            current = next;
        }

        return Directory.Exists(ToAbsolute(current));
    }

    private static string ToAbsolute(string projectRelative) =>
        Path.Combine(Path.GetDirectoryName(Application.dataPath) ?? string.Empty, projectRelative);

    /// <summary>
    /// Writes the annotated clip at <paramref name="assetPath"/>, updating the asset already there
    /// rather than replacing it. Configs reference clips by GUID, so a re-run must not mint new
    /// ones; an existing slice and any authored components are left alone.
    /// </summary>
    public static AnnotatedAnimationClip CreateOrUpdate(AnimationClip clip, Transform rootBone,
        string assetPath, out bool created)
    {
        var asset = AssetDatabase.LoadAssetAtPath<AnnotatedAnimationClip>(assetPath);
        created = asset == null;
        if (created) asset = ScriptableObject.CreateInstance<AnnotatedAnimationClip>();

        asset.SetSource(clip, rootBone);

        // The cast is load-bearing: AnnotatedAnimationClip shadows FrameCount with the slice-local
        // count, and the slice is what we are about to set.
        var clipFrameCount = ((SkeletonAnimation)asset).FrameCount;
        if (created || asset.endFrame <= 0 || asset.endFrame > clipFrameCount)
        {
            asset.endFrame = clipFrameCount;
        }

        asset.startFrame = Mathf.Clamp(asset.startFrame, 0, asset.endFrame);

        if (created) AssetDatabase.CreateAsset(asset, assetPath);
        else EditorUtility.SetDirty(asset);

        return asset;
    }

    /// <summary>
    /// Creates an annotated clip for every take of <paramref name="model"/> whose name matches
    /// <paramref name="nameFilter"/>, into <paramref name="outputFolder"/>.
    /// </summary>
    /// <remarks>
    /// Gait phase is added only to clips that carry no <see cref="GaitPhaseComponent"/> yet, so a
    /// re-run cannot overwrite anchors someone corrected by hand.
    /// </remarks>
    public static BatchReport CreateBatch(GameObject model, string nameFilter, string outputFolder,
        bool addGaitPhase, bool detectFootfalls)
    {
        var report = new BatchReport();

        if (!TryResolveModel(model, out var rootBone, out var clips, out var error))
        {
            report.Error = error;
            return report;
        }

        report.TotalClips = clips.Count;

        var matched = new List<AnimationClip>();
        foreach (var clip in clips)
        {
            if (MatchesFilter(clip.name, nameFilter)) matched.Add(clip);
        }

        report.MatchedClips = matched.Count;
        if (matched.Count == 0)
        {
            report.Error = $"No take matches \"{nameFilter}\" ({clips.Count} takes in the model).";
            return report;
        }

        if (!EnsureFolder(outputFolder))
        {
            report.Error = $"Could not create the folder '{outputFolder}'. It must be under Assets.";
            return report;
        }

        try
        {
            AssetDatabase.StartAssetEditing();

            for (var i = 0; i < matched.Count; i++)
            {
                var clip = matched[i];
                EditorUtility.DisplayProgressBar("Create Annotated Clips", clip.name,
                    (float)i / matched.Count);

                var path = AssetPathFor(outputFolder, clip.name);
                var asset = CreateOrUpdate(clip, rootBone, path, out var created);
                if (created) report.Created++;
                else report.Updated++;

                if (!addGaitPhase || asset.TryGetComponent<GaitPhaseComponent>(out _)) continue;

                var phase = new GaitPhaseComponent();
                asset.components.Add(phase);
                EditorUtility.SetDirty(asset);

                if (!detectFootfalls) continue;

                if (phase.TryDetect(asset, out _, out var detectError))
                {
                    report.Detected++;
                }
                else
                {
                    report.DetectionFailed++;
                    Debug.LogError($"Footfall detection failed on \"{clip.name}\": {detectError}", asset);
                }
            }
        }
        finally
        {
            EditorUtility.ClearProgressBar();
            AssetDatabase.StopAssetEditing();
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        report.Ok = true;
        return report;
    }
}
}
