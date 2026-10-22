using System;
using System.Collections.Generic;
using System.IO;
using AnimationTools;
using AnimationTools.Editor;
using GameplayTags;
using GameplayTags.Editor;
using UnityEditor;
using UnityEngine;

namespace Style.Editor
{
/// <summary>
/// Prepares retargeted 100STYLE clips for a <see cref="StyleTransferConfig"/>: a whole-clip
/// <c>style.&lt;Name&gt;</c> and <c>gait.&lt;Code&gt;</c> tag parsed from each clip's name, the
/// slice the dataset's frame cuts give, and footfalls where the clip has none.
/// </summary>
/// <remarks>
/// The tags are authored data from then on, so a mislabelled take is fixed in the clip editor, not
/// here. Tags are created through <see cref="GameplayTagConfig"/>, which keeps them under
/// <c>Assets/Resources/GameplayTags</c>, the only folder the tag picker and lookup read.
/// </remarks>
public static class HundredStyleClipSetup
{
    /// <summary>What a run did, for the console and the closing dialog.</summary>
    public sealed class Report
    {
        public int Clips;
        public int ChannelsAdded;
        public int Sliced;
        public int NoFrameCut;
        public int FootfallsDetected;
        public int FootfallsSkipped;
        public int FootfallsFailed;
        public readonly List<string> Unparsed = new();
        public bool Cancelled;

        public string Summary()
        {
            var summary = $"{Clips} clips set up: {ChannelsAdded} tag channels added, {Sliced} sliced " +
                          $"from the frame cuts ({NoFrameCut} had none), footfalls detected on " +
                          $"{FootfallsDetected} ({FootfallsSkipped} already had them, {FootfallsFailed} failed).";
            if (Unparsed.Count > 0) summary += $" {Unparsed.Count} clip names could not be parsed.";
            return Cancelled ? summary + " Cancelled part way; what had run was kept." : summary;
        }
    }

    public const string StyleRoot = "style";
    public const string GaitRoot = "gait";

    /// <summary>The rate the dataset's frame cuts are numbered at.</summary>
    public const float SourceFrameRate = 60f;

    /// <summary>
    /// Longest slice kept per take. Every take is cyclic locomotion, so 20 s holds well over ten
    /// strides; the whole dataset at full length would not bake within Editor memory.
    /// </summary>
    public const float MaxTakeSeconds = 20f;

    /// <summary>
    /// Toe speed below which a foot counts as planted, in m/s. The 0.15 default reads this 60 fps
    /// capture's Neutral walk as planted 30% of the time and its run 2%; 0.5 gives 58% and 38%.
    /// </summary>
    public const float ContactVelocityThreshold = 0.5f;

    private static readonly Dictionary<string, string> GaitDescriptions = new()
    {
        { "FW", "Walking forward." },
        { "BW", "Walking backward." },
        { "SW", "Walking sideways." },
        { "FR", "Running forward." },
        { "BR", "Running backward." },
        { "SR", "Running sideways." },
        { "ID", "Standing idle." },
        { "TR1", "Transitions between gaits, take 1." },
        { "TR2", "Transitions between gaits, take 2." },
        { "TR3", "Transitions between gaits, take 3." },
    };

    private const string AssetsPath = "Assets/MoSynth/Set Up 100STYLE Clips";
    private const string MenuPath = "MoSynth/Style/Set Up 100STYLE Clips";

    private static string DatasetFolder =>
        Path.GetFullPath(Path.Combine(Application.dataPath, "..", "External", "LFS", "100STYLE"));

    [MenuItem(AssetsPath, true)]
    [MenuItem(MenuPath, true)]
    private static bool ValidateRun() => Selected().Length > 0;

    [MenuItem(AssetsPath)]
    [MenuItem(MenuPath)]
    private static void Run()
    {
        var clips = Selected();
        if (clips.Length == 0) return;

        if (!TryLoadDataset(out var dataset, out var error))
        {
            EditorUtility.DisplayDialog("Set Up 100STYLE Clips", error, "OK");
            return;
        }

        var report = new Report();
        SetUp(clips, dataset, report);

        Debug.Log($"[Style] Set Up 100STYLE Clips: {report.Summary()}");
        if (report.Unparsed.Count > 0)
        {
            Debug.LogWarning("[Style] Clip names not of the form [prefix_]<Style>_<Gait>, or naming a " +
                             $"style the dataset does not list: {string.Join(", ", report.Unparsed)}");
        }

        EditorUtility.DisplayDialog("Set Up 100STYLE Clips", report.Summary(), "OK");
    }

    /// <summary>The dataset's style descriptions and frame cuts, read from its CSVs.</summary>
    public sealed class Dataset
    {
        public Dictionary<string, string> Descriptions;
        public Dictionary<string, Dictionary<string, (int start, int stop)>> FrameCuts;
    }

    /// <summary>Read the dataset's CSVs. False, with a message, when either is missing.</summary>
    public static bool TryLoadDataset(out Dataset dataset, out string error)
    {
        dataset = null;
        var descriptionsPath = Path.Combine(DatasetFolder, "Dataset_List.csv");
        var frameCutsPath = Path.Combine(DatasetFolder, "Frame_Cuts.csv");
        if (!File.Exists(descriptionsPath) || !File.Exists(frameCutsPath))
        {
            error = $"Dataset_List.csv and Frame_Cuts.csv are expected in {DatasetFolder}. Fetch the " +
                    "dataset first.";
            return false;
        }

        dataset = new Dataset
        {
            Descriptions = ParseDescriptions(File.ReadAllLines(descriptionsPath)),
            FrameCuts = ParseFrameCuts(File.ReadAllLines(frameCutsPath)),
        };
        error = null;
        return true;
    }

    /// <summary>
    /// Tag, slice and detect footfalls on <paramref name="clips"/>, adding what was done to
    /// <paramref name="report"/>, so several calls can share one report. Shows no dialog.
    /// </summary>
    public static void SetUp(IReadOnlyList<AnnotatedAnimationClip> clips, Dataset dataset, Report report)
    {
        var plans = new List<(AnnotatedAnimationClip clip, string style, string gait)>();
        foreach (var clip in clips)
        {
            if (TryParseClipName(clip.name, dataset.Descriptions.Keys, out var style, out var gait))
            {
                plans.Add((clip, style, gait));
            }
            else
            {
                report.Unparsed.Add(clip.name);
            }
        }

        // Before asset editing starts: a tag created inside it cannot be loaded back until it ends.
        var tags = EnsureTags(plans, dataset.Descriptions, report);
        ApplyAll(plans, tags, dataset.FrameCuts, report);
    }

    /// <summary><c>DeepAssets</c> is what makes selecting a folder work.</summary>
    private static AnnotatedAnimationClip[] Selected() =>
        Selection.GetFiltered<AnnotatedAnimationClip>(SelectionMode.Assets | SelectionMode.DeepAssets);

    // --- Parsing ------------------------------------------------------------------------------------

    /// <summary>
    /// Splits <c>[prefix_]&lt;Style&gt;_&lt;Gait&gt;</c>. Style names hold no underscore, so the
    /// last two tokens decide and anything before them is a prefix a retarget may have added.
    /// </summary>
    /// <param name="knownStyles">The dataset's style names; the result takes their spelling.</param>
    public static bool TryParseClipName(string clipName, IEnumerable<string> knownStyles,
        out string style, out string gait)
    {
        style = null;
        gait = null;
        if (string.IsNullOrEmpty(clipName)) return false;

        var tokens = clipName.Split('_');
        if (tokens.Length < 2) return false;

        var gaitToken = tokens[^1].ToUpperInvariant();
        if (!GaitDescriptions.ContainsKey(gaitToken)) return false;

        var styleToken = tokens[^2];
        foreach (var known in knownStyles)
        {
            if (!string.Equals(known, styleToken, StringComparison.OrdinalIgnoreCase)) continue;

            style = known;
            gait = gaitToken;
            return true;
        }

        return false;
    }

    /// <summary>Style name to description, from <c>Dataset_List.csv</c>.</summary>
    public static Dictionary<string, string> ParseDescriptions(IReadOnlyList<string> lines)
    {
        var descriptions = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = 1; i < lines.Count; i++)
        {
            var fields = SplitCsv(lines[i]);
            if (fields.Count < 2 || string.IsNullOrWhiteSpace(fields[0])) continue;

            descriptions[fields[0].Trim()] = fields[1].Trim();
        }

        return descriptions;
    }

    /// <summary>
    /// Style name to gait code to its <c>[start, stop)</c> source frames, from <c>Frame_Cuts.csv</c>.
    /// A gait whose cut reads <c>N/A</c> is absent.
    /// </summary>
    public static Dictionary<string, Dictionary<string, (int start, int stop)>> ParseFrameCuts(
        IReadOnlyList<string> lines)
    {
        var cuts = new Dictionary<string, Dictionary<string, (int, int)>>(StringComparer.OrdinalIgnoreCase);
        if (lines.Count == 0) return cuts;

        var header = SplitCsv(lines[0]);
        for (var i = 1; i < lines.Count; i++)
        {
            var fields = SplitCsv(lines[i]);
            if (fields.Count == 0 || string.IsNullOrWhiteSpace(fields[0])) continue;

            var byGait = new Dictionary<string, (int, int)>(StringComparer.OrdinalIgnoreCase);
            foreach (var gait in GaitDescriptions.Keys)
            {
                var startColumn = header.IndexOf(gait + "_START");
                var stopColumn = header.IndexOf(gait + "_STOP");
                if (startColumn < 0 || stopColumn < 0 ||
                    startColumn >= fields.Count || stopColumn >= fields.Count) continue;

                if (int.TryParse(fields[startColumn].Trim(), out var start) &&
                    int.TryParse(fields[stopColumn].Trim(), out var stop))
                {
                    byGait[gait] = (start, stop);
                }
            }

            cuts[fields[0].Trim()] = byGait;
        }

        return cuts;
    }

    /// <summary>
    /// A cut in source frames as a slice of a clip baked at <paramref name="clipFrameRate"/>. False
    /// when the cut does not land inside the clip.
    /// </summary>
    public static bool TryScaleSlice(int sourceStart, int sourceStop, float clipFrameRate,
        int clipFrameCount, out int startFrame, out int endFrame)
    {
        var scale = clipFrameRate / SourceFrameRate;
        startFrame = Mathf.Max(0, Mathf.RoundToInt(sourceStart * scale));
        endFrame = Mathf.Min(clipFrameCount, Mathf.RoundToInt(sourceStop * scale));
        return endFrame - startFrame >= 2;
    }

    /// <summary>The end of a slice cut to at most <paramref name="maxFrames"/> frames from its start.</summary>
    public static int CapSliceEnd(int startFrame, int endFrame, int maxFrames) =>
        Mathf.Min(endFrame, startFrame + maxFrames);

    /// <summary>Comma-separated fields, honouring double quotes around a field with a comma in it.</summary>
    private static List<string> SplitCsv(string line)
    {
        var fields = new List<string>();
        if (line == null) return fields;

        var field = new System.Text.StringBuilder();
        var quoted = false;
        for (var i = 0; i < line.Length; i++)
        {
            var character = line[i];
            if (character == '"')
            {
                if (quoted && i + 1 < line.Length && line[i + 1] == '"')
                {
                    field.Append('"');
                    i++;
                }
                else
                {
                    quoted = !quoted;
                }
            }
            else if (character == ',' && !quoted)
            {
                fields.Add(field.ToString());
                field.Clear();
            }
            else
            {
                field.Append(character);
            }
        }

        fields.Add(field.ToString());
        return fields;
    }

    // --- Tags ---------------------------------------------------------------------------------------

    /// <summary>Find or create every style and gait tag the plans need, keyed by full tag name.</summary>
    private static Dictionary<string, GameplayTagSO> EnsureTags(
        List<(AnnotatedAnimationClip clip, string style, string gait)> plans,
        Dictionary<string, string> descriptions, Report report)
    {
        var tags = new Dictionary<string, GameplayTagSO>();
        var tagConfig = GameplayTagConfig.instance;

        for (var i = plans.Count - 1; i >= 0; i--)
        {
            var (clip, style, gait) = plans[i];
            var styleName = $"{StyleRoot}.{style}";
            var gaitName = $"{GaitRoot}.{gait}";

            if (!TryEnsureTag(tagConfig, tags, styleName, descriptions.GetValueOrDefault(style)) ||
                !TryEnsureTag(tagConfig, tags, gaitName, GaitDescriptions[gait]))
            {
                // AddTag refuses names that are not alphanumeric, and has already said why.
                report.Unparsed.Add(clip.name);
                plans.RemoveAt(i);
            }
        }

        return tags;
    }

    private static bool TryEnsureTag(GameplayTagConfig tagConfig, Dictionary<string, GameplayTagSO> tags,
        string fullName, string description)
    {
        if (tags.ContainsKey(fullName)) return true;

        var tag = tagConfig.GetTag(fullName) ?? tagConfig.AddTag(fullName);
        if (tag == null) return false;

        // Only filled when empty, so a description edited by hand survives a re-run.
        if (string.IsNullOrEmpty(tag.Description) && !string.IsNullOrEmpty(description))
        {
            var serialized = new SerializedObject(tag);
            serialized.FindProperty("description").stringValue = description;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(tag);
        }

        tags[fullName] = tag;
        return true;
    }

    // --- Clips --------------------------------------------------------------------------------------

    private static void ApplyAll(List<(AnnotatedAnimationClip clip, string style, string gait)> plans,
        Dictionary<string, GameplayTagSO> tags,
        Dictionary<string, Dictionary<string, (int start, int stop)>> frameCuts, Report report)
    {
        var undoGroup = Undo.GetCurrentGroup();
        Undo.SetCurrentGroupName("Set up 100STYLE clips");

        AssetDatabase.StartAssetEditing();
        try
        {
            for (var i = 0; i < plans.Count; i++)
            {
                var (clip, style, gait) = plans[i];
                if (EditorUtility.DisplayCancelableProgressBar("Set Up 100STYLE Clips", clip.name,
                        (float)i / plans.Count))
                {
                    report.Cancelled = true;
                    break;
                }

                // A managed-reference list edit is not tracked reliably by RecordObject's diffing,
                // so the whole object is snapshotted.
                Undo.RegisterCompleteObjectUndo(clip, "Set up 100STYLE clip");

                AddWholeClipChannel(clip, tags[$"{StyleRoot}.{style}"], report);
                AddWholeClipChannel(clip, tags[$"{GaitRoot}.{gait}"], report);

                // Before detection, which reads only the slice.
                if (frameCuts.TryGetValue(style, out var byGait) && byGait.TryGetValue(gait, out var cut))
                {
                    ApplySlice(clip, cut.start, cut.stop, report);
                }
                else
                {
                    report.NoFrameCut++;
                    CapWholeClip(clip);
                }

                DetectFootfallsIfMissing(clip, report);
                // Detection baked the clip; a folder of long takes would otherwise hold every bake.
                clip.ReleaseLoadedData();

                report.Clips++;
                EditorUtility.SetDirty(clip);
            }
        }
        finally
        {
            EditorUtility.ClearProgressBar();
            AssetDatabase.StopAssetEditing();
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        Undo.CollapseUndoOperations(undoGroup);

        // Written straight to the assets, so an open editor is still showing what was there before.
        AnnotatedClipEditorWindow.RefreshOpenWindows();
    }

    /// <summary>A channel on from frame 0 for the whole clip, unless the tag already has one.</summary>
    private static void AddWholeClipChannel(AnnotatedAnimationClip clip, GameplayTagSO tag, Report report)
    {
        if (!clip.TryGetComponent<AnimationTagComponent>(out var component))
        {
            component = new AnimationTagComponent();
            clip.components.Add(component);
        }

        foreach (var channel in component.channels)
        {
            if (channel?.tag == tag) return;
        }

        component.channels.Add(new AnimationTagging.TagChannel { tag = tag, toggles = new List<int> { 0 } });
        report.ChannelsAdded++;
    }

    private static int MaxTakeFrames(AnnotatedAnimationClip clip) =>
        Mathf.RoundToInt(MaxTakeSeconds * clip.Clip.frameRate);

    private static void CapWholeClip(AnnotatedAnimationClip clip)
    {
        if (clip.Clip == null) return;

        clip.startFrame = 0;
        clip.endFrame = CapSliceEnd(0, ((SkeletonAnimation)clip).FrameCount, MaxTakeFrames(clip));
    }

    private static void ApplySlice(AnnotatedAnimationClip clip, int sourceStart, int sourceStop, Report report)
    {
        if (clip.Clip == null) return;

        var fullFrameCount = ((SkeletonAnimation)clip).FrameCount;
        if (!TryScaleSlice(sourceStart, sourceStop, clip.Clip.frameRate, fullFrameCount,
                out var startFrame, out var endFrame))
        {
            Debug.LogWarning($"[Style] {clip.name}: the frame cut [{sourceStart}, {sourceStop}) at " +
                             $"{SourceFrameRate} fps does not fit the clip's {fullFrameCount} frames; " +
                             "kept from its first frame instead.", clip);
            report.NoFrameCut++;
            CapWholeClip(clip);
            return;
        }

        clip.startFrame = startFrame;
        clip.endFrame = CapSliceEnd(startFrame, endFrame, MaxTakeFrames(clip));
        report.Sliced++;
    }

    /// <summary>Detect footfalls on a clip with none, leaving any existing anchors alone.</summary>
    private static void DetectFootfallsIfMissing(AnnotatedAnimationClip clip, Report report)
    {
        var seeded = !clip.TryGetComponent<GaitPhaseComponent>(out var phase);

        // Anchors detected at another threshold are this setup's own earlier output, not hand
        // corrections, so they are detected again rather than kept.
        var detectedElsewise = !seeded &&
                               !Mathf.Approximately(phase.contactVelocityThreshold, ContactVelocityThreshold);
        if (!detectedElsewise && !DetectFootfallsMenu.ShouldDetect(clip, overwriteExisting: false))
        {
            report.FootfallsSkipped++;
            return;
        }

        if (seeded)
        {
            phase = new GaitPhaseComponent();
            clip.components.Add(phase);
        }

        phase.contactVelocityThreshold = ContactVelocityThreshold;

        if (phase.TryDetect(clip, out _, out var error))
        {
            report.FootfallsDetected++;
            return;
        }

        report.FootfallsFailed++;
        Debug.LogError($"[Style] {clip.name}: {error}", clip);

        // Do not leave the clip with an empty component it did not have before.
        if (seeded) clip.components.Remove(phase);
    }
}
}
