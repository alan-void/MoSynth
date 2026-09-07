using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Runs footfall detection over every <see cref="AnnotatedAnimationClip"/> in the Project-window
/// selection, so a whole retargeted dataset can be re-detected without opening each clip.
/// </summary>
/// <remarks>
/// Anchors are correctable by hand — that is why <see cref="GaitPhaseComponent"/> stores markers
/// rather than a baked curve — and nothing records that a clip was corrected. So a clip that already
/// has footfalls is skipped unless the prompt is answered otherwise; that prompt is the only thing
/// standing between a bulk run and losing that work.
/// </remarks>
public static class DetectFootfallsMenu
{
    /// <summary>What a run did, for the console and the closing dialog.</summary>
    public sealed class Report
    {
        public int Detected;    // clips whose anchors were detected
        public int Seeded;      // of those, ones that had no component until this run added one
        public int Skipped;     // clips left alone because they already had anchors
        public int Failed;
        public bool Cancelled;

        public string Summary()
        {
            var parts = new List<string>();
            if (Detected > 0) parts.Add($"{Detected} detected");
            if (Seeded > 0) parts.Add($"{Seeded} of those newly given a gait phase component");
            if (Skipped > 0) parts.Add($"{Skipped} skipped (already had footfalls)");
            if (Failed > 0) parts.Add($"{Failed} failed");

            var summary = parts.Count == 0 ? "Nothing to do" : string.Join(", ", parts);
            return Cancelled ? summary + ". Cancelled part way; what had run was kept." : summary + ".";
        }
    }

    private const string AssetsPath = "Assets/MoSynth/Detect Footfalls";
    private const string MenuPath = "MoSynth/Animation/Detect Footfalls In Selection";

    [MenuItem(AssetsPath, true)]
    [MenuItem(MenuPath, true, priority = 101)]
    private static bool ValidateRun() => Selected().Length > 0;

    [MenuItem(AssetsPath)]
    [MenuItem(MenuPath, priority = 101)]
    private static void Run()
    {
        var clips = Selected();
        if (clips.Length == 0) return;

        if (!TryAskAboutExistingAnchors(clips, out var overwriteExisting)) return;

        var report = DetectAll(clips, overwriteExisting);

        Debug.Log($"[GaitPhase] Detect Footfalls: {report.Summary()}");
        EditorUtility.DisplayDialog("Detect Footfalls", report.Summary(), "OK");
    }

    /// <summary>
    /// The clips a run would touch. <c>DeepAssets</c> is what makes selecting a folder work, so
    /// nothing here has to scan the project itself.
    /// </summary>
    private static AnnotatedAnimationClip[] Selected() =>
        Selection.GetFiltered<AnnotatedAnimationClip>(SelectionMode.Assets | SelectionMode.DeepAssets);

    /// <summary>Whether this clip is a candidate, given whether the run is overwriting.</summary>
    /// <remarks>
    /// A clip with no component is one this run will add one to. A component with an empty anchor
    /// list holds nothing to lose, so it is detected either way.
    /// </remarks>
    public static bool ShouldDetect(AnnotatedAnimationClip clip, bool overwriteExisting)
    {
        if (clip == null) return false;
        if (overwriteExisting) return true;

        return !clip.TryGetComponent<GaitPhaseComponent>(out var phase)
               || phase.footfalls == null
               || phase.footfalls.Count == 0;
    }

    /// <summary>
    /// Asks what to do about clips that already carry anchors. False means the run was cancelled.
    /// </summary>
    private static bool TryAskAboutExistingAnchors(IReadOnlyList<AnnotatedAnimationClip> clips,
        out bool overwriteExisting)
    {
        overwriteExisting = false;

        var withAnchors = 0;
        foreach (var clip in clips)
        {
            if (!ShouldDetect(clip, overwriteExisting: false)) withAnchors++;
        }

        if (withAnchors == 0) return true;

        var choice = EditorUtility.DisplayDialogComplex("Detect Footfalls",
            $"{withAnchors} of {clips.Count} selected clips already have footfalls, which may have " +
            "been corrected by hand. Redetecting replaces them.",
            "Skip Them", "Cancel", "Redetect All");

        switch (choice)
        {
            case 0: return true;                              // skip the ones that have anchors
            case 2: overwriteExisting = true; return true;
            default: return false;
        }
    }

    private static Report DetectAll(IReadOnlyList<AnnotatedAnimationClip> clips, bool overwriteExisting)
    {
        var report = new Report();
        var undoGroup = Undo.GetCurrentGroup();
        Undo.SetCurrentGroupName("Detect footfalls");

        AssetDatabase.StartAssetEditing();
        try
        {
            for (var i = 0; i < clips.Count; i++)
            {
                var clip = clips[i];
                if (clip == null) continue;

                if (!ShouldDetect(clip, overwriteExisting))
                {
                    report.Skipped++;
                    continue;
                }

                if (EditorUtility.DisplayCancelableProgressBar("Detect Footfalls", clip.name,
                        (float)i / clips.Count))
                {
                    report.Cancelled = true;
                    break;
                }

                Detect(clip, report);
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

        // Written straight to the asset, so an open editor is still showing the anchors from before.
        AnnotatedClipEditorWindow.RefreshOpenWindows();
        return report;
    }

    private static void Detect(AnnotatedAnimationClip clip, Report report)
    {
        // TryDetect assigns the anchor list as a plain field, and this may also add an element to
        // the managed-reference component list — neither is something RecordObject's diffing
        // tracks reliably, so the whole object is snapshotted. Snapshotting does not dirty the
        // asset; SetDirty below is separate.
        Undo.RegisterCompleteObjectUndo(clip, "Detect footfalls");

        var seeded = !clip.TryGetComponent<GaitPhaseComponent>(out var phase);
        if (seeded)
        {
            phase = new GaitPhaseComponent();
            clip.components.Add(phase);
        }

        if (phase.TryDetect(clip, out _, out var error))
        {
            if (seeded) report.Seeded++;
            report.Detected++;
            EditorUtility.SetDirty(clip);
        }
        else
        {
            report.Failed++;
            Debug.LogError($"[GaitPhase] {clip.name}: {error}", clip);

            // A component added for a detection that then failed would leave the clip carrying an
            // empty lane it did not have before.
            if (seeded) clip.components.Remove(phase);
        }
    }
}
}
