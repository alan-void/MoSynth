using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>Menu entry for running a benchmark sweep from a live Editor session.</summary>
public static class SynthesisBenchmarkMenu
{
    private const string LastConfigPrefsKey = "MoSynth.Benchmark.LastConfigPath";

    [MenuItem("MoSynth/Benchmark/Run Sweep", priority = 100)]
    public static void RunSweep()
    {
        var config = ResolveConfig();
        if (config == null)
        {
            EditorUtility.DisplayDialog(
                "Run Sweep",
                "No SynthesisBenchmarkConfig found. Create one via Assets > Create > MoSynth > Synthesis Benchmark Config, " +
                "then select it in the Project window and run this again.",
                "OK");
            return;
        }

        // The sweep replaces whatever scene is open, so the usual save prompt has to come first.
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

        EditorPrefs.SetString(LastConfigPrefsKey, AssetDatabase.GetAssetPath(config));

        var outputDirectory = SynthesisBenchmarkLauncher.ResolveOutputDirectory(
            Path.Combine(config.outputDirectory, SynthesisBenchmarkLauncher.TimestampFolderName()));

        SynthesisBenchmarkLauncher.Launch(config, outputDirectory, exitEditorWhenDone: false);
    }

    [MenuItem("MoSynth/Benchmark/Run Sweep", validate = true)]
    private static bool ValidateRunSweep() => !EditorApplication.isPlaying;

    /// <summary>
    /// The config to run: whatever is selected, else the one used last, else the only one in the
    /// project. Falling back this way means the common case — one benchmark config, run repeatedly —
    /// needs no selection at all.
    /// </summary>
    private static SynthesisBenchmarkConfig ResolveConfig()
    {
        if (Selection.activeObject is SynthesisBenchmarkConfig selected) return selected;

        var lastPath = EditorPrefs.GetString(LastConfigPrefsKey, null);
        if (!string.IsNullOrEmpty(lastPath))
        {
            var last = AssetDatabase.LoadAssetAtPath<SynthesisBenchmarkConfig>(lastPath);
            if (last != null) return last;
        }

        var guids = AssetDatabase.FindAssets($"t:{nameof(SynthesisBenchmarkConfig)}");
        if (guids.Length == 0) return null;

        if (guids.Length > 1)
        {
            Debug.LogWarning($"[Benchmark] {guids.Length} benchmark configs in the project and none selected; " +
                             "using the first. Select one in the Project window to choose.");
        }

        return AssetDatabase.LoadAssetAtPath<SynthesisBenchmarkConfig>(AssetDatabase.GUIDToAssetPath(guids[0]));
    }
}
}
