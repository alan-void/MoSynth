using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Gets a sweep from "here is a config" to "play mode is running it". Shared by the menu item and
/// the command-line entry point so both take exactly the same route into
/// <see cref="SynthesisBenchmarkDriver"/>.
/// </summary>
public static class SynthesisBenchmarkLauncher
{
    /// <summary>
    /// Validates the config, writes the plan, replaces the open scene with an empty one and enters
    /// play mode. Returns false when the sweep could not be started, having already logged why.
    /// </summary>
    /// <param name="exitEditorWhenDone">Quit the Editor with a status code at the end of the sweep.</param>
    public static bool Launch(SynthesisBenchmarkConfig config, string outputDirectory, bool exitEditorWhenDone)
    {
        if (config == null)
        {
            Debug.LogError("[Benchmark] No config supplied.");
            return false;
        }

        if (!config.TryValidate(out var error))
        {
            Debug.LogError($"[Benchmark] \"{config.name}\" is not runnable: {error}");
            return false;
        }

        var configAssetPath = AssetDatabase.GetAssetPath(config);
        if (string.IsNullOrEmpty(configAssetPath))
        {
            Debug.LogError("[Benchmark] The config must be a saved asset; the sweep reloads it after entering play mode.");
            return false;
        }

        if (EditorApplication.isPlaying)
        {
            Debug.LogError("[Benchmark] Already in play mode. Stop it first — the sweep owns the play session.");
            return false;
        }

        try
        {
            Directory.CreateDirectory(outputDirectory);
        }
        catch (Exception e)
        {
            Debug.LogError($"[Benchmark] Could not create \"{outputDirectory}\": {e.Message}");
            return false;
        }

        var plan = new SynthesisBenchmarkPlan
        {
            configAssetPath = configAssetPath,
            outputDirectory = outputDirectory,
            exitEditorWhenDone = exitEditorWhenDone
        };

        Directory.CreateDirectory(Path.GetDirectoryName(SynthesisBenchmarkPlan.PlanPath) ?? "Temp");
        File.WriteAllText(SynthesisBenchmarkPlan.PlanPath, JsonUtility.ToJson(plan, true));

        // A fresh scene: anything else open would compete for the frame the cost metrics measure.
        EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // Armed on both sides of the play-mode domain reload: this call drives the sweep when domain
        // reload is disabled, otherwise the driver's static constructor picks the plan file back up.
        SynthesisBenchmarkDriver.Arm();
        EditorApplication.EnterPlaymode();
        return true;
    }

    /// <summary>
    /// Resolves an output directory the way the rest of the project resolves user-supplied paths:
    /// absolute ones as given, relative ones against the project root next to Assets.
    /// </summary>
    public static string ResolveOutputDirectory(string path)
    {
        return Path.IsPathRooted(path)
            ? path
            : Path.Combine(Directory.GetParent(Application.dataPath)!.FullName, path);
    }

    /// <summary>A sweep folder name that sorts chronologically and is safe on every filesystem.</summary>
    public static string TimestampFolderName() => DateTime.Now.ToString("yyyyMMdd_HHmmss");
}
}
