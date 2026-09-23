using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Splines;

namespace AnimationTools.Editor
{
/// <summary>
/// Inspector for <see cref="SynthesisBenchmarkConfig"/>: the default fields, a preview of the paths
/// the folder currently resolves to, and the button that runs the sweep.
/// </summary>
/// <remarks>
/// <see cref="SynthesisBenchmarkConfig.pathPrefabFolder"/> is resolved at sweep start, so the
/// preview is the only way to see what a run will cover before it starts.
/// </remarks>
[CustomEditor(typeof(SynthesisBenchmarkConfig))]
public class SynthesisBenchmarkConfigEditor : UnityEditor.Editor
{
    private bool _showPaths = true;

    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var config = (SynthesisBenchmarkConfig)target;

        EditorGUILayout.Space();

        var runnable = config.TryValidate(out var error);
        if (!runnable) EditorGUILayout.HelpBox(error, MessageType.Error);

        var paths = PreviewPaths(config, out var skipped);
        _showPaths = EditorGUILayout.Foldout(_showPaths, $"Paths ({paths.Count})", true);
        if (_showPaths)
        {
            using (new EditorGUI.IndentLevelScope())
            {
                if (paths.Count == 0)
                {
                    EditorGUILayout.HelpBox(
                        "No path prefabs with a SplineContainer. Point the folder at one, or fill in the explicit list.",
                        MessageType.Warning);
                }

                foreach (var path in paths) EditorGUILayout.LabelField("• " + path.name);

                foreach (var name in skipped)
                    EditorGUILayout.LabelField($"• {name}", "skipped: no SplineContainer");
            }
        }

        if (runnable && paths.Count > 0)
        {
            EditorGUILayout.HelpBox(
                $"{config.methods.Count} method(s) x {paths.Count} path(s) = {config.methods.Count * paths.Count} runs, " +
                $"each at least {config.settleTime:0.#} s settle plus {config.lapsRequired:0.##} lap(s), " +
                $"capped at {config.maxRunSeconds:0.#} s.",
                MessageType.Info);
        }

        EditorGUILayout.Space();

        using (new EditorGUI.DisabledScope(!runnable || paths.Count == 0 || EditorApplication.isPlaying))
        {
            if (GUILayout.Button("Run Sweep", GUILayout.Height(28)))
            {
                Selection.activeObject = config;
                // Deferred: replacing the scene and entering play mode do not belong in OnInspectorGUI.
                EditorApplication.delayCall += SynthesisBenchmarkMenu.RunSweep;
            }
        }

        if (EditorApplication.isPlaying)
            EditorGUILayout.HelpBox("Stop play mode to run a sweep; it owns the play session.", MessageType.Info);
    }

    /// <summary>
    /// The same resolution the driver performs, minus the logging, since this runs every repaint.
    /// </summary>
    private static List<GameObject> PreviewPaths(SynthesisBenchmarkConfig config, out List<string> skipped)
    {
        skipped = new List<string>();
        var candidates = new List<GameObject>();

        if (!string.IsNullOrWhiteSpace(config.pathPrefabFolder) && AssetDatabase.IsValidFolder(config.pathPrefabFolder))
        {
            var guids = AssetDatabase.FindAssets("t:Prefab", new[] { config.pathPrefabFolder });
            var assetPaths = new List<string>(guids.Length);
            foreach (var guid in guids) assetPaths.Add(AssetDatabase.GUIDToAssetPath(guid));
            assetPaths.Sort(StringComparer.Ordinal);

            foreach (var assetPath in assetPaths)
            {
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);
                if (prefab != null) candidates.Add(prefab);
            }
        }
        else if (config.pathPrefabs != null)
        {
            foreach (var prefab in config.pathPrefabs)
            {
                if (prefab != null) candidates.Add(prefab);
            }
        }

        var usable = new List<GameObject>(candidates.Count);
        foreach (var prefab in candidates)
        {
            if (prefab.GetComponentInChildren<SplineContainer>(true) != null) usable.Add(prefab);
            else skipped.Add(prefab.name);
        }

        return usable;
    }
}
}
