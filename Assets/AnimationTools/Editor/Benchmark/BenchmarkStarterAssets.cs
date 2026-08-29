using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Splines;

namespace AnimationTools.Editor
{
/// <summary>
/// One-shot setup: turns the characters and splines already wired up in a scene into the path and
/// method prefabs a sweep runs on, plus a config pointing at them.
/// </summary>
/// <remarks>
/// The alternative is dragging half a dozen objects out of <c>ExampleSplines 1.unity</c> by hand and
/// getting the folder layout right, which is exactly the kind of setup that is wrong the first time
/// and silently wrong the second. Run it once; after that, paths are added by dropping a prefab in
/// the folder.
/// <para>
/// Character prefabs lose their scene spline reference on save, which is intended — the driver
/// assigns each run's path. Unity logs that as a warning.
/// </para>
/// </remarks>
public static class BenchmarkStarterAssets
{
    public const string SourceScene = "Assets/Scenes/ExampleSplines 1.unity";
    public const string RootFolder = "Assets/Benchmarks";
    public const string PathsFolder = RootFolder + "/Paths";
    public const string RandomPathsFolder = PathsFolder + "/Random";
    public const string MethodsFolder = RootFolder + "/Methods";
    public const string ConfigPath = RootFolder + "/DefaultBenchmark.asset";

    [MenuItem("MoSynth/Benchmark/Create Starter Assets", priority = 200)]
    public static void CreateStarterAssetsMenu()
    {
        if (!File.Exists(SourceScene))
        {
            EditorUtility.DisplayDialog("Create Starter Assets",
                $"\"{SourceScene}\" not found. Extract the prefabs by hand from whichever scene has your characters.", "OK");
            return;
        }

        if (!EditorUtility.DisplayDialog("Create Starter Assets",
                $"This opens \"{SourceScene}\" and saves its characters and splines as prefabs under {RootFolder}, " +
                "then creates a benchmark config pointing at them.\n\nThe scene itself is not modified.",
                "Go ahead", "Cancel"))
        {
            return;
        }

        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

        var config = Create();
        if (config == null) return;

        Selection.activeObject = config;
        EditorGUIUtility.PingObject(config);
    }

    /// <summary>
    /// Does the extraction, with no dialogs and no save prompt — the caller owns both, which is what
    /// lets this run unattended as well as from the menu. Replaces whatever scene is open with
    /// <see cref="SourceScene"/>. Returns the config, or null if the source scene is missing.
    /// </summary>
    public static SynthesisBenchmarkConfig Create()
    {
        if (!File.Exists(SourceScene))
        {
            Debug.LogError($"[Benchmark] \"{SourceScene}\" not found.");
            return null;
        }

        BenchmarkPathAssets.EnsureFolder(RootFolder);
        BenchmarkPathAssets.EnsureFolder(PathsFolder);
        BenchmarkPathAssets.EnsureFolder(MethodsFolder);

        var scene = EditorSceneManager.OpenScene(SourceScene, OpenSceneMode.Single);

        var pathPrefabs = new List<GameObject>();
        var methodPrefabs = new List<GameObject>();
        var seen = new HashSet<GameObject>();
        var report = new StringBuilder();

        foreach (var root in scene.GetRootGameObjects())
        {
            foreach (var container in root.GetComponentsInChildren<SplineContainer>(true))
            {
                var target = container.transform.root.gameObject;
                if (!seen.Add(target)) continue;

                var prefab = SavePrefab(target, PathsFolder);
                pathPrefabs.Add(prefab);
                report.AppendLine($"  path   \"{prefab.name}\" (knots {container.Spline?.Count ?? 0}, " +
                                  $"closed {container.Spline?.Closed}) from scene object \"{target.name}\"");
            }

            foreach (var synthesizer in root.GetComponentsInChildren<MotionSynthesisComponent>(true))
            {
                var target = synthesizer.transform.root.gameObject;
                if (!seen.Add(target)) continue;

                // Only characters that can be steered along a path are benchmarkable here.
                var input = FindSplineInput(target);
                if (input == null)
                {
                    Debug.LogWarning($"[Benchmark] Skipping \"{target.name}\": no IMotionSynthesisSplineControlInput, " +
                                     "so nothing would tell it where to go.");
                    continue;
                }

                var prefab = SavePrefab(target, MethodsFolder);
                methodPrefabs.Add(prefab);

                var stages = new List<string>();
                foreach (var stage in synthesizer.stages)
                {
                    if (stage != null) stages.Add(stage.GetType().Name);
                }

                report.AppendLine($"  method \"{prefab.name}\" via {input.GetType().Name}, " +
                                  $"stages [{string.Join(", ", stages)}]");
            }
        }

        var config = CreateConfig(methodPrefabs);
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        Debug.Log($"[Benchmark] Starter assets: {pathPrefabs.Count} path(s), {methodPrefabs.Count} method(s), " +
                  $"config at {ConfigPath}.\n{report}" +
                  "Character prefabs lose their scene spline reference on save; the sweep assigns it per run.");

        return config;
    }

    private static SynthesisBenchmarkConfig CreateConfig(List<GameObject> methodPrefabs)
    {
        var config = AssetDatabase.LoadAssetAtPath<SynthesisBenchmarkConfig>(ConfigPath);
        var isNew = config == null;
        if (isNew) config = ScriptableObject.CreateInstance<SynthesisBenchmarkConfig>();

        config.pathPrefabFolder = PathsFolder;

        // Regenerating must not silently drop the overrides that turn one prefab into several
        // methods, so an entry already pointing at this prefab keeps its own.
        var existing = config.methods ?? new List<BenchmarkMethod>();
        config.methods = new List<BenchmarkMethod>();
        foreach (var prefab in methodPrefabs)
        {
            var method = new BenchmarkMethod { name = prefab.name, characterPrefab = prefab };
            foreach (var previous in existing)
            {
                if (previous == null || previous.characterPrefab != prefab) continue;
                method.overrides = previous.overrides;
                break;
            }

            config.methods.Add(method);
        }

        if (isNew) AssetDatabase.CreateAsset(config, ConfigPath);
        else EditorUtility.SetDirty(config);

        return config;
    }

    /// <summary>
    /// Writes the prefab at a name-derived path, overwriting any prefab already there rather than
    /// making a numbered sibling. Overwriting in place keeps the asset's GUID, so a config already
    /// pointing at it survives a regeneration.
    /// </summary>
    private static GameObject SavePrefab(GameObject sceneObject, string folder)
    {
        var path = $"{folder}/{Sanitize(sceneObject.name)}.prefab";
        return PrefabUtility.SaveAsPrefabAsset(sceneObject, path);
    }

    private static IMotionSynthesisSplineControlInput FindSplineInput(GameObject root)
    {
        foreach (var behaviour in root.GetComponentsInChildren<MonoBehaviour>(true))
        {
            if (behaviour is IMotionSynthesisSplineControlInput input) return input;
        }

        return null;
    }

    private static string Sanitize(string name)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var chars = name.ToCharArray();
        for (var i = 0; i < chars.Length; i++)
        {
            if (System.Array.IndexOf(invalid, chars[i]) >= 0) chars[i] = '_';
        }

        return new string(chars).Trim();
    }
}
}
