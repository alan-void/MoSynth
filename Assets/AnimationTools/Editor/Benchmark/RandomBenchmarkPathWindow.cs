using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Parameters for a batch of random benchmark paths, and the button that writes it.
/// </summary>
/// <remarks>
/// A window rather than a bare menu item because the batch has parameters worth seeing before
/// committing to a sweep that runs one character per path: the seed that reproduces it, and the
/// longest lap, which is what decides whether a run fits inside the config's time limit.
/// </remarks>
public class RandomBenchmarkPathWindow : EditorWindow
{
    /// <summary>Slots are named with a two-digit index, which is also all a report row needs to stay readable.</summary>
    private const int MaxCount = 99;

    [SerializeField] private int seed = 1;
    [SerializeField] private int count = 8;
    [SerializeField] private float smoothRatio = 0.5f;
    [SerializeField] private float closedRatio = 0.5f;
    [SerializeField] private Vector2 extent = new(5f, 9f);
    [SerializeField] private float minTurnRadius = 1.5f;
    [SerializeField] private bool clearExisting = true;

    private Vector2 _scroll;

    [MenuItem("MoSynth/Benchmark/Create Random Paths...", priority = 202)]
    public static void Open()
    {
        var window = GetWindow<RandomBenchmarkPathWindow>(true, "Random Benchmark Paths");
        window.minSize = new Vector2(420f, 380f);
        window.Show();
    }

    [MenuItem("MoSynth/Benchmark/Delete Random Paths", priority = 203)]
    public static void DeleteRandomPaths()
    {
        if (!AssetDatabase.IsValidFolder(BenchmarkStarterAssets.RandomPathsFolder))
        {
            EditorUtility.DisplayDialog("Delete Random Paths", "There are no random paths to delete.", "OK");
            return;
        }

        if (!EditorUtility.DisplayDialog("Delete Random Paths",
                $"Delete {BenchmarkStarterAssets.RandomPathsFolder} and every path in it?",
                "Delete", "Cancel"))
        {
            return;
        }

        if (RandomBenchmarkPathGenerator.DeleteAll()) Debug.Log("[Benchmark] Random paths deleted.");
    }

    [MenuItem("MoSynth/Benchmark/Delete Random Paths", validate = true)]
    private static bool ValidateDeleteRandomPaths() => !EditorApplication.isPlaying;

    private void OnGUI()
    {
        _scroll = EditorGUILayout.BeginScrollView(_scroll);

        EditorGUILayout.HelpBox(
            "Random paths widen what a sweep covers without giving up traceability: the seed and slot " +
            "are in every prefab name, so a results.csv row identifies the exact curve, and " +
            "regenerating with the same seed reproduces it.",
            MessageType.Info);

        EditorGUILayout.LabelField("Batch", EditorStyles.boldLabel);

        // Zero is not a valid RNG seed, and a negative one would put a '-' in every file name.
        seed = Mathf.Clamp(EditorGUILayout.IntField(
            new GUIContent("Seed", "Reproduces this exact batch. Appears in every prefab name."), seed), 1, int.MaxValue);
        count = Mathf.Clamp(EditorGUILayout.IntField(
            new GUIContent("Count", "Paths to generate. Every one of them adds a run per method to the sweep."), count), 1, MaxCount);

        smoothRatio = EditorGUILayout.Slider(
            new GUIContent("Smooth ratio", "Share of the batch with rounded turns; the rest have sharp corners."), smoothRatio, 0f, 1f);
        closedRatio = EditorGUILayout.Slider(
            new GUIContent("Closed ratio", "Share of each family that loops. Open paths finish at the far end instead of counting laps."), closedRatio, 0f, 1f);

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Shape", EditorStyles.boldLabel);

        extent.x = EditorGUILayout.FloatField(new GUIContent("Extent min (m)", "Smallest base radius or step scale."), extent.x);
        extent.y = EditorGUILayout.FloatField(new GUIContent("Extent max (m)", "Largest base radius or step scale."), extent.y);
        extent.x = Mathf.Max(minTurnRadius * 2f, extent.x);
        extent.y = Mathf.Max(extent.x, extent.y);

        minTurnRadius = Mathf.Max(0.25f, EditorGUILayout.FloatField(
            new GUIContent("Min turn radius (m)", "Tightest turn a smooth path may demand. Candidates below it are rejected and redrawn."), minTurnRadius));

        EditorGUILayout.Space();
        clearExisting = EditorGUILayout.Toggle(
            new GUIContent("Clear existing", "Delete the previous batch first. Leaving it in place silently adds its runs to every later sweep."), clearExisting);

        if (!clearExisting)
        {
            EditorGUILayout.HelpBox(
                "The previous batch will be kept, so both batches run in every sweep from now on.",
                MessageType.Warning);
        }

        EditorGUILayout.Space();
        DrawPreview();

        EditorGUILayout.Space();
        if (GUILayout.Button($"Generate {count} paths into {BenchmarkStarterAssets.RandomPathsFolder}", GUILayout.Height(28f)))
        {
            Generate();
        }

        EditorGUILayout.EndScrollView();
    }

    private void DrawPreview()
    {
        EditorGUILayout.LabelField("Will write", EditorStyles.boldLabel);

        var preview = new System.Text.StringBuilder();
        for (var index = 0; index < count; index++)
        {
            var kind = RandomPathShapes.KindForIndex(index, count, smoothRatio, closedRatio);
            preview.AppendLine(RandomBenchmarkPathGenerator.NameFor(seed, index, kind));
        }

        EditorGUILayout.SelectableLabel(preview.ToString(),
            EditorStyles.textArea, GUILayout.Height(Mathf.Min(160f, 16f * count + 8f)));

        // A big loop can outlast the config's per-run limit, which surfaces as a timeout rather than
        // as anything that looks like a path problem.
        var longestLap = 2f * Mathf.PI * extent.y;
        EditorGUILayout.HelpBox(
            $"Longest loop is roughly {longestLap:0} m, about {longestLap:0} s per lap at 1 m/s. " +
            "Check that against the benchmark config's Max Run Seconds and Laps Required.",
            MessageType.None);
    }

    private void Generate()
    {
        // Generating replaces whatever scene is open, so the usual save prompt has to come first.
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

        var settings = RandomPathSettings.Default;
        settings.extentMin = extent.x;
        settings.extentMax = extent.y;
        settings.minTurnRadius = minTurnRadius;

        var created = RandomBenchmarkPathGenerator.Create(seed, count, smoothRatio, closedRatio, settings, clearExisting);
        if (created.Count > 0) EditorGUIUtility.PingObject(created[0]);
    }
}
}
