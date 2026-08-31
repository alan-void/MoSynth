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

    /// <summary>A loop needs three knots to be a polygon; two would be a there-and-back.</summary>
    private const int MinKnots = RandomPathShapes.MinKnotCount;

    private const int MaxKnots = 30;

    [SerializeField] private int seed = 1;
    [SerializeField] private int count = 12;
    [SerializeField] private float smoothRatio = 0.5f;
    [SerializeField] private float closedRatio = 0.34f;
    [SerializeField] private float walkRatio = 0.5f;
    [SerializeField] private Vector2 extent = new(5f, 9f);
    [SerializeField] private Vector2Int knots = new(5, 12);
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
        walkRatio = EditorGUILayout.Slider(
            new GUIContent("Walk ratio", "Share of what does not loop that wanders freely rather than running down a corridor."), walkRatio, 0f, 1f);

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Shape", EditorStyles.boldLabel);

        extent.x = EditorGUILayout.FloatField(new GUIContent("Extent min (m)", "Smallest overall path size: a loop's radius, and the length an open path is scaled to match."), extent.x);
        extent.y = EditorGUILayout.FloatField(new GUIContent("Extent max (m)", "Largest overall path size."), extent.y);
        extent.x = Mathf.Max(1f, extent.x);
        extent.y = Mathf.Max(extent.x, extent.y);

        knots.x = EditorGUILayout.IntField(new GUIContent("Knots min", "Fewest control points a path may be built from. More knots inside the same extent means a more convoluted path."), knots.x);
        knots.y = EditorGUILayout.IntField(new GUIContent("Knots max", "Most control points a path may be built from."), knots.y);
        knots.x = Mathf.Clamp(knots.x, MinKnots, MaxKnots);
        knots.y = Mathf.Clamp(knots.y, knots.x, MaxKnots);

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
            var kind = RandomPathShapes.KindForIndex(index, count, smoothRatio, closedRatio, walkRatio);
            preview.AppendLine(RandomBenchmarkPathGenerator.NameFor(seed, index, kind));
        }

        EditorGUILayout.SelectableLabel(preview.ToString(),
            EditorStyles.textArea, GUILayout.Height(Mathf.Min(160f, 16f * count + 8f)));

        // A big path can outlast the config's per-run limit, which surfaces as a timeout rather than
        // as anything that looks like a path problem. Open paths are scaled to the same length as a
        // loop of the same extent, so one figure covers every kind.
        var longest = 2f * Mathf.PI * extent.y;
        EditorGUILayout.HelpBox(
            $"Longest path is roughly {longest:0} m, about {longest:0} s to cover at 1 m/s. " +
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
        settings.knotCountMin = knots.x;
        settings.knotCountMax = knots.y;

        var created = RandomBenchmarkPathGenerator.Create(
            seed, count, smoothRatio, closedRatio, walkRatio, settings, clearExisting);
        if (created.Count > 0) EditorGUIUtility.PingObject(created[0]);
    }
}
}
