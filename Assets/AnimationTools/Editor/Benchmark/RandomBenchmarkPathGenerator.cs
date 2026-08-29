using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Writes a seeded batch of random benchmark paths as <c>SplineContainer</c> prefabs.
/// </summary>
/// <remarks>
/// The standard suite answers "what was this method asked to do" by construction — four shapes, each
/// isolating one demand. A random batch keeps that answerable a different way: the seed and slot are
/// in the prefab name, and geometry is a pure function of the two, so a report row identifies the
/// exact curve it was measured on.
/// </remarks>
public static class RandomBenchmarkPathGenerator
{
    /// <summary>
    /// The prefab name for one slot. Zero-padding the index keeps the driver's ordinal sort in
    /// numeric order; the seed is left unpadded because a reader retypes it from a report.
    /// </summary>
    public static string NameFor(int seed, int index, RandomPathKind kind) => $"Random_s{seed}_{index:D2}_{kind}";

    /// <summary>
    /// Writes the batch, overwriting any prefab of the same name. No dialogs and no save prompt — the
    /// caller owns both. Works in a throwaway scene so nothing the user has open is dirtied.
    /// </summary>
    public static List<GameObject> Create(
        int seed,
        int count,
        float smoothRatio,
        float closedRatio,
        RandomPathSettings settings,
        bool clearExisting)
    {
        // The generated objects have to live in some scene to be saved as prefabs. An empty
        // untitled one keeps that side effect off whatever the user was working in.
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        if (clearExisting) DeleteAll();

        var folder = BenchmarkStarterAssets.RandomPathsFolder;
        BenchmarkPathAssets.EnsureFolder(folder);

        var report = new StringBuilder();
        var created = new List<GameObject>();
        for (var index = 0; index < count; index++)
        {
            var kind = RandomPathShapes.KindForIndex(index, count, smoothRatio, closedRatio);
            var spline = RandomPathShapes.Generate(kind, seed, index, settings);
            if (spline == null)
            {
                Debug.LogError($"[Benchmark] No {kind} candidate for seed {seed} slot {index} met the " +
                               "path settings; skipping it rather than writing a path nothing can follow. " +
                               "Loosen the minimum turn radius or widen the extents.");
                continue;
            }

            created.Add(BenchmarkPathAssets.Save(NameFor(seed, index, kind), spline, folder, report));
        }

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        Debug.Log($"[Benchmark] {created.Count} random path(s) for seed {seed} written to {folder}:\n{report}");
        return created;
    }

    /// <summary>
    /// Removes the whole random paths folder. Deleting through the AssetDatabase rather than the file
    /// system is what keeps the .meta files from being orphaned.
    /// </summary>
    public static bool DeleteAll()
    {
        var folder = BenchmarkStarterAssets.RandomPathsFolder;
        return AssetDatabase.IsValidFolder(folder) && AssetDatabase.DeleteAsset(folder);
    }
}
}
