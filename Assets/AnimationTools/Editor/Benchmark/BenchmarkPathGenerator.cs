using System.Collections.Generic;
using System.Text;
using Unity.Mathematics;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Splines;

namespace AnimationTools.Editor
{
/// <summary>
/// Generates the standard suite of benchmark paths as <see cref="SplineContainer"/> prefabs.
/// </summary>
/// <remarks>
/// Parametric and regenerable, each isolating one demand. All are closed and lie in the XZ plane
/// with an unscaled container, which the metrics and MotionFieldSplineControlInput both assume.
/// </remarks>
public static class BenchmarkPathGenerator
{
    [MenuItem("MoSynth/Benchmark/Create Standard Paths", priority = 201)]
    public static void CreateStandardPathsMenu()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

        var created = Create();
        Debug.Log($"[Benchmark] {created.Count} standard path prefab(s) written to {BenchmarkStarterAssets.PathsFolder}.");

        if (created.Count > 0) EditorGUIUtility.PingObject(created[0]);
    }

    /// <summary>
    /// Writes the suite, overwriting any prefab of the same name. No dialogs and no save prompt —
    /// the caller owns both. Works in a throwaway scene so nothing the user has open is dirtied.
    /// </summary>
    public static List<GameObject> Create()
    {
        // Prefabs are saved from scene objects; an empty scene keeps that off the user's own.
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        var folder = BenchmarkStarterAssets.PathsFolder;
        BenchmarkPathAssets.EnsureFolder(folder);

        var report = new StringBuilder();
        var created = new List<GameObject>
        {
            BenchmarkPathAssets.Save("Circle", Circle(4f), folder, report),
            BenchmarkPathAssets.Save("Oval", Oval(7f, 3f), folder, report),
            BenchmarkPathAssets.Save("FigureEight", FigureEight(6f), folder, report),
            BenchmarkPathAssets.Save("SharpCorners", SharpCorners(6f, 4f), folder, report)
        };

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        Debug.Log($"[Benchmark] Standard paths:\n{report}");
        return created;
    }

    /// <summary>Constant curvature: the easiest thing to follow, and the baseline everything else is worse than.</summary>
    private static Spline Circle(float radius)
    {
        var points = new float3[8];
        for (var i = 0; i < points.Length; i++)
        {
            var angle = 2f * math.PI * i / points.Length;
            points[i] = new float3(math.cos(angle) * radius, 0f, math.sin(angle) * radius);
        }

        return Smooth(points);
    }

    /// <summary>Straights joined by tight ends, so acceleration into and out of a turn is separable from the turn itself.</summary>
    private static Spline Oval(float halfLength, float halfWidth)
    {
        return Smooth(new[]
        {
            new float3(halfLength, 0f, 0f),
            new float3(halfLength * 0.4f, 0f, halfWidth),
            new float3(-halfLength * 0.4f, 0f, halfWidth),
            new float3(-halfLength, 0f, 0f),
            new float3(-halfLength * 0.4f, 0f, -halfWidth),
            new float3(halfLength * 0.4f, 0f, -halfWidth)
        });
    }

    /// <summary>
    /// A lemniscate: the only path here that reverses its turn direction, which is where a policy
    /// that has committed to a turn shows whether it can commit to the opposite one.
    /// </summary>
    private static Spline FigureEight(float scale)
    {
        var points = new float3[12];
        for (var i = 0; i < points.Length; i++)
        {
            var t = 2f * math.PI * i / points.Length;
            var denominator = 1f + math.sin(t) * math.sin(t);
            points[i] = new float3(
                scale * math.cos(t) / denominator,
                0f,
                scale * math.sin(t) * math.cos(t) / denominator);
        }

        return Smooth(points);
    }

    /// <summary>
    /// Linear tangents, so heading is discontinuous at the corners and overshoot dominates the
    /// trajectory error.
    /// </summary>
    private static Spline SharpCorners(float halfLength, float halfWidth)
    {
        var spline = new Spline { Closed = true };
        float3[] points =
        {
            new(halfLength, 0f, halfWidth),
            new(-halfLength, 0f, halfWidth),
            new(-halfLength, 0f, -halfWidth),
            new(halfLength, 0f, -halfWidth)
        };

        foreach (var point in points) spline.Add(new BezierKnot(point), TangentMode.Linear);
        return spline;
    }

    private static Spline Smooth(IReadOnlyList<float3> points)
    {
        var spline = new Spline { Closed = true };
        foreach (var point in points) spline.Add(new BezierKnot(point), TangentMode.AutoSmooth);
        return spline;
    }
}
}
