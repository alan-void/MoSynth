using System;
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
/// Hand-drawn splines are fine for looking at a character, but they make a poor benchmark: nobody
/// can say afterwards what a method was actually being asked to do. These are parametric and
/// regenerable, and each one isolates a different demand — a constant-curvature circle, straights
/// joined by tight ends, a path that reverses its turn direction, and corners sharp enough that no
/// locomotion clip can follow them without overshooting.
/// <para>
/// All four are closed, because a lap is the run length and an open path can only be traversed once.
/// All lie in the XZ plane with an unscaled container, which is what the metrics and
/// MotionFieldSplineControlInput both assume.
/// </para>
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
        // The generated objects have to live in some scene to be saved as prefabs. An empty
        // untitled one keeps that side effect off whatever the user was working in.
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        var report = new StringBuilder();
        var created = new List<GameObject>
        {
            Save("Circle", Circle(4f), report),
            Save("Oval", Oval(7f, 3f), report),
            Save("FigureEight", FigureEight(6f), report),
            Save("SharpCorners", SharpCorners(6f, 4f), report)
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
    /// Linear tangents, so the corners are genuinely discontinuous in heading. No locomotion clip can
    /// turn that fast, so this is the path where corner overshoot dominates the trajectory error —
    /// which is the point of including it.
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

    private static GameObject Save(string name, Spline spline, StringBuilder report)
    {
        var go = new GameObject(name);
        var container = go.AddComponent<SplineContainer>();
        container.Spline = spline;

        var path = $"{BenchmarkStarterAssets.PathsFolder}/{name}.prefab";
        var prefab = PrefabUtility.SaveAsPrefabAsset(go, path);
        UnityEngine.Object.DestroyImmediate(go);

        report.AppendLine($"  {name,-14} knots {spline.Count,2}, closed {spline.Closed}, " +
                          $"length {spline.GetLength():0.00} m -> {path}");
        return prefab;
    }
}
}
