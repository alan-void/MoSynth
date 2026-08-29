using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;
using UnityEngine.Splines;

namespace AnimationTools.Editor
{
/// <summary>
/// Writing a spline out as a benchmark path prefab, shared by the generators that produce them.
/// </summary>
internal static class BenchmarkPathAssets
{
    /// <summary>Creates <paramref name="folder"/> and any missing parent, so a generator can write into a folder that has never existed.</summary>
    internal static void EnsureFolder(string folder)
    {
        if (AssetDatabase.IsValidFolder(folder)) return;

        var parent = Path.GetDirectoryName(folder)!.Replace('\\', '/');
        EnsureFolder(parent);
        AssetDatabase.CreateFolder(parent, Path.GetFileName(folder));
    }

    /// <summary>
    /// Saves a spline as a path prefab and appends one line describing it to <paramref name="report"/>.
    /// </summary>
    /// <remarks>
    /// Overwrites any prefab already at the name rather than making a numbered sibling: overwriting in
    /// place keeps the asset's GUID, so a config already pointing at it survives a regeneration.
    /// </remarks>
    internal static GameObject Save(string name, Spline spline, string folder, StringBuilder report)
    {
        var go = new GameObject(name);
        var container = go.AddComponent<SplineContainer>();
        container.Spline = spline;

        var path = $"{folder}/{name}.prefab";
        var prefab = PrefabUtility.SaveAsPrefabAsset(go, path);
        Object.DestroyImmediate(go);

        report.AppendLine($"  {name,-14} knots {spline.Count,2}, closed {spline.Closed}, " +
                          $"length {spline.GetLength():0.00} m -> {path}");
        return prefab;
    }
}
}
