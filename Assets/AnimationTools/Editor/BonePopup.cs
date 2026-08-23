using System.Collections.Generic;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Shared bits of the bone-picking dropdowns: an indented depth-first listing of a rig's
/// Transforms, with duplicate names disambiguated. Used by <see cref="SkeletonDrawer"/> to pick a
/// skeleton's root bone and by <see cref="SkeletonBoneDrawer"/> to pick a bone within one.
/// </summary>
internal static class BonePopup
{
    public const string NoneOption = "(none)";

    private const int IndentSpacesPerDepth = 4;

    public static void Collect(Transform root, List<Transform> transforms, List<int> depths)
    {
        CollectRecursive(root, 0, transforms, depths);
    }

    private static void CollectRecursive(Transform transform, int depth, List<Transform> transforms, List<int> depths)
    {
        transforms.Add(transform);
        depths.Add(depth);
        for (var i = 0; i < transform.childCount; i++)
        {
            CollectRecursive(transform.GetChild(i), depth + 1, transforms, depths);
        }
    }

    /// <summary>
    /// Options for a bone popup. With <paramref name="includeNone"/> (the default) index 0 is
    /// <see cref="NoneOption"/> and option <c>i + 1</c> corresponds to <c>transforms[i]</c>;
    /// without it the array maps one-to-one onto <paramref name="transforms"/>.
    /// </summary>
    /// <remarks>
    /// "(none)" is only worth offering where being unset means something. It does for a
    /// <see cref="SkeletonBone"/> — an unset root-motion bone falls back to the skeleton root, and
    /// unset contact bones fall back to <see cref="BoneNameConventions"/> — but a skeleton with no
    /// root is simply unusable, so <see cref="SkeletonDrawer"/> leaves the entry out.
    /// </remarks>
    public static string[] BuildOptions(List<Transform> transforms, List<int> depths, bool includeNone = true)
    {
        var nameCounts = new Dictionary<string, int>();
        foreach (var transform in transforms)
        {
            nameCounts.TryGetValue(transform.name, out var count);
            nameCounts[transform.name] = count + 1;
        }

        var offset = includeNone ? 1 : 0;
        var seenCounts = new Dictionary<string, int>();
        var options = new string[transforms.Count + offset];
        if (includeNone) options[0] = NoneOption;

        for (var i = 0; i < transforms.Count; i++)
        {
            var transform = transforms[i];
            var indent = new string(' ', depths[i] * IndentSpacesPerDepth);

            string displayName;
            if (nameCounts[transform.name] > 1)
            {
                seenCounts.TryGetValue(transform.name, out var seen);
                seen += 1;
                seenCounts[transform.name] = seen;
                displayName = $"{transform.name} ({seen})";
            }
            else
            {
                displayName = transform.name;
            }

            options[i + offset] = indent + displayName;
        }

        return options;
    }
}
}
