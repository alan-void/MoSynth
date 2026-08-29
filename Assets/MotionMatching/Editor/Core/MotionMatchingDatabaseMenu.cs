using UnityEditor;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// Regenerates every <see cref="MotionMatchingData"/> asset in the project from a menu item.
/// </summary>
/// <remarks>
/// The generated databases are unversioned and are not detected as stale until something reads
/// them, so a change to an extraction format or a feature definition has to be answered by a
/// regeneration of everything. Doing that one inspector button at a time means finding every asset
/// by hand and missing one.
/// </remarks>
public static class MotionMatchingDatabaseMenu
{
    [MenuItem("MoSynth/Database/Regenerate Motion Matching Databases")]
    public static void RegenerateAll()
    {
        var guids = AssetDatabase.FindAssets($"t:{nameof(MotionMatchingData)}");
        if (guids.Length == 0)
        {
            Debug.LogWarning("[MotionMatching] No MotionMatchingData assets found.");
            return;
        }

        foreach (var guid in guids)
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            var mmData = AssetDatabase.LoadAssetAtPath<MotionMatchingData>(path);
            if (mmData == null) continue;

            if (!mmData.TryValidate(out var error))
            {
                Debug.LogError($"[MotionMatching] Skipping \"{mmData.name}\": {error}", mmData);
                continue;
            }

            Debug.Log($"[MotionMatching] Regenerating \"{mmData.name}\"...", mmData);
            MotionMatchingDataEditor.GenerateDatabases(mmData);
        }
    }
}
}
