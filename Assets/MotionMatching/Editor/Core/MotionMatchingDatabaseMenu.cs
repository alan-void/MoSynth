using UnityEditor;
using UnityEngine;

namespace MotionMatching
{
/// <summary>
/// Regenerates every <see cref="MotionMatchingData"/> asset in the project from a menu item.
/// </summary>
/// <remarks>
/// The generated databases are unversioned, so any change to an extraction format or a feature
/// definition requires regenerating all of them.
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
