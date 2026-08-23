using System.IO;
using UnityEngine;

namespace AnimationTools
{
/// <summary>
/// Builds the pose database an <see cref="IPoseSetSource"/> describes: validating the source,
/// extracting poses from its clips, and reading or writing the serialized form on disk.
/// </summary>
public static class PoseSetImporter
{
    /// <summary>
    /// Checks that the source can produce a pose database: a skeleton, and at least one clip whose
    /// skeleton is structurally equal to it. Returns false with a message suitable for an Inspector
    /// HelpBox. Never logs — inspectors call it every repaint.
    /// </summary>
    public static bool TryValidate(IPoseSetSource source, out string error)
    {
        var skeleton = source.Skeleton;
        if (skeleton == null || !skeleton.IsSet)
        {
            error = "No skeleton assigned. Drop the rig's root bone here; the pose skeleton is " +
                    "identical to each clip's skeleton.";
            return false;
        }

        var clips = source.AnimationClips;
        if (clips == null || clips.Count == 0)
        {
            error = "Assign at least one animation clip.";
            return false;
        }

        for (var i = 0; i < clips.Count; i++)
        {
            var clip = clips[i];
            if (clip == null)
            {
                error = $"Clip {i} is empty.";
                return false;
            }

            if (!clip.TryValidate(out var clipError))
            {
                error = $"Clip \"{clip.name}\": {clipError}";
                return false;
            }

            if (!Skeleton.StructurallyEqual(skeleton, clip.Skeleton))
            {
                error = $"Clip \"{clip.name}\" ({clip.Skeleton.BoneCount} bones) is not structurally equal to " +
                        $"this asset's skeleton ({skeleton.BoneCount} bones).";
                return false;
            }
        }

        error = null;
        return true;
    }

    /// <summary>
    /// Extracts the pose database from the source's clips, in memory. Null when
    /// <see cref="TryValidate"/> fails; a single clip that does not line up is skipped rather than
    /// failing the whole extraction.
    /// </summary>
    public static PoseSet Import(IPoseSetSource source)
    {
        if (!TryValidate(source, out var error))
        {
            Debug.LogError($"[PoseSet] '{source.name}': {error}");
            return null;
        }

        var skeleton = source.Skeleton;
        var clips = source.AnimationClips;

        var poseSet = new PoseSet();
        poseSet.SetSkeleton(skeleton);

        for (var i = 0; i < clips.Count; i++)
        {
            var clip = clips[i];
            if (clip == null || clip.Skeleton == null || !Skeleton.StructurallyEqual(skeleton, clip.Skeleton))
            {
                Debug.LogError($"[PoseSet] '{source.name}': clip {i} (\"{(clip != null ? clip.name : "null")}\")'s " +
                               "skeleton is not structurally equal to this asset's skeleton; skipping.");
                continue;
            }

            if (!PoseExtractor.Extract(clip, poseSet, source))
            {
                Debug.LogWarning($"[PoseSet] '{source.name}': failed to extract poses from clip {i}.");
            }
        }

        poseSet.ConvertTagsToNativeArrays();
        Debug.Log($"[PoseSet] '{source.name}': {poseSet.NumberPoses} poses.");
        return poseSet;
    }

    /// <summary>
    /// The serialized pose database, extracting it from the clips when no file is on disk. Null
    /// when <see cref="TryValidate"/> fails, which is why it stays silent in that case: callers are
    /// reached from OnValidate every Inspector repaint, and report through the inspector instead.
    /// A set extracted at runtime is written back to disk in the editor, so the fallback pays for
    /// itself only once.
    /// </summary>
    public static PoseSet GetOrImport(IPoseSetSource source)
    {
        if (!TryValidate(source, out _)) return null;

        var serializer = new PoseSerializer();
        if (serializer.Deserialize(source.GetAssetPath(), source.name, source.Skeleton, out var poseSet))
        {
            return poseSet;
        }

        Debug.LogWarning($"[PoseSet] No serialized pose set for '{source.name}'. Extracting at runtime. " +
                         "Press Generate Pose Database on the asset to avoid this.");

        poseSet = Import(source);

#if UNITY_EDITOR
        if (poseSet != null)
        {
            serializer.Serialize(poseSet, source.GetAssetPath(), source.name);
        }
#endif

        return poseSet;
    }

    /// <summary>
    /// Directory a source's generated files live in: a folder per asset under
    /// <paramref name="rootFolder"/> in StreamingAssets. Created if missing (editor only).
    /// </summary>
    public static string GetDatabasePath(string rootFolder, string sourceName)
    {
        var path = Path.Combine(Application.streamingAssetsPath, rootFolder, sourceName);
#if UNITY_EDITOR
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
#endif
        return path;
    }
}
}
