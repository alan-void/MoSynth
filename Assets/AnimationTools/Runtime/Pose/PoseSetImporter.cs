using System.Collections.Generic;
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
    /// Checks what can be checked without loading a clip: the skeleton, its rest pose, a non-empty
    /// clip list, and the contact-bone list. Returns false with a message suitable for an Inspector HelpBox. Never
    /// logs, and cheap enough for every repaint.
    /// </summary>
    public static bool TryValidateSettings(IPoseSetSource source, out string error)
    {
        var skeleton = source.Skeleton;
        if (skeleton == null || !skeleton.IsSet)
        {
            error = "No skeleton assigned. Drop the imported model on the Skeleton field, then pick " +
                    "the rig's root bone from the dropdown.";
            return false;
        }

        if (!skeleton.TryValidateRestPose(out error)) return false;

        var clips = source.AnimationClips;
        if (clips == null || clips.Count == 0)
        {
            error = "Assign at least one animation clip.";
            return false;
        }

        return TryValidateContactBones(skeleton, source.ContactBoneNames, source.MirrorClips, out error);
    }

    /// <summary>
    /// Every contact bone is set, is in <paramref name="skeleton"/>, and is listed once. When clips
    /// are mirrored, each bone's left/right counterpart must be listed too, because a mirrored clip
    /// writes a slot from its counterpart's measurement.
    /// </summary>
    public static bool TryValidateContactBones(Skeleton skeleton, IReadOnlyList<string> contactBoneNames,
        bool mirrorClips, out string error)
    {
        error = null;
        if (contactBoneNames == null) return true;

        for (var slot = 0; slot < contactBoneNames.Count; slot++)
        {
            var boneName = contactBoneNames[slot];
            if (string.IsNullOrEmpty(boneName))
            {
                error = $"Contact bone {slot} is unset. Pick a bone or remove the entry.";
                return false;
            }

            if (skeleton.IndexOfName(boneName) < 0)
            {
                error = $"Contact bone \"{boneName}\" is not in skeleton \"{skeleton.Name}\".";
                return false;
            }

            for (var other = 0; other < slot; other++)
            {
                if (contactBoneNames[other] != boneName) continue;

                error = $"Contact bone \"{boneName}\" is listed twice.";
                return false;
            }

            if (!mirrorClips) continue;

            var counterpart = PoseMirror.CounterpartName(boneName);
            if (counterpart == boneName || skeleton.IndexOfName(counterpart) < 0) continue;

            var listed = false;
            for (var other = 0; other < contactBoneNames.Count; other++)
            {
                if (contactBoneNames[other] == counterpart) listed = true;
            }

            if (!listed)
            {
                error = $"Mirror Clips is on, so contact bone \"{boneName}\" needs its counterpart " +
                        $"\"{counterpart}\" in the contact list too.";
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// <see cref="TryValidateSettings"/>, plus every clip checked against the skeleton. Loads every
    /// clip and the animation it samples, so call it when about to extract, never per repaint.
    /// </summary>
    public static bool TryValidate(IPoseSetSource source, out string error)
    {
        if (!TryValidateSettings(source, out error)) return false;

        var skeleton = source.Skeleton;
        var clips = source.AnimationClips;
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

        PoseMirror mirror = null;
        if (source.MirrorClips && !PoseMirror.TryCreate(skeleton, out mirror, out var mirrorError))
        {
            Debug.LogError($"[PoseSet] '{source.name}': cannot mirror clips. {mirrorError}");
            return null;
        }

        var poseSet = new PoseSet();
        poseSet.SetSkeleton(skeleton, source.ContactBoneNames);

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
                continue;
            }

            if (mirror != null && !PoseExtractor.Extract(clip, poseSet, source, mirror))
            {
                Debug.LogWarning($"[PoseSet] '{source.name}': failed to extract the mirrored copy of clip {i}.");
            }
        }

        poseSet.ConvertTagsToNativeArrays();
        var mirrorNote = mirror != null ? ", every clip followed by its mirrored copy" : "";
        Debug.Log($"[PoseSet] '{source.name}': {poseSet.NumberPoses} poses{mirrorNote}.");
        return poseSet;
    }

    /// <summary>
    /// The serialized pose database, extracting it from the clips (and, in the editor, writing it
    /// back) when no file is on disk. Silently null when <see cref="TryValidateSettings"/> fails,
    /// since callers reach it from OnValidate every Inspector repaint.
    /// </summary>
    /// <remarks>
    /// Reading the file only needs the settings check, since the file carries its own skeleton to
    /// compare against; the clips are loaded only when there is no file and they must be extracted.
    /// </remarks>
    public static PoseSet GetOrImport(IPoseSetSource source)
    {
        if (!TryValidateSettings(source, out _)) return null;

        var serializer = new PoseSerializer();
        if (serializer.Deserialize(source.GetAssetPath(), source.name, source.Skeleton, source.ContactBoneNames,
                out var poseSet))
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
