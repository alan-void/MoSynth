using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Inspector pieces every <see cref="IPoseSetSource"/> asset needs: reporting what stops it
/// producing a pose database, and the rig its bone pickers list.
/// </summary>
public static class PoseSetSourceGUI
{
    /// <summary>
    /// Draws one HelpBox per problem the skeleton or an animation clip has. Returns true when at
    /// least one was drawn, which is the caller's cue to disable whatever generates the database.
    /// </summary>
    public static bool DrawSkeletonValidation(IPoseSetSource source)
    {
        var skeleton = source.Skeleton;
        if (skeleton == null || !skeleton.IsSet)
        {
            EditorGUILayout.HelpBox(
                "No skeleton assigned. Drop the imported model on the Skeleton field, then pick " +
                "the rig's identity armature node from the dropdown.",
                MessageType.Error);
            return true;
        }

        var clips = source.AnimationClips;
        var anyError = false;
        for (var i = 0; i < clips.Count; i++)
        {
            var clip = clips[i];
            if (clip == null)
            {
                EditorGUILayout.HelpBox($"Animation clip {i} is not assigned.", MessageType.Error);
                anyError = true;
                continue;
            }

            if (clip.Skeleton == null)
            {
                EditorGUILayout.HelpBox($"Clip \"{clip.name}\" has no resolvable skeleton.", MessageType.Error);
                anyError = true;
                continue;
            }

            if (!clip.TryValidate(out var error))
            {
                EditorGUILayout.HelpBox($"Clip \"{clip.name}\": {error}", MessageType.Error);
                anyError = true;
                continue;
            }

            if (!skeleton.MatchesFrom(1, clip.Skeleton))
            {
                EditorGUILayout.HelpBox(
                    $"Clip \"{clip.name}\"'s skeleton does not match this asset's skeleton from bone 1 (Hips) onward.",
                    MessageType.Error);
                anyError = true;
            }
        }

        return anyError;
    }

    /// <summary>
    /// The rig a bone picker draws its dropdown from: the asset's skeleton root. Null when no
    /// skeleton is assigned.
    /// </summary>
    public static Transform GetRigRoot(Skeleton skeleton)
    {
        return skeleton != null && skeleton.IsSet ? skeleton.Root : null;
    }
}
}
