using System.IO;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
public static class CreateAnnotatedClipMenu
{
    [MenuItem("Assets/Create/MoSynth/Annotated Clip From Selection", true)]
    private static bool ValidateCreateFromSelection()
    {
        var selected = Selection.activeObject;
        if (selected is AnimationClip) return true;

        if (selected is GameObject && AssetDatabase.Contains(selected))
        {
            var path = AssetDatabase.GetAssetPath(selected);
            return AssetDatabase.LoadMainAssetAtPath(path) == selected;
        }

        return false;
    }

    [MenuItem("Assets/Create/MoSynth/Annotated Clip From Selection")]
    private static void CreateFromSelection()
    {
        if (!TryResolveClipAndRig(Selection.activeObject, out var clip, out var rig, out var assetPath))
            return;

        if (AnnotatedClipFactory.IsHumanoid(assetPath))
        {
            EditorUtility.DisplayDialog("Annotated Clip",
                "Humanoid rigs are not supported — muscle clips can't sample onto a plain Transform hierarchy. Set the rig to Generic.",
                "OK");
            return;
        }

        // The asset stores the guessed root, so a rig the heuristic cannot read has to be caught
        // here rather than failing later with an empty skeleton.
        var rootBone = AnnotatedClipFactory.GuessRootBone(rig);
        if (rootBone == null)
        {
            EditorUtility.DisplayDialog("Annotated Clip",
                $"Could not find a root bone in \"{rig.name}\" — no descendant named '*Hips' and no " +
                "single-child chain to follow. Create the asset from a rig that has one, then assign " +
                "the skeleton root by hand.",
                "OK");
            return;
        }

        var dir = Path.GetDirectoryName(assetPath);
        var newAssetPath = AssetDatabase.GenerateUniqueAssetPath(Path.Combine(dir, clip.name + "_annotated.asset"));
        var asset = AnnotatedClipFactory.CreateOrUpdate(clip, rootBone, newAssetPath, out _);

        Selection.activeObject = asset;
        EditorGUIUtility.PingObject(asset);

        if (AnnotatedClipFactory.UsesKeyframeReduction(assetPath))
        {
            Debug.Log($"'{assetPath}' uses keyframe reduction (Anim. Compression is not Off); setting Anim. Compression " +
                      "to Off on the model importer improves bake fidelity.");
        }
    }

    private static bool TryResolveClipAndRig(Object selected, out AnimationClip clip, out Transform rig, out string assetPath)
    {
        clip = null;
        rig = null;
        assetPath = null;

        if (selected is AnimationClip selectedClip)
        {
            clip = selectedClip;
            assetPath = AssetDatabase.GetAssetPath(clip);
            var go = AssetDatabase.LoadMainAssetAtPath(assetPath) as GameObject;
            rig = go?.transform;
            if (rig == null)
            {
                EditorUtility.DisplayDialog("Annotated Clip", "This AnimationClip's asset has no GameObject as its main object.", "OK");
                return false;
            }

            return true;
        }

        if (selected is GameObject selectedRig)
        {
            rig = selectedRig.transform;
            assetPath = AssetDatabase.GetAssetPath(rig);

            var clips = AnnotatedClipFactory.LoadClips(assetPath);
            if (clips.Count == 0)
            {
                EditorUtility.DisplayDialog("Annotated Clip", "No AnimationClip found in this asset.", "OK");
                return false;
            }

            clip = clips[0];
            return true;
        }

        return false;
    }
}
}
