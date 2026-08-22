using System.IO;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
public static class CreateAnnotatedClipMenu
{
    [MenuItem("Assets/Create/MotionMatching/Annotated Clip From Selection", true)]
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

    [MenuItem("Assets/Create/MotionMatching/Annotated Clip From Selection")]
    private static void CreateFromSelection()
    {
        if (!TryResolveClipAndRig(Selection.activeObject, out var clip, out var rig, out var assetPath))
            return;

        var modelImporter = AssetImporter.GetAtPath(assetPath) as ModelImporter;
        if (modelImporter != null && modelImporter.animationType == ModelImporterAnimationType.Human)
        {
            EditorUtility.DisplayDialog("Annotated Clip",
                "Humanoid rigs are not supported — muscle clips can't sample onto a plain Transform hierarchy. Set the rig to Generic.",
                "OK");
            return;
        }

        // The only place the root bone is guessed. The asset stores the result, so a rig this
        // heuristic cannot read has to be caught here rather than failing later with an empty
        // skeleton.
        var rootBone = GuessRootBone(rig.transform);
        if (rootBone == null)
        {
            EditorUtility.DisplayDialog("Annotated Clip",
                $"Could not find a root bone in \"{rig.name}\" — no descendant named '*Hips' and no " +
                "single-child chain to follow. Create the asset from a rig that has one, then assign " +
                "the skeleton root by hand.",
                "OK");
            return;
        }

        var asset = ScriptableObject.CreateInstance<AnnotatedAnimationClip>();
        asset.SetSource(clip, rootBone);
        asset.endFrame = ((SkeletonAnimation)asset).FrameCount;

        var dir = Path.GetDirectoryName(assetPath);
        var newAssetPath = AssetDatabase.GenerateUniqueAssetPath(Path.Combine(dir, clip.name + "_annotated.asset"));
        AssetDatabase.CreateAsset(asset, newAssetPath);

        Selection.activeObject = asset;
        EditorGUIUtility.PingObject(asset);

        if (modelImporter != null && modelImporter.animationCompression != ModelImporterAnimationCompression.Off)
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

            foreach (var representation in AssetDatabase.LoadAllAssetRepresentationsAtPath(assetPath))
            {
                if (representation is AnimationClip representationClip)
                {
                    clip = representationClip;
                    break;
                }
            }

            if (clip == null)
            {
                EditorUtility.DisplayDialog("Annotated Clip", "No AnimationClip found in this asset.", "OK");
                return false;
            }

            return true;
        }

        return false;
    }

    private static Transform GuessRootBone(Transform rig)
    {
        var hips = FindHipsRecursive(rig);
        if (hips != null) return hips;

        if (rig.childCount != 1) return null;

        var current = rig.GetChild(0);
        while (current.childCount == 1)
        {
            current = current.GetChild(0);
        }

        return current;
    }

    private static Transform FindHipsRecursive(Transform transform)
    {
        for (var i = 0; i < transform.childCount; i++)
        {
            var child = transform.GetChild(i);
            if (child.name.EndsWith("Hips", System.StringComparison.OrdinalIgnoreCase)) return child;

            var found = FindHipsRecursive(child);
            if (found != null) return found;
        }

        return null;
    }
}
}
