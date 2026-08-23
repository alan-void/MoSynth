using AnimationTools;
using UnityEditor;
using UnityEngine;
using SkeletonBone = AnimationTools.SkeletonBone;

namespace MotionMatching.Editor
{
/// <summary>
/// Migration tool: builds a scene rig matching a <see cref="MotionMatchingData"/>'s skeleton,
/// bone for bone. Use it to get a rig to assign to feature/contact <see cref="SkeletonBone"/>
/// fields when no imported rig asset already provides one.
/// </summary>
public static class RigFromMmDataMenu
{
    [MenuItem("MotionMatching/Create Rig From MotionMatchingData")]
    private static void CreateRigFromMotionMatchingData()
    {
        var data = (MotionMatchingData)Selection.activeObject;
        var skeleton = data.Skeleton;

        if (skeleton == null || !skeleton.IsSet)
        {
            Debug.LogError($"[RigFromMmDataMenu] \"{data.name}\" has no skeleton assigned.");
            return;
        }

        var rigRoot = SkeletonRigBuilder.CreateHierarchy(skeleton);
        Undo.RegisterCreatedObjectUndo(rigRoot.gameObject, "Create Rig From MotionMatchingData");
        Selection.activeGameObject = rigRoot.gameObject;
    }

    [MenuItem("MotionMatching/Create Rig From MotionMatchingData", true)]
    private static bool ValidateCreateRigFromMotionMatchingData()
    {
        return Selection.activeObject is MotionMatchingData;
    }
}
}
