using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Draws a <see cref="Skeleton"/> as a rig object field plus a root-bone dropdown.
/// </summary>
/// <remarks>
/// A bone inside an imported rig is not reachable from the Project window or the object picker —
/// only the asset's main object is — so a bare Transform field cannot express "this skeleton starts
/// at Model:Hips". Dropping the rig in seeds the field, and the dropdown then reaches any Transform
/// beneath it. Which node is the right one differs by asset (a clip's skeleton starts at the root
/// bone, a config's at the armature node that doubles as the SimulationBone), so this deliberately
/// does not guess on assignment.
/// <para/>
/// Scene objects are refused: a skeleton reads its rest pose live off these Transforms, so a scene
/// rig would report whatever pose it is currently animated to. <see cref="SkeletonBoneOverrides"/>
/// is what binds a skeleton to a live rig.
/// </remarks>
[CustomPropertyDrawer(typeof(Skeleton))]
public class SkeletonDrawer : PropertyDrawer
{
    private static readonly GUIContent RootBoneLabel = new("Root Bone",
        "The skeleton is this Transform and every Transform beneath it, in depth-first order.");

    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        var line = EditorGUIUtility.singleLineHeight;
        return ResolveRig(property) == null ? line : line * 2 + EditorGUIUtility.standardVerticalSpacing;
    }

    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        var rootProp = property.FindPropertyRelative("root");
        var rig = ResolveRig(property);

        EditorGUI.BeginProperty(position, label, property);

        var line = new Rect(position.x, position.y, position.width, EditorGUIUtility.singleLineHeight);

        EditorGUI.BeginChangeCheck();
        var newRig = EditorGUI.ObjectField(line, label, rig, typeof(Transform), false) as Transform;
        if (EditorGUI.EndChangeCheck())
        {
            rootProp.objectReferenceValue = newRig;
            rig = newRig;
        }

        if (rig != null)
        {
            line.y += EditorGUIUtility.singleLineHeight + EditorGUIUtility.standardVerticalSpacing;
            using (new EditorGUI.IndentLevelScope())
            {
                DrawRootBonePopup(line, rootProp, rig);
            }
        }

        EditorGUI.EndProperty();
    }

    /// <summary>
    /// The asset the root bone belongs to, which is its topmost ancestor — the same derivation
    /// <see cref="AnimationClipBaker"/> uses to find the object a clip's curve paths are relative
    /// to. Null when nothing is assigned, or when the assignment is a missing reference: Unity
    /// reports one of those as non-null, so reaching through it would throw on every repaint.
    /// </summary>
    private static Transform ResolveRig(SerializedProperty property)
    {
        var root = property.FindPropertyRelative("root").objectReferenceValue as Transform;
        if (root == null) return null;

        try
        {
            return root.root;
        }
        catch (MissingReferenceException)
        {
            return null;
        }
    }

    private static void DrawRootBonePopup(Rect position, SerializedProperty rootProp, Transform rig)
    {
        var transforms = new List<Transform>();
        var depths = new List<int>();
        BonePopup.Collect(rig, transforms, depths);

        var options = BonePopup.BuildOptions(transforms, depths);
        var current = rootProp.objectReferenceValue as Transform;
        var resolvedIndex = current != null ? transforms.IndexOf(current) : -1;

        EditorGUI.BeginChangeCheck();
        var newIndex = EditorGUI.Popup(position, RootBoneLabel.text, resolvedIndex + 1, options);
        if (!EditorGUI.EndChangeCheck()) return;

        rootProp.objectReferenceValue = newIndex == 0 ? null : transforms[newIndex - 1];
    }
}
}
