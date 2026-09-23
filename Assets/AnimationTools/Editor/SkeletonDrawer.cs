using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Draws a <see cref="Skeleton"/> as a Transform field over its root bone, plus a dropdown of every
/// Transform in the rig that bone belongs to. Both show the same value; they differ only in what
/// they can reach.
/// </summary>
/// <remarks>
/// An imported model exposes only its top-level children as sub-assets, so a deeper bone is
/// reachable only through the dropdown. The field is how a skeleton gets cleared, so the dropdown
/// offers no "(none)".
/// <para/>
/// Scene objects are refused: a skeleton reads its rest pose live off these Transforms. See
/// openwiki/animation-tools/skeletons-and-rig-binding.md.
/// </remarks>
[CustomPropertyDrawer(typeof(Skeleton))]
public class SkeletonDrawer : PropertyDrawer
{
    private static readonly GUIContent PickLabel = new("Pick from Rig",
        "An imported model exposes only its topmost bone to the Project window, so a bone any " +
        "deeper has to be chosen from this list rather than dragged.");

    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        var line = EditorGUIUtility.singleLineHeight;
        return ResolveRig(property.FindPropertyRelative("root")) == null
            ? line
            : line * 2 + EditorGUIUtility.standardVerticalSpacing;
    }

    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        var rootProp = property.FindPropertyRelative("root");

        EditorGUI.BeginProperty(position, label, property);

        var line = new Rect(position.x, position.y, position.width, EditorGUIUtility.singleLineHeight);

        // Before the object field, which would otherwise swallow the drag and normalise it.
        HandleDrop(line, rootProp);

        EditorGUI.BeginChangeCheck();
        var assigned = EditorGUI.ObjectField(
            line, label, rootProp.objectReferenceValue as Transform, typeof(Transform), false) as Transform;
        if (EditorGUI.EndChangeCheck()) rootProp.objectReferenceValue = assigned;

        var rig = ResolveRig(rootProp);
        if (rig != null)
        {
            line.y += EditorGUIUtility.singleLineHeight + EditorGUIUtility.standardVerticalSpacing;
            using (new EditorGUI.IndentLevelScope())
            {
                DrawBonePopup(line, PickLabel, rootProp, rig);
            }
        }

        EditorGUI.EndProperty();
    }

    /// <summary>
    /// Assigns whatever is dropped on the control, verbatim.
    /// </summary>
    /// <remarks>
    /// Handled by hand because <see cref="EditorGUI.ObjectField(Rect,GUIContent,Object,System.Type,bool)"/>
    /// normalises a dragged sub-asset up to its model's main asset.
    /// </remarks>
    private static void HandleDrop(Rect position, SerializedProperty rootProp)
    {
        var e = Event.current;
        if (e.type != EventType.DragUpdated && e.type != EventType.DragPerform) return;
        if (!position.Contains(e.mousePosition)) return;

        var dropped = FirstAssetTransform(DragAndDrop.objectReferences);
        if (dropped == null) return;

        DragAndDrop.visualMode = DragAndDropVisualMode.Link;

        if (e.type == EventType.DragPerform)
        {
            DragAndDrop.AcceptDrag();
            rootProp.objectReferenceValue = dropped;
            rootProp.serializedObject.ApplyModifiedProperties();
        }

        e.Use();
    }

    /// <summary>
    /// The first dragged object that is a Transform belonging to an imported asset. A GameObject
    /// stands in for its Transform, which is what a drag from the Project window carries.
    /// </summary>
    private static Transform FirstAssetTransform(Object[] dragged)
    {
        foreach (var candidate in dragged)
        {
            var transform = candidate as Transform;
            if (transform == null && candidate is GameObject gameObject) transform = gameObject.transform;

            if (transform != null && EditorUtility.IsPersistent(transform)) return transform;
        }

        return null;
    }

    /// <summary>
    /// The root bone's topmost ancestor, as <see cref="AnimationClipBaker"/> derives it. Null when
    /// nothing is assigned or the reference is missing (which Unity reports as non-null).
    /// </summary>
    private static Transform ResolveRig(SerializedProperty rootProp)
    {
        if (rootProp.objectReferenceValue is not Transform root || root == null) return null;

        try
        {
            return root.root;
        }
        catch (MissingReferenceException)
        {
            return null;
        }
    }

    private static void DrawBonePopup(Rect position, GUIContent label, SerializedProperty rootProp, Transform rig)
    {
        var transforms = new List<Transform>();
        var depths = new List<int>();
        BonePopup.Collect(rig, transforms, depths);

        // No "(none)": a skeleton without a root is unusable, so the list maps 1:1 onto transforms.
        var options = BonePopup.BuildOptions(transforms, depths, includeNone: false);
        var current = Mathf.Max(0, transforms.IndexOf(rootProp.objectReferenceValue as Transform));

        EditorGUI.BeginChangeCheck();
        var selected = EditorGUI.Popup(position, label.text, current, options);
        if (!EditorGUI.EndChangeCheck()) return;

        rootProp.objectReferenceValue = transforms[selected];
    }
}
}
