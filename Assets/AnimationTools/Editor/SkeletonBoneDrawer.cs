using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
/// <summary>
/// Draws a <see cref="SkeletonBone"/> as a bone-name dropdown over the transforms of a resolved
/// rig, instead of raw Transform/skeleton fields. Resolves the rig root to populate the dropdown
/// from, in order, a sibling field named by a <see cref="BoneFromAttribute"/> on the reference
/// field, then an <see cref="ISkeletonProvider"/> on the owning object.
/// </summary>
[CustomPropertyDrawer(typeof(SkeletonBone))]
public class SkeletonBoneDrawer : PropertyDrawer
{
    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        return EditorGUIUtility.singleLineHeight;
    }

    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        var rigRoot = ResolveRigRoot(property);

        EditorGUI.BeginProperty(position, label, property);
        DrawCore(position, label, property, rigRoot);
        EditorGUI.EndProperty();
    }

    /// <summary>
    /// Draws a <see cref="SkeletonBone"/> popup for a caller that already knows the rig root
    /// (e.g. a custom editor whose asset resolves the rig from its own clips), bypassing
    /// [BoneFrom]/<see cref="ISkeletonProvider"/> resolution. <paramref name="rigRoot"/> may be
    /// null, which draws the same disabled fallback as an unresolved rig in <see cref="OnGUI"/>.
    /// </summary>
    public static void DrawLayout(GUIContent label, SerializedProperty boneTransformProperty, Transform rigRoot)
    {
        var position = EditorGUILayout.GetControlRect(true, EditorGUIUtility.singleLineHeight);

        EditorGUI.BeginProperty(position, label, boneTransformProperty);
        DrawCore(position, label, boneTransformProperty, rigRoot);
        EditorGUI.EndProperty();
    }

    // --- Rig resolution ------------------------------------------------------------------------

    private Transform ResolveRigRoot(SerializedProperty property)
    {
        var boneFrom = fieldInfo?.GetCustomAttribute<BoneFromAttribute>();
        if (boneFrom != null)
        {
            var rigRoot = ResolveFromSibling(property, boneFrom.SkeletonMemberName);
            if (rigRoot != null) return rigRoot;
        }

        if (property.serializedObject.targetObject is ISkeletonProvider provider)
            return provider.CharacterRig?.Root;

        return null;
    }

    private static Transform ResolveFromSibling(SerializedProperty property, string siblingName)
    {
        var siblingPath = GetSiblingPath(property.propertyPath, siblingName);

        // Sibling is a Skeleton or a SkeletonBoneOverrides: both hold the rig Transform directly
        // in a field named "root".
        var rootProp = property.serializedObject.FindProperty(siblingPath + ".root");
        if (rootProp != null && rootProp.objectReferenceValue is Transform skeletonRootTransform)
            return skeletonRootTransform;

        // Sibling is an object reference to an AnnotatedAnimationClip or SkeletonAnimation asset.
        //
        // The type check is what keeps this from firing on the case above: a nested Skeleton is a
        // Generic property, and reading a PPtr off one does not throw -- Unity logs
        // "type is not a supported pptr value" from native code, once per GUI event, forever. An
        // empty ".root" there means the rig is unresolved, not that another strategy should be tried.
        var siblingProp = property.serializedObject.FindProperty(siblingPath);
        if (siblingProp == null || siblingProp.propertyType != SerializedPropertyType.ObjectReference)
            return null;

        return RigRootFromAsset(siblingProp.objectReferenceValue);
    }

    private static Transform RigRootFromAsset(UnityEngine.Object obj)
    {
        // The animation's skeleton root, not its rig: a rig's root also carries mesh and helper
        // nodes, which are not bones and must not appear in a bone dropdown.
        if (obj is SkeletonAnimation animation) return animation.RootBone;

        // The sibling may be the rig itself rather than an asset that references one.
        if (obj is GameObject rig) return rig.transform;

        return obj as Transform;
    }

    /// <summary>
    /// Maps a property path to a same-level sibling, stripping any trailing array index so a
    /// <c>[BoneFrom]</c> on a list element still finds a field on the element's own owner (e.g.
    /// <c>features.Array.data[0].bone</c> -&gt; <c>features.Array.data[0].animationClips</c>).
    /// </summary>
    private static string GetSiblingPath(string propertyPath, string siblingName)
    {
        var path = propertyPath;
        if (path.EndsWith("]"))
        {
            var arrayIdx = path.LastIndexOf(".Array.data[", StringComparison.Ordinal);
            if (arrayIdx >= 0) path = path.Substring(0, arrayIdx);
        }

        var lastDot = path.LastIndexOf('.');
        return lastDot < 0 ? siblingName : path.Substring(0, lastDot + 1) + siblingName;
    }

    // --- Drawing ---------------------------------------------------------------------------------

    private static void DrawCore(Rect position, GUIContent label, SerializedProperty property, Transform rigRoot)
    {
        var boneProp = property.FindPropertyRelative("bone");
        var skeletonRootProp = property.FindPropertyRelative("skeleton.root");

        if (rigRoot == null)
        {
            DrawUnresolved(position, label);
            return;
        }

        var transforms = new List<Transform>();
        var depths = new List<int>();
        BonePopup.Collect(rigRoot, transforms, depths);

        DrawPopup(position, label, transforms, depths, boneProp, skeletonRootProp, rigRoot);
    }

    private static void DrawUnresolved(Rect position, GUIContent label)
    {
        var content = new GUIContent(BonePopup.NoneOption,
            "No rig found — add [BoneFrom] or implement ISkeletonProvider.");

        using (new EditorGUI.DisabledScope(true))
        {
            EditorGUI.Popup(position, label, 0, new[] { content });
        }
    }

    private static void DrawPopup(
        Rect position, GUIContent label, List<Transform> transforms, List<int> depths,
        SerializedProperty boneProp, SerializedProperty skeletonRootProp, Transform rigRoot)
    {
        var options = BonePopup.BuildOptions(transforms, depths);

        var boneTransform = boneProp.objectReferenceValue as Transform;
        var resolvedIndex = boneTransform != null ? transforms.IndexOf(boneTransform) : -1;
        var currentIndex = resolvedIndex >= 0 ? resolvedIndex + 1 : 0;

        EditorGUI.BeginChangeCheck();
        var newIndex = EditorGUI.Popup(position, label.text, currentIndex, options);
        var changed = EditorGUI.EndChangeCheck();

        if (!changed) return;

        if (newIndex == 0)
        {
            boneProp.objectReferenceValue = null;
            if (skeletonRootProp != null) skeletonRootProp.objectReferenceValue = null;
        }
        else
        {
            var transform = transforms[newIndex - 1];
            boneProp.objectReferenceValue = transform;
            if (skeletonRootProp != null) skeletonRootProp.objectReferenceValue = rigRoot;
        }
    }

}
}
