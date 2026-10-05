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
    /// Draws a HelpBox for the first problem <see cref="PoseSetImporter.TryValidateSettings"/>
    /// finds. Returns true when one was drawn, which is the caller's cue to disable whatever
    /// generates the database.
    /// </summary>
    /// <remarks>
    /// The clips themselves are checked only when the database is generated: checking them here
    /// would load every clip, and the animation each one samples, for as long as the asset is
    /// inspected.
    /// </remarks>
    public static bool DrawSkeletonValidation(IPoseSetSource source)
    {
        if (PoseSetImporter.TryValidateSettings(source, out var error)) return false;

        EditorGUILayout.HelpBox(error, MessageType.Error);
        return true;
    }

    /// <summary>
    /// Draws a config's ordered contact-bone list with a bone picker per slot, plus Add and
    /// "Find Toes". Edits go through <paramref name="contactBones"/>, so the caller's
    /// <c>ApplyModifiedProperties</c> commits them.
    /// </summary>
    /// <remarks>
    /// "Find Toes" is the only place the contact-bone name heuristic runs: the list it writes is
    /// authored data from then on, and nothing downstream guesses.
    /// </remarks>
    public static void DrawContactBones(SerializedProperty contactBones, Skeleton skeleton)
    {
        var rigRoot = GetRigRoot(skeleton);

        EditorGUILayout.LabelField(new GUIContent("Contact Bones",
            "Bones whose contact the database flags, one channel each, in this order. Every database " +
            "on one character must list the same bones. Empty means no contact channels."),
            EditorStyles.boldLabel);

        // Deleting mid-loop would change the control count between the layout and repaint passes.
        var removeIndex = -1;
        for (var i = 0; i < contactBones.arraySize; i++)
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                SkeletonBoneDrawer.DrawLayout(new GUIContent($"Slot {i}"),
                    contactBones.GetArrayElementAtIndex(i), rigRoot);
                if (GUILayout.Button("x", GUILayout.Width(20))) removeIndex = i;
            }
        }

        if (removeIndex >= 0) contactBones.DeleteArrayElementAtIndex(removeIndex);

        if (contactBones.arraySize == 0)
        {
            EditorGUILayout.LabelField(" ", "No contact channels.", EditorStyles.miniLabel);
        }

        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Add Contact Bone"))
            {
                contactBones.arraySize++;
                SetBone(contactBones.GetArrayElementAtIndex(contactBones.arraySize - 1), null, null);
            }

            using (new EditorGUI.DisabledScope(rigRoot == null))
            {
                if (GUILayout.Button(new GUIContent("Find Toes",
                        "Replace the list with the left and right toe, found by name.")))
                {
                    FindToes(contactBones, skeleton);
                }
            }
        }
    }

    private static void FindToes(SerializedProperty contactBones, Skeleton skeleton)
    {
        contactBones.arraySize = 0;
        foreach (var left in new[] { true, false })
        {
            if (!BoneNameConventions.TryFindContactBone(skeleton, left, out var index))
            {
                Debug.LogWarning($"Find Toes: no {(left ? "left" : "right")} toe or foot bone found by name " +
                                 $"in \"{skeleton.Name}\". Pick it by hand.");
                continue;
            }

            contactBones.arraySize++;
            SetBone(contactBones.GetArrayElementAtIndex(contactBones.arraySize - 1), skeleton.Root,
                skeleton.GetBone(index).Transform);
        }
    }

    /// <summary>Writes a <see cref="SkeletonBone"/> property: the skeleton's root and the bone.</summary>
    public static void SetBone(SerializedProperty skeletonBone, Transform skeletonRoot, Transform bone)
    {
        skeletonBone.FindPropertyRelative("skeleton.root").objectReferenceValue = skeletonRoot;
        skeletonBone.FindPropertyRelative("bone").objectReferenceValue = bone;
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
