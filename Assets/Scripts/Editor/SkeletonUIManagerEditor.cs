using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(SkeletonUIManager))]
public class SkeletonUIManagerEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var manager = (SkeletonUIManager)target;

        GUILayout.Space(15);
        if (GUILayout.Button("Open Bone Selector Window", GUILayout.Height(30)))
        {
            BoneSelectorWindow.ShowWindow(manager);
        }
    }
}