using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Inspector for an <see cref="AnnotatedAnimationClip"/>: the asset's own fields, whether it
    /// validates, and a way into the clip editor.
    /// </summary>
    /// <remarks>
    /// The annotation itself is viewed and edited in <see cref="AnnotatedClipEditorWindow"/>, which
    /// has the room for a timeline and a preview side by side. This stays the plain data view for a
    /// clip picked in the Project window.
    /// </remarks>
    [CustomEditor(typeof(AnnotatedAnimationClip))]
    public class AnnotatedAnimationClipEditor : SkeletonAnimationEditor
    {
        private SerializedProperty _clipProp;
        private SerializedProperty _skeletonProp;
        private SerializedProperty _rootMotionBoneProp;
        private SerializedProperty _startFrameProp;
        private SerializedProperty _endFrameProp;
        private SerializedProperty _componentsProp;

        protected override void OnEnable()
        {
            base.OnEnable();

            _clipProp = serializedObject.FindProperty("clip");
            _skeletonProp = serializedObject.FindProperty("skeleton");
            _rootMotionBoneProp = serializedObject.FindProperty("rootMotionBone");
            _startFrameProp = serializedObject.FindProperty("startFrame");
            _endFrameProp = serializedObject.FindProperty("endFrame");
            _componentsProp = serializedObject.FindProperty("components");
        }

        public override void OnInspectorGUI()
        {
            var clip = (AnnotatedAnimationClip)target;

            serializedObject.Update();

            EditorGUILayout.PropertyField(_clipProp);
            EditorGUILayout.PropertyField(_skeletonProp);
            EditorGUILayout.PropertyField(_rootMotionBoneProp);
            EditorGUILayout.PropertyField(_startFrameProp);
            EditorGUILayout.PropertyField(_endFrameProp);

            EditorGUILayout.Space();

            // The list is [SerializeReference] [SubclassSelector], so the package drawer supplies
            // the type dropdown and the per-element foldout; nothing here has to build them.
            EditorGUILayout.PropertyField(_componentsProp, true);

            serializedObject.ApplyModifiedProperties();

            var raw = (SkeletonAnimation)target;

            // Error, not Warning: a TryValidate failure means the asset cannot be baked at all.
            if (!raw.TryValidate(out var error))
            {
                EditorGUILayout.HelpBox(error, MessageType.Error);
            }

            using (new EditorGUI.DisabledScope(true))
            {
                EditorGUILayout.LabelField($"Raw frames: {raw.FrameCount}   Frame time: {raw.FrameTime:F4}s");
                var skeleton = raw.Skeleton;
                var resolvedRootBone = raw.RootBone;
                EditorGUILayout.LabelField(
                    $"Root bone: {(resolvedRootBone != null ? resolvedRootBone.name : "<unresolved>")}" +
                    $"   Bones: {(skeleton != null ? skeleton.BoneCount.ToString() : "-")}");
            }

            EditorGUILayout.Space();

            if (GUILayout.Button("Open in Clip Editor", GUILayout.Height(24f)))
            {
                AnnotatedClipEditorWindow.Open(clip);
            }
        }
    }
}
