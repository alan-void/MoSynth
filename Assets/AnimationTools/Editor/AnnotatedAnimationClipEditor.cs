using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace AnimationTools.Editor
{
    [CustomEditor(typeof(AnnotatedAnimationClip))]
    public class AnimationDataEditor : SkeletonAnimationEditor
    {
        private bool _tagsFoldout;

        private SerializedProperty _clipProp;
        private SerializedProperty _skeletonProp;
        private SerializedProperty _rootMotionBoneProp;
        private SerializedProperty _startFrameProp;
        private SerializedProperty _endFrameProp;

        protected override void OnEnable()
        {
            base.OnEnable();

            _clipProp = serializedObject.FindProperty("clip");
            _skeletonProp = serializedObject.FindProperty("skeleton");
            _rootMotionBoneProp = serializedObject.FindProperty("rootMotionBone");
            _startFrameProp = serializedObject.FindProperty("startFrame");
            _endFrameProp = serializedObject.FindProperty("endFrame");
        }

        public override void OnInspectorGUI()
        {
            AnnotatedAnimationClip clip = (AnnotatedAnimationClip)target;

            serializedObject.Update();

            EditorGUILayout.PropertyField(_clipProp);
            EditorGUILayout.PropertyField(_skeletonProp);
            EditorGUILayout.PropertyField(_rootMotionBoneProp);
            EditorGUILayout.PropertyField(_startFrameProp);
            EditorGUILayout.PropertyField(_endFrameProp);

            serializedObject.ApplyModifiedProperties();

            var raw = (SkeletonAnimation)target;

            if (!raw.TryValidate(out var error))
            {
                EditorGUILayout.HelpBox(error, MessageType.Warning);
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

            // Save
            if (GUI.changed)
            {
                EditorUtility.SetDirty(target);
                EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
            }
        }
    }
}
