using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace AnimationTools.Editor
{
    [CustomEditor(typeof(AnnotatedAnimationClip))]
    public class AnimationDataEditor : SkeletonAnimationEditor
    {
        private SerializedProperty _clipProp;
        private SerializedProperty _skeletonProp;
        private SerializedProperty _rootMotionBoneProp;
        private SerializedProperty _startFrameProp;
        private SerializedProperty _endFrameProp;
        private SerializedProperty _componentsProp;

        private GaitPhaseStrip _phaseStrip;

        protected override void OnEnable()
        {
            base.OnEnable();

            _clipProp = serializedObject.FindProperty("clip");
            _skeletonProp = serializedObject.FindProperty("skeleton");
            _rootMotionBoneProp = serializedObject.FindProperty("rootMotionBone");
            _startFrameProp = serializedObject.FindProperty("startFrame");
            _endFrameProp = serializedObject.FindProperty("endFrame");
            _componentsProp = serializedObject.FindProperty("components");

            _phaseStrip = new GaitPhaseStrip();
        }

        private void OnDisable()
        {
            _phaseStrip?.Dispose();
            _phaseStrip = null;
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

            EditorGUILayout.Space();

            // The list is [SerializeReference] [SubclassSelector], so the package drawer supplies
            // the type dropdown and the per-element foldout; nothing here has to build them.
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(_componentsProp, true);
            if (EditorGUI.EndChangeCheck()) _phaseStrip.Invalidate();

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

            DrawGaitPhaseSection(clip);

            // Save
            if (GUI.changed)
            {
                EditorUtility.SetDirty(target);
                EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
            }
        }

        /// <summary>
        /// Detection, a summary, and the timeline — shown only when the clip actually carries a
        /// gait phase component, so a clip annotated for something else is not cluttered by it.
        /// </summary>
        private void DrawGaitPhaseSection(AnnotatedAnimationClip clip)
        {
            if (!clip.TryGetComponent<GaitPhaseComponent>(out var phase)) return;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Gait Phase", EditorStyles.boldLabel);

            if (GUILayout.Button("Detect Footfalls", GUILayout.Height(22)))
            {
                Undo.RecordObject(clip, "Detect footfalls");
                if (phase.TryDetect(clip, out var contacts, out var detectError))
                {
                    _phaseStrip.SetContacts(contacts);
                    EditorUtility.SetDirty(clip);
                    Debug.Log($"[GaitPhase] {clip.name}: {phase.Describe()}.", clip);
                }
                else
                {
                    Debug.LogError($"[GaitPhase] {clip.name}: {detectError}", clip);
                }

                _phaseStrip.Invalidate();
            }

            _phaseStrip.Draw(clip, phase, SeekToFrame);
        }
    }
}
