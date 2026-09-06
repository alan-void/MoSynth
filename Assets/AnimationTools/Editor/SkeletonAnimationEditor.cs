using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Inspector for a <see cref="SkeletonAnimation"/>, with a scrubbable preview that draws the
    /// skeleton as lines in an off-screen render, so a baked animation can be checked without
    /// instantiating a rig in the scene.
    /// </summary>
    [CustomEditor(typeof(SkeletonAnimation))]
    public class SkeletonAnimationEditor : UnityEditor.Editor
    {
        private SkeletonAnimation _skeletonAnimation;

        // The rendering lives in SkeletonPreview; this class only decides which frame it shows.
        private SkeletonPreview _preview;

        private bool _isPlaying;
        private float _currentTime;
        private float _previousTime;

        protected virtual void OnEnable()
        {
            _skeletonAnimation = (SkeletonAnimation)target;

            _preview = new SkeletonPreview { Source = _skeletonAnimation };

            EditorApplication.update += UpdateSimulation;
            _previousTime = (float)EditorApplication.timeSinceStartup;
        }

        /// <remarks>
        /// Virtual because a subclass that declares its own <c>OnDisable</c> would otherwise hide
        /// this one rather than extend it, and Unity dispatches only to the most-derived version -
        /// leaving the preview and its native buffers to leak for every inspector opened.
        /// </remarks>
        protected virtual void OnDisable()
        {
            EditorApplication.update -= UpdateSimulation;

            _preview?.Dispose();
            _preview = null;
        }

        private bool CanPreview => _preview != null && _preview.CanRender;

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            // Error, not Warning: a TryValidate failure means the asset cannot be baked at all.
            if (_skeletonAnimation != null && !_skeletonAnimation.TryValidate(out var error))
            {
                EditorGUILayout.HelpBox(error, MessageType.Error);
            }
        }

        private void UpdateSimulation()
        {
            var time = (float)EditorApplication.timeSinceStartup;
            var deltaTime = time - _previousTime;
            _previousTime = time;

            if (!_isPlaying || !CanPreview) return;

            _currentTime += deltaTime;
            var duration = _skeletonAnimation.FrameCount * _skeletonAnimation.FrameTime;
            if (_currentTime >= duration) _currentTime %= duration;

            Repaint();
        }

        public override bool HasPreviewGUI() => CanPreview;

        public override GUIContent GetPreviewTitle() => new("Skeleton Animation Preview");

        /// <summary>
        /// Moves the preview playhead onto a clip frame and pauses, so a subclass can drive the
        /// pose from its own UI.
        /// </summary>
        /// <param name="frameIndex">
        /// A frame of the <em>whole</em> clip. The preview does not apply an
        /// <see cref="AnnotatedAnimationClip"/>'s start/end slice, so a caller working in sliced
        /// frames must add the start frame itself.
        /// </param>
        protected void SeekToFrame(int frameIndex)
        {
            if (!CanPreview) return;

            _isPlaying = false;
            _currentTime = Mathf.Clamp(frameIndex, 0, _skeletonAnimation.FrameCount - 1) *
                           _skeletonAnimation.FrameTime;
            Repaint();
        }

        public override void OnPreviewSettings()
        {
            if (GUILayout.Button(_isPlaying ? "Pause" : "Play", EditorStyles.toolbarButton))
            {
                _isPlaying = !_isPlaying;
                if (_isPlaying) _previousTime = (float)EditorApplication.timeSinceStartup;
            }

            if (!CanPreview) return;

            var duration = _skeletonAnimation.FrameCount * _skeletonAnimation.FrameTime;
            EditorGUI.BeginChangeCheck();
            _currentTime = GUILayout.HorizontalSlider(_currentTime, 0f, duration, GUILayout.Width(150));
            if (EditorGUI.EndChangeCheck()) Repaint();
        }

        public override void OnInteractivePreviewGUI(Rect r, GUIStyle background)
        {
            if (!CanPreview) return;

            _preview.Source = _skeletonAnimation;
            _preview.Frame = Mathf.FloorToInt(_currentTime / _skeletonAnimation.FrameTime);
            _preview.Draw(r, background);
        }
    }
}
