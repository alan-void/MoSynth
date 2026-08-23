using UnityEditor;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using AnimationTools;
using System.Collections.Generic;
using UnityEditor.AnimatedValues;
using UnityEditor.SceneManagement;

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
        // Off-screen scene used to draw the preview: its own camera, lights and render texture.
        private PreviewRenderUtility _previewRenderUtility;
        private SkeletonAnimation _skeletonAnimation;
        private bool _isPlaying = false;
        private float _currentTime = 0f;
        private float _previousTime = 0f;

        private Vector2 _drag;
        private float _distance = 5f;
        private Vector3 _targetPos = Vector3.zero;

        private Material _lineMaterial;
        private Mesh _skeletonMesh;
        private List<Vector3> _lineVertices = new List<Vector3>();
        private List<int> _lineIndices = new List<int>();

        private Material _gridMaterial;
        private Mesh _gridMesh;

        private Skeleton _skeleton;
        private int _skeletonContentHash;
        private SkeletonData _skeletonData;
        private NativeArray<float3> _fkPositions;
        private NativeArray<quaternion> _fkRotations;

        protected virtual void OnEnable()
        {
            _skeletonAnimation = (SkeletonAnimation)target;
            EditorApplication.update += UpdateSimulation;
            _previousTime = (float)EditorApplication.timeSinceStartup;

            RefreshSkeletonCache();

            Shader shader = Shader.Find("Hidden/Internal-Colored");
            if (shader != null)
            {
                _lineMaterial = new Material(shader);
                _lineMaterial.hideFlags = HideFlags.HideAndDontSave;
                _lineMaterial.SetInt("_ZWrite", 1);
                _lineMaterial.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.LessEqual);
                _lineMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            }

            _skeletonMesh = new Mesh();
            _skeletonMesh.hideFlags = HideFlags.HideAndDontSave;
            _skeletonMesh.MarkDynamic();

            if (shader != null)
            {
                _gridMaterial = new Material(shader);
                _gridMaterial.hideFlags = HideFlags.HideAndDontSave;
                _gridMaterial.SetInt("_ZWrite", 1);
                _gridMaterial.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.LessEqual);
                _gridMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            }
            CreateGridMesh();
        }

        /// <summary>
        /// Reallocates the FK scratch buffers when the asset's resolved <see cref="Skeleton"/>
        /// changes (including to/from null, e.g. an unconfigured rig/clip or one edited in the
        /// Inspector). Cheap no-op otherwise.
        /// </summary>
        /// <remarks>
        /// Keyed on the bone tree's content as well as the root's identity. Root identity alone
        /// misses a rig whose bones changed underneath it, which used to leave this cache and the
        /// asset's own baked <c>PoseSequence</c> describing two different skeletons — and the FK
        /// pass indexing one by the other's bone count.
        /// </remarks>
        private void RefreshSkeletonCache()
        {
            var skeleton = _skeletonAnimation != null ? _skeletonAnimation.Skeleton : null;
            var contentHash = skeleton?.ContentHash ?? 0;
            if (skeleton?.Root == _skeleton?.Root && contentHash == _skeletonContentHash) return;

            if (_fkPositions.IsCreated) _fkPositions.Dispose();
            if (_fkRotations.IsCreated) _fkRotations.Dispose();

            _skeleton = skeleton;
            _skeletonContentHash = contentHash;
            _skeletonData = default;

            if (_skeleton == null) return;

            _skeletonData = _skeleton.GetSkeletonData();
            _fkPositions = new NativeArray<float3>(_skeleton.BoneCount, Allocator.Persistent);
            _fkRotations = new NativeArray<quaternion>(_skeleton.BoneCount, Allocator.Persistent);
        }

        private void CreateGridMesh()
        {
            _gridMesh = new Mesh();
            _gridMesh.hideFlags = HideFlags.HideAndDontSave;

            List<Vector3> verts = new List<Vector3>();
            List<int> indices = new List<int>();
            List<Color> colors = new List<Color>();

            int gridSize = 10;
            float step = 1f;
            Color darkGray = new Color(0.3f, 0.3f, 0.3f, 1f);
            Color lightGray = new Color(0.5f, 0.5f, 0.5f, 1f);

            for (int i = -gridSize; i <= gridSize; i++)
            {
                verts.Add(new Vector3(i * step, 0, -gridSize * step));
                verts.Add(new Vector3(i * step, 0, gridSize * step));

                verts.Add(new Vector3(-gridSize * step, 0, i * step));
                verts.Add(new Vector3(gridSize * step, 0, i * step));

                Color c = (i % 5 == 0) ? lightGray : darkGray;
                colors.Add(c); colors.Add(c);
                colors.Add(c); colors.Add(c);

                int count = indices.Count;
                indices.Add(count); indices.Add(count + 1);
                indices.Add(count + 2); indices.Add(count + 3);
            }

            _gridMesh.SetVertices(verts);
            _gridMesh.SetColors(colors);
            _gridMesh.SetIndices(indices, MeshTopology.Lines, 0);
        }

        private void OnDisable()
        {
            EditorApplication.update -= UpdateSimulation;

            if (_previewRenderUtility != null)
            {
                _previewRenderUtility.Cleanup();
                _previewRenderUtility = null;
            }

            if (_lineMaterial != null)
            {
                DestroyImmediate(_lineMaterial);
            }
            if (_skeletonMesh != null)
            {
                DestroyImmediate(_skeletonMesh);
            }
            if (_gridMaterial != null)
            {
                DestroyImmediate(_gridMaterial);
            }
            if (_gridMesh != null)
            {
                DestroyImmediate(_gridMesh);
            }

            if (_fkPositions.IsCreated)
            {
                _fkPositions.Dispose();
            }
            if (_fkRotations.IsCreated)
            {
                _fkRotations.Dispose();
            }
        }

        private bool CanPreview =>
            _skeletonAnimation != null && _skeletonAnimation.TryValidate(out _) &&
            _skeletonAnimation.PoseSequence != null && _skeletonAnimation.FrameCount > 0;

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
            float time = (float)EditorApplication.timeSinceStartup;
            float deltaTime = time - _previousTime;
            _previousTime = time;

            if (_isPlaying && CanPreview)
            {
                _currentTime += deltaTime;
                float duration = _skeletonAnimation.FrameCount * _skeletonAnimation.FrameTime;
                if (_currentTime >= duration)
                {
                    _currentTime = _currentTime % duration;
                }
                Repaint();
            }
        }

        public override bool HasPreviewGUI()
        {
            return CanPreview;
        }

        public override GUIContent GetPreviewTitle()
        {
            return new GUIContent("Skeleton Animation Preview");
        }

        public override void OnPreviewSettings()
        {
            GUIStyle buttonStyle = new GUIStyle(EditorStyles.toolbarButton);

            if (GUILayout.Button(_isPlaying ? "Pause" : "Play", buttonStyle))
            {
                _isPlaying = !_isPlaying;
                if (_isPlaying)
                {
                    _previousTime = (float)EditorApplication.timeSinceStartup;
                }
            }

            if (CanPreview)
            {
                float duration = _skeletonAnimation.FrameCount * _skeletonAnimation.FrameTime;
                EditorGUI.BeginChangeCheck();
                _currentTime = GUILayout.HorizontalSlider(_currentTime, 0f, duration, GUILayout.Width(150));
                if (EditorGUI.EndChangeCheck())
                {
                    // Update preview if scrubbed manually
                    Repaint();
                }
            }
        }

        public override void OnInteractivePreviewGUI(Rect r, GUIStyle background)
        {
            if (_previewRenderUtility == null)
            {
                _previewRenderUtility = new PreviewRenderUtility();
                _previewRenderUtility.camera.fieldOfView = 30f;
                _previewRenderUtility.camera.nearClipPlane = 0.01f;
                _previewRenderUtility.camera.farClipPlane = 1000f;
            }

            HandleCameraControls(r);

            _previewRenderUtility.BeginPreview(r, background);

            Texture resultRender;
            // Anything thrown between Begin and End leaves the preview unbalanced, and every later
            // repaint then reports "Previous BeginPreview() was not closed" instead of the actual
            // problem — one real error turning into an unreadable stream of two.
            try
            {
                if (_gridMaterial != null && _gridMesh != null)
                {
                    _previewRenderUtility.DrawMesh(_gridMesh, Matrix4x4.identity, _gridMaterial, 0);
                }

                DrawSkeleton();

                _previewRenderUtility.camera.Render();
            }
            finally
            {
                resultRender = _previewRenderUtility.EndPreview();
            }

            GUI.DrawTexture(r, resultRender, ScaleMode.StretchToFill, false);
        }

        private void HandleCameraControls(Rect r)
        {
            Event e = Event.current;

            if (r.Contains(e.mousePosition))
            {
                if (e.type == EventType.MouseDrag && e.button == 0)
                {
                    _drag.x += e.delta.x * 0.5f;
                    _drag.y += e.delta.y * 0.5f;
                    e.Use();
                }
                else if (e.type == EventType.MouseDrag && e.button == 2)
                {
                    // Pan
                    Vector3 right = _previewRenderUtility.camera.transform.right;
                    Vector3 up = _previewRenderUtility.camera.transform.up;
                    _targetPos -= (right * e.delta.x - up * e.delta.y) * (0.01f * _distance);
                    e.Use();
                }
                else if (e.type == EventType.ScrollWheel)
                {
                    _distance += e.delta.y * 0.1f;
                    _distance = Mathf.Max(0.1f, _distance);
                    e.Use();
                }
            }
        }

        private void DrawSkeleton()
        {
            // The rig/clip fields can be edited live in the Inspector, so re-check the cache
            // every draw instead of only on OnEnable.
            RefreshSkeletonCache();

            if (!CanPreview) return;
            if (!_fkPositions.IsCreated) return;

            int frameIndex = Mathf.FloorToInt(_currentTime / _skeletonAnimation.FrameTime);
            frameIndex = Mathf.Clamp(frameIndex, 0, _skeletonAnimation.FrameCount - 1);

            var boneCount = _skeletonData.BoneCount;

            // First preview access triggers the one-time clip bake.
            var pose = _skeletonAnimation.GetFrame(frameIndex);

            // The bake and this cache are invalidated by different things, so draw nothing rather
            // than let PoseFK throw on a repaint if they ever disagree again.
            if (pose.Layout.RotationCount != boneCount) return;

            PoseFK.LocalToCharacter(pose, _skeletonData, _fkPositions, _fkRotations);

            _lineVertices.Clear();
            _lineIndices.Clear();

            // Root to children
            for (var i = 1; i < boneCount; i++)
            {
                var parentIndex = _skeletonData.ParentIndices[i];

                _lineVertices.Add(_fkPositions[parentIndex]);
                _lineVertices.Add(_fkPositions[i]);

                var indexCount = _lineIndices.Count;
                _lineIndices.Add(indexCount);
                _lineIndices.Add(indexCount + 1);
            }

            _skeletonMesh.Clear();
            _skeletonMesh.SetVertices(_lineVertices);
            _skeletonMesh.SetIndices(_lineIndices, MeshTopology.Lines, 0);

            // Setup Camera
            Vector3 rootPos = _fkPositions[0];

            // Adjust target if needed, maybe slowly lerp?
            // For now, center around rootPos + _targetPos (pan offset)
            Vector3 camTarget = rootPos + _targetPos;

            Quaternion camRotation = Quaternion.Euler(_drag.y, _drag.x, 0);
            _previewRenderUtility.camera.transform.position = camTarget - camRotation * Vector3.forward * _distance;
            _previewRenderUtility.camera.transform.rotation = camRotation;

            Color[] colors = new Color[_lineVertices.Count];
            for(int i = 0; i < colors.Length; i++) colors[i] = Color.green;
            _skeletonMesh.SetColors(colors);

            Matrix4x4 matrix = Matrix4x4.identity;

            if (_lineMaterial != null)
            {
                _previewRenderUtility.DrawMesh(_skeletonMesh, matrix, _lineMaterial, 0);
            }
        }
    }
}
