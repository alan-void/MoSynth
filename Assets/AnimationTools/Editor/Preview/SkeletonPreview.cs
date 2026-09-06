using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// What an overlay subscriber is handed for one repaint: the drawing surface, and the pose
    /// already resolved into character space so an overlay never has to run FK itself.
    /// </summary>
    /// <remarks>
    /// <see cref="BonePositions"/> and <see cref="BoneRotations"/> are the preview's own buffers and
    /// are overwritten on the next repaint. Read them during the callback; never store them.
    /// </remarks>
    public readonly struct SkeletonPreviewFrame
    {
        public readonly ISkeletonPreviewDrawer Draw;
        public readonly Camera Camera;
        public readonly SkeletonData Skeleton;
        public readonly NativeArray<float3> BonePositions;
        public readonly NativeArray<quaternion> BoneRotations;

        /// <summary>The whole-clip frame currently posed.</summary>
        public readonly int Frame;

        public SkeletonPreviewFrame(ISkeletonPreviewDrawer draw, Camera camera, SkeletonData skeleton,
            NativeArray<float3> bonePositions, NativeArray<quaternion> boneRotations, int frame)
        {
            Draw = draw;
            Camera = camera;
            Skeleton = skeleton;
            BonePositions = bonePositions;
            BoneRotations = boneRotations;
            Frame = frame;
        }
    }

    /// <summary>
    /// An off-screen render of a <see cref="SkeletonAnimation"/> posed at one frame, drawn as bone
    /// lines over a ground grid, with an orbit camera and a hook for overlay geometry.
    /// </summary>
    /// <remarks>
    /// Holds no playback state: the host decides which frame is showing and this draws it, so an
    /// inspector's preview settings and a window's transport bar can drive the same code.
    /// <para>
    /// One instance per host, never shared. Two hosts sharing a <see cref="PreviewRenderUtility"/>
    /// would interleave their Begin/End pairs, and an unbalanced pair reports
    /// "Previous BeginPreview() was not closed" on every later repaint instead of the real problem.
    /// </para>
    /// </remarks>
    public sealed class SkeletonPreview : IDisposable
    {
        private const int GridHalfExtent = 10;

        private PreviewRenderUtility _previewRenderUtility;

        private Material _lineMaterial;
        private Mesh _skeletonMesh;
        private readonly List<Vector3> _lineVertices = new();
        private readonly List<Color> _lineColours = new();
        private readonly List<int> _lineIndices = new();

        private Material _gridMaterial;
        private Mesh _gridMesh;

        private readonly SkeletonPreviewOverlay _overlay = new();

        private Vector2 _drag;
        private float _distance = 5f;
        private Vector3 _targetPos = Vector3.zero;
        private bool _frameOnNextDraw;

        private Skeleton _skeleton;
        private int _skeletonContentHash;
        private SkeletonData _skeletonData;
        private NativeArray<float3> _fkPositions;
        private NativeArray<quaternion> _fkRotations;

        /// <summary>
        /// The asset to pose. Deliberately typed as the base <see cref="SkeletonAnimation"/>:
        /// <see cref="AnnotatedAnimationClip"/> shadows <c>FrameCount</c> and <c>GetFrame</c> with
        /// slice-local versions, and this preview works in whole-clip frames.
        /// </summary>
        public SkeletonAnimation Source { get; set; }

        /// <summary>The whole-clip frame to pose. Clamped when drawn, so a stale value is harmless.</summary>
        public int Frame { get; set; }

        public bool CanRender =>
            Source != null && Source.TryValidate(out _) &&
            Source.PoseSequence != null && Source.FrameCount > 0;

        public Color SkeletonColour { get; set; } = Color.green;

        /// <summary>
        /// Raised once per repaint, after the skeleton mesh is built and before the camera renders,
        /// so a subscriber's geometry lands in the same off-screen image.
        /// </summary>
        public event Action<SkeletonPreviewFrame> DrawingOverlays;

        /// <summary>Renders into <paramref name="rect"/> and handles its camera input. Call from OnGUI.</summary>
        public void Draw(Rect rect, GUIStyle background)
        {
            EnsureResources();
            HandleCameraControls(rect);

            _previewRenderUtility.BeginPreview(rect, background);

            Texture rendered;
            // Anything thrown between Begin and End leaves the preview unbalanced, and every later
            // repaint then reports "Previous BeginPreview() was not closed" instead of the actual
            // problem - one real error turning into an unreadable stream of two.
            try
            {
                if (_gridMaterial != null && _gridMesh != null)
                {
                    _previewRenderUtility.DrawMesh(_gridMesh, Matrix4x4.identity, _gridMaterial, 0);
                }

                if (TryPoseSkeleton(out var frame))
                {
                    UpdateCamera();
                    DrawSkeletonLines();
                    DrawOverlays(frame);
                }

                _previewRenderUtility.camera.Render();
            }
            finally
            {
                rendered = _previewRenderUtility.EndPreview();
            }

            GUI.DrawTexture(rect, rendered, ScaleMode.StretchToFill, false);
        }

        /// <summary>Re-frames the camera on the whole skeleton at the next repaint.</summary>
        public void FrameAll() => _frameOnNextDraw = true;

        private void EnsureResources()
        {
            // Built on the first repaint rather than in the constructor: a PreviewRenderUtility
            // creates a scene and a camera, which is not something to do while deserializing.
            if (_previewRenderUtility == null)
            {
                _previewRenderUtility = new PreviewRenderUtility();
                _previewRenderUtility.camera.fieldOfView = 30f;
                _previewRenderUtility.camera.nearClipPlane = 0.01f;
                _previewRenderUtility.camera.farClipPlane = 1000f;
            }

            if (_skeletonMesh == null)
            {
                _skeletonMesh = new Mesh { hideFlags = HideFlags.HideAndDontSave };
                _skeletonMesh.MarkDynamic();
            }

            if (_lineMaterial == null) _lineMaterial = CreateLineMaterial();
            if (_gridMaterial == null) _gridMaterial = CreateLineMaterial();
            if (_gridMesh == null) _gridMesh = CreateGridMesh();
        }

        private static Material CreateLineMaterial()
        {
            var shader = Shader.Find("Hidden/Internal-Colored");
            if (shader == null) return null;

            var material = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
            material.SetInt("_ZWrite", 1);
            material.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.LessEqual);
            material.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            return material;
        }

        private static Mesh CreateGridMesh()
        {
            var mesh = new Mesh { hideFlags = HideFlags.HideAndDontSave };

            var vertices = new List<Vector3>();
            var indices = new List<int>();
            var colours = new List<Color>();

            var darkGray = new Color(0.3f, 0.3f, 0.3f, 1f);
            var lightGray = new Color(0.5f, 0.5f, 0.5f, 1f);

            for (var i = -GridHalfExtent; i <= GridHalfExtent; i++)
            {
                vertices.Add(new Vector3(i, 0f, -GridHalfExtent));
                vertices.Add(new Vector3(i, 0f, GridHalfExtent));
                vertices.Add(new Vector3(-GridHalfExtent, 0f, i));
                vertices.Add(new Vector3(GridHalfExtent, 0f, i));

                var colour = i % 5 == 0 ? lightGray : darkGray;
                colours.Add(colour);
                colours.Add(colour);
                colours.Add(colour);
                colours.Add(colour);

                var count = indices.Count;
                indices.Add(count);
                indices.Add(count + 1);
                indices.Add(count + 2);
                indices.Add(count + 3);
            }

            mesh.SetVertices(vertices);
            mesh.SetColors(colours);
            mesh.SetIndices(indices, MeshTopology.Lines, 0);
            return mesh;
        }

        /// <summary>
        /// Reallocates the FK scratch buffers when the source's resolved <see cref="Skeleton"/>
        /// changes (including to or from null). Cheap no-op otherwise.
        /// </summary>
        /// <remarks>
        /// Keyed on the bone tree's content as well as the root's identity. Root identity alone
        /// misses a rig whose bones changed underneath it, which used to leave this cache and the
        /// asset's own baked pose sequence describing two different skeletons - and the FK pass
        /// indexing one by the other's bone count.
        /// </remarks>
        private void RefreshSkeletonCache()
        {
            var skeleton = Source != null ? Source.Skeleton : null;
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

        private bool TryPoseSkeleton(out int frame)
        {
            frame = 0;

            // The rig and clip can be edited live, so re-check the cache every draw.
            RefreshSkeletonCache();

            if (!CanRender || !_fkPositions.IsCreated) return false;

            frame = Mathf.Clamp(Frame, 0, Source.FrameCount - 1);

            // First access triggers the one-time clip bake.
            var pose = Source.GetFrame(frame);

            // The bake and this cache are invalidated by different things, so draw nothing rather
            // than let FK throw on a repaint if they ever disagree again.
            if (pose.Layout.RotationCount != _skeletonData.BoneCount) return false;

            _skeletonData.LocalSpaceToCharacterSpace(pose, _fkPositions, _fkRotations);
            return true;
        }

        private void UpdateCamera()
        {
            if (_frameOnNextDraw)
            {
                _frameOnNextDraw = false;
                FrameBonesNow();
            }

            var camTarget = (Vector3)_fkPositions[0] + _targetPos;
            var camRotation = Quaternion.Euler(_drag.y, _drag.x, 0f);

            var camera = _previewRenderUtility.camera;
            camera.transform.position = camTarget - camRotation * Vector3.forward * _distance;
            camera.transform.rotation = camRotation;
        }

        private void FrameBonesNow()
        {
            var bounds = new Bounds(_fkPositions[0], Vector3.zero);
            for (var i = 1; i < _fkPositions.Length; i++) bounds.Encapsulate(_fkPositions[i]);

            _targetPos = bounds.center - (Vector3)_fkPositions[0];
            _distance = Mathf.Max(0.5f, bounds.extents.magnitude * 2.5f);
        }

        private void DrawSkeletonLines()
        {
            _lineVertices.Clear();
            _lineColours.Clear();
            _lineIndices.Clear();

            for (var i = 1; i < _skeletonData.BoneCount; i++)
            {
                var parentIndex = _skeletonData.ParentIndices[i];

                _lineVertices.Add(_fkPositions[parentIndex]);
                _lineVertices.Add(_fkPositions[i]);
                _lineColours.Add(SkeletonColour);
                _lineColours.Add(SkeletonColour);

                var indexCount = _lineIndices.Count;
                _lineIndices.Add(indexCount);
                _lineIndices.Add(indexCount + 1);
            }

            _skeletonMesh.Clear();
            _skeletonMesh.SetVertices(_lineVertices);
            _skeletonMesh.SetColors(_lineColours);
            _skeletonMesh.SetIndices(_lineIndices, MeshTopology.Lines, 0);

            if (_lineMaterial != null)
            {
                _previewRenderUtility.DrawMesh(_skeletonMesh, Matrix4x4.identity, _lineMaterial, 0);
            }
        }

        private void DrawOverlays(int frame)
        {
            var subscribers = DrawingOverlays;
            if (subscribers == null) return;

            _overlay.Begin(_previewRenderUtility, _lineMaterial);
            subscribers(new SkeletonPreviewFrame(_overlay, _previewRenderUtility.camera, _skeletonData,
                _fkPositions, _fkRotations, frame));
            _overlay.Flush();
        }

        private void HandleCameraControls(Rect rect)
        {
            var e = Event.current;
            if (!rect.Contains(e.mousePosition)) return;

            if (e.type == EventType.MouseDrag && e.button == 0)
            {
                _drag.x += e.delta.x * 0.5f;
                _drag.y += e.delta.y * 0.5f;
                e.Use();
            }
            else if (e.type == EventType.MouseDrag && e.button == 2)
            {
                var right = _previewRenderUtility.camera.transform.right;
                var up = _previewRenderUtility.camera.transform.up;
                _targetPos -= (right * e.delta.x - up * e.delta.y) * (0.01f * _distance);
                e.Use();
            }
            else if (e.type == EventType.ScrollWheel)
            {
                _distance = Mathf.Max(0.1f, _distance + e.delta.y * 0.1f);
                e.Use();
            }
        }

        /// <summary>
        /// Releases everything this owns. The <see cref="SkeletonData"/> and the source's pose
        /// sequence are <c>Allocator.Domain</c> and owned elsewhere - only the two FK buffers here
        /// are ours. Safe to call twice.
        /// </summary>
        public void Dispose()
        {
            if (_previewRenderUtility != null)
            {
                _previewRenderUtility.Cleanup();
                _previewRenderUtility = null;
            }

            _overlay.Dispose();

            if (_lineMaterial != null) UnityEngine.Object.DestroyImmediate(_lineMaterial);
            if (_skeletonMesh != null) UnityEngine.Object.DestroyImmediate(_skeletonMesh);
            if (_gridMaterial != null) UnityEngine.Object.DestroyImmediate(_gridMaterial);
            if (_gridMesh != null) UnityEngine.Object.DestroyImmediate(_gridMesh);

            _lineMaterial = null;
            _skeletonMesh = null;
            _gridMaterial = null;
            _gridMesh = null;

            if (_fkPositions.IsCreated) _fkPositions.Dispose();
            if (_fkRotations.IsCreated) _fkRotations.Dispose();

            _skeleton = null;
            _skeletonContentHash = 0;
            _skeletonData = default;
        }
    }
}
