using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Collects every overlay's line segments for one repaint into a single dynamic mesh, so an
    /// arbitrary number of overlays costs one draw call rather than one each.
    /// </summary>
    public sealed class SkeletonPreviewOverlay : ISkeletonPreviewDrawer, IDisposable
    {
        private const int CircleSegments = 16;

        private readonly List<Vector3> _vertices = new();
        private readonly List<Color> _colours = new();
        private readonly List<int> _indices = new();

        private PreviewRenderUtility _target;
        private Material _lineMaterial;
        private Mesh _mesh;

        /// <summary>Starts a repaint's collection. Drawing outside a Begin/Flush pair is ignored.</summary>
        public void Begin(PreviewRenderUtility target, Material lineMaterial)
        {
            _target = target;
            _lineMaterial = lineMaterial;
            _vertices.Clear();
            _colours.Clear();
            _indices.Clear();
        }

        /// <summary>Submits everything collected since <see cref="Begin"/> to the preview camera.</summary>
        public void Flush()
        {
            if (_target == null || _lineMaterial == null || _vertices.Count == 0) return;

            if (_mesh == null)
            {
                _mesh = new Mesh { hideFlags = HideFlags.HideAndDontSave };
                _mesh.MarkDynamic();
            }

            _mesh.Clear();
            _mesh.SetVertices(_vertices);
            _mesh.SetColors(_colours);
            _mesh.SetIndices(_indices, MeshTopology.Lines, 0);

            _target.DrawMesh(_mesh, Matrix4x4.identity, _lineMaterial, 0);
        }

        public void DrawLine(float3 from, float3 to, Color colour)
        {
            if (_target == null) return;
            AddSegment(from, to, colour);
        }

        public void DrawPolyline(IReadOnlyList<float3> points, Color colour)
        {
            if (_target == null || points == null || points.Count < 2) return;

            for (var i = 1; i < points.Count; i++)
            {
                AddSegment(points[i - 1], points[i], colour);
            }
        }

        public void DrawWireSphere(float3 centre, float radius, Color colour)
        {
            if (_target == null) return;

            AddCircle(centre, radius, new float3(1f, 0f, 0f), new float3(0f, 1f, 0f), colour);
            AddCircle(centre, radius, new float3(1f, 0f, 0f), new float3(0f, 0f, 1f), colour);
            AddCircle(centre, radius, new float3(0f, 1f, 0f), new float3(0f, 0f, 1f), colour);
        }

        public void DrawArrow(float3 from, float3 direction, Color colour)
        {
            if (_target == null) return;

            var length = math.length(direction);
            if (length <= math.EPSILON) return;

            var tip = from + direction;
            AddSegment(from, tip, colour);

            // A barb pair in whichever plane is least degenerate for this direction.
            var forward = direction / length;
            var reference = math.abs(forward.y) > 0.99f ? new float3(1f, 0f, 0f) : new float3(0f, 1f, 0f);
            var side = math.normalize(math.cross(forward, reference)) * (length * 0.15f);
            var back = tip - forward * (length * 0.25f);

            AddSegment(tip, back + side, colour);
            AddSegment(tip, back - side, colour);
        }

        public void DrawMesh(Mesh mesh, Matrix4x4 matrix, Material material)
        {
            if (_target == null || mesh == null || material == null) return;
            _target.DrawMesh(mesh, matrix, material, 0);
        }

        private void AddSegment(float3 from, float3 to, Color colour)
        {
            var index = _vertices.Count;

            _vertices.Add(from);
            _vertices.Add(to);
            _colours.Add(colour);
            _colours.Add(colour);
            _indices.Add(index);
            _indices.Add(index + 1);
        }

        private void AddCircle(float3 centre, float radius, float3 axisA, float3 axisB, Color colour)
        {
            var previous = centre + axisA * radius;

            for (var i = 1; i <= CircleSegments; i++)
            {
                var angle = i / (float)CircleSegments * 2f * math.PI;
                var point = centre + (axisA * math.cos(angle) + axisB * math.sin(angle)) * radius;
                AddSegment(previous, point, colour);
                previous = point;
            }
        }

        public void Dispose()
        {
            if (_mesh != null) UnityEngine.Object.DestroyImmediate(_mesh);
            _mesh = null;
            _target = null;
            _lineMaterial = null;
        }
    }
}
