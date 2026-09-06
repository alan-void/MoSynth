using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Line drawing into a <see cref="SkeletonPreview"/>'s off-screen render, in the same
    /// character space the FK'd bone positions are given in.
    /// </summary>
    /// <remarks>
    /// <see cref="UnityEditor.Handles"/> draws into the current GUI or Scene view camera, not the
    /// preview's own, so an overlay cannot use it. Everything here becomes geometry submitted to
    /// the preview camera instead.
    /// </remarks>
    public interface ISkeletonPreviewDrawer
    {
        void DrawLine(float3 from, float3 to, Color colour);

        /// <summary>Draws a connected run of segments. Fewer than two points draws nothing.</summary>
        void DrawPolyline(IReadOnlyList<float3> points, Color colour);

        /// <summary>Three axis-aligned circles, the cheapest readable marker for a point.</summary>
        void DrawWireSphere(float3 centre, float radius, Color colour);

        /// <summary>A shaft along <paramref name="direction"/> with two barbs at its tip.</summary>
        void DrawArrow(float3 from, float3 direction, Color colour);

        /// <summary>
        /// Escape hatch for an overlay that needs real geometry rather than lines. The mesh and
        /// material stay the caller's to own and destroy.
        /// </summary>
        void DrawMesh(Mesh mesh, Matrix4x4 matrix, Material material);
    }
}
