// Editor-only, and the guard must enclose the usings: `using UnityEditor` fails to resolve in a player build.
#if UNITY_EDITOR
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace MotionMatching
{
    public static class GizmosExtensions
    {
        public static void DrawLine(Vector3 startPosition, Vector3 endPosition, float thickness, bool useDepth = true)
        {
            if (Camera.current == null) return;
            float d = Mathf.Min(Vector3.Distance(Camera.current.transform.position, startPosition), Vector3.Distance(Camera.current.transform.position, endPosition));
            Handles.DrawBezier(startPosition, endPosition, startPosition, endPosition, Gizmos.color, null, thickness * Mathf.Min(1.0f, (3.0f / d)));
        }

        /// <summary>Draws a wire cube with a given rotation.</summary>
        public static void DrawWireCube(Vector3 center, Vector3 size, Quaternion rotation = default(Quaternion))
        {
            var old = Gizmos.matrix;
            if (rotation.Equals(default(Quaternion)))
                rotation = Quaternion.identity;
            Gizmos.matrix = Matrix4x4.TRS(center, rotation, size);
            Gizmos.DrawWireCube(Vector3.zero, Vector3.one);
            Gizmos.matrix = old;
        }

        public static void DrawArrow(Vector3 from, Vector3 to, float arrowHeadLength = 0.25f, float arrowHeadAngle = 20.0f, float thickness = 3.0f, bool useDepth = true)
        {
            DrawLine(from, to, thickness, useDepth);
            var direction = to - from;
            if (direction.magnitude < 0.001f) return;
            var right = Quaternion.LookRotation(direction) * Quaternion.Euler(0, 180 + arrowHeadAngle, 0) * new Vector3(0, 0, 1);
            var left = Quaternion.LookRotation(direction) * Quaternion.Euler(0, 180 - arrowHeadAngle, 0) * new Vector3(0, 0, 1);
            DrawLine(to, to + right * arrowHeadLength, thickness, useDepth);
            DrawLine(to, to + left * arrowHeadLength, thickness, useDepth);
        }

        public static void DrawWireSphere(Vector3 center, float radius, Quaternion rotation = default(Quaternion))
        {
            var old = Gizmos.matrix;
            if (rotation.Equals(default(Quaternion)))
                rotation = Quaternion.identity;
            Gizmos.matrix = Matrix4x4.TRS(center, rotation, Vector3.one);
            Gizmos.DrawWireSphere(Vector3.zero, radius);
            Gizmos.matrix = old;
        }

        /// <summary>Draws a flat wire circle facing up.</summary>
        public static void DrawWireCircle(Vector3 center, float radius, Quaternion rotation, int segments = 20)
        {
            DrawWireArc(center, radius, 360, rotation, segments);
        }

        /// <summary>Draws a flat arc rotated around its center.</summary>
        /// <param name="angle">Angle in degrees.</param>
        public static void DrawWireArc(Vector3 center, float radius, float angle, Quaternion rotation, int segments = 20)
        {
            var old = Gizmos.matrix;

            Gizmos.matrix = Matrix4x4.TRS(center, rotation, Vector3.one);
            Vector3 from = Vector3.forward * radius;
            var step = Mathf.Max(Mathf.RoundToInt(angle / segments), 1);
            for (int i = 0; i <= angle; i += step)
            {
                var to = new Vector3(radius * Mathf.Sin(i * Mathf.Deg2Rad), 0, radius * Mathf.Cos(i * Mathf.Deg2Rad));
                Gizmos.DrawLine(from, to);
                from = to;
            }

            Gizmos.matrix = old;
        }

        /// <summary>Draws a flat arc rotated around an arbitrary center of rotation.</summary>
        /// <param name="angle">Angle in degrees.</param>
        public static void DrawWireArc(Vector3 center, float radius, float angle, int segments, Quaternion rotation, Vector3 centerOfRotation)
        {
            var old = Gizmos.matrix;
            if (rotation.Equals(default(Quaternion)))
                rotation = Quaternion.identity;
            Gizmos.matrix = Matrix4x4.TRS(centerOfRotation, rotation, Vector3.one);
            var deltaTranslation = centerOfRotation - center;
            Vector3 from = deltaTranslation + Vector3.forward * radius;
            var step = Mathf.Max(Mathf.RoundToInt(angle / segments), 1);
            for (int i = 0; i <= angle; i += step)
            {
                var to = new Vector3(radius * Mathf.Sin(i * Mathf.Deg2Rad), 0, radius * Mathf.Cos(i * Mathf.Deg2Rad)) + deltaTranslation;
                Gizmos.DrawLine(from, to);
                from = to;
            }

            Gizmos.matrix = old;
        }

        /// <summary>Draws a flat arc under an arbitrary gizmo matrix.</summary>
        /// <param name="angle">Angle in degrees.</param>
        public static void DrawWireArc(Matrix4x4 matrix, float radius, float angle, int segments)
        {
            var old = Gizmos.matrix;
            Gizmos.matrix = matrix;
            Vector3 from = Vector3.forward * radius;
            var step = Mathf.Max(Mathf.RoundToInt(angle / segments), 1);
            for (int i = 0; i <= angle; i += step)
            {
                var to = new Vector3(radius * Mathf.Sin(i * Mathf.Deg2Rad), 0, radius * Mathf.Cos(i * Mathf.Deg2Rad));
                Gizmos.DrawLine(from, to);
                from = to;
            }

            Gizmos.matrix = old;
        }

        /// <summary>Draws an upright wire cylinder rotated around its center.</summary>
        public static void DrawWireCylinder(Vector3 center, float radius, Quaternion rotation, float height)
        {
            var old = Gizmos.matrix;
            if (rotation.Equals(default(Quaternion)))
                rotation = Quaternion.identity;
            Gizmos.matrix = Matrix4x4.TRS(center, rotation, Vector3.one);
            var half = height / 2;

            Gizmos.DrawLine(Vector3.right * radius - Vector3.up * half, Vector3.right * radius + Vector3.up * half);
            Gizmos.DrawLine(-Vector3.right * radius - Vector3.up * half, -Vector3.right * radius + Vector3.up * half);
            Gizmos.DrawLine(Vector3.forward * radius - Vector3.up * half, Vector3.forward * radius + Vector3.up * half);
            Gizmos.DrawLine(-Vector3.forward * radius - Vector3.up * half, -Vector3.forward * radius + Vector3.up * half);

            // The caps rotate around the cylinder's center, not their own.
            DrawWireArc(center + Vector3.up * half, radius, 360, 20, rotation, center);
            DrawWireArc(center + Vector3.down * half, radius, 360, 20, rotation, center);
            Gizmos.matrix = old;
        }

        /// <summary>Draws an upright wire capsule.</summary>
        public static void DrawWireCapsule(Vector3 center, float radius, float height, Quaternion rotation)
        {
            if (rotation.Equals(default(Quaternion)))
                rotation = Quaternion.identity;
            var old = Gizmos.matrix;
            Gizmos.matrix = Matrix4x4.TRS(center, rotation, Vector3.one);
            var half = height / 2 - radius;

            DrawWireCylinder(center, radius, rotation, height - radius * 2);

            // Each cap is two perpendicular half-circle arcs.
            var mat = Matrix4x4.Translate(center + rotation * Vector3.up * half) * Matrix4x4.Rotate(rotation * Quaternion.AngleAxis(90, Vector3.forward));
            DrawWireArc(mat, radius, 180, 20);
            mat = Matrix4x4.Translate(center + rotation * Vector3.up * half) * Matrix4x4.Rotate(rotation * Quaternion.AngleAxis(90, Vector3.up) * Quaternion.AngleAxis(90, Vector3.forward));
            DrawWireArc(mat, radius, 180, 20);

            mat = Matrix4x4.Translate(center + rotation * Vector3.down * half) * Matrix4x4.Rotate(rotation * Quaternion.AngleAxis(90, Vector3.up) * Quaternion.AngleAxis(-90, Vector3.forward));
            DrawWireArc(mat, radius, 180, 20);
            mat = Matrix4x4.Translate(center + rotation * Vector3.down * half) * Matrix4x4.Rotate(rotation * Quaternion.AngleAxis(-90, Vector3.forward));
            DrawWireArc(mat, radius, 180, 20);

            Gizmos.matrix = old;
        }

        /// <summary>Draws a flat wire ellipse from two ground-plane semi-axes (direction and length).</summary>
        public static void DrawWireEllipse(float3 center, float2 primaryAxis, float2 secondaryAxis, Quaternion rotation, int segments = 20, float thickness = 1.5f)
        {
            var old = Gizmos.matrix;
            Gizmos.matrix = Matrix4x4.TRS(center, rotation, Vector3.one);

            Vector3 from = CalculateEllipsePoint(primaryAxis, secondaryAxis, 0);
            var step = Mathf.Max(Mathf.RoundToInt(360f / segments), 1);

            for (int i = step; i <= 360; i += step)
            {
                Vector3 to = CalculateEllipsePoint(primaryAxis, secondaryAxis, i);
                Gizmos.DrawLine(from, to);
                from = to;
            }
            Gizmos.DrawLine(from, CalculateEllipsePoint(primaryAxis, secondaryAxis, 0));

            Gizmos.matrix = old;
        }

        private static Vector3 CalculateEllipsePoint(float2 primaryAxis, float2 secondaryAxis, float angleDegrees)
        {
            float angleRad = angleDegrees * Mathf.Deg2Rad;
            float x = primaryAxis.x * Mathf.Cos(angleRad) + secondaryAxis.x * Mathf.Sin(angleRad);
            float y = primaryAxis.y * Mathf.Cos(angleRad) + secondaryAxis.y * Mathf.Sin(angleRad);
            return new Vector3(x, 0, y);
        }
    }
}
#endif