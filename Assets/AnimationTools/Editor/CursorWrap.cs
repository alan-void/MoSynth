using System.Runtime.InteropServices;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Keeps a view drag's cursor inside the panel it is moving, wrapping it to the opposite edge
    /// rather than letting it wander off across the display.
    /// </summary>
    /// <remarks>
    /// Unity's <c>SetWantsMouseJumping</c> only wraps at the screen edge. Callers leave it on as the
    /// fallback for platforms where <see cref="Move"/> fails.
    /// </remarks>
    public struct CursorWrap
    {
        private Vector2 _pendingWarp;

        /// <summary>
        /// The pointer motion this event carries, after which the cursor is pulled back inside
        /// <paramref name="rect"/> if it has left it. Call once per drag event and use the result in
        /// place of <c>Event.delta</c>.
        /// </summary>
        /// <remarks>
        /// Unity measures delta against the position it last saw, so the event after a wrap reports
        /// the jump as motion; this subtracts it back out.
        /// </remarks>
        public Vector2 DeltaWithin(Rect rect, Event e)
        {
            var delta = e.delta - _pendingWarp;
            _pendingWarp = Vector2.zero;

            var offset = Offset(rect, e.mousePosition);
            if (offset != Vector2.zero && Move(offset)) _pendingWarp = offset;

            return delta;
        }

        /// <summary>
        /// How far to move a cursor that has left <paramref name="rect"/> to bring it back in at the
        /// opposite edge. Zero while it is inside.
        /// </summary>
        public static Vector2 Offset(Rect rect, Vector2 mousePosition)
        {
            var offset = Vector2.zero;

            if (mousePosition.x < rect.xMin) offset.x = rect.width;
            else if (mousePosition.x > rect.xMax) offset.x = -rect.width;

            if (mousePosition.y < rect.yMin) offset.y = rect.height;
            else if (mousePosition.y > rect.yMax) offset.y = -rect.height;

            return offset;
        }

        /// <summary>
        /// Moves the operating system's cursor by an offset in editor points. False where that cannot
        /// be done, which leaves the drag unwrapped rather than misplaced.
        /// </summary>
        /// <remarks>
        /// Relative rather than absolute: editor points and physical pixels disagree under display
        /// scaling and between differently-scaled monitors, and a relative move needs no origin.
        /// </remarks>
        public static bool Move(Vector2 offsetInPoints)
        {
#if UNITY_EDITOR_WIN
            if (!GetCursorPos(out var position)) return false;

            var scale = EditorGUIUtility.pixelsPerPoint;
            return SetCursorPos(
                position.x + Mathf.RoundToInt(offsetInPoints.x * scale),
                position.y + Mathf.RoundToInt(offsetInPoints.y * scale));
#else
            return false;
#endif
        }

#if UNITY_EDITOR_WIN
        [StructLayout(LayoutKind.Sequential)]
        private struct NativePoint
        {
            public int x;
            public int y;
        }

        [DllImport("user32.dll")]
        private static extern bool GetCursorPos(out NativePoint point);

        [DllImport("user32.dll")]
        private static extern bool SetCursorPos(int x, int y);
#endif
    }
}
