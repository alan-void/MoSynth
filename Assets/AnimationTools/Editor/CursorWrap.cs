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
    /// Blender wraps within the region being panned, and that containment is most of what makes the
    /// gesture feel like it belongs to the panel. Unity's <c>SetWantsMouseJumping</c> wraps at the
    /// edge of the screen instead and there is no region-scoped variant in the Editor API, so the
    /// wrap is done here. Callers leave <c>SetWantsMouseJumping</c> on underneath as the fallback for
    /// platforms <see cref="Move"/> cannot serve; on Windows it never fires, because a cursor kept
    /// inside a panel never reaches the edge of the screen.
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
        /// the jump as motion. Taking it back out is what <c>SetWantsMouseJumping</c> does for itself
        /// internally, and doing both halves here is what stops a caller getting the order wrong.
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
        /// Relative rather than absolute on purpose: <c>GUIUtility.GUIToScreenPoint</c> answers in
        /// editor points and the OS wants physical pixels, and those disagree under display scaling
        /// and again between monitors scaled differently. Moving <em>by</em> a panel width needs no
        /// origin at all and one scale factor. <c>user32.dll</c> is part of Windows, so this costs a
        /// fresh clone of the project nothing.
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
