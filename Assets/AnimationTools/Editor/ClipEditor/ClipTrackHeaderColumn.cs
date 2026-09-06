using System;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// The gutter left of the lanes: the clip's component list. Each row carries a track's name and
    /// its two toggles, with an Add Component button under the last of them.
    /// </summary>
    /// <remarks>
    /// Drawn inside the same scroll view as the lanes so the two can never scroll out of step, and
    /// so the Add Component button sits under the list the way the Inspector's does. The column's
    /// width lives on <see cref="ClipTimelineView"/>, which owns the drag that changes it.
    /// </remarks>
    public static class ClipTrackHeaderColumn
    {
        public const float AddComponentRowHeight = 32f;

        private const float ResizeGripHeight = 4f;
        private const float MinLaneHeight = 16f;
        private const float MaxLaneHeight = 400f;

        private static readonly Color Background = new(0.19f, 0.19f, 0.20f, 1f);
        private static readonly Color FocusedBackground = new(0.24f, 0.28f, 0.34f, 1f);
        private static readonly Color Separator = new(0.12f, 0.12f, 0.13f, 1f);

        /// <summary>
        /// Draws one row's header. <paramref name="onFocus"/> is raised when the row is clicked,
        /// and returns whether anything about the view state changed.
        /// </summary>
        public static bool Draw(Rect rect, ClipTrackRow row, bool isFocused, Action onFocus)
        {
            var changed = false;

            if (Event.current.type == EventType.Repaint)
            {
                // No right edge: ClipTimelineView draws one continuous line down the whole
                // gutter, which a per-row edge would only double up on.
                EditorGUI.DrawRect(rect, isFocused ? FocusedBackground : Background);
                EditorGUI.DrawRect(new Rect(rect.x, rect.yMax - 1f, rect.width, 1f), Separator);
            }

            var line = new Rect(rect.x + 4f, rect.y + 3f, rect.width - 8f, EditorGUIUtility.singleLineHeight);

            var visibleRect = new Rect(line.x, line.y, 18f, line.height);

            var newVisible = GUI.Toggle(visibleRect, row.Visible,
                new GUIContent(row.Visible ? "◉" : "○", "Show this lane"), EditorStyles.label);
            if (newVisible != row.Visible)
            {
                row.Visible = newVisible;
                changed = true;
            }

            var component = row.Track?.Component;
            if (component != null)
            {
                var enabledRect = new Rect(visibleRect.xMax, line.y, 18f, line.height);
                var newEnabled = GUI.Toggle(enabledRect, component.isEnabled,
                    new GUIContent(string.Empty, "Whether this annotation is enabled on the clip"));

                if (newEnabled != component.isEnabled && row.Track.Editor != null)
                {
                    var enabledProperty = row.Track.ComponentProperty?.FindPropertyRelative("isEnabled");
                    if (enabledProperty != null)
                    {
                        enabledProperty.boolValue = newEnabled;
                        row.Track.Editor.Commit();
                    }
                }
            }

            var titleRect = new Rect(line.x + 40f, line.y, line.width - 40f, line.height);
            var title = row.Track?.Title ?? "(missing)";
            var titleStyle = new GUIStyle(EditorStyles.label)
            {
                fontStyle = isFocused ? FontStyle.Bold : FontStyle.Normal,
                clipping = TextClipping.Clip
            };
            GUI.Label(titleRect, new GUIContent(title, title), titleStyle);

            // Second line: the component's own one-line summary, when the row is tall enough.
            if (rect.height > EditorGUIUtility.singleLineHeight * 2f && component != null)
            {
                var describeRect = new Rect(line.x + 4f, line.yMax, rect.width - 12f,
                    EditorGUIUtility.singleLineHeight);
                GUI.Label(describeRect, component.Describe(), EditorStyles.miniLabel);
            }

            changed |= HandleResize(rect, row);
            HandleFocusClick(rect, onFocus);
            return changed;
        }

        private static bool HandleResize(Rect rect, ClipTrackRow row)
        {
            var grip = new Rect(rect.x, rect.yMax - ResizeGripHeight, rect.width, ResizeGripHeight);
            EditorGUIUtility.AddCursorRect(grip, MouseCursor.ResizeVertical);

            var controlId = GUIUtility.GetControlID(FocusType.Passive);
            var e = Event.current;

            switch (e.GetTypeForControl(controlId))
            {
                case EventType.MouseDown when e.button == 0 && grip.Contains(e.mousePosition):
                    GUIUtility.hotControl = controlId;
                    e.Use();
                    return false;

                case EventType.MouseDrag when GUIUtility.hotControl == controlId:
                    row.LaneHeight = Mathf.Clamp(row.LaneHeight + e.delta.y, MinLaneHeight, MaxLaneHeight);
                    row.HeightIsUserSet = true;
                    e.Use();
                    return true;

                case EventType.MouseUp when GUIUtility.hotControl == controlId:
                    GUIUtility.hotControl = 0;
                    e.Use();
                    return true;
            }

            return false;
        }

        private static void HandleFocusClick(Rect rect, Action onFocus)
        {
            var e = Event.current;
            if (e.type != EventType.MouseDown || e.button != 0) return;
            if (!rect.Contains(e.mousePosition)) return;

            onFocus?.Invoke();
            e.Use();
        }

        /// <summary>
        /// The Inspector's Add Component button, in the Inspector's place: under the component list
        /// rather than off in the toolbar.
        /// </summary>
        public static void DrawAddComponent(Rect rect, Action onClick)
        {
            if (onClick == null) return;

            if (Event.current.type == EventType.Repaint) EditorGUI.DrawRect(rect, Background);

            var button = new Rect(rect.x + 12f, rect.y + 6f, Mathf.Max(40f, rect.width - 24f), 20f);
            if (GUI.Button(button, "Add Component")) onClick();
        }
    }
}
