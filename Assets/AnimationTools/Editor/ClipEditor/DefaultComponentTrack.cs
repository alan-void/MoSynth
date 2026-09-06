using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// What a component with no registered track gets: a lane showing its own
    /// <see cref="AnimationClipComponent.Describe"/> line, and a fully editable inspector.
    /// </summary>
    /// <remarks>
    /// The inspector is one <c>PropertyField</c> over the managed reference, which the
    /// SubclassSelector drawer expands into every field the component declares. So a component type
    /// added tomorrow is authorable in this window immediately, with no editor code at all.
    /// </remarks>
    public sealed class DefaultComponentTrack : AnimationClipComponentTrack
    {
        private static readonly Color EnabledFill = new(0.36f, 0.42f, 0.5f, 1f);
        private static readonly Color DisabledFill = new(0.24f, 0.25f, 0.27f, 1f);

        public override float DefaultLaneHeight => 22f;

        public override void DrawTrack(in TrackDrawContext context)
        {
            if (Event.current.type != EventType.Repaint) return;

            // No per-frame data to place, so the bar spans the slice: that is the extent the
            // component's annotation is about, and pretending to more would be a lie.
            var editor = context.Editor;
            var axis = context.Axis;

            var left = axis.FrameToX(editor.StartFrame);
            var right = axis.FrameToX(Mathf.Max(editor.StartFrame + 1, editor.EndFrame));

            var bar = new Rect(left, context.LaneRect.y + 3f, Mathf.Max(2f, right - left),
                context.LaneRect.height - 6f);
            bar.xMin = Mathf.Max(bar.xMin, context.LaneRect.xMin);
            bar.xMax = Mathf.Min(bar.xMax, context.LaneRect.xMax);
            if (bar.width <= 0f) return;

            var enabled = Component != null && Component.isEnabled;
            EditorGUI.DrawRect(bar, enabled ? EnabledFill : DisabledFill);

            var label = Component != null ? Component.Describe() : "(missing)";
            var style = new GUIStyle(EditorStyles.miniLabel)
            {
                normal = { textColor = enabled ? Color.white : new Color(0.6f, 0.6f, 0.6f) },
                clipping = TextClipping.Clip
            };

            GUI.Label(new Rect(bar.x + 4f, bar.y, bar.width - 8f, bar.height), label, style);
        }

        public override void DrawInspector(in TrackInspectorContext context)
        {
            if (context.ComponentProperty == null) return;

            EditorGUILayout.PropertyField(context.ComponentProperty, true);
        }
    }
}
