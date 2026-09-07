using UnityEditor;

namespace AnimationTools.Editor
{
    /// <summary>
    /// The clip editor's view of one <see cref="AnimationClipComponent"/>: a timeline lane, an
    /// inspector, and optionally geometry drawn into the animation preview.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Declare a subclass, tag it with <see cref="ClipComponentTrackAttribute"/>, and the window
    /// finds it - the same registration-free seam the components themselves use. A component with
    /// no track still gets a lane and a fully editable inspector from
    /// <see cref="DefaultComponentTrack"/>, so writing one of these is an upgrade rather than a
    /// prerequisite.
    /// </para>
    /// <para>
    /// Drawing and input share one method per surface because that is how IMGUI works and lanes
    /// never overlap. Persist edits through <see cref="ComponentProperty"/> and
    /// <see cref="ClipEditorContext.Commit"/>, once per interaction - see
    /// <c>openwiki/animation-tools/clip-editor.md</c> for why a commit per mouse-move is expensive.
    /// </para>
    /// </remarks>
    public abstract class AnimationClipComponentTrack
    {
        /// <summary>Set before <see cref="OnEnable"/> and non-null for the track's whole life.</summary>
        public ClipEditorContext Editor { get; internal set; }

        public AnimationClipComponent Component { get; internal set; }

        /// <summary>
        /// The component's element of the clip's <c>components</c> list, and the only path an edit
        /// should take. Re-resolved by the window when the list changes, so never cache it.
        /// </summary>
        public SerializedProperty ComponentProperty { get; internal set; }

        public virtual string Title =>
            Component == null ? "(missing)" : ObjectNames.NicifyVariableName(Component.GetType().Name);

        /// <summary>The lane's height in pixels before the user resizes it.</summary>
        public virtual float DefaultLaneHeight => 44f;

        /// <summary>
        /// The height this track's current content needs, or 0 for "no opinion". Applied every
        /// repaint until the user resizes the lane by hand, after which their height wins.
        /// </summary>
        /// <remarks>
        /// <see cref="DefaultLaneHeight"/> is read once, when the row is built, and rows are only
        /// rebuilt when the clip's components list changes shape. A track whose content count is
        /// authored - a lane per tag channel, say - would otherwise gain rows it has no room to
        /// draw.
        /// </remarks>
        public virtual float RequestedLaneHeight => 0f;

        /// <summary>
        /// Gates <see cref="DrawPreviewOverlay"/>, so a track that draws nothing in 3D costs
        /// nothing per preview repaint.
        /// </summary>
        public virtual bool DrawsPreviewOverlay => false;

        /// <summary>
        /// The clip-frame range the track's selection covers, for the timeline's "frame selected".
        /// False when the track has no selection concept, or nothing is selected.
        /// </summary>
        /// <remarks>
        /// The timeline asks rather than the track acting, because <see cref="TrackDrawContext"/>
        /// hands out a copy of the axis - a track cannot zoom or scroll the view itself.
        /// </remarks>
        public virtual bool TryGetSelectionRange(out int firstClipFrame, out int lastClipFrame)
        {
            firstClipFrame = 0;
            lastClipFrame = 0;
            return false;
        }

        /// <summary>
        /// The clip-frame range the track's own annotation covers, for the timeline's "frame all".
        /// False when the track holds nothing to frame.
        /// </summary>
        /// <remarks>
        /// The sibling of <see cref="TryGetSelectionRange"/>, and asked of every visible lane rather
        /// than only the focused one: Home frames the window's content, not one component's.
        /// </remarks>
        public virtual bool TryGetContentRange(out int firstClipFrame, out int lastClipFrame)
        {
            firstClipFrame = 0;
            lastClipFrame = 0;
            return false;
        }

        public virtual void OnEnable() { }

        public virtual void OnDisable() { }

        /// <summary>Draws and handles input for the lane. Called once per IMGUI event.</summary>
        public virtual void DrawTrack(in TrackDrawContext context) { }

        /// <summary>Draws the right-hand panel with plain <c>EditorGUILayout</c>.</summary>
        public virtual void DrawInspector(in TrackInspectorContext context) { }

        /// <summary>
        /// Draws into the preview's off-screen render. <c>Handles</c> will not work here - it draws
        /// into the current GUI camera, not the preview's; use the context's drawer instead.
        /// </summary>
        public virtual void DrawPreviewOverlay(in TrackOverlayContext context) { }
    }
}
