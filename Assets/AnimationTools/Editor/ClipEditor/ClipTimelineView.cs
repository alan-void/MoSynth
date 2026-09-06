using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// The timeline: a frame ruler, the slice the clip is trimmed to, one lane per visible track,
    /// and the playhead over all of it.
    /// </summary>
    /// <remarks>
    /// The axis runs over <em>whole-clip</em> frames rather than the slice, because the slice's own
    /// handles live on this ruler - on a slice-local axis, dragging the start handle would slide
    /// every lane sideways under the cursor. Tracks convert to their own frame space through
    /// <see cref="ClipEditorContext.SliceToClipFrame"/>.
    /// </remarks>
    public sealed class ClipTimelineView
    {
        public const float RulerHeight = 20f;

        private const float ScrollbarWidth = 14f;
        private const float SliceHandleWidth = 7f;
        private const float ZoomPerScrollNotch = 0.08f;

        private static readonly Color RulerBackground = new(0.16f, 0.16f, 0.17f, 1f);
        private static readonly Color LaneBackground = new(0.22f, 0.22f, 0.23f, 1f);
        private static readonly Color MajorTick = new(0.62f, 0.62f, 0.63f, 1f);
        private static readonly Color MinorTick = new(0.36f, 0.36f, 0.37f, 1f);
        private static readonly Color OutsideSlice = new(0f, 0f, 0f, 0.45f);
        private static readonly Color SliceHandle = new(0.42f, 0.62f, 0.86f, 1f);
        private static readonly Color Playhead = new(0.95f, 0.55f, 0.20f, 1f);
        private static readonly Color GutterEdge = new(0.10f, 0.10f, 0.11f, 1f);
        private static readonly Color RowGapFill = new(0.15f, 0.15f, 0.16f, 1f);

        public const float MinHeaderWidth = 120f;
        public const float MaxHeaderWidth = 480f;

        /// <summary>Lane width the gutter may never squeeze below.</summary>
        private const float LaneWidthFloor = 80f;

        /// <summary>Gap between one component's lane and the next, so they read as separate.</summary>
        private const float RowGap = 6f;

        /// <summary>Frame-to-pixel mapping. The window owns the value so it survives a domain reload.</summary>
        public ClipTimeAxis Axis;

        /// <summary>Width of the component gutter. Owned by the window so a resize survives a reload.</summary>
        public float HeaderWidth = 168f;

        public Vector2 Scroll;

        /// <summary>Which row owns the inspector panel, or -1 for none.</summary>
        public int FocusedRowIndex = -1;

        public bool LabelsAsSeconds;

        /// <summary>Set when a lane height or visibility changed and the view state needs saving.</summary>
        public bool ViewStateChanged { get; private set; }

        // Held only so keyboard handling, which runs after the rows are drawn, can ask the focused
        // track about its selection.
        private IReadOnlyList<ClipTrackRow> _rows;

        private int _draggingSliceHandle; // 0 none, -1 start, 1 end
        private int _sliceHandleControlId;
        private int _scrubControlId;
        private int _headerResizeControlId;

        /// <summary>
        /// A keyboard control the lane area claims when clicked, and the whole reason tracks receive
        /// key events at all.
        /// </summary>
        /// <remarks>
        /// This is a UI Toolkit window whose IMGUI runs inside <c>IMGUIContainer</c>s, so keys go to
        /// whichever element the panel has focused. A container only takes that focus once an IMGUI
        /// control inside it owns <c>GUIUtility.keyboardControl</c> - and until this existed, nothing
        /// in the timeline ever claimed it, so focus lived permanently in the inspector pane's
        /// property fields and no lane binding could ever fire.
        /// </remarks>
        private int _laneKeyboardControlId;

        /// <summary>
        /// Draws the whole timeline. <paramref name="rect"/> must be the drawing surface's own rect,
        /// as an <c>IMGUIContainer</c> supplies it: the scroll view below is set up so that a lane
        /// inside it and the ruler outside it share one horizontal coordinate space.
        /// </summary>
        public void Draw(Rect rect, ClipEditorContext editor, IReadOnlyList<ClipTrackRow> rows,
            Action addComponent = null)
        {
            ViewStateChanged = false;
            _rows = rows;

            var frameCount = editor?.ClipFrameCount ?? 0;
            if (frameCount <= 0)
            {
                EditorGUI.DrawRect(rect, LaneBackground);
                GUI.Label(rect, "No clip to show.", CentredLabel());
                return;
            }

            ApplyRequestedHeights(rows);

            HeaderWidth = ClampHeaderWidth(HeaderWidth, rect.width);

            _laneKeyboardControlId = GUIUtility.GetControlID(FocusType.Keyboard);

            var contentHeight = ContentHeight(rows) + ClipTrackHeaderColumn.AddComponentRowHeight;
            var bodyHeight = Mathf.Max(0f, rect.height - RulerHeight);
            var scrollbar = contentHeight > bodyHeight ? ScrollbarWidth : 0f;

            var laneAreaRect = new Rect(
                rect.x + HeaderWidth, rect.y,
                Mathf.Max(1f, rect.width - HeaderWidth - scrollbar), rect.height);

            Axis.Prepare(laneAreaRect, frameCount);

            var rulerRect = new Rect(rect.x, rect.y, rect.width, RulerHeight);
            var bodyRect = new Rect(rect.x, rect.y + RulerHeight, rect.width, bodyHeight);

            DrawRuler(rulerRect, laneAreaRect, editor, frameCount);

            // Before the rows: the grip overlaps the header rect, and the header consumes any click
            // inside it to focus the row.
            HandleHeaderResize(new Rect(rect.x + HeaderWidth - 3f, bodyRect.y, 6f, bodyRect.height),
                rect, editor);

            DrawRows(rect, bodyRect, laneAreaRect, editor, rows, scrollbar, contentHeight, frameCount,
                addComponent);

            var overlayRect = new Rect(laneAreaRect.x, rect.y, laneAreaRect.width, rect.height);
            DrawOutsideSlice(overlayRect, editor, frameCount);
            DrawPlayhead(overlayRect, editor);

            // Drawn last and outside the scroll view, so the component list is divided from the
            // timeline by one unbroken line rather than by each row's own right edge.
            if (Event.current.type == EventType.Repaint)
            {
                EditorGUI.DrawRect(new Rect(rect.x + HeaderWidth - 1f, rect.y, 1f, rect.height),
                    GutterEdge);
            }

            HandleTimelineInput(rect, laneAreaRect, bodyRect, editor, frameCount);
        }

        private static float ContentHeight(IReadOnlyList<ClipTrackRow> rows)
        {
            var total = 0f;
            if (rows == null) return total;

            foreach (var row in rows)
            {
                if (row.Visible) total += row.LaneHeight + RowGap;
            }

            return total;
        }

        /// <summary>
        /// Lets a track whose content grew ask for a taller lane. Runs before the height is
        /// measured for the scroll view, so the request takes effect on the same repaint.
        /// </summary>
        private static void ApplyRequestedHeights(IReadOnlyList<ClipTrackRow> rows)
        {
            if (rows == null) return;

            foreach (var row in rows)
            {
                if (row.Track == null || row.HeightIsUserSet) continue;

                var requested = row.Track.RequestedLaneHeight;
                if (requested > 0f) row.LaneHeight = requested;
            }
        }

        private void DrawRuler(Rect rulerRect, Rect laneAreaRect, ClipEditorContext editor, int frameCount)
        {
            if (Event.current.type == EventType.Repaint)
            {
                EditorGUI.DrawRect(rulerRect, RulerBackground);

                ClipTimelineTicks.Choose(Axis.pixelsPerFrame, out var major, out var minor);
                var first = Axis.FirstVisibleFrame(frameCount);
                var last = Axis.LastVisibleFrame(frameCount);

                if (minor > 0) DrawTicks(rulerRect, laneAreaRect, first, last, minor, 4f, MinorTick, false, editor);
                DrawTicks(rulerRect, laneAreaRect, first, last, major, 8f, MajorTick, true, editor);
            }

            DrawSliceHandles(rulerRect, laneAreaRect, editor, frameCount);
            HandleRulerScrub(rulerRect, laneAreaRect, editor, frameCount);
        }

        private void DrawTicks(Rect rulerRect, Rect laneAreaRect, int firstFrame, int lastFrame, int step,
            float height, Color colour, bool labelled, ClipEditorContext editor)
        {
            var labelStyle = new GUIStyle(EditorStyles.miniLabel) { normal = { textColor = colour } };

            for (var frame = ClipTimelineTicks.FirstTickAtOrAfter(firstFrame, step);
                 frame <= lastFrame;
                 frame += step)
            {
                var x = Mathf.Round(Axis.FrameToX(frame));
                if (x < laneAreaRect.xMin || x > laneAreaRect.xMax) continue;

                EditorGUI.DrawRect(new Rect(x, rulerRect.yMax - height, 1f, height), colour);

                if (!labelled) continue;

                var label = ClipTimelineTicks.Label(frame, editor.FrameTime, LabelsAsSeconds);
                GUI.Label(new Rect(x + 2f, rulerRect.y - 1f, 60f, rulerRect.height - 4f), label, labelStyle);
            }
        }

        /// <summary>
        /// The two grips that trim the clip. Dragging the end handle can push footfall anchors out
        /// of range, and the asset deletes those on commit, so the count at risk is shown live.
        /// </summary>
        private void DrawSliceHandles(Rect rulerRect, Rect laneAreaRect, ClipEditorContext editor,
            int frameCount)
        {
            var startX = Axis.FrameToX(editor.StartFrame);
            var endX = Axis.FrameToX(editor.EndFrame);

            if (Event.current.type == EventType.Repaint)
            {
                DrawHandle(startX, rulerRect, laneAreaRect, true);
                DrawHandle(endX, rulerRect, laneAreaRect, false);
            }

            _sliceHandleControlId = GUIUtility.GetControlID(FocusType.Passive);
            var e = Event.current;

            switch (e.GetTypeForControl(_sliceHandleControlId))
            {
                case EventType.MouseDown when e.button == 0 && rulerRect.Contains(e.mousePosition):
                    if (Mathf.Abs(e.mousePosition.x - startX) <= SliceHandleWidth) _draggingSliceHandle = -1;
                    else if (Mathf.Abs(e.mousePosition.x - endX) <= SliceHandleWidth) _draggingSliceHandle = 1;
                    else return;

                    GUIUtility.hotControl = _sliceHandleControlId;
                    e.Use();
                    return;

                case EventType.MouseDrag when GUIUtility.hotControl == _sliceHandleControlId:
                    DragSliceHandle(e.mousePosition.x, editor, frameCount);
                    e.Use();
                    return;

                case EventType.MouseUp when GUIUtility.hotControl == _sliceHandleControlId:
                    GUIUtility.hotControl = 0;
                    _draggingSliceHandle = 0;
                    editor.Commit();
                    e.Use();
                    return;
            }
        }

        private static void DrawHandle(float x, Rect rulerRect, Rect laneAreaRect, bool pointsRight)
        {
            if (x < laneAreaRect.xMin - SliceHandleWidth || x > laneAreaRect.xMax + SliceHandleWidth) return;

            var body = pointsRight
                ? new Rect(x, rulerRect.y, SliceHandleWidth, rulerRect.height)
                : new Rect(x - SliceHandleWidth, rulerRect.y, SliceHandleWidth, rulerRect.height);

            EditorGUI.DrawRect(body, SliceHandle);
            EditorGUIUtility.AddCursorRect(body, MouseCursor.ResizeHorizontal);
        }

        private void DragSliceHandle(float mouseX, ClipEditorContext editor, int frameCount)
        {
            var frame = Mathf.Clamp(Mathf.RoundToInt(Axis.XToFrame(mouseX)), 0, frameCount);

            if (_draggingSliceHandle < 0)
            {
                editor.StartFrameProperty.intValue = Mathf.Min(frame, editor.EndFrame);
            }
            else
            {
                editor.EndFrameProperty.intValue = Mathf.Max(frame, editor.StartFrame);
            }

            // Applied without committing so the drag stays live without re-baking the clip on every
            // mouse-move; the commit happens once on mouse-up.
            editor.SerializedClip.ApplyModifiedPropertiesWithoutUndo();
            editor.Repaint();
        }

        private void HandleRulerScrub(Rect rulerRect, Rect laneAreaRect, ClipEditorContext editor,
            int frameCount)
        {
            _scrubControlId = GUIUtility.GetControlID(FocusType.Passive);
            var e = Event.current;

            var inScrubArea = rulerRect.Contains(e.mousePosition) &&
                              e.mousePosition.x >= laneAreaRect.xMin;

            switch (e.GetTypeForControl(_scrubControlId))
            {
                case EventType.MouseDown when e.button == 0 && inScrubArea:
                    GUIUtility.hotControl = _scrubControlId;
                    ScrubTo(e.mousePosition.x, editor, frameCount);
                    e.Use();
                    return;

                case EventType.MouseDrag when GUIUtility.hotControl == _scrubControlId:
                    ScrubTo(e.mousePosition.x, editor, frameCount);
                    e.Use();
                    return;

                case EventType.MouseUp when GUIUtility.hotControl == _scrubControlId:
                    GUIUtility.hotControl = 0;
                    e.Use();
                    return;
            }
        }

        private void ScrubTo(float x, ClipEditorContext editor, int frameCount) =>
            editor.SeekToClipFrame(Mathf.Clamp(Mathf.RoundToInt(Axis.XToFrame(x)), 0, frameCount - 1));

        private void DrawRows(Rect rect, Rect bodyRect, Rect laneAreaRect, ClipEditorContext editor,
            IReadOnlyList<ClipTrackRow> rows, float scrollbar, float contentHeight, int frameCount,
            Action addComponent)
        {
            if (Event.current.type == EventType.Repaint) EditorGUI.DrawRect(bodyRect, LaneBackground);
            rows ??= System.Array.Empty<ClipTrackRow>();

            ClaimKeyboardFocus(laneAreaRect, bodyRect);

            // Content x starts at rect.x, so a lane drawn in here and the ruler drawn outside share
            // the same horizontal coordinates and cannot drift apart.
            var contentWidth = rect.width - scrollbar;
            var contentRect = new Rect(rect.x, 0f, contentWidth, contentHeight);
            Scroll = GUI.BeginScrollView(bodyRect, Scroll, contentRect, false, false);

            var firstVisible = Axis.FirstVisibleFrame(frameCount);
            var lastVisible = Axis.LastVisibleFrame(frameCount);

            var y = 0f;
            for (var i = 0; i < rows.Count; i++)
            {
                var row = rows[i];
                if (!row.Visible) continue;

                var headerRect = new Rect(rect.x, y, HeaderWidth, row.LaneHeight);
                var laneRect = new Rect(laneAreaRect.x, y, laneAreaRect.width, row.LaneHeight);

                // Clicking a lane focuses its row, so the inspector follows what you are editing and
                // the track's own key bindings become reachable. Deliberately not consumed - the
                // track still needs this click.
                if (Event.current.type == EventType.MouseDown && Event.current.button == 0 &&
                    laneRect.Contains(Event.current.mousePosition))
                {
                    FocusedRowIndex = i;
                }

                var isFocused = i == FocusedRowIndex;
                var index = i;
                ViewStateChanged |= ClipTrackHeaderColumn.Draw(headerRect, row, isFocused,
                    () => FocusedRowIndex = index);

                if (row.Track != null)
                {
                    row.Track.DrawTrack(new TrackDrawContext(laneRect, Axis, firstVisible, lastVisible,
                        editor.PlayheadClipFrame, isFocused, editor));
                }

                if (Event.current.type == EventType.Repaint)
                {
                    EditorGUI.DrawRect(new Rect(rect.x, y + row.LaneHeight, contentWidth, RowGap),
                        RowGapFill);
                }

                y += row.LaneHeight + RowGap;
            }

            ClipTrackHeaderColumn.DrawAddComponent(
                new Rect(rect.x, y, HeaderWidth, ClipTrackHeaderColumn.AddComponentRowHeight),
                addComponent);

            if (rows.Count == 0 && Event.current.type == EventType.Repaint)
            {
                GUI.Label(new Rect(laneAreaRect.x, 0f, laneAreaRect.width,
                        ClipTrackHeaderColumn.AddComponentRowHeight),
                    "This clip has no components.", CentredLabel());
            }

            GUI.EndScrollView();
        }

        /// <summary>
        /// Parks IMGUI keyboard focus on the lane area when it is clicked, which is what lets this
        /// container receive key events at all. Never consumes the click.
        /// </summary>
        private void ClaimKeyboardFocus(Rect laneAreaRect, Rect bodyRect)
        {
            var e = Event.current;
            if (e.type != EventType.MouseDown || e.button != 0) return;
            if (!laneAreaRect.Contains(e.mousePosition) || !bodyRect.Contains(e.mousePosition)) return;

            GUIUtility.keyboardControl = _laneKeyboardControlId;
        }

        /// <summary>
        /// Keeps the gutter usable and leaves room for the lanes beside it.
        /// </summary>
        /// <remarks>
        /// The available width is ignored until the pane has actually been measured. During a
        /// <see cref="EventType.Layout"/> pass <c>GUILayoutUtility.GetRect</c> hands back a
        /// degenerate rect, and clamping against that would reset the user's width on every pass.
        /// </remarks>
        private static float ClampHeaderWidth(float width, float availableWidth)
        {
            var ceiling = availableWidth > MinHeaderWidth + LaneWidthFloor
                ? Mathf.Min(MaxHeaderWidth, availableWidth - LaneWidthFloor)
                : MaxHeaderWidth;

            return Mathf.Clamp(width, MinHeaderWidth, ceiling);
        }

        /// <summary>Drags the boundary between the component gutter and the lanes.</summary>
        private void HandleHeaderResize(Rect grip, Rect rect, ClipEditorContext editor)
        {
            EditorGUIUtility.AddCursorRect(grip, MouseCursor.ResizeHorizontal);

            _headerResizeControlId = GUIUtility.GetControlID(FocusType.Passive);
            var e = Event.current;

            switch (e.GetTypeForControl(_headerResizeControlId))
            {
                case EventType.MouseDown when e.button == 0 && grip.Contains(e.mousePosition):
                    GUIUtility.hotControl = _headerResizeControlId;
                    e.Use();
                    return;

                case EventType.MouseDrag when GUIUtility.hotControl == _headerResizeControlId:
                    HeaderWidth = ClampHeaderWidth(e.mousePosition.x - rect.x, rect.width);
                    ViewStateChanged = true;
                    e.Use();
                    editor.Repaint();
                    return;

                case EventType.MouseUp when GUIUtility.hotControl == _headerResizeControlId:
                    GUIUtility.hotControl = 0;
                    e.Use();
                    return;
            }
        }

        private void DrawOutsideSlice(Rect overlayRect, ClipEditorContext editor, int frameCount)
        {
            if (Event.current.type != EventType.Repaint) return;

            var startX = Mathf.Clamp(Axis.FrameToX(editor.StartFrame), overlayRect.xMin, overlayRect.xMax);
            var endX = Mathf.Clamp(Axis.FrameToX(editor.EndFrame), overlayRect.xMin, overlayRect.xMax);

            if (startX > overlayRect.xMin)
            {
                EditorGUI.DrawRect(
                    new Rect(overlayRect.x, overlayRect.y, startX - overlayRect.xMin, overlayRect.height),
                    OutsideSlice);
            }

            if (endX < overlayRect.xMax)
            {
                EditorGUI.DrawRect(
                    new Rect(endX, overlayRect.y, overlayRect.xMax - endX, overlayRect.height),
                    OutsideSlice);
            }
        }

        private void DrawPlayhead(Rect overlayRect, ClipEditorContext editor)
        {
            if (Event.current.type != EventType.Repaint) return;

            var x = Mathf.Round(Axis.FrameToX(editor.PlayheadClipFrame));
            if (x < overlayRect.xMin || x > overlayRect.xMax) return;

            EditorGUI.DrawRect(new Rect(x, overlayRect.y, 1f, overlayRect.height), Playhead);
            EditorGUI.DrawRect(new Rect(x - 3f, overlayRect.y, 7f, 4f), Playhead);
        }

        private void HandleTimelineInput(Rect rect, Rect laneAreaRect, Rect bodyRect,
            ClipEditorContext editor, int frameCount)
        {
            var e = Event.current;
            var overLanes = laneAreaRect.Contains(e.mousePosition);

            switch (e.type)
            {
                case EventType.ScrollWheel when overLanes:
                    if (e.control || e.command)
                    {
                        Scroll.y = Mathf.Max(0f, Scroll.y + e.delta.y * 10f);
                    }
                    else
                    {
                        Axis.ZoomAt(e.mousePosition.x, Mathf.Exp(-e.delta.y * ZoomPerScrollNotch), frameCount);
                    }

                    e.Use();
                    editor.Repaint();
                    return;

                case EventType.MouseDrag when overLanes && (e.button == 2 || e.alt):
                    Axis.PanPixels(-e.delta.x, frameCount);
                    e.Use();
                    editor.Repaint();
                    return;

                // Anything a track did not claim scrubs, so clicking a lane always moves the pose.
                case EventType.MouseDown when e.button == 0 && overLanes && bodyRect.Contains(e.mousePosition):
                    ScrubTo(e.mousePosition.x, editor, frameCount);
                    e.Use();
                    return;

                case EventType.KeyDown when e.keyCode == KeyCode.F:
                    Axis.FrameRange(editor.StartFrame, Mathf.Max(editor.StartFrame, editor.EndFrame - 1),
                        frameCount);
                    e.Use();
                    editor.Repaint();
                    return;

                case EventType.KeyDown when e.keyCode == KeyCode.Home:
                    Axis.FrameRange(0, frameCount - 1, frameCount);
                    e.Use();
                    editor.Repaint();
                    return;

                // Home and "." rather than A, because A is select-all in a keyframe lane. Both are
                // Blender's own view keys, so the window moves towards that keymap, not away.
                case EventType.KeyDown when e.keyCode is KeyCode.Period or KeyCode.KeypadPeriod:
                    if (TryGetFocusedSelection(out var firstSelected, out var lastSelected))
                    {
                        Axis.FrameRange(firstSelected, lastSelected, frameCount);
                    }
                    else
                    {
                        Axis.FrameRange(0, frameCount - 1, frameCount);
                    }

                    e.Use();
                    editor.Repaint();
                    return;
            }
        }

        private bool TryGetFocusedSelection(out int firstClipFrame, out int lastClipFrame)
        {
            firstClipFrame = 0;
            lastClipFrame = 0;

            if (_rows == null || FocusedRowIndex < 0 || FocusedRowIndex >= _rows.Count) return false;

            var track = _rows[FocusedRowIndex].Track;
            return track != null && track.TryGetSelectionRange(out firstClipFrame, out lastClipFrame);
        }

        private static GUIStyle CentredLabel() =>
            new(EditorStyles.centeredGreyMiniLabel) { alignment = TextAnchor.MiddleCenter };
    }
}
