using System.Collections.Generic;
using GameplayTags;
using GameplayTags.Editor;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// The clip editor's view of an <see cref="AnimationTagComponent"/>: a dope-sheet row per tag
    /// channel, with the keys that switch it on and off as draggable handles.
    /// </summary>
    /// <remarks>
    /// The keymap is Blender's, via <see cref="TimelineKeymap"/> - box select, <c>G</c> to move,
    /// <c>S</c> to scale, numbers to type an exact value. None of it is discoverable, which is why
    /// the inspector carries the list.
    /// <para>
    /// A running mode only ever previews. Nothing reaches the asset until it is confirmed, so an
    /// abandoned drag costs neither an undo entry nor the clip re-bake that a commit triggers.
    /// </para>
    /// </remarks>
    [ClipComponentTrack(typeof(AnimationTagComponent))]
    public sealed class AnimationTagTrack : AnimationClipComponentTrack
    {
        private const float RowHeight = 20f;
        private const float RowPadding = 2f;
        private const float BarInset = 5f;
        private const float DiamondSize = 9f;

        private static readonly Color RowBackground = new(0.17f, 0.17f, 0.18f, 1f);
        private static readonly Color RowBackgroundAlt = new(0.20f, 0.20f, 0.21f, 1f);
        private static readonly Color SelectionOutline = new(1f, 0.85f, 0.3f, 1f);
        private static readonly Color BoxFill = new(0.4f, 0.6f, 0.9f, 0.22f);
        private static readonly Color MatchBand = new(1f, 0.78f, 0.25f, 0.30f);
        private static readonly Color UntaggedBar = new(0.45f, 0.45f, 0.47f, 1f);

        private readonly TimelineKeySelection _selection = new();
        private readonly List<AnimationTagging.TagSpan> _spans = new();
        private readonly List<int> _rowFrames = new();
        private readonly List<int> _scratch = new();
        private readonly List<AnimationClipSegment> _queryMatches = new();
        private readonly HashSet<int> _previewSelected = new();

        private SerializedObject _queryScratch;
        private int _controlId;
        private int _activeRow;
        private bool _queryRun;

        private AnimationTagComponent Tags => Component as AnimationTagComponent;

        private List<AnimationTagging.TagChannel> Channels => Tags?.channels;

        public override float DefaultLaneHeight => LaneHeightFor(Channels?.Count ?? 0);

        public override float RequestedLaneHeight => LaneHeightFor(Channels?.Count ?? 0);

        private static float LaneHeightFor(int channels) =>
            Mathf.Max(RowHeight, channels * RowHeight) + RowPadding * 2f;

        public override void OnDisable()
        {
            if (Editor != null && Editor.HasModal(this)) Editor.CancelModal();
        }

        public override bool TryGetSelectionRange(out int firstClipFrame, out int lastClipFrame)
        {
            firstClipFrame = 0;
            lastClipFrame = 0;
            if (_selection.Count == 0 || Editor == null) return false;

            var lowest = int.MaxValue;
            var highest = int.MinValue;
            foreach (var key in _selection)
            {
                lowest = Mathf.Min(lowest, key.Frame);
                highest = Mathf.Max(highest, key.Frame);
            }

            firstClipFrame = Editor.SliceToClipFrame(lowest);
            lastClipFrame = Editor.SliceToClipFrame(highest);
            return true;
        }

        // ---- lane ----

        public override void DrawTrack(in TrackDrawContext context)
        {
            var channels = Channels;
            if (channels == null || context.Editor.SliceFrameCount <= 0) return;

            // Allocated unconditionally: an id handed out only when a key is pressed would shift
            // every id after it between Layout and Repaint.
            _controlId = GUIUtility.GetControlID(FocusType.Passive);

            var modal = context.Editor.HasModal(this) ? context.Editor.Modal : null;

            if (Event.current.type == EventType.Repaint)
            {
                DrawQueryMatches(context);

                for (var row = 0; row < channels.Count; row++)
                {
                    DrawRow(context, row, channels[row], RowRect(context.LaneRect, row), modal);
                }

                DrawModalOverlay(context, modal);
            }

            HandleInput(context, channels);
        }

        private static Rect RowRect(Rect laneRect, int row) =>
            new(laneRect.x, laneRect.y + RowPadding + row * RowHeight, laneRect.width, RowHeight);

        private void DrawRow(in TrackDrawContext context, int row, AnimationTagging.TagChannel channel,
            Rect rowRect, TimelineModalOperator modal)
        {
            EditorGUI.DrawRect(rowRect, row % 2 == 0 ? RowBackground : RowBackgroundAlt);

            var toggles = PreviewToggles(context, row, channel, modal);
            var colour = ChannelColour(channel);

            AnimationTagging.Spans(toggles, context.Editor.SliceFrameCount, _spans);
            foreach (var span in _spans)
            {
                var left = SliceFrameToX(context, span.StartFrame);
                var right = SliceFrameToX(context, span.EndFrame);

                var bar = Rect.MinMaxRect(Mathf.Max(left, rowRect.xMin), rowRect.y + BarInset,
                    Mathf.Min(right, rowRect.xMax), rowRect.yMax - BarInset);

                if (bar.width > 0f) EditorGUI.DrawRect(bar, colour);
            }

            DrawChannelLabel(rowRect, channel, toggles);

            // Selection is keyed by a key's stored frame, so while a mode previews the keys
            // somewhere else the highlight has to be looked up by where they are being drawn.
            PreviewSelected(context, row, modal);

            foreach (var frame in toggles)
            {
                var clipFrame = context.Editor.SliceToClipFrame(frame);
                if (clipFrame < context.FirstVisibleClipFrame || clipFrame > context.LastVisibleClipFrame)
                {
                    continue;
                }

                DrawDiamond(new Vector2(context.Axis.FrameToX(clipFrame), rowRect.center.y),
                    _previewSelected.Contains(frame), colour);
            }
        }

        /// <summary>Where this row's selected keys are currently drawn, previewed mode included.</summary>
        private void PreviewSelected(in TrackDrawContext context, int row, TimelineModalOperator modal)
        {
            _selection.FramesIn(row, _rowFrames);

            if (modal != null)
            {
                var frameCount = context.Editor.SliceFrameCount;
                if (modal.Kind == TimelineModalKind.Grab)
                {
                    Shift(_rowFrames,
                        AnimationTagEdits.ClampDelta(_rowFrames, modal.FrameDelta, frameCount));
                }
                else if (modal.Kind == TimelineModalKind.Scale)
                {
                    Scaled(_rowFrames, modal.PivotFrame, modal.ScaleFactor);
                }
            }

            _previewSelected.Clear();
            foreach (var frame in _rowFrames) _previewSelected.Add(frame);
        }

        /// <summary>
        /// The keys as they should be drawn: the stored ones, or where a running mode would put
        /// them. Preview only - nothing here reaches the asset.
        /// </summary>
        private List<int> PreviewToggles(in TrackDrawContext context, int row,
            AnimationTagging.TagChannel channel, TimelineModalOperator modal)
        {
            if (modal == null || !_selection.HasAnyIn(row)) return channel.toggles;

            _selection.FramesIn(row, _rowFrames);
            var frameCount = context.Editor.SliceFrameCount;

            return modal.Kind switch
            {
                TimelineModalKind.Grab => AnimationTagEdits.Move(channel.toggles, _rowFrames,
                    AnimationTagEdits.ClampDelta(_rowFrames, modal.FrameDelta, frameCount), frameCount),
                TimelineModalKind.Scale => AnimationTagEdits.Scale(channel.toggles, _rowFrames,
                    modal.PivotFrame, modal.ScaleFactor, frameCount),
                _ => channel.toggles
            };
        }

        private static void DrawChannelLabel(Rect rowRect, AnimationTagging.TagChannel channel,
            IReadOnlyList<int> toggles)
        {
            var label = channel.tag == null ? "(no tag)" : TagLabel(channel.tag);
            var style = new GUIStyle(EditorStyles.miniLabel)
            {
                normal = { textColor = new Color(0.88f, 0.88f, 0.9f) },
                clipping = TextClipping.Clip
            };

            // Pinned to the left edge rather than to the first span, so a channel whose keys are all
            // scrolled off screen still says which tag the empty row belongs to.
            var text = new Rect(rowRect.x + 4f, rowRect.y, 220f, rowRect.height);
            GUI.Label(text, toggles.Count == 0 ? $"{label} — double-click to add a key" : label, style);
        }

        private void DrawQueryMatches(in TrackDrawContext context)
        {
            if (!_queryRun) return;

            foreach (var match in _queryMatches)
            {
                if (match.Clip != context.Editor.Clip) continue;

                var left = SliceFrameToX(context, match.StartFrame);
                var right = SliceFrameToX(context, match.EndFrame);

                var band = Rect.MinMaxRect(Mathf.Max(left, context.LaneRect.xMin), context.LaneRect.y,
                    Mathf.Min(right, context.LaneRect.xMax), context.LaneRect.yMax);

                if (band.width > 0f) EditorGUI.DrawRect(band, MatchBand);
            }
        }

        private void DrawModalOverlay(in TrackDrawContext context, TimelineModalOperator modal)
        {
            if (modal == null) return;

            if (modal.Kind == TimelineModalKind.BoxSelect)
            {
                EditorGUI.DrawRect(modal.BoxRect, BoxFill);
                return;
            }

            var status = modal.StatusText;
            if (string.IsNullOrEmpty(status)) return;

            var box = new Rect(context.LaneRect.x + 6f, context.LaneRect.y + 2f, 180f, 16f);
            EditorGUI.DrawRect(box, new Color(0f, 0f, 0f, 0.65f));
            GUI.Label(new Rect(box.x + 4f, box.y, box.width - 8f, box.height), status,
                EditorStyles.miniLabel);
        }

        private static void DrawDiamond(Vector2 centre, bool selected, Color colour)
        {
            var half = DiamondSize * 0.5f;
            var matrix = GUI.matrix;
            GUIUtility.RotateAroundPivot(45f, centre);

            if (selected)
            {
                EditorGUI.DrawRect(new Rect(centre.x - half - 1.5f, centre.y - half - 1.5f,
                    DiamondSize + 3f, DiamondSize + 3f), SelectionOutline);
            }

            EditorGUI.DrawRect(new Rect(centre.x - half, centre.y - half, DiamondSize, DiamondSize),
                colour * 1.25f);

            GUI.matrix = matrix;
        }

        /// <summary>
        /// A stable hue per tag, so channels are told apart with nothing to author. Falls back to the
        /// asset name because <c>TagFullName</c> is empty until the tag's root has been loaded.
        /// </summary>
        private static Color ChannelColour(AnimationTagging.TagChannel channel)
        {
            if (channel.tag == null) return UntaggedBar;

            var hash = TagLabel(channel.tag).GetHashCode();
            return Color.HSVToRGB((hash & 0xFFFF) / 65535f, 0.45f, 0.62f);
        }

        private static string TagLabel(GameplayTagSO tag) =>
            string.IsNullOrEmpty(tag.TagFullName) ? tag.name : tag.TagFullName;

        private static float SliceFrameToX(in TrackDrawContext context, int sliceFrame) =>
            context.Axis.FrameToX(context.Editor.SliceToClipFrame(sliceFrame));

        /// <summary>
        /// The slice-frame-to-pixel mapping as a closure. Built here because the hit testers take a
        /// delegate, and a readonly ref parameter cannot be captured by one.
        /// </summary>
        private static System.Func<int, float> FrameToPixels(in TrackDrawContext context)
        {
            var axis = context.Axis;
            var editor = context.Editor;
            return sliceFrame => axis.FrameToX(editor.SliceToClipFrame(sliceFrame));
        }

        // ---- input ----

        private void HandleInput(in TrackDrawContext context, List<AnimationTagging.TagChannel> channels)
        {
            var editor = context.Editor;
            var e = Event.current;

            if (editor.HasModal(this))
            {
                var result = editor.Modal.HandleEvent(e);
                if (result == TimelineModalResult.Confirmed) ConfirmModal(context, channels);
                if (result is TimelineModalResult.Confirmed or TimelineModalResult.Cancelled)
                {
                    editor.CancelModal();
                }

                if (result != TimelineModalResult.None) editor.Repaint();
                return;
            }

            var laneRect = context.LaneRect;

            // Key events go through the raw event type, never GetTypeForControl: that method
            // returns Ignore for a key unless GUIUtility.keyboardControl is this control, and the
            // lane's control is Passive so it never can be. Filtering clicks by it is still right -
            // they must respect hotControl.
            if (e.type == EventType.KeyDown && context.IsFocused &&
                !EditorGUIUtility.editingTextField)
            {
                HandleKey(context, channels, e);
                return;
            }

            switch (e.GetTypeForControl(_controlId))
            {
                case EventType.MouseDown when e.button == 0 && laneRect.Contains(e.mousePosition):
                    HandleClick(context, channels, e);
                    return;

                case EventType.ContextClick when laneRect.Contains(e.mousePosition):
                    ShowContextMenu(context, channels, e.mousePosition);
                    e.Use();
                    return;
            }
        }

        private void HandleClick(in TrackDrawContext context, List<AnimationTagging.TagChannel> channels,
            Event e)
        {
            var row = RowAt(context.LaneRect, e.mousePosition.y, channels.Count);
            var picked = -1;

            if (row >= 0)
            {
                _activeRow = row;
                picked = TimelineKeyHitTester.Pick(channels[row].toggles, e.mousePosition.x,
                    FrameToPixels(context));
            }

            if (picked < 0)
            {
                // Double-click makes a key, as it does on a gait lane. Without it the only way to
                // create one is the I binding, and a lane with no keys is a lane where selecting,
                // moving and deleting all silently do nothing.
                if (e.clickCount == 2 && row >= 0)
                {
                    InsertKeyAt(context, row,
                        context.Editor.ClipToSliceFrame(
                            Mathf.RoundToInt(context.Axis.XToFrame(e.mousePosition.x))));

                    e.Use();
                    return;
                }

                // Blender's tweak-select: a drag from empty space is a box. Started from anywhere in
                // the lane, not only from a channel row - below the last row and a component with no
                // channels at all are exactly where you reach for a box.
                BeginModal(context, TimelineModalKind.BoxSelect, e, extend: e.shift);
                return;
            }

            if (e.shift) _selection.Toggle(row, picked);
            else _selection.SetTo(row, picked);

            context.Editor.SeekToClipFrame(context.Editor.SliceToClipFrame(picked));
            e.Use();
        }

        private void HandleKey(in TrackDrawContext context, List<AnimationTagging.TagChannel> channels,
            Event e)
        {
            var editor = context.Editor;

            var action = TimelineKeymap.Resolve(e);

            switch (action)
            {
                case TimelineKeyAction.SelectAll:
                    for (var row = 0; row < channels.Count; row++)
                    {
                        foreach (var frame in channels[row].toggles) _selection.Add(row, frame);
                    }

                    break;

                case TimelineKeyAction.DeselectAll:
                    _selection.Clear();
                    break;

                case TimelineKeyAction.InvertSelection:
                    InvertSelection(channels);
                    break;

                case TimelineKeyAction.BeginBoxSelect:
                    BeginModal(context, TimelineModalKind.BoxSelect, e, extend: e.shift);
                    break;

                case TimelineKeyAction.BeginGrab when _selection.Count > 0:
                    BeginModal(context, TimelineModalKind.Grab, e);
                    break;

                case TimelineKeyAction.BeginScale when _selection.Count > 0:
                    BeginModal(context, TimelineModalKind.Scale, e);
                    break;

                case TimelineKeyAction.Duplicate when _selection.Count > 0:
                    Duplicate(context, channels);
                    break;

                case TimelineKeyAction.Delete when _selection.Count > 0:
                    ApplyToSelectedRows(context, channels, "Delete tag keys",
                        (toggles, frames, frameCount) =>
                            AnimationTagEdits.Delete(toggles, frames, frameCount));
                    _selection.Clear();
                    break;

                case TimelineKeyAction.InsertKey:
                    InsertKeyAtPlayhead(context, channels);
                    break;

                case TimelineKeyAction.StepBack:
                    editor.SeekToClipFrame(editor.PlayheadClipFrame - 1);
                    break;

                case TimelineKeyAction.StepForward:
                    editor.SeekToClipFrame(editor.PlayheadClipFrame + 1);
                    break;

                case TimelineKeyAction.PreviousKey:
                case TimelineKeyAction.NextKey:
                    JumpToKey(context, channels, action == TimelineKeyAction.NextKey);
                    break;

                // Home and "." belong to the timeline, which owns the axis a track only sees a copy
                // of. Left unclaimed on purpose.
                default:
                    return;
            }

            e.Use();
            editor.Repaint();
        }

        private void InvertSelection(List<AnimationTagging.TagChannel> channels)
        {
            var inverted = new TimelineKeySelection();
            for (var row = 0; row < channels.Count; row++)
            {
                foreach (var frame in channels[row].toggles)
                {
                    if (!_selection.Contains(row, frame)) inverted.Add(row, frame);
                }
            }

            _selection.Clear();
            foreach (var key in inverted) _selection.Add(key.Row, key.Frame);
        }

        private void BeginModal(in TrackDrawContext context, TimelineModalKind kind, Event e,
            bool extend = false)
        {
            var editor = context.Editor;
            var pivotX = context.Axis.FrameToX(editor.PlayheadClipFrame);

            editor.ClaimModal(this).Begin(kind, _controlId, e.mousePosition,
                context.Axis.pixelsPerFrame, editor.ClipToSliceFrame(editor.PlayheadClipFrame),
                pivotX, extend);

            e.Use();
            editor.Repaint();
        }

        private void ConfirmModal(in TrackDrawContext context, List<AnimationTagging.TagChannel> channels)
        {
            var modal = context.Editor.Modal;

            switch (modal.Kind)
            {
                case TimelineModalKind.BoxSelect:
                    ConfirmBoxSelect(context, channels, modal);
                    return;

                case TimelineModalKind.Grab:
                {
                    var delta = modal.FrameDelta;
                    ApplyToSelectedRows(context, channels, "Move tag keys",
                        (toggles, frames, frameCount) => AnimationTagEdits.Move(toggles, frames,
                            AnimationTagEdits.ClampDelta(frames, delta, frameCount), frameCount),
                        (frames, frameCount) => Shift(frames,
                            AnimationTagEdits.ClampDelta(frames, delta, frameCount)));
                    return;
                }

                case TimelineModalKind.Scale:
                {
                    var pivot = modal.PivotFrame;
                    var factor = modal.ScaleFactor;
                    ApplyToSelectedRows(context, channels, "Scale tag keys",
                        (toggles, frames, frameCount) =>
                            AnimationTagEdits.Scale(toggles, frames, pivot, factor, frameCount),
                        (frames, _) => Scaled(frames, pivot, factor));
                    return;
                }
            }
        }

        private void ConfirmBoxSelect(in TrackDrawContext context,
            List<AnimationTagging.TagChannel> channels, TimelineModalOperator modal)
        {
            var box = modal.BoxRect;
            if (!modal.Extend) _selection.Clear();

            var toX = FrameToPixels(context);

            for (var row = 0; row < channels.Count; row++)
            {
                var rowRect = RowRect(context.LaneRect, row);
                if (rowRect.yMax < box.yMin || rowRect.yMin > box.yMax) continue;

                TimelineKeyHitTester.PickRange(channels[row].toggles, box.xMin, box.xMax, toX,
                    _scratch);

                foreach (var frame in _scratch) _selection.Add(row, frame);
            }
        }

        private void Duplicate(in TrackDrawContext context, List<AnimationTagging.TagChannel> channels)
        {
            // Duplicated in place and then grabbed, so the copies are what you drag - Blender's
            // Shift+D. One frame of offset keeps them from cancelling against their originals.
            ApplyToSelectedRows(context, channels, "Duplicate tag keys",
                (toggles, frames, frameCount) =>
                    AnimationTagEdits.Duplicate(toggles, frames, 1, frameCount),
                (frames, _) => Shift(frames, 1));

            BeginModal(context, TimelineModalKind.Grab, Event.current);
        }

        private void InsertKeyAtPlayhead(in TrackDrawContext context,
            List<AnimationTagging.TagChannel> channels)
        {
            if (channels.Count == 0) return;

            InsertKeyAt(context, Mathf.Clamp(_activeRow, 0, channels.Count - 1),
                context.Editor.ClipToSliceFrame(context.Editor.PlayheadClipFrame));
        }

        /// <summary>Adds a key to one channel and selects it, so it can be moved straight away.</summary>
        private void InsertKeyAt(in TrackDrawContext context, int row, int sliceFrame)
        {
            var channels = Channels;
            if (channels == null || row < 0 || row >= channels.Count) return;

            var editor = context.Editor;
            if (sliceFrame < 0 || sliceFrame >= editor.SliceFrameCount) return;

            WriteChannel(editor, row,
                AnimationTagEdits.Insert(channels[row].toggles, sliceFrame, editor.SliceFrameCount));

            Undo.SetCurrentGroupName("Insert tag key");
            editor.Commit();

            _activeRow = row;
            _selection.SetTo(row, sliceFrame);
        }

        private void JumpToKey(in TrackDrawContext context, List<AnimationTagging.TagChannel> channels,
            bool forward)
        {
            var editor = context.Editor;
            var from = editor.ClipToSliceFrame(editor.PlayheadClipFrame);

            var best = -1;
            foreach (var channel in channels)
            {
                foreach (var frame in channel.toggles)
                {
                    if (forward ? frame <= from : frame >= from) continue;
                    if (best >= 0 && (forward ? frame >= best : frame <= best)) continue;

                    best = frame;
                }
            }

            if (best >= 0) editor.SeekToClipFrame(editor.SliceToClipFrame(best));
        }

        /// <summary>
        /// The channel row under a lane-space y, or -1 for none - the padding, past the last row, or
        /// a component with no channels at all. Those are the places a box select starts, so the
        /// caller must treat -1 as "empty space", never as "ignore the click".
        /// </summary>
        public static int RowAt(Rect laneRect, float y, int rowCount)
        {
            var row = Mathf.FloorToInt((y - laneRect.y - RowPadding) / RowHeight);
            return row >= 0 && row < rowCount ? row : -1;
        }

        // ---- writing ----

        private delegate List<int> ChannelEdit(IReadOnlyList<int> toggles, IReadOnlyList<int> selected,
            int frameCount);

        private delegate void SelectionEdit(List<int> selected, int frameCount);

        /// <summary>
        /// Applies one edit to every row holding a selection, writes them all, and commits once.
        /// </summary>
        /// <remarks>
        /// The selection is re-derived from what the edit actually produced, never from what it was
        /// asked for: keys that cancelled against each other are gone, and keeping them selected
        /// would leave the next edit addressing frames that no longer hold a key.
        /// </remarks>
        private void ApplyToSelectedRows(in TrackDrawContext context,
            List<AnimationTagging.TagChannel> channels, string undoName, ChannelEdit edit,
            SelectionEdit moveSelection = null)
        {
            var editor = context.Editor;
            var frameCount = editor.SliceFrameCount;
            var changed = false;

            for (var row = 0; row < channels.Count; row++)
            {
                if (!_selection.HasAnyIn(row)) continue;

                _selection.FramesIn(row, _rowFrames);
                var edited = edit(channels[row].toggles, _rowFrames, frameCount);

                WriteChannel(editor, row, edited);
                changed = true;

                if (moveSelection == null)
                {
                    _scratch.Clear();
                    _selection.ReplaceRow(row, _scratch);
                    continue;
                }

                moveSelection(_rowFrames, frameCount);
                _selection.ReplaceRow(row, _rowFrames);
                _selection.Intersect(row, edited);
            }

            if (!changed) return;

            Undo.SetCurrentGroupName(undoName);
            editor.Commit();
        }

        private void WriteChannel(ClipEditorContext editor, int row, IReadOnlyList<int> toggles)
        {
            var channels = ComponentProperty?.FindPropertyRelative("channels");
            if (channels == null || row >= channels.arraySize) return;

            var array = channels.GetArrayElementAtIndex(row).FindPropertyRelative("toggles");
            array.arraySize = toggles.Count;
            for (var i = 0; i < toggles.Count; i++)
            {
                array.GetArrayElementAtIndex(i).intValue = toggles[i];
            }
        }

        private static void Shift(List<int> frames, int delta)
        {
            for (var i = 0; i < frames.Count; i++) frames[i] += delta;
        }

        private static void Scaled(List<int> frames, int pivot, float factor)
        {
            for (var i = 0; i < frames.Count; i++)
            {
                frames[i] = pivot + Mathf.RoundToInt((frames[i] - pivot) * factor);
            }
        }

        // ---- context menu ----

        private void ShowContextMenu(in TrackDrawContext context,
            List<AnimationTagging.TagChannel> channels, Vector2 mouse)
        {
            // Copied out of the "in" parameter because a menu item is a closure, and a readonly
            // ref cannot be captured by one.
            var drawContext = context;
            var editor = context.Editor;
            var row = RowAt(context.LaneRect, mouse.y, channels.Count);
            var frame = editor.ClipToSliceFrame(Mathf.RoundToInt(context.Axis.XToFrame(mouse.x)));

            var menu = new GenericMenu();

            if (row >= 0)
            {
                _activeRow = row;
                var target = row;
                menu.AddItem(new GUIContent("Insert Key Here"), false, () =>
                {
                    WriteChannel(editor, target,
                        AnimationTagEdits.Insert(channels[target].toggles, frame,
                            editor.SliceFrameCount));

                    Undo.SetCurrentGroupName("Insert tag key");
                    editor.Commit();
                });

                menu.AddItem(new GUIContent($"Remove Channel '{ChannelName(channels[target])}'"), false,
                    () => RemoveChannel(editor, target));
            }
            else
            {
                menu.AddDisabledItem(new GUIContent("Insert Key Here"));
            }

            menu.AddSeparator(string.Empty);

            if (_selection.Count > 0)
            {
                menu.AddItem(new GUIContent("Delete Selected Keys"), false, () =>
                {
                    ApplyToSelectedRows(drawContext, channels, "Delete tag keys",
                        (toggles, frames, frameCount) =>
                            AnimationTagEdits.Delete(toggles, frames, frameCount));
                    _selection.Clear();
                });
            }
            else
            {
                menu.AddDisabledItem(new GUIContent("Delete Selected Keys"));
            }

            menu.ShowAsContext();
        }

        private static string ChannelName(AnimationTagging.TagChannel channel) =>
            channel.tag == null ? "(no tag)" : TagLabel(channel.tag);

        private void RemoveChannel(ClipEditorContext editor, int row)
        {
            var channels = ComponentProperty?.FindPropertyRelative("channels");
            if (channels == null || row >= channels.arraySize) return;

            channels.DeleteArrayElementAtIndex(row);
            Undo.SetCurrentGroupName("Remove tag channel");
            editor.Commit();

            _selection.Clear();
        }

        // ---- inspector ----

        public override void DrawInspector(in TrackInspectorContext context)
        {
            var tags = Tags;
            if (tags == null || context.ComponentProperty == null) return;

            DrawChannelList(context, tags);
            EditorGUILayout.Space();
            DrawAddChannel(context);
            EditorGUILayout.Space();
            DrawQueryPanel(context, tags);
            EditorGUILayout.Space();
            DrawSummary(context, tags);
        }

        private static void DrawChannelList(in TrackInspectorContext context, AnimationTagComponent tags)
        {
            EditorGUILayout.LabelField("Channels", EditorStyles.boldLabel);

            var channels = context.ComponentProperty.FindPropertyRelative("channels");
            if (channels.arraySize == 0)
            {
                EditorGUILayout.LabelField("None yet.", EditorStyles.miniLabel);
                return;
            }

            for (var i = 0; i < channels.arraySize; i++)
            {
                using (new EditorGUILayout.HorizontalScope())
                {
                    var element = channels.GetArrayElementAtIndex(i);
                    EditorGUILayout.PropertyField(element.FindPropertyRelative("tag"), GUIContent.none);

                    var keys = tags.channels[i]?.toggles?.Count ?? 0;
                    EditorGUILayout.LabelField($"{keys} keys", EditorStyles.miniLabel,
                        GUILayout.Width(56f));
                }
            }
        }

        /// <summary>
        /// Picks the tag first, then makes the channel - the same searchable dropdown a serialized
        /// tag field uses, with a New Tag entry so a vocabulary can grow without leaving the window.
        /// </summary>
        private void DrawAddChannel(in TrackInspectorContext context)
        {
            var button = GUILayoutUtility.GetRect(new GUIContent("Add Channel"), GUI.skin.button);
            if (!GUI.Button(button, "Add Channel")) return;

            // Copied out of the "in" parameter: the picker answers later, through a closure.
            var property = context.ComponentProperty;
            var editor = context.Editor;

            GameplayTagPicker.ShowWithCreate(button, tag => AddChannel(property, editor, tag));
        }

        private static void AddChannel(SerializedProperty componentProperty, ClipEditorContext editor,
            GameplayTagSO tag)
        {
            if (tag == null) return;

            var channels = componentProperty.FindPropertyRelative("channels");
            channels.arraySize++;

            var added = channels.GetArrayElementAtIndex(channels.arraySize - 1);
            added.FindPropertyRelative("tag").objectReferenceValue = tag;
            added.FindPropertyRelative("toggles").arraySize = 0;

            Undo.SetCurrentGroupName("Add tag channel");
            editor.Commit();
        }

        private void DrawQueryPanel(in TrackInspectorContext context, AnimationTagComponent tags)
        {
            EditorGUILayout.LabelField("Query", EditorStyles.boldLabel);

            _queryScratch ??= ClipTagQueryScratch.instance.Serialized();
            _queryScratch.Update();
            EditorGUILayout.PropertyField(_queryScratch.FindProperty("query"), new GUIContent("Match"),
                true);
            if (_queryScratch.ApplyModifiedProperties()) ClipTagQueryScratch.instance.Save();

            if (GUILayout.Button("Find Segments", GUILayout.Height(20f)))
            {
                _queryMatches.Clear();
                tags.FindSegments(context.Editor.Clip, ClipTagQueryScratch.instance.Query,
                    _queryMatches);

                _queryRun = true;
                context.Editor.Repaint();
            }

            if (!_queryRun) return;

            if (_queryMatches.Count == 0)
            {
                EditorGUILayout.LabelField("No matching segments.", EditorStyles.miniLabel);
                return;
            }

            foreach (var match in _queryMatches)
            {
                var label = $"frames {match.StartFrame}–{match.EndFrame}  ({match.FrameCount})";
                if (!GUILayout.Button(label, EditorStyles.miniButton)) continue;

                context.Editor.SeekToClipFrame(context.Editor.SliceToClipFrame(match.StartFrame));
            }
        }

        /// <summary>
        /// What the annotation currently says, and the two ways it goes wrong quietly: a channel
        /// with no tag answers no query, and a clip re-trimmed under its keys loses them.
        /// </summary>
        private void DrawSummary(in TrackInspectorContext context, AnimationTagComponent tags)
        {
            var untagged = 0;
            var empty = 0;
            foreach (var channel in tags.channels)
            {
                if (channel == null) continue;
                if (channel.tag == null) untagged++;
                if (channel.toggles.Count == 0) empty++;
            }

            EditorGUILayout.LabelField(tags.Describe(), EditorStyles.miniLabel);

            if (untagged > 0)
            {
                EditorGUILayout.HelpBox(
                    $"{untagged} channels have no tag. Their keys are stored but answer no query, " +
                    "and OnValidate folds them nowhere - assign a tag or remove the channel.",
                    MessageType.Warning);
            }

            if (empty > 0)
            {
                EditorGUILayout.HelpBox($"{empty} channels have no keys, so their tag is never on.",
                    MessageType.Info);
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Keys", EditorStyles.boldLabel);
            foreach (var line in TimelineKeymap.Summary)
            {
                EditorGUILayout.LabelField(line, EditorStyles.miniLabel);
            }
        }
    }
}
