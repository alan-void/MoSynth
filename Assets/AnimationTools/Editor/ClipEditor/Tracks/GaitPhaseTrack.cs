using System.Collections.Generic;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace AnimationTools.Editor
{
    /// <summary>
    /// The clip editor's view of a <see cref="GaitPhaseComponent"/>: the phase and the contacts it
    /// was read from as a strip, and the footfall anchors as things you can drag.
    /// </summary>
    /// <remarks>
    /// The dense bands are a cached texture and the anchors are vector-drawn, which is the split
    /// that matters: a band never needs hit-testing and would cost a primitive per frame if drawn
    /// as one, while an anchor needs a pixel-accurate hit rect at any zoom and has to follow a drag
    /// without the texture being rebuilt.
    /// </remarks>
    [ClipComponentTrack(typeof(GaitPhaseComponent))]
    public sealed class GaitPhaseTrack : AnimationClipComponentTrack
    {
        private const float AnchorBandHeight = 14f;
        private const float AnchorHalfWidth = 3.5f;
        private const int TrajectoryLookahead = 20;

        private static readonly Color SelectionOutline = new(1f, 0.85f, 0.3f, 1f);
        private static readonly Color RubberBand = new(0.4f, 0.6f, 0.9f, 0.25f);

        private readonly TimelineSignalTexture _signal = new();
        private readonly TimelineKeySelection _selection = new();
        private readonly List<int> _selectedFrames = new();
        private readonly List<int> _scratchFrames = new();
        private readonly HashSet<int> _selectedIndices = new();
        private readonly List<float3> _trajectory = new();

        private float[] _phase;
        private float[] _phaseRate;
        private bool[] _contacts;
        private int _cachedVersion = int.MinValue;

        // Bumped by a detection pass, which changes the contact bands without touching the asset.
        private int _contactsVersion;

        private int _controlId;

        public override float DefaultLaneHeight => GaitPhaseSignalBuilder.Height + AnchorBandHeight + 4f;

        public override bool DrawsPreviewOverlay => true;

        private GaitPhaseComponent Phase => Component as GaitPhaseComponent;

        public override void OnDisable()
        {
            if (Editor != null && Editor.HasModal(this)) Editor.CancelModal();

            _signal.Dispose();
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

            firstClipFrame = lowest;
            lastClipFrame = highest;
            return true;
        }

        public override bool TryGetContentRange(out int firstClipFrame, out int lastClipFrame) =>
            FootfallEdits.ContentRange(Phase?.footfalls, out firstClipFrame, out lastClipFrame);

        public override void DrawTrack(in TrackDrawContext context)
        {
            var phase = Phase;
            if (phase == null || context.Editor.ClipFrameCount <= 0) return;

            RefreshCaches(context.Editor);

            var bandRect = new Rect(context.LaneRect.x, context.LaneRect.y,
                context.LaneRect.width, context.LaneRect.height - AnchorBandHeight);
            var anchorRect = new Rect(context.LaneRect.x, bandRect.yMax,
                context.LaneRect.width, AnchorBandHeight);

            DrawSignal(context, bandRect);
            DrawAnchors(context, anchorRect, phase);
            HandleAnchorInput(context, anchorRect, phase);
        }

        private void RefreshCaches(ClipEditorContext editor)
        {
            var version = editor.DataVersion * 397 + _contactsVersion;
            if (version == _cachedVersion && _phase != null) return;

            _cachedVersion = version;
            Phase.Evaluate(editor.Clip, out _phase, out _phaseRate);

            var clipFrames = editor.ClipFrameCount;
            var phaseValues = _phase;
            var phaseRates = _phaseRate;
            var contacts = _contacts;

            _signal.Rebuild(clipFrames, GaitPhaseSignalBuilder.Height, version,
                GaitPhaseSignalBuilder.Background,
                (pixels, width, _) => GaitPhaseSignalBuilder.Fill(pixels, width, phaseValues, phaseRates,
                    contacts, _signal.ColumnToFrame));
        }

        private void DrawSignal(in TrackDrawContext context, Rect bandRect)
        {
            if (Event.current.type != EventType.Repaint) return;

            // Spans the whole clip, because the phase does: anchors are clip frames and an anchor
            // past the trim still gives the frames before it a cycle.
            if (TimelineSignalTexture.TryMapRange(context.Axis, bandRect, 0,
                    context.Editor.ClipFrameCount, out var destination, out var uv))
            {
                _signal.Draw(destination, uv);
            }
        }

        private void DrawAnchors(in TrackDrawContext context, Rect anchorRect, GaitPhaseComponent phase)
        {
            if (Event.current.type != EventType.Repaint) return;

            EditorGUI.DrawRect(anchorRect, new Color(0.14f, 0.14f, 0.15f, 1f));

            var modal = context.Editor.HasModal(this) ? context.Editor.Modal : null;
            if (modal is { Kind: TimelineModalKind.BoxSelect })
            {
                EditorGUI.DrawRect(modal.BoxRect, RubberBand);
            }

            var delta = PreviewDelta(context, phase, modal);

            for (var i = 0; i < phase.footfalls.Count; i++)
            {
                var footfall = phase.footfalls[i];
                var selected = _selection.Contains(0, footfall.frame);
                var clipFrame = footfall.frame + (selected ? delta : 0);
                if (clipFrame < context.FirstVisibleClipFrame || clipFrame > context.LastVisibleClipFrame)
                {
                    continue;
                }

                var x = context.Axis.FrameToX(clipFrame);
                var repeated = i > 0 && phase.footfalls[i - 1].foot == footfall.foot;
                var body = new Rect(x - AnchorHalfWidth, anchorRect.y + 2f,
                    AnchorHalfWidth * 2f, anchorRect.height - 4f);

                if (selected)
                {
                    EditorGUI.DrawRect(new Rect(body.x - 1f, body.y - 1f, body.width + 2f, body.height + 2f),
                        SelectionOutline);
                }

                EditorGUI.DrawRect(body, GaitPhaseSignalBuilder.AnchorColour(footfall.foot, repeated));
            }

            DrawModalStatus(context, modal);
        }

        /// <summary>
        /// How far a running grab would move the selection, clamped to the clip. A preview only:
        /// nothing reaches the asset until the mode is confirmed.
        /// </summary>
        private int PreviewDelta(in TrackDrawContext context, GaitPhaseComponent phase,
            TimelineModalOperator modal)
        {
            if (modal is not { Kind: TimelineModalKind.Grab }) return 0;

            return FootfallEdits.ClampDelta(phase.footfalls, SelectedIndices(phase), modal.FrameDelta,
                context.Editor.ClipFrameCount);
        }

        private static void DrawModalStatus(in TrackDrawContext context, TimelineModalOperator modal)
        {
            var status = modal?.StatusText;
            if (string.IsNullOrEmpty(status)) return;

            var box = new Rect(context.LaneRect.x + 6f, context.LaneRect.y + 2f, 180f, 16f);
            EditorGUI.DrawRect(box, new Color(0f, 0f, 0f, 0.65f));
            GUI.Label(new Rect(box.x + 4f, box.y, box.width - 8f, box.height), status,
                EditorStyles.miniLabel);
        }

        private void HandleAnchorInput(in TrackDrawContext context, Rect anchorRect,
            GaitPhaseComponent phase)
        {
            // Allocated unconditionally so ids do not shift between Layout and Repaint.
            _controlId = GUIUtility.GetControlID(FocusType.Passive);

            var editor = context.Editor;
            var e = Event.current;

            if (editor.HasModal(this))
            {
                var result = editor.Modal.HandleEvent(e);
                if (result == TimelineModalResult.Confirmed) ConfirmModal(context, phase);
                if (result is TimelineModalResult.Confirmed or TimelineModalResult.Cancelled)
                {
                    editor.CancelModal();
                }

                if (result != TimelineModalResult.None) editor.Repaint();
                return;
            }

            // Key events go through the raw event type, never GetTypeForControl: that method
            // returns Ignore for a key unless GUIUtility.keyboardControl is this control, and the
            // lane's control is Passive so it never can be. Filtering clicks by it is still right -
            // they must respect hotControl.
            if (e.type == EventType.KeyDown && context.IsFocused &&
                !EditorGUIUtility.editingTextField)
            {
                HandleKey(context, phase, e);
                return;
            }

            switch (e.GetTypeForControl(_controlId))
            {
                case EventType.MouseDown when e.button == 0 && anchorRect.Contains(e.mousePosition):
                    HandleClick(context, phase, e);
                    return;

                case EventType.ContextClick when anchorRect.Contains(e.mousePosition):
                    ShowContextMenu(editor, phase, e.mousePosition.x, context.Axis);
                    e.Use();
                    return;
            }
        }

        private void HandleClick(in TrackDrawContext context, GaitPhaseComponent phase, Event e)
        {
            var editor = context.Editor;
            var axis = context.Axis;
            var picked = FootfallHitTester.Pick(phase.footfalls, e.mousePosition.x,
                clipFrame => axis.FrameToX(clipFrame));

            if (e.clickCount == 2 && picked < 0)
            {
                AddAnchorAt(editor, Mathf.RoundToInt(axis.XToFrame(e.mousePosition.x)));
                e.Use();
                return;
            }

            if (picked < 0)
            {
                BeginModal(context, TimelineModalKind.BoxSelect, e, extend: e.shift);
                return;
            }

            var frame = phase.footfalls[picked].frame;
            if (e.shift) _selection.Toggle(0, frame);
            else _selection.SetTo(0, frame);

            e.Use();
        }

        private void HandleKey(in TrackDrawContext context, GaitPhaseComponent phase, Event e)
        {
            var editor = context.Editor;
            var action = TimelineKeymap.Resolve(e);

            switch (action)
            {
                case TimelineKeyAction.SelectAll:
                    foreach (var footfall in phase.footfalls) _selection.Add(0, footfall.frame);
                    break;

                case TimelineKeyAction.DeselectAll:
                    _selection.Clear();
                    break;

                case TimelineKeyAction.InvertSelection:
                    InvertSelection(phase);
                    break;

                case TimelineKeyAction.BeginBoxSelect:
                    BeginModal(context, TimelineModalKind.BoxSelect, e, extend: e.shift);
                    return;

                case TimelineKeyAction.BeginGrab when _selection.Count > 0:
                    BeginModal(context, TimelineModalKind.Grab, e);
                    return;

                case TimelineKeyAction.Delete when _selection.Count > 0:
                    WriteFootfalls(editor, FootfallEdits.Delete(phase.footfalls, SelectedIndices(phase)),
                        "Delete footfalls");
                    _selection.Clear();
                    break;

                case TimelineKeyAction.InsertKey:
                    AddAnchorAt(editor, editor.PlayheadClipFrame);
                    break;

                case TimelineKeyAction.StepBack:
                    editor.SeekToClipFrame(editor.PlayheadClipFrame - 1);
                    break;

                case TimelineKeyAction.StepForward:
                    editor.SeekToClipFrame(editor.PlayheadClipFrame + 1);
                    break;

                case TimelineKeyAction.PreviousKey:
                case TimelineKeyAction.NextKey:
                    JumpToAnchor(editor, phase, action == TimelineKeyAction.NextKey);
                    break;

                // Scale has no meaning for anchors whose spacing is the gait, and Home / "." belong
                // to the timeline, which owns the axis a track only sees a copy of.
                default:
                    return;
            }

            e.Use();
            editor.Repaint();
        }

        private void InvertSelection(GaitPhaseComponent phase)
        {
            _scratchFrames.Clear();
            foreach (var footfall in phase.footfalls)
            {
                if (!_selection.Contains(0, footfall.frame)) _scratchFrames.Add(footfall.frame);
            }

            _selection.ReplaceRow(0, _scratchFrames);
        }

        private void BeginModal(in TrackDrawContext context, TimelineModalKind kind, Event e,
            bool extend = false)
        {
            var editor = context.Editor;

            editor.ClaimModal(this).Begin(kind, _controlId, e.mousePosition,
                context.Axis.pixelsPerFrame, editor.PlayheadClipFrame,
                context.Axis.FrameToX(editor.PlayheadClipFrame), extend);

            e.Use();
            editor.Repaint();
        }

        private void ConfirmModal(in TrackDrawContext context, GaitPhaseComponent phase)
        {
            var editor = context.Editor;
            var modal = editor.Modal;

            if (modal.Kind == TimelineModalKind.BoxSelect)
            {
                var box = modal.BoxRect;
                var axis = context.Axis;
                if (!modal.Extend) _selection.Clear();

                _scratchFrames.Clear();
                FootfallHitTester.PickRange(phase.footfalls, box.xMin, box.xMax,
                    clipFrame => axis.FrameToX(clipFrame), _scratchFrames);

                // PickRange reports indices; the selection is keyed by frame.
                foreach (var index in _scratchFrames) _selection.Add(0, phase.footfalls[index].frame);
                return;
            }

            if (modal.Kind != TimelineModalKind.Grab) return;

            var delta = FootfallEdits.ClampDelta(phase.footfalls, SelectedIndices(phase),
                modal.FrameDelta, editor.ClipFrameCount);
            if (delta == 0) return;

            _selection.FramesIn(0, _selectedFrames);
            WriteFootfalls(editor,
                FootfallEdits.Move(phase.footfalls, SelectedIndices(phase), delta, editor.ClipFrameCount),
                "Move footfalls");

            for (var i = 0; i < _selectedFrames.Count; i++) _selectedFrames[i] += delta;
            _selection.ReplaceRow(0, _selectedFrames);
        }

        /// <summary>
        /// The selection translated back to list indices, which is what <see cref="FootfallEdits"/>
        /// still speaks - an anchor carries a foot, so its edits are not the tag lane's key
        /// arithmetic. Re-derived per use because anchors are re-sorted on every write.
        /// </summary>
        private HashSet<int> SelectedIndices(GaitPhaseComponent phase)
        {
            _selectedIndices.Clear();
            for (var i = 0; i < phase.footfalls.Count; i++)
            {
                if (_selection.Contains(0, phase.footfalls[i].frame)) _selectedIndices.Add(i);
            }

            return _selectedIndices;
        }

        private void JumpToAnchor(ClipEditorContext editor, GaitPhaseComponent phase, bool forward)
        {
            var from = editor.PlayheadClipFrame;

            var best = -1;
            foreach (var footfall in phase.footfalls)
            {
                if (forward ? footfall.frame <= from : footfall.frame >= from) continue;
                if (best >= 0 && (forward ? footfall.frame >= best : footfall.frame <= best)) continue;

                best = footfall.frame;
            }

            if (best >= 0) editor.SeekToClipFrame(best);
        }

        private void AddAnchorAt(ClipEditorContext editor, int clipFrame)
        {
            WriteFootfalls(editor, FootfallEdits.Add(Phase.footfalls, clipFrame, editor.ClipFrameCount),
                "Add footfall");
            _selection.SetTo(0, clipFrame);
        }

        private void ShowContextMenu(ClipEditorContext editor, GaitPhaseComponent phase, float mouseX,
            ClipTimeAxis axis)
        {
            var clipFrame = Mathf.RoundToInt(axis.XToFrame(mouseX));
            var picked = FootfallHitTester.Pick(phase.footfalls, mouseX,
                clipFrame => axis.FrameToX(clipFrame));

            var menu = new GenericMenu();
            menu.AddItem(new GUIContent("Add Footfall Here"), false, () => AddAnchorAt(editor, clipFrame));

            if (picked >= 0)
            {
                menu.AddItem(new GUIContent("Delete"), false, () =>
                {
                    WriteFootfalls(editor, FootfallEdits.Delete(phase.footfalls, new HashSet<int> { picked }),
                        "Delete footfall");
                    _selection.Clear();
                });

                menu.AddSeparator(string.Empty);

                menu.AddItem(new GUIContent("Set Foot/Left"), false, () => SetFoot(editor, picked, GaitPhase.Foot.Left));
                menu.AddItem(new GUIContent("Set Foot/Right"), false, () => SetFoot(editor, picked, GaitPhase.Foot.Right));
            }
            else
            {
                menu.AddDisabledItem(new GUIContent("Delete"));
            }

            menu.ShowAsContext();
        }

        private void SetFoot(ClipEditorContext editor, int index, GaitPhase.Foot foot)
        {
            var footfalls = new List<GaitPhase.Footfall>(Phase.footfalls);
            var footfall = footfalls[index];
            footfall.foot = foot;
            footfalls[index] = footfall;

            WriteFootfalls(editor, footfalls, "Set footfall foot");
        }

        /// <summary>
        /// Replaces the anchor list through the serialized property, which is what makes the edit
        /// undoable, and commits once.
        /// </summary>
        private void WriteFootfalls(ClipEditorContext editor, List<GaitPhase.Footfall> footfalls,
            string undoName)
        {
            var array = ComponentProperty?.FindPropertyRelative("footfalls");
            if (array == null) return;

            array.arraySize = footfalls.Count;
            for (var i = 0; i < footfalls.Count; i++)
            {
                var element = array.GetArrayElementAtIndex(i);
                element.FindPropertyRelative("frame").intValue = footfalls[i].frame;
                element.FindPropertyRelative("foot").enumValueIndex = (int)footfalls[i].foot;
            }

            Undo.SetCurrentGroupName(undoName);
            editor.Commit();
        }

        public override void DrawInspector(in TrackInspectorContext context)
        {
            var phase = Phase;
            if (phase == null) return;

            var editor = context.Editor;
            var skeleton = editor.Clip.Skeleton;

            EditorGUILayout.LabelField("Detection", EditorStyles.boldLabel);

            DrawBoneField(context, "leftContactBoneName", "Left Contact Bone", skeleton);
            DrawBoneField(context, "rightContactBoneName", "Right Contact Bone", skeleton);

            var thresholdProperty = context.ComponentProperty.FindPropertyRelative("contactVelocityThreshold");
            EditorGUILayout.Slider(thresholdProperty,
                0f, 1f, new GUIContent("Contact Velocity", "Character-space speed below which a foot counts as planted, in m/s"));

            var smoothingProperty = context.ComponentProperty.FindPropertyRelative("smoothingRadius");
            EditorGUILayout.IntSlider(smoothingProperty,
                0, 20, new GUIContent("Smoothing Radius", "Median filter half-width, in frames"));

            if (GUILayout.Button("Detect Footfalls", GUILayout.Height(22f)))
            {
                Detect(editor, phase);
            }

            EditorGUILayout.Space();
            DrawSummary(editor, phase);
            DrawKeymap();
        }

        /// <summary>
        /// The one place that mutates the component directly rather than through a serialized
        /// property, because <see cref="GaitPhaseComponent.TryDetect"/> rewrites the anchor list
        /// itself. Anything pending is committed first, and the serialized copy re-read after, so
        /// the two views never disagree.
        /// </summary>
        private void Detect(ClipEditorContext editor, GaitPhaseComponent phase)
        {
            editor.Commit();

            Undo.RecordObject(editor.Clip, "Detect footfalls");

            if (phase.TryDetect(editor.Clip, out var contacts, out var error))
            {
                _contacts = contacts;
                _contactsVersion++;
                EditorUtility.SetDirty(editor.Clip);
                Debug.Log($"[GaitPhase] {editor.Clip.name}: {phase.Describe()}.", editor.Clip);
            }
            else
            {
                Debug.LogError($"[GaitPhase] {editor.Clip.name}: {error}", editor.Clip);
            }

            _selection.Clear();
            editor.RefreshFromAsset();
        }

        /// <summary>The bindings, since none of them are discoverable.</summary>
        private static void DrawKeymap()
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Keys", EditorStyles.boldLabel);
            foreach (var line in TimelineKeymap.Summary)
            {
                EditorGUILayout.LabelField(line, EditorStyles.miniLabel);
            }
        }

        private static void DrawBoneField(in TrackInspectorContext context, string propertyName,
            string label, Skeleton skeleton)
        {
            var property = context.ComponentProperty.FindPropertyRelative(propertyName);
            var tooltip = "Leave empty to fall back to the toe or foot bone name conventions";

            if (skeleton == null || skeleton.Root == null)
            {
                EditorGUILayout.PropertyField(property, new GUIContent(label, tooltip));
                return;
            }

            var transforms = new List<Transform>();
            var depths = new List<int>();
            BonePopup.Collect(skeleton.Root, transforms, depths);
            var options = BonePopup.BuildOptions(transforms, depths);

            var selected = 0;
            for (var i = 0; i < transforms.Count; i++)
            {
                if (transforms[i].name != property.stringValue) continue;

                selected = i + 1;
                break;
            }

            EditorGUI.BeginChangeCheck();
            var chosen = EditorGUILayout.Popup(new GUIContent(label, tooltip), selected, options);
            if (!EditorGUI.EndChangeCheck()) return;

            property.stringValue = chosen == 0 ? string.Empty : transforms[chosen - 1].name;
        }

        /// <summary>
        /// Counts, stride, and the two ways the phase is known to go wrong. This is the part of the
        /// tool that tells you whether the detection worked, so it stays prominent.
        /// </summary>
        private void DrawSummary(ClipEditorContext editor, GaitPhaseComponent phase)
        {
            RefreshCaches(editor);

            var frameCount = Mathf.Max(1, editor.ClipFrameCount);
            var noCycle = 0;
            foreach (var rate in _phaseRate)
            {
                if (rate == 0f) noCycle++;
            }

            var repeated = FootfallEdits.RepeatedFeet(phase.footfalls);
            var left = 0;
            foreach (var footfall in phase.footfalls)
            {
                if (footfall.foot == GaitPhase.Foot.Left) left++;
            }

            EditorGUILayout.LabelField(
                $"{phase.footfalls.Count} footfalls ({left} L / {phase.footfalls.Count - left} R)\n" +
                $"stride {StrideSeconds(_phaseRate, editor.FrameTime):0.00}s\n" +
                $"no cycle on {100f * noCycle / frameCount:0.#}% of {frameCount} frames",
                EditorStyles.miniLabel, GUILayout.Height(42f));

            if (_contacts == null)
            {
                EditorGUILayout.HelpBox(
                    "The contact bands are empty until you run detection: contact flags are not " +
                    "stored on the clip, only the footfalls they produced.", MessageType.None);
            }

            if (repeated.Count > 0)
            {
                EditorGUILayout.HelpBox(
                    $"{repeated.Count} footfalls are the same foot twice running. Each one is a contact the " +
                    "detection missed, and the phase jumps a whole cycle there instead of half. Raise " +
                    "the velocity threshold, or add the missing footfall by hand.",
                    MessageType.Warning);

                if (GUILayout.Button("Select Repeated Footfalls"))
                {
                    _selection.Clear();
                    foreach (var index in repeated) _selection.Add(0, phase.footfalls[index].frame);

                    editor.SeekToClipFrame(phase.footfalls[repeated[0]].frame);
                    editor.Repaint();
                }
            }

            if (noCycle > 0)
            {
                EditorGUILayout.HelpBox(
                    $"{noCycle} frames sit outside the first and last footfall and have no measurable " +
                    "cycle. Training drops them. Trimming the clip to the walking part is usually the " +
                    "right fix.",
                    MessageType.Info);
            }
        }

        /// <summary>Mean stride period over the frames that have a cycle, in seconds.</summary>
        private static float StrideSeconds(float[] phaseRates, float frameTime)
        {
            var total = 0f;
            var counted = 0;
            foreach (var rate in phaseRates)
            {
                if (rate <= 0f) continue;

                total += rate;
                counted++;
            }

            return counted == 0 || total <= 0f ? 0f : GaitPhase.Tau / (total / counted);
        }

        /// <summary>
        /// Marks each contact bone and traces where it goes next, so a suspicious anchor can be
        /// checked against the motion instead of against the strip that produced it.
        /// </summary>
        public override void DrawPreviewOverlay(in TrackOverlayContext context)
        {
            var skeleton = context.Editor.Clip.Skeleton;
            if (skeleton == null) return;

            DrawFootOverlay(context, skeleton, true);
            DrawFootOverlay(context, skeleton, false);
        }

        private void DrawFootOverlay(in TrackOverlayContext context, Skeleton skeleton, bool left)
        {
            if (!TryResolveContactBone(skeleton, left, out var boneIndex)) return;
            if (boneIndex < 0 || boneIndex >= context.BonePositions.Length) return;

            var colour = left
                ? (Color)GaitPhaseSignalBuilder.LeftContact
                : (Color)GaitPhaseSignalBuilder.RightContact;

            var planted = IsPlantedOn(context.Editor, context.ClipFrame, left);
            context.Draw.DrawWireSphere(context.BonePositions[boneIndex], planted ? 0.06f : 0.035f, colour);

            _trajectory.Clear();
            var lastFrame = Mathf.Min(context.Editor.ClipFrameCount - 1,
                context.ClipFrame + TrajectoryLookahead);

            for (var frame = context.ClipFrame; frame <= lastFrame; frame++)
            {
                var pose = context.Editor.GetClipFrame(frame);
                if (pose.Layout.RotationCount != context.Skeleton.BoneCount) return;

                _trajectory.Add(context.Skeleton.CharacterSpacePosition(pose, boneIndex));
            }

            context.Draw.DrawPolyline(_trajectory, colour);
        }

        private bool IsPlantedOn(ClipEditorContext editor, int clipFrame, bool left)
        {
            if (_contacts == null) return false;

            var index = clipFrame * 2 + (left ? 0 : 1);
            return index >= 0 && index < _contacts.Length && _contacts[index];
        }

        /// <summary>Mirrors what detection does: an explicit name first, then the name conventions.</summary>
        private bool TryResolveContactBone(Skeleton skeleton, bool left, out int index)
        {
            var name = left ? Phase.leftContactBoneName : Phase.rightContactBoneName;
            if (!string.IsNullOrEmpty(name))
            {
                index = skeleton.IndexOfName(name);
                return index >= 0;
            }

            return BoneNameConventions.TryFindContactBone(skeleton, left, out index);
        }
    }
}
