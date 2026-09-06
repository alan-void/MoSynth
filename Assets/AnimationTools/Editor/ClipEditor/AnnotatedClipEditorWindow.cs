using System;
using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEditor.Callbacks;
using UnityEditor.UIElements;
using UnityEngine;
using UnityEngine.UIElements;

namespace AnimationTools.Editor
{
    /// <summary>
    /// Views and edits an <see cref="AnnotatedAnimationClip"/>: a scrubbable preview of the motion,
    /// a zoomable timeline with one lane per clip component, and an inspector for whichever lane is
    /// focused.
    /// </summary>
    /// <remarks>
    /// The shell is UI Toolkit for its splitters and toolbar; the preview, timeline and inspector
    /// are each an <c>IMGUIContainer</c>. That split is deliberate:
    /// <see cref="PreviewRenderUtility"/> is an IMGUI API in all but name, and drawing a lane is the
    /// public seam other people extend, so it speaks the IMGUI the rest of this project's editors
    /// already use. See <c>openwiki/animation-tools/clip-editor.md</c>.
    /// </remarks>
    public sealed class AnnotatedClipEditorWindow : EditorWindow
    {
        private const float InspectorWidth = 320f;
        private const float MinPreviewHeight = 120f;

        [SerializeField] private AnnotatedAnimationClip _clip;
        [SerializeField] private bool _locked;
        [SerializeField] private int _playheadClipFrame;
        [SerializeField] private ClipTimeAxis _axis;
        [SerializeField] private Vector2 _timelineScroll;
        [SerializeField] private float _headerWidth = 168f;
        [SerializeField] private int _focusedRowIndex = -1;
        [SerializeField] private bool _labelsAsSeconds;

        [NonSerialized] private ClipEditorContext _context;
        [NonSerialized] private SkeletonPreview _preview;
        [NonSerialized] private ClipTimelineView _timeline;
        [NonSerialized] private IMGUIContainer _timelinePane;
        [NonSerialized] private List<ClipTrackRow> _rows = new();
        [NonSerialized] private ClipTrackViewState _viewState;
        [NonSerialized] private string _trackSignature;
        [NonSerialized] private string _assetGuid = string.Empty;

        [NonSerialized] private bool _isPlaying;
        [NonSerialized] private double _lastUpdateTime;
        [NonSerialized] private float _playbackTime;

        private ObjectField _clipField;
        private Label _frameLabel;

        [MenuItem("MoSynth/Animation/Clip Editor...", priority = 100)]
        public static void Open() => Open(Selection.activeObject as AnnotatedAnimationClip);

        public static void Open(AnnotatedAnimationClip clip)
        {
            var window = GetWindow<AnnotatedClipEditorWindow>();
            window.titleContent = new GUIContent("Clip Editor");
            window.minSize = new Vector2(720f, 420f);

            if (clip != null) window.SetClip(clip);
            window.Focus();
        }

        [OnOpenAsset]
        private static bool OnOpenAsset(int instanceId, int line)
        {
            if (EditorUtility.InstanceIDToObject(instanceId) is not AnnotatedAnimationClip clip) return false;

            Open(clip);
            return true;
        }

        private void OnEnable()
        {
            // Cheap, and it gives lanes a MouseMove when the platform sends one. It is not what
            // makes modal operators follow the cursor - this window has no OnGUI, so its IMGUI runs
            // in IMGUIContainers where that flag is not on the delivery path. TimelineModalOperator
            // samples the cursor on repaint instead.
            wantsMouseMove = true;

            _timeline = new ClipTimelineView();
            _preview = new SkeletonPreview();
            _preview.DrawingOverlays += DrawTrackOverlays;

            Undo.undoRedoPerformed += OnUndoRedo;
            EditorApplication.update += OnEditorUpdate;
            Selection.selectionChanged += OnSelectionChanged;

            _lastUpdateTime = EditorApplication.timeSinceStartup;
            RebuildContext();
        }

        private void OnDisable()
        {
            _context?.CancelModal();

            Undo.undoRedoPerformed -= OnUndoRedo;
            EditorApplication.update -= OnEditorUpdate;
            Selection.selectionChanged -= OnSelectionChanged;

            if (_preview != null) _preview.DrawingOverlays -= DrawTrackOverlays;
            _preview?.Dispose();
            _preview = null;

            DisposeTracks();
            SaveViewState();
        }

        public void CreateGUI()
        {
            rootVisualElement.Clear();
            rootVisualElement.style.flexDirection = FlexDirection.Column;

            rootVisualElement.Add(BuildToolbar());

            var split = new TwoPaneSplitView(1, InspectorWidth, TwoPaneSplitViewOrientation.Horizontal);
            split.style.flexGrow = 1f;

            var leftSplit = new TwoPaneSplitView(0, 260f, TwoPaneSplitViewOrientation.Vertical);
            leftSplit.style.flexGrow = 1f;

            var previewPane = new IMGUIContainer(DrawPreview);
            previewPane.style.flexGrow = 1f;
            previewPane.style.minHeight = MinPreviewHeight;

            _timelinePane = new IMGUIContainer(DrawTimeline);
            _timelinePane.style.flexGrow = 1f;
            _timelinePane.style.minHeight = 100f;

            // Keys reach a lane only if the panel has focused this container. The timeline claims
            // IMGUI keyboard focus on a lane click (ClipTimelineView), and this is the other half:
            // without it the inspector pane's property fields keep panel focus for good.
            _timelinePane.focusable = true;
            _timelinePane.RegisterCallback<PointerDownEvent>(_ => _timelinePane.Focus());

            leftSplit.Add(previewPane);
            leftSplit.Add(_timelinePane);

            var inspectorScroll = new ScrollView { style = { flexGrow = 1f } };
            inspectorScroll.Add(new IMGUIContainer(DrawInspector));

            split.Add(leftSplit);
            split.Add(inspectorScroll);
            rootVisualElement.Add(split);
        }

        private VisualElement BuildToolbar()
        {
            var toolbar = new Toolbar();

            _clipField = new ObjectField
            {
                objectType = typeof(AnnotatedAnimationClip),
                allowSceneObjects = false,
                value = _clip,
                style = { width = 220f }
            };
            _clipField.RegisterValueChangedCallback(evt => SetClip(evt.newValue as AnnotatedAnimationClip));
            toolbar.Add(_clipField);

            var lockToggle = new ToolbarToggle { text = "Lock", value = _locked, tooltip = "Stop following the Project selection" };
            lockToggle.RegisterValueChangedCallback(evt => _locked = evt.newValue);
            toolbar.Add(lockToggle);

            toolbar.Add(new ToolbarButton(() => StepFrames(-1)) { text = "◀" });
            toolbar.Add(new ToolbarButton(TogglePlayback) { text = "▶ / ❚❚" });
            toolbar.Add(new ToolbarButton(() => StepFrames(1)) { text = "▶" });

            _frameLabel = new Label { style = { unityTextAlign = TextAnchor.MiddleLeft, minWidth = 150f, marginLeft = 6f } };
            toolbar.Add(_frameLabel);

            toolbar.Add(new ToolbarSpacer { flex = true });

            var secondsToggle = new ToolbarToggle { text = "Seconds", value = _labelsAsSeconds };
            secondsToggle.RegisterValueChangedCallback(evt =>
            {
                _labelsAsSeconds = evt.newValue;
                Repaint();
            });
            toolbar.Add(secondsToggle);

            // Add Component lives under the component list in the timeline's gutter, where the
            // Inspector puts it, rather than here.
            toolbar.Add(new ToolbarMenu { text = "Lanes" }.WithDeferredMenu(BuildLanesMenu));

            return toolbar;
        }

        private void BuildLanesMenu(DropdownMenu menu)
        {
            if (_rows.Count == 0)
            {
                menu.AppendAction("(no components)", _ => { }, DropdownMenuAction.Status.Disabled);
                return;
            }

            foreach (var row in _rows)
            {
                var captured = row;
                menu.AppendAction(captured.Track?.Title ?? "(missing)",
                    _ =>
                    {
                        captured.Visible = !captured.Visible;
                        SaveViewState();
                        Repaint();
                    },
                    _ => captured.Visible ? DropdownMenuAction.Status.Checked : DropdownMenuAction.Status.Normal);
            }
        }

        /// <summary>Opens the component type menu, from the button under the component list.</summary>
        private void ShowAddComponentMenu()
        {
            var menu = new GenericMenu();

            if (_context == null)
            {
                menu.AddDisabledItem(new GUIContent("(no clip)"));
                menu.ShowAsContext();
                return;
            }

            foreach (var type in AddableComponentTypes())
            {
                var captured = type;
                menu.AddItem(new GUIContent(ObjectNames.NicifyVariableName(type.Name)), false,
                    () => AddComponent(captured));
            }

            menu.ShowAsContext();
        }

        /// <summary>
        /// Every component the Add menu offers. A type with no parameterless constructor cannot be
        /// created as a managed reference, so it is skipped rather than offered and then failing.
        /// </summary>
        private static IEnumerable<Type> AddableComponentTypes()
        {
            foreach (var type in TypeCache.GetTypesDerivedFrom<AnimationClipComponent>())
            {
                if (type.IsAbstract || type.GetConstructor(Type.EmptyTypes) == null) continue;

                yield return type;
            }
        }

        private void AddComponent(Type componentType)
        {
            var components = _context.ComponentsProperty;
            components.arraySize++;

            var element = components.GetArrayElementAtIndex(components.arraySize - 1);
            element.managedReferenceValue = Activator.CreateInstance(componentType);

            Undo.SetCurrentGroupName($"Add {componentType.Name}");
            _context.Commit();

            RebuildTracks();
            _focusedRowIndex = _rows.Count - 1;

            // A component you just added is always shown. Lane visibility is remembered per
            // component type, so without this a type whose lane was hidden earlier comes back
            // hidden - which looks exactly like the component not having been added at all.
            if (_focusedRowIndex >= 0)
            {
                _rows[_focusedRowIndex].Visible = true;
                SaveViewState();
            }

            Repaint();
        }

        private void RemoveFocusedComponent()
        {
            if (_context == null || _focusedRowIndex < 0 || _focusedRowIndex >= _rows.Count) return;

            var componentIndex = _rows[_focusedRowIndex].ComponentIndex;
            _context.ComponentsProperty.DeleteArrayElementAtIndex(componentIndex);

            Undo.SetCurrentGroupName("Remove clip component");
            _context.Commit();

            RebuildTracks();
            _focusedRowIndex = Mathf.Min(_focusedRowIndex, _rows.Count - 1);
            Repaint();
        }

        private void SetClip(AnnotatedAnimationClip clip)
        {
            if (_clip == clip) return;

            SaveViewState();

            _clip = clip;
            if (_clipField != null && _clipField.value != clip) _clipField.value = clip;

            _focusedRowIndex = -1;
            _timelineScroll = Vector2.zero;
            _axis = default;
            _playheadClipFrame = 0;

            RebuildContext();
            Repaint();
        }

        private void OnSelectionChanged()
        {
            if (_locked) return;
            if (Selection.activeObject is not AnnotatedAnimationClip clip) return;

            SetClip(clip);
        }

        private void RebuildContext()
        {
            DisposeTracks();

            _context = null;
            _trackSignature = null;
            _assetGuid = string.Empty;

            if (_clip == null) return;

            _assetGuid = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(_clip));
            _viewState = ClipTrackViewState.Load(_assetGuid);

            _context = new ClipEditorContext(_clip, Repaint);
            _context.SeekToClipFrame(_playheadClipFrame);

            RebuildTracks();
        }

        /// <summary>
        /// Rebuilds the row list when the components list has changed shape. Cheap to call every
        /// repaint: the signature comparison is what decides whether anything is actually rebuilt.
        /// </summary>
        private void RebuildTracksIfStale()
        {
            if (_context == null) return;

            var signature = BuildTrackSignature();
            if (signature == _trackSignature) return;

            RebuildTracks();
        }

        private string BuildTrackSignature()
        {
            var builder = new StringBuilder();
            foreach (var component in _clip.components)
            {
                builder.Append(component?.GetType().FullName ?? "null").Append('|');
            }

            return builder.ToString();
        }

        private void RebuildTracks()
        {
            DisposeTracks();

            _rows = new List<ClipTrackRow>();
            if (_context == null || _clip == null) return;

            _trackSignature = BuildTrackSignature();

            for (var i = 0; i < _clip.components.Count; i++)
            {
                var component = _clip.components[i];
                if (component == null) continue;

                var track = ClipComponentTrackRegistry.Create(component);
                track.Editor = _context;
                track.Component = component;

                var row = new ClipTrackRow
                {
                    Track = track,
                    ComponentIndex = i,
                    LaneHeight = track.DefaultLaneHeight
                };

                _viewState?.Apply(row);
                _rows.Add(row);

                track.OnEnable();
            }

            RefreshTrackProperties();
        }

        /// <summary>
        /// Re-resolves each track's array element. Array-element properties are invalidated by any
        /// insertion, deletion or <c>Update()</c>, so they are resolved per repaint, never cached.
        /// </summary>
        private void RefreshTrackProperties()
        {
            if (_context == null) return;

            var components = _context.ComponentsProperty;
            if (components == null) return;

            foreach (var row in _rows)
            {
                row.Track.ComponentProperty = row.ComponentIndex < components.arraySize
                    ? components.GetArrayElementAtIndex(row.ComponentIndex)
                    : null;
            }
        }

        private void DisposeTracks()
        {
            if (_rows == null) return;

            foreach (var row in _rows)
            {
                row.Track?.OnDisable();
            }

            _rows.Clear();
        }

        private void SaveViewState()
        {
            if (_viewState == null || string.IsNullOrEmpty(_assetGuid) || _rows == null) return;

            foreach (var row in _rows) _viewState.Record(row);
            _viewState.Save(_assetGuid);
        }

        private void OnUndoRedo()
        {
            if (_context == null) return;

            _context.RefreshFromAsset();
            RebuildTracks();
            Repaint();
        }

        private void OnEditorUpdate()
        {
            var now = EditorApplication.timeSinceStartup;
            var deltaTime = (float)(now - _lastUpdateTime);
            _lastUpdateTime = now;

            if (!_isPlaying || _context == null || !_context.IsValid) return;

            // Playback loops the slice rather than the clip: the slice is the part that is kept.
            var sliceFrames = _context.SliceFrameCount;
            if (sliceFrames <= 1) return;

            _playbackTime += deltaTime;
            var duration = sliceFrames * _context.FrameTime;
            if (_playbackTime >= duration) _playbackTime %= duration;

            var sliceFrame = Mathf.Clamp(Mathf.FloorToInt(_playbackTime / _context.FrameTime),
                0, sliceFrames - 1);
            _context.SeekToClipFrame(_context.SliceToClipFrame(sliceFrame));
            Repaint();
        }

        private void TogglePlayback()
        {
            _isPlaying = !_isPlaying;
            if (!_isPlaying || _context == null) return;

            _playbackTime = _context.ClipToSliceFrame(_context.PlayheadClipFrame) * _context.FrameTime;
            _lastUpdateTime = EditorApplication.timeSinceStartup;
        }

        private void StepFrames(int delta)
        {
            if (_context == null) return;

            _isPlaying = false;
            _context.SeekToClipFrame(_context.PlayheadClipFrame + delta);
            Repaint();
        }

        private void DrawPreview()
        {
            var rect = GUILayoutUtility.GetRect(10f, 10000f, 10f, 10000f);

            if (_context == null || !_context.IsValid)
            {
                GUI.Label(rect, "Select an annotated clip.", EditorStyles.centeredGreyMiniLabel);
                return;
            }

            _preview.Source = _clip;
            _preview.Frame = _context.PlayheadClipFrame;
            _preview.Draw(rect, GUIStyle.none);
        }

        private void DrawTrackOverlays(SkeletonPreviewFrame frame)
        {
            if (_context == null || _rows == null) return;

            var context = new TrackOverlayContext(frame, _context);
            foreach (var row in _rows)
            {
                if (!row.Visible || row.Track == null || !row.Track.DrawsPreviewOverlay) continue;
                row.Track.DrawPreviewOverlay(context);
            }
        }

        private void DrawTimeline()
        {
            if (_context == null || !_context.IsValid)
            {
                RebuildIfClipReappeared();
                return;
            }

            RebuildTracksIfStale();
            _context.SerializedClip.Update();
            RefreshTrackProperties();

            _timeline.Axis = _axis;
            _timeline.Scroll = _timelineScroll;
            _timeline.HeaderWidth = _headerWidth;
            _timeline.FocusedRowIndex = _focusedRowIndex;
            _timeline.LabelsAsSeconds = _labelsAsSeconds;

            var rect = GUILayoutUtility.GetRect(10f, 10000f, 10f, 10000f);
            _timeline.Draw(rect, _context, _rows, ShowAddComponentMenu);

            _axis = _timeline.Axis;
            _timelineScroll = _timeline.Scroll;
            _headerWidth = _timeline.HeaderWidth;
            _focusedRowIndex = _timeline.FocusedRowIndex;
            _playheadClipFrame = _context.PlayheadClipFrame;

            if (_timeline.ViewStateChanged) SaveViewState();

            // A running mode follows the cursor by sampling it on repaint, so it needs repaints to
            // keep coming while no event does. See TimelineModalOperator.
            if (_context.Modal.IsActive) Repaint();

            if (_frameLabel != null)
            {
                _frameLabel.text = $"frame {_context.PlayheadClipFrame} / {_context.ClipFrameCount - 1}" +
                                   $"   slice {_context.StartFrame}-{_context.EndFrame}";
            }
        }

        private void RebuildIfClipReappeared()
        {
            if (_clip == null) return;
            if (_context != null && _context.IsValid) return;

            RebuildContext();
        }

        private void DrawInspector()
        {
            if (_context == null || !_context.IsValid)
            {
                EditorGUILayout.HelpBox("Select an AnnotatedAnimationClip to edit.", MessageType.Info);
                return;
            }

            _context.SerializedClip.Update();
            RefreshTrackProperties();

            EditorGUILayout.LabelField("Clip", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(_context.SerializedClip.FindProperty("clip"));
            EditorGUILayout.PropertyField(_context.SerializedClip.FindProperty("skeleton"));
            EditorGUILayout.PropertyField(_context.SerializedClip.FindProperty("rootMotionBone"));
            EditorGUILayout.PropertyField(_context.StartFrameProperty);
            EditorGUILayout.PropertyField(_context.EndFrameProperty);

            // Error, not Warning: a TryValidate failure means the asset cannot be baked at all.
            if (!((SkeletonAnimation)_clip).TryValidate(out var error))
            {
                EditorGUILayout.HelpBox(error, MessageType.Error);
            }

            using (new EditorGUI.DisabledScope(true))
            {
                EditorGUILayout.LabelField(
                    $"Clip frames: {_context.ClipFrameCount}   Slice: {_context.SliceFrameCount}   " +
                    $"Frame time: {_context.FrameTime:F4}s", EditorStyles.miniLabel);
            }

            DrawSeparator();
            DrawFocusedTrackInspector();

            _context.Commit();
        }

        /// <summary>A rule between the clip's own fields and the focused component's.</summary>
        private static void DrawSeparator()
        {
            EditorGUILayout.Space(6f);

            var rule = GUILayoutUtility.GetRect(1f, 1f, GUILayout.ExpandWidth(true));
            EditorGUI.DrawRect(rule, new Color(0.12f, 0.12f, 0.13f, 1f));

            EditorGUILayout.Space(6f);
        }

        private void DrawFocusedTrackInspector()
        {
            if (_focusedRowIndex < 0 || _focusedRowIndex >= _rows.Count)
            {
                EditorGUILayout.HelpBox("Select a track in the timeline to edit it.", MessageType.None);
                return;
            }

            var row = _rows[_focusedRowIndex];
            if (row.Track?.ComponentProperty == null) return;

            using (new EditorGUILayout.HorizontalScope())
            {
                EditorGUILayout.LabelField(row.Track.Title, EditorStyles.boldLabel);
                if (GUILayout.Button("Remove", EditorStyles.miniButton, GUILayout.Width(64f)))
                {
                    RemoveFocusedComponent();
                    return;
                }
            }

            row.Track.DrawInspector(new TrackInspectorContext(_context, row.Track.ComponentProperty, _preview));
        }
    }

    /// <summary>Lets a toolbar menu build its items each time it opens rather than once at startup.</summary>
    internal static class ToolbarMenuExtensions
    {
        public static ToolbarMenu WithDeferredMenu(this ToolbarMenu toolbarMenu, Action<DropdownMenu> build)
        {
            toolbarMenu.RegisterCallback<PointerDownEvent>(_ =>
            {
                toolbarMenu.menu.ClearItems();
                build(toolbarMenu.menu);
            }, TrickleDown.TrickleDown);

            return toolbarMenu;
        }
    }
}
