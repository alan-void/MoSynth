---
type: "Reference"
title: "Authoring a clip editor track"
openwiki_generated: true
sources:
  - id: openwiki-source-6b78588b55bb7110b93b174c
    resource: repo://Assets/AnimationTools/Editor/AnimationTools.Editor.asmdef
  - id: openwiki-source-a3b16f062ded3c00fc8975a4
    resource: repo://Assets/AnimationTools/Editor/AnnotatedAnimationClipEditor.cs
  - id: openwiki-source-f76a8e2afff29d4f4712a9e9
    resource: repo://Assets/AnimationTools/Editor/ClipEditor/ClipComponentTrackRegistry.cs
  - id: openwiki-source-a8d5cc43d012a805fdf4f2cf
    resource: repo://Assets/AnimationTools/Editor/ClipEditor/ClipTimeAxis.cs
  - id: openwiki-source-d0d15f9248a1d20396b252f2
    resource: repo://Assets/AnimationTools/Editor/Preview/ISkeletonPreviewDrawer.cs
  - id: openwiki-source-cb1b2ed1367e7e03ba0b1bee
    resource: repo://Assets/AnimationTools/Editor/Preview/SkeletonPreview.cs
  - id: openwiki-source-fbb65ec0e00c40c94f5ee81b
    resource: repo://Assets/AnimationTools/Editor/Preview/SkeletonPreviewOverlay.cs
  - id: openwiki-source-cc5f1479e3a4f206524b6c7c
    resource: repo://Assets/AnimationTools/Editor/SkeletonAnimationEditor.cs
  - id: openwiki-source-c61936b38c469ff3a1dff0b4
    resource: repo://Assets/AnimationTools/Tests/Editor/AnimationTools.Tests.asmdef
  - id: openwiki-source-00a29a5658285575f3a87255
    resource: repo://Assets/AnimationTools/Tests/Editor/ClipComponentTrackRegistryTests.cs
  - id: openwiki-source-8c51bed27ae9d7ec98d865e0
    resource: repo://Assets/AnimationTools/Tests/Editor/ClipTimeAxisTests.cs
generated: {by: "claude-code", at: "2026-09-02T21:44:37.029Z"}
---

# Authoring a clip editor track

Operational reference for `Assets/AnimationTools/Editor/ClipEditor/`. For what the window is and why
it is shaped this way, read [The annotated clip editor](../../animation-tools/clip-editor.md) first.

## The contract

```csharp
[ClipComponentTrack(typeof(MyComponent))]
public sealed class MyTrack : AnimationClipComponentTrack
{
    public override float DefaultLaneHeight => 44f;
    public override float RequestedLaneHeight => 0f;     // 0 = no opinion; see below
    public override bool DrawsPreviewOverlay => false;   // gate; false skips the overlay entirely

    public override bool TryGetSelectionRange(out int first, out int last) => false;
    public override bool TryGetContentRange(out int first, out int last) => false;

    public override void OnEnable() { }                  // Editor/Component already assigned
    public override void OnDisable() { }                 // dispose textures HERE

    public override void DrawTrack(in TrackDrawContext context) { }
    public override void DrawInspector(in TrackInspectorContext context) { }
    public override void DrawPreviewOverlay(in TrackOverlayContext context) { }
}
```

Requirements enforced by `ClipComponentTrackRegistry`, each of which logs a warning and skips the
type rather than throwing: derives from `AnimationClipComponentTrack`, non-abstract, public
parameterless constructor, and the attribute's type is an `AnimationClipComponent`. Lookup walks the
component's base chain up to but excluding `AnimationClipComponent`, so a track registered on an
abstract component family covers its subclasses. Two tracks on one component type resolve by
`Type.FullName` order with a warning — deterministic rather than dependent on assembly load order.

Everything lives in the existing `AnimationTools.Editor` asmdef. **Do not add a new one**: a
separate assembly would have to be referenced by anyone authoring a track, and `AnimationTools.Tests`
already references `AnimationTools.Editor`, which is what makes the pure types testable with no
plumbing. No `MackySoft.SerializeReferenceExtensions` reference is needed — the `[SubclassSelector]`
drawer resolves at runtime from the serialized field.

## Lane height: `DefaultLaneHeight` is read once

`AnnotatedClipEditorWindow` seeds `ClipTrackRow.LaneHeight` from `DefaultLaneHeight` when the row is
built, and rows are only rebuilt when the clip's **components list changes shape**
(`BuildTrackSignature`). A track whose content count is authored inside the component - a lane per
tag channel - would otherwise gain rows with no room to draw them and no way to say so.

`RequestedLaneHeight` is the answer: return the height the current content needs, and
`ClipTimelineView.ApplyRequestedHeights` applies it every repaint until the user drags the lane
themselves, at which point `ClipTrackRow.HeightIsUserSet` latches and their height wins for good.
Return 0 to opt out entirely, which every other track does.

## The shared keyframe keymap

`Editor/ClipEditor/Keys/` holds the machinery a lane of draggable keyframes needs. `AnimationTagTrack`
and `GaitPhaseTrack` both use it, and a third such track should too rather than hand-rolling drags.

- **`TimelineKeySelection`** - selection as a set of `(row, frame)` pairs. **Not indices.** Every
  edit re-sorts its list, so an index means something different afterwards; that is why the old
  footfall selection had to be cleared after every move. A frame is unique within a row, so it *is*
  the key's identity: after a move the selection is the old frames plus the delta, and a key that
  collided and cancelled falls out on its own via `Intersect`.
- **`TimelineKeyHitTester`** - `Pick` / `PickRange`, 6px tolerance, generalised over rows.
- **`TimelineModalOperator`** - the Blender modal state machine: `Begin`, then feed it events until
  it returns `Confirmed` or `Cancelled`. Grab, Scale and BoxSelect, with numeric entry.
- **`TimelineKeymap`** - the single key-to-action table, plus `Summary` for the inspector.

### How a key reaches a lane at all

This is a **UI Toolkit** window - `CreateGUI` builds three `IMGUIContainer`s and there is no
`OnGUI`. Keyboard events go to the panel's focused element, and an `IMGUIContainer` only becomes
that element once an IMGUI control inside it owns `GUIUtility.keyboardControl`. Three things have to
line up, and when they did not, **every** binding in every lane was silently dead:

1. **Handle keys off raw `e.type`, never `e.GetTypeForControl(id)`.** That method returns `Ignore`
   for a key event unless `keyboardControl` is that control, and a lane's control is
   `FocusType.Passive` so it never can be. Clicks *should* still go through `GetTypeForControl` -
   they must respect `hotControl`. This one bug killed `GaitPhaseTrack`'s `Delete` from the day it
   was written.
2. **`ClipTimelineView` claims IMGUI keyboard focus** on a lane click, via a `FocusType.Keyboard`
   control (`_laneKeyboardControlId`), without consuming the click. Without this, focus lives
   permanently in the inspector pane's property fields. `AnnotatedClipEditorWindow` does the UI
   Toolkit half by calling `Focus()` on the timeline container from a `PointerDownEvent`.
3. **A lane click focuses its row.** Key cases are gated on `context.IsFocused`, and before this the
   only thing that set `FocusedRowIndex` was a click on the 168px gutter.

Also: skip key handling while `EditorGUIUtility.editingTextField`, or typing in an inspector field
fires lane operators.

### Three more things about modal operators

1. **A mode follows the cursor by sampling `Event.current.mousePosition` on `Repaint`**, and the
   window repaints while `Modal.IsActive`. `wantsMouseMove` is *not* what carries this - it is an
   `OnGUI` window flag and this window has no `OnGUI`. Never call `Use()` on a repaint.
2. **Only one operator runs at a time**, owned by `ClipEditorContext` (`ClaimModal` / `HasModal` /
   `CancelModal`). Claiming cancels whoever had it. Two IMGUI operators both holding `hotControl`
   would deadlock the window.
3. **The control id is allocated by the track, unconditionally, every repaint**, and passed into
   `Begin`. Allocating one inside `Begin` would hand out a different id depending on whether a key
   had been pressed - which is exactly how IMGUI control ids drift between Layout and Repaint.
   `Begin` must **not** clear `keyboardControl`: that throws away the focus Enter, Escape and typed
   digits depend on.

**A cancel must write nothing.** The track previews the operator's value while it runs and commits
once on `Confirmed`. Committing and then undoing is not equivalent: a commit drops the clip's bake
(see below) and leaves an undo entry the user did not ask for.

### The gutter is the component list

`ClipTimelineView.HeaderWidth` is draggable and serialized on the window; the old
`ClipTrackHeaderColumn.Width` constant is gone. `Add Component` is drawn under the last row by
`ClipTrackHeaderColumn.DrawAddComponent`, inside the scroll view so it scrolls with the list, and
its height is part of `ContentHeight`. It is no longer in the toolbar.

The resize grip overlaps the header rect, and `HandleFocusClick` consumes any click inside that
rect - so `HandleHeaderResize` must run **before** `DrawRows`, not after.

### `A` is select-all, not frame-all

`ClipTimelineView` used to bind `A` to "frame the whole clip". It is now `Home`, with `.` for frame
selected - Blender's own view keys - because `A` is select-all in a keyframe lane and a Blender user
reaches for it constantly. `F` still frames the slice.

Both framing keys ask the tracks rather than the tracks acting on the axis: `TrackDrawContext.Axis`
is a **copy** of the `ClipTimeAxis` struct, so a track cannot zoom or scroll the view. A track that
wants framing must answer the question, not do the work.

- `.` asks the focused row via `TryGetSelectionRange`.
- `Home` asks **every visible row** via `TryGetContentRange` and frames the union, falling back to
  the whole clip when nothing answers. That is Blender's View All: it frames the keys, not the
  range. `Shift+Home` always frames the whole clip.

The min/max arithmetic behind both content ranges lives in the pure helpers - `AnimationTagEdits`
and `FootfallEdits` - not in the tracks, so it is unit-tested with the rest of the edit maths.

A track handling keys must therefore leave `Home` and `.` unclaimed - `TimelineKeymap` resolves them
so they can appear in the summary, but both tag and gait tracks fall through without `e.Use()`.

### Panning is a drag, not a `MouseDrag` case

`HandlePan` runs **before** `DrawRuler` and `DrawRows`, because alt+left is a pan to the timeline and
a box select to a track, and whichever sees the `MouseDown` first wins it. It takes `hotControl` like
every other drag in that file.

The cursor is wrapped by `CursorWrap`, which keeps it inside the lane rect and brings it back in at
the opposite edge - Blender's region wrap. Unity's own `SetWantsMouseJumping` wraps at the edge of the
*screen* and has no region-scoped variant, so it stays on underneath only as the fallback for
platforms `CursorWrap.Move` cannot serve; on Windows it can never fire. The window's `OnDisable`
calls `SetWantsMouseJumping(0)` in case it is closed mid-drag.

**Take the delta from `CursorWrap.DeltaWithin`, never from `Event.delta`, in a drag that wraps.**
Unity measures delta against the position it last saw, so the event after a wrap reports the jump as
motion - a visible lurch mid-pan. `DeltaWithin` subtracts it back out, which is why the wrap and the
delta live behind one call.

Neither the scroll nor the zoom is clamped to the clip any more: `ClipTimeAxis.ClampToContent` is
gone. Draw the ruler from `LeftEdgeFrame`/`RightEdgeFrame`, which report what is actually on screen
and go negative; index per-frame data with `FirstVisibleFrame`/`LastVisibleFrame`, which stay clamped
inside the clip and are what a lane is handed.

## Rules that lose data silently if broken

1. **Never commit during a drag.** `ClipEditorContext.Commit()` calls `ApplyModifiedProperties`,
   which fires `AnnotatedAnimationClip.OnValidate` → `ClearRuntimeCaches()` → `_poseSequence = null`,
   and the next preview repaint re-runs `AnimationClipBaker.Bake` over the whole clip. Hold a
   transient delta, render at `value + delta`, commit once on `MouseUp`. The symptom to watch for in
   the Profiler is `AnimationClipBaker.Bake` appearing once per mouse-move.
2. **Never delete annotation because the range moved.** A component's `OnValidate` runs on every
   validate, including one caused by a slice-handle drag, so a range clamp there destroys anchors
   permanently and silently. Both components used to do exactly this and it cost a real clip 44
   footfalls. Annotation frames are clip-local; readers ignore what they were not asked about, and
   `GaitPhase.UsableAnchors` was already filtering out-of-range anchors anyway. Clamp the frames an
   *edit produces* instead — see `AnimationTagEdits.ClampDelta` and its `InRange`.
3. **Keep anchor lists ascending.** `GaitPhase.UsableAnchors` drops any anchor not strictly later
   than its predecessor, with no message. `FootfallEdits.Move` re-sorts; selection indices are
   therefore invalid after a commit and must be cleared.
4. **Never mix write paths.** With pending serialized modifications outstanding, an
   `Undo.RecordObject` + direct field mutation is overwritten by the next `ApplyModifiedProperties`.
   `GaitPhaseTrack.Detect` is the only direct mutator in the codebase (because
   `GaitPhaseComponent.TryDetect` rewrites `footfalls` itself) and is bracketed:
   `Commit()` → `Undo.RecordObject` → `TryDetect` → `SetDirty` → `RefreshFromAsset()`.
5. **Never cache a `SerializedProperty` across repaints.** Array-element properties are invalidated
   by any insertion, deletion or `Update()`. `ComponentProperty` is re-resolved by the window every
   repaint in `RefreshTrackProperties`.

## Frame spaces

`ClipEditorContext` exposes `ClipFrameCount` and `SliceFrameCount` under those names because
`AnnotatedAnimationClip` shadows `FrameCount` and `GetFrame` with slice-local versions, and which one
binds depends on the expression's static type. **Do not write `FrameCount` or `GetFrame` on an
`AnnotatedAnimationClip`-typed expression in new code.** Use:

- `context.Editor.ClipFrameCount` / `SliceFrameCount`
- `context.Editor.SliceToClipFrame(n)` / `ClipToSliceFrame(n)` — for *pose* access only. **Annotation
  is in clip frames**, so a track drawing or editing it converts nothing; both existing tracks call
  `Axis.FrameToX(frame)` directly.
- `context.Editor.GetClipFrame(clipFrame)` — casts to `SkeletonAnimation` internally
- `SkeletonPreview.Source` is typed `SkeletonAnimation` for the same reason; leave it that way

The timeline axis is in **clip** frames. `TrackDrawContext.LaneRect` is authoritative for y;
`TrackDrawContext.Axis` is authoritative for x, and nothing else may derive an x from a frame.

## Drawing a lane

- Cull to `context.FirstVisibleClipFrame` / `LastVisibleClipFrame`. Anything per-frame that is not
  culled makes cost scale with clip length.
- A dense per-frame signal goes through `TimelineSignalTexture`: `Rebuild(frameCount, height,
  dataVersion, background, fill)` rasterises one column per frame (capped at 4096, downsampled
  beyond — use `ColumnToFrame`), then `TryMapRange(axis, laneRect, firstFrame, lastFrame, out dest,
  out uv)` and `Draw(dest, uv)`. Rebuild only when `Editor.DataVersion` moves; combine with a local
  counter for state that is not in the asset (`GaitPhaseTrack._contactsVersion` does this for
  detection contacts).
- Dispose every texture in `OnDisable`. Tracks are disposed when the clip changes, when the
  component list changes shape, and in the window's `OnDisable`.
- Draw and input share the method. Take `GUIUtility.hotControl` with a
  `GUIUtility.GetControlID(FocusType.Passive)` for a drag so it survives the cursor leaving the lane,
  and call `GetControlID` unconditionally so IDs stay aligned between Layout and Repaint.
- Anything the track does not `Use()` falls through to the timeline's own handling: zoom, pan and the
  view-framing keys. It does **not** scrub — the playhead moves from the ruler and the step keys only,
  so a click in a lane is free to mean selection. Do not seek from a click handler; both tracks used
  to, and selecting an anchor moving the pose under it is the bug that got them stopped.

## Drawing a preview overlay

`UnityEditor.Handles` **will not work** — it draws into the current GUI or Scene view camera, not the
preview's off-screen one. Use `context.Draw` (`ISkeletonPreviewDrawer`), whose segments are batched
into one mesh per repaint by `SkeletonPreviewOverlay`.

`context.BonePositions` / `BoneRotations` are the preview's own persistent buffers, already FK'd for
`context.ClipFrame`. They are overwritten on the next repaint: read them during the call, never store
them. `context.Skeleton` is a `SkeletonData` from `Skeleton.GetSkeletonData()` — `Allocator.Domain`,
owned by the `Skeleton`, **never dispose it**. The same goes for `SkeletonAnimation.PoseSequence`.
The only buffers `SkeletonPreview` owns are its two `Allocator.Persistent` FK arrays.

Guard `pose.Layout.RotationCount != skeleton.BoneCount` before any FK call:
`SkeletonData.LocalSpaceToCharacterSpace` throws `ArgumentException` through `CheckFullPoseLayout`
when the counts disagree, and a throw inside a repaint between `BeginPreview` and `EndPreview` leaves
the preview unbalanced, after which every later repaint reports "Previous BeginPreview() was not
closed" instead of the real problem.

## PreviewRenderUtility lifetime

One `SkeletonPreview` per host, never static or shared — two hosts interleaving Begin/End pairs
produce the unbalanced-preview failure above. It is created lazily inside a repaint, never in a
constructor. `Dispose` must be called from the host's `OnDisable` and is idempotent.

`SkeletonAnimationEditor.OnDisable` is `protected virtual`, and this is load-bearing. It was
previously `private`, and `AnnotatedAnimationClipEditor` declared its own `private void OnDisable()`
to dispose the gait strip — which *hid* rather than overrode the base method. Unity dispatches only
to the most-derived version, so on an annotated clip the base cleanup never ran and every inspector
opened leaked a `PreviewRenderUtility` and two `Allocator.Persistent` NativeArrays. The derived
inspector no longer declares one at all, so the base runs unshadowed; if you add one back, it must
be `protected override` and call `base.OnDisable()`.

## Domain reload

Only `[SerializeField]` fields on the window survive: `_clip`, `_locked`, `_playheadClipFrame`,
`_axis` (a `[Serializable]` struct whose `ViewRect` is `[NonSerialized]` and recomputed from layout),
`_timelineScroll`, `_focusedRowIndex`, `_labelsAsSeconds`. `SerializedObject`, `SkeletonPreview`,
tracks, textures and NativeArrays are `[NonSerialized]` and rebuilt in `OnEnable`. **Never hold a
`SerializedObject` across a reload.** Lane heights and visibility go to `SessionState` keyed by asset
GUID and component type name — keyed by type rather than list index so reordering components does not
shuffle which lanes are hidden.

The registry is built lazily on first use rather than from `[InitializeOnLoad]`, which can run before
every assembly is ready; `TypeCache` is rebuilt by the same reload that clears the static.

## Tag channels specifically

`AnimationTagComponent.OnValidate` runs `AnimationTagging.Normalise` per channel, which sorts and
cancels same-frame pairs and **drops nothing for being out of range**. A key beyond the clip's slice
is kept, drawn, selectable and deletable like any other. A *negative* key is still dropped, and that
is not a range rule: `IsOn` counts it for every frame from 0 up, so one inverts the whole channel.

Three more that look like bugs and are not:

- **Keys on the same frame cancel.** That is the parity rule, not an error to reject. A drag that
  lands one key on another is supposed to remove both, and `AnimationTagEdits` returns a normalised
  list so the caller can see what actually survived. Do not "fix" this by rejecting collisions.
- **A trailing unpaired key is on until the end of the clip.** `AnimationTagging.Spans` runs it to
  `clipFrameCount`. Do not append a closing key to "repair" it - that would make the annotation
  depend on the trim.
- **A key outside the slice is not stale data.** It is annotation about a part of the clip that is
  not currently extracted. Leave it; the user trims and untrims freely, and that is the whole point
  of the frame space.

`AnimationTagEdits` never mutates in place: every method returns a new normalised list, precisely so
a caller can compare what it asked for against what it got.

## Verification

Unit-testable (in `AnimationTools.Tests`, which already references `AnimationTools.Editor`):
`ClipTimeAxisTests`, `ClipTimelineTickTests`, `ClipComponentTrackRegistryTests`, `FootfallEditTests`.
Run with `MoSynth/Tests/Run EditMode Tests` and read `Temp/animtools_test_results.txt` — not the
console, which the domain reload clears.

`ClipComponentTrackRegistryTests` is the guard against the silent failure the attribute has: renaming
or moving a component type leaves its track registered against a type that no longer exists, and the
window falls back to `DefaultComponentTrack` with no error, because nothing can tell that apart from
a component that simply has no track. This is the same identity hazard `AnimationClipComponent`
documents for serialized data.

Manual checks that catch what the tests cannot:

1. Cross-check a pose against the clip's Inspector preview at the same frame number — this is the
   direct test that the shadowed `GetFrame` has not been got wrong. An off-by-`startFrame` shows here
   and almost nowhere else.
2. Force a domain reload with the window open, then again with the window *and* the Inspector open on
   the same asset (the two-`PreviewRenderUtility` case). Expect no NativeArray leak warnings and no
   "Previous BeginPreview() was not closed".
3. Delete the clip asset with the window open: it must degrade to the empty state, not throw
   "SerializedObject of destroyed Object" every repaint.
4. Add a component with no registered track and confirm `DefaultComponentTrack` renders it fully
   editable.
