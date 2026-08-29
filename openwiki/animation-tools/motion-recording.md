---
type: Architecture Guide
title: Motion recording
description: Capturing a run to a flat float file plus a self-describing manifest, and the channel model that decides what lands in it.
tags: [recording, channels, evaluation, python-interop]
sources:
  - id: openwiki-source-e574d54dea1fdc6eed27d71c
    resource: repo://Assets/AnimationTools/Runtime/Recording/BuiltInRecorderChannels.cs
  - id: openwiki-source-bdd63f48a3fa038e3b180e9a
    resource: repo://Assets/AnimationTools/Runtime/Recording/EvaluationRecorderChannels.cs
  - id: openwiki-source-56fd5b04c784f0b9ee34f2cb
    resource: repo://Assets/AnimationTools/Runtime/Recording/MotionRecorder.cs
  - id: openwiki-source-683a6624a343d6c52deaa3a4
    resource: repo://Assets/AnimationTools/Runtime/Recording/RecorderChannel.cs
  - id: openwiki-source-5fe68df00820295cf5339d41
    resource: repo://Assets/AnimationTools/Runtime/Recording/RecordingReader.cs
  - id: openwiki-source-7773ac20f5f53c3dfba969e8
    resource: repo://Assets/AnimationTools/Tests/Editor/RecordingLayoutTests.cs
  - id: openwiki-source-99e8bee4ddb5a1d470ccd6f5
    resource: repo://Assets/AnimationTools/Tests/Editor/RecordingReaderTests.cs
  - id: openwiki-source-2fd528172fa38333a2639e4d
    resource: repo://Assets/MotionMatching/Runtime/Recording/MotionMatchingRecorderChannels.cs
  - id: openwiki-source-a1c21aa9e5c5e3bd36ff227c
    resource: repo://Assets/Scripts/JointLivePlotter.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
---

# Motion recording

`MotionRecorder` writes a configurable list of channels to a flat float32 `.bin` plus a JSON
manifest, driven by `MotionSynthesisComponent.OnPoseApplied`. It owns file lifecycle, layout
construction and manifest emission — and nothing else. It does not interpret the data;
`RecordingReader` reads it back and the metric calculators consume that.

Channels own their own width and their own sampling. The recorder concatenates.

## The channel model

Recorder channels build on [the layout system](channel-layout-system.md), but diverge from pose
channels in two ways.

**One contiguous tail section.** Every recorder channel — time, bone transforms, the full pose,
custom ones — lives in the single `RecorderSections.Recorded` section, which sits after every
built-in pose section. Recorder channels have no reason to be ordered by kind the way a
`PoseLayout`'s sections do, so offsets follow **authored order** directly.

**Identity includes the name.** Where a `BoneChannelDescriptor` is identified by (type, bone, usage),
a recorder channel is identified by (concrete type, name, config) — because these are
inspector-serialized through a `[SerializeReference]` list and therefore need default constructors
and mutable fields. Two `BoneWorldPositionChannel`s differing only by `name` coexist and get distinct
offsets.

**Binding happens before the layout is built.** A channel's `FloatCount` may depend on state that
binding resolves — `FullPoseChannel` throws outright if its width is read while unbound, because it
cannot know the width until it has the synthesizer's `PoseLayout`.

## What gets recorded

Three files' worth of channels, split by intent:

- **`BuiltInRecorderChannels`** — generic and replayable: `TimeChannel`,
  `BoneWorldPositionChannel`, `BoneWorldForwardChannel`, `FullPoseChannel`.
- **`EvaluationRecorderChannels`** — things that only make sense while benchmarking:
  `StageCostChannel`, `FootContactChannel`, `PoseDiscontinuityChannel`,
  `ContactBoneWorldPositionChannel`. Two of them read state the synthesis component publishes solely
  for a harness.
- **`MotionMatchingRecorderChannels`** — those needing a control input, which only exists in a motion
  matching stack: `PathTargetPositionChannel`, `TargetSpeedChannel`.

### Execution order is load-bearing

`MotionRecorder` carries `[DefaultExecutionOrder(1000)]` so it runs after every other synthesis-tied
behaviour. That is what makes a bone transform's world position **this tick's answer, not last
tick's** — sampling always happens after the pose has been applied to the Transforms.

### NaN, never the origin

When a lookup cannot be resolved, the channel records `float.NaN` rather than zero, **so a broken
setup shows up in the data rather than reading as the origin**. `TargetSpeedChannel` does the same
with a scalar. Both also log an error at bind time.

This matters because a `(0,0,0)` in a path-target column is a perfectly plausible measurement, and a
NaN is not.

### Two channels that reach into the harness

`StageCostChannel` switches `MotionSynthesisComponent.MeasureStageCost` on when it binds — that is
what makes the component take the per-stage timestamps at all — and **nothing turns it back off**.
Acceptable only because the component is torn down at the end of a benchmark run. Its profiler
recorder is released through a `DisposeRecorder` call that is deliberately outside the
`RecorderChannel` contract; the benchmark driver calls it after stopping the recorder.

`ContactBoneWorldPositionChannel` is deliberately **not** a `BoneWorldPositionChannel` with a
hand-picked bone. Footskate is the distance a bone travels while its own flag says planted, so the
position and the flag have to name the same bone. Going through `BoneNameConventions` — exactly as
[`PoseLayoutBuilder`](pose-buffers.md) does — is what guarantees that, rather than leaving it to
whoever fills in the inspector.

## Lifecycle

```
StartRecording()
  ├─ guards: already recording / no synthesizer / PoseLayout null / empty channel list
  ├─ channel.Bind(synthesizer)   for every channel      ← before the layout
  ├─ Layout = new RecordingLayout(channels); bind one handle per channel
  ├─ allocate StateBuffer (Persistent) + float[] + byte[] scratch
  ├─ resolve output dir, open <name>_<yyyyMMdd_HHmmss>.bin
  └─ subscribe OnPoseApplied
per sample → channel.Sample(...) → BlockCopy float[] → byte[] → one Stream.Write
           → flush every flushEveryFrames (default 300; 0 = only on stop)
StopRecording() → unsubscribe, flush, WriteManifest(), dispose
```

`RecordTrigger` selects `EverySynthesisUpdate` (write immediately from the callback) or
`EveryUnityFrame` (cache the pose, write from `LateUpdate` with `Time.deltaTime`). Because
`OnPoseApplied` does not fire on frames the synthesis rate limiter skips, the two triggers produce
different row counts for the same run.

The `Synthesizer` property refuses reassignment while recording. `StopRecording` is idempotent and is
called from both `OnDisable` and `OnDestroy`.

Failure semantics degrade rather than throw: an unresolved bone falls back to `Hips` then to bone 0
with a warning; an uncreated pose zero-fills; an unavailable profiler recorder writes NaN.

## Reading it back

`RecordingReader.Load(manifestPath)` reads the JSON, resolves the data file relative to it, and loads
the floats. Channels are looked up by name.

**Frame count comes from the file length, not the manifest** — integer division discards a partial
trailing frame, so a recording survives a crash mid-write. The manifest's own `frameCount` is written
but deliberately ignored; the reader always trusts the bytes.

Or skip C# entirely — the manifest is self-describing enough to load with numpy alone. See
[on-disk formats](on-disk-formats.md) for the documented recipe and the full manifest schema.

## Operations

Two ways a recording is produced:

- **Authored** — drop a `MotionRecorder` on the character alongside the synthesis component, fill the
  channel list in the inspector, leave `autoStartOnPlay` on. Output lands in
  `<projectRoot>/Recordings/<name>_<timestamp>.{bin,json}`.
- **Harness-constructed** — [`PathFollowingMetric`](path-following-metrics.md) and the
  [benchmark driver](benchmarking.md) both `AddComponent<MotionRecorder>()`, set
  `autoStartOnPlay = false`, assign channels programmatically and call `StartRecording()` explicitly.
  The benchmark's fixed set is time, root, root forward, both feet, contacts, discontinuity and cost,
  plus the full pose when asked.

## Not part of this

`Assets/Scripts/JointLivePlotter.cs` is a standalone live-charting tool: it finite-differences one
`Transform` at an interval and pushes into XCharts line charts. It shares no type, namespace or file
with the recording stack, persists nothing, and does not use `RecorderChannel`. It also uses
`UnityEditor` from a runtime-folder MonoBehaviour, so it would not compile into a player build, and
its cleanup is commented out. Treat it as an unrelated, unfinished debug tool.

## Source map

| Concern | File |
| --- | --- |
| Recorder lifecycle and file writing | `Assets/AnimationTools/Runtime/Recording/MotionRecorder.cs` |
| Channel base and section key | `Assets/AnimationTools/Runtime/Recording/RecorderChannel.cs` |
| Layout | `Assets/AnimationTools/Runtime/Recording/RecordingLayout.cs` |
| Generic channels | `Assets/AnimationTools/Runtime/Recording/BuiltInRecorderChannels.cs` |
| Benchmark-only channels | `Assets/AnimationTools/Runtime/Recording/EvaluationRecorderChannels.cs` |
| Control-input channels | `Assets/MotionMatching/Runtime/Recording/MotionMatchingRecorderChannels.cs` |
| Read-back | `Assets/AnimationTools/Runtime/Recording/RecordingReader.cs` |

**Tests.** `RecordingLayoutTests` asserts offsets follow authored order, that channels differing only
by name coexist, that duplicate identities throw, and that the recorder section really does sit after
every pose section. `RecordingReaderTests` covers channel lookup, row-major addressing and the
partial-trailing-frame truncation. There is no test for `MotionRecorder` itself or for any channel's
`Sample` body.
