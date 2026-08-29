---
type: Architecture Guide
title: Debug visualizers
description: Four tools for looking at what the database holds and what the pipeline produced — two of them ordinary stages, which makes them a worked example of the extension seam.
tags: [debugging, gizmos, stages, profiling]
sources:
  - id: openwiki-source-c9ec04c146f7125c2e260739
    resource: repo://Assets/AnimationTools/Runtime/Core/MoSynthStage.cs
  - id: openwiki-source-fc0680cc4001c01cff2aea7d
    resource: repo://Assets/MotionMatching/Runtime/Core/ContactVisualizerStage.cs
  - id: openwiki-source-00352c9469db3e499c0b37c1
    resource: repo://Assets/MotionMatching/Runtime/Debugging/Visualisers/MotionMatchingDataVisualiser.cs
  - id: openwiki-source-afa2f1f4898ba41b5a1d8807
    resource: repo://Assets/MotionMatching/Runtime/Pose/PoseSetVisualizerStage.cs
  - id: openwiki-source-a764d6e44a1cfe129ac2be56
    resource: repo://Assets/MotionMatching/Runtime/Utils/GizmosExtensions.cs
  - id: openwiki-source-652de6e8eae5cbeaf5f5f4ed
    resource: repo://Assets/MotionMatching/Runtime/Utils/PROFILE.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
---

# Debug visualizers

Four tools, answering different questions at different points in the pipeline.

| Tool | Kind | Inspects |
| --- | --- | --- |
| `MotionMatchingDataVisualiser` | MonoBehaviour, gizmos | a **baked database** — poses and extracted features, before any synthesis |
| `PoseSetVisualizerStage` | **`MoSynthStage`** | a pose set played straight through |
| `ContactVisualizerStage` | **`MoSynthStage`** | the pose the **whole pipeline** produced |
| `GizmosExtensions`, `PROFILE` | static utilities | drawing primitives; hot-path timing |

## The visualizer stages are ordinary stages

`PoseSetVisualizerStage` and `ContactVisualizerStage` are `[Serializable]` classes with parameterless
constructors, authored into the same `[SerializeReference]` stage list as `MotionMatchingStage`. They
override `Init` and `Apply` and nothing else.

`PoseSetVisualizerStage` documents itself as able to **stand in as the only stage on a component** —
which is about the strongest demonstration of
[the extension seam](../animation-tools/synthesis-pipeline.md) available: a debug tool is a
first-class pipeline participant, not a special case bolted on beside one.

## Looking at a baked database

`MotionMatchingDataVisualiser` plays a `MotionMatchingData`'s baked contents back frame by frame as
gizmos, on a throwaway rig it builds itself.

Its value is specific: it shows the **features** as well as the poses, so a trajectory feature
pointing the wrong way — the usual cause of a search picking odd frames — **is visible here and
nowhere else**. It catches extraction mistakes before any synthesis is involved.

Direction gizmos for simulation-frame channels are anchored to the corresponding predicted *position*
rather than to the character, so the arrows sit where the trajectory says the character will be. With
playback off, the rig is parked at the origin in its rest pose so it stays readable.

Two things to know about it: it is declared in `namespace MotionMatching.Editor` while living in the
**runtime** assembly, so the namespace is misleading; and it disposes the pose set and feature set it
obtained from the shared cached accessors, in both `OnDestroy` and `OnApplicationQuit` — disposing
assets it does not necessarily own.

## Looking at a pose set

`PoseSetVisualizerStage` plays a database straight through, looping, with **no search and no control
input**. It is the way to check an import, an extraction or a retarget with the rest of the pipeline
out of the way.

Like the matching stage it plays poses stored over the database's own rig, so it refuses to run
unless the component's skeleton is that rig — with three distinct, actionable failure messages. On
any failure `Apply` becomes a no-op returning `true`, so the pipeline continues with the pose
untouched.

It keeps a float playhead so playback stays at database speed at any synthesis rate.

## Looking at the pipeline's output

`ContactVisualizerStage` estimates foot contact for a set of bones and draws it as a cross per bone —
green for contact, red otherwise. It **never mutates the pose**.

It applies the *same* velocity-threshold test that
[`PoseExtractor`](../animation-tools/pose-database.md) uses when baking contacts, over the same
skeleton. But what it sees is not the same input: **this is the pose the whole pipeline produced,
after searching and blending.** So it answers *"is this foot planted right now"*, not *"which frames
will the next bake mark"*.

It carries a **private pose buffer in its own layout** — the full pose channel set plus one bool
channel per configured bone — because the pipeline's layout only carries the two built-in foot
contacts, and a read-only stage must not change the layout everything else shares. That is a neat
illustration of the `BuildFullPoseChannels` seam described in
[pose buffers](../animation-tools/pose-buffers.md).

It appears in no live scene or prefab; `PoseSetVisualizerStage` appears in five.

## Drawing

`GizmosExtensions` exists because Unity's gizmo lines have no thickness. It routes through
`Handles.DrawBezier` with a degenerate control polygon to get a thick line, and scales thickness by
camera distance so gizmos do not turn into slabs when you get close.

The entire file — including the `namespace` — is wrapped in `#if UNITY_EDITOR`, so it exists only in
Editor builds and every caller must guard its own use.

## Profiling

`PROFILE` is hand-rolled timing for the motion matching hot path, kept separate from Unity's profiler
because it **accumulates min, max and a rolling average across frames** rather than showing one frame
at a time — which is what matters for a search whose cost swings with the query.

It compiles out entirely unless `PROFILE_MOTION_MATCHING` is defined at the top of the file, so
instrumentation can be left in permanently at zero cost. The `SHOUTING_CASE` names are intentional:
they make instrumentation stand out from the code being measured.

**The define is currently commented out**, so every helper compiles to an empty body and the data
accessor always returns null.

## Source map

| Concern | File |
| --- | --- |
| Baked database gizmos | `Assets/MotionMatching/Runtime/Debugging/Visualisers/MotionMatchingDataVisualiser.cs` |
| Straight playback stage | `Assets/MotionMatching/Runtime/Pose/PoseSetVisualizerStage.cs` |
| Contact estimation stage | `Assets/MotionMatching/Runtime/Core/ContactVisualizerStage.cs` |
| Thick gizmo lines | `Assets/MotionMatching/Runtime/Utils/GizmosExtensions.cs` |
| Hot-path timing | `Assets/MotionMatching/Runtime/Utils/PROFILE.cs` |

The motion field has its own, much larger visualizer with a different purpose — drawing the field as
a point cloud. See [the pose manifold](../motion-field/pose-manifold-embedding.md).
