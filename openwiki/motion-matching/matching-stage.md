---
type: Architecture Guide
title: The motion matching stage
description: Synthesis by search — playing a database frame by frame while periodically asking whether a better frame exists.
tags: [motion-matching, stage, search, playback]
sources:
  - id: openwiki-source-c7050471458b93768d777c1b
    resource: repo://Assets/MotionMatching/Runtime/Core/MotionMatchingSearch/MotionMatchingSearch.cs
  - id: openwiki-source-f84c8bceda0edfaac6926af8
    resource: repo://Assets/MotionMatching/Runtime/Core/MotionMatchingStage.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# The motion matching stage

`MotionMatchingStage` is synthesis by search. Every tick it plays the next frame of an animation
database, and periodically asks *"is there a better frame to be playing?"* — comparing what the
control input wants against every frame's precomputed feature vector, and jumping if the answer is a
clear yes.

It has three replaceable parts:

| Part | Field | Covered in |
| --- | --- | --- |
| the database | `mmData` | [pose database](../animation-tools/pose-database.md), [feature vectors](feature-vectors.md) |
| what the character is being asked to do | `controlInput` | [control inputs](control-inputs.md) |
| how the database is searched | `mmSearch` | [search backends](search-backends.md) |

It is an ordinary [`MoSynthStage`](../animation-tools/synthesis-pipeline.md) — a plain
`[Serializable]` class in the component's stage list.

## Two clocks

Playback and search run independently:

```csharp
if (_searchTimeLeft <= 0) { SearchForBetterFrame(); _searchTimeLeft = searchInterval; }
else                      { _searchTimeLeft -= deltaTime; }
AdvancePlayback(deltaTime);
pose.CopyFrom(_poseSet.GetPoseBuffer(CurrentFrame));
```

The playhead advances every tick; a search runs at most once per `searchInterval`, which defaults to
`10f / 60f` — ten database frames at 60 Hz, about 0.167 s. Searching is the expensive half, and
between searches the character simply keeps playing the clip it landed on.

A coarse interval only works because there is an escape hatch: a control input raises
`OnHighInputChange` when the player does something sudden, and the stage subscribes to it and zeroes
its timer, searching immediately rather than waiting out the interval.

Playback keeps a float playhead so fractional frame time carries across ticks, which keeps database
speed correct at any synthesis rate. The integer part is re-seeded from `CurrentFrame` each tick, so
a search jump is respected while the sub-frame remainder survives.

## The jump gate

A search does not simply take the best frame.

First, the stage **scores the frame already playing** and passes that distance in as the initial
best, so the search reports only something strictly better and returns −1 when nothing beats it —
"keep playing what is playing".

Second, a winner closer than `minFrameSwitchDistance` (default **20**) to the current frame is
ignored. Nearby frames look almost identical, so jumping to one costs a discontinuity and buys
nothing.

There is a real limitation in that gate: it compares **raw frame indices**, and indices are global
across the concatenated pose set. So a genuinely better frame five indices away *in a different clip*
is rejected as though it were adjacent.

When the stage does jump it sets `PoseDiscontinuity` so downstream blending can re-anchor. It sets it
for a second case too: **running off the end of a clip into the next one is a pose jump, same as a
search switch.**

## The query vector

`FillQueryVector` builds what the search compares against. Two things about it are worth knowing.

**Pose features come from the frame playing, not from the character.** That keeps the query in the
database's own pose space. Using the character's current pose would mean inverting the retargeting
first — there is a standing TODO saying exactly that.

**Bone trajectory channels are optional constraints**, switched on by a control input supplying a
target and switched off by weight-masking rather than by filling in a plausible value. That mechanism
is the subject of [feature vectors](feature-vectors.md), and it is the single most important thing to
understand about how the query behaves.

## Startup

`Init` loads the pose set and feature set, then refuses to run in two cases:

- Either set is unusable — it folds `mmData.TryValidate`'s message into the error and returns.
- **The component's skeleton is not the rig the database was built over.** The database's poses are
  stored over its own rig, so this is checked with `Skeleton.StructurallyEqual` and the error names
  the fix: assign that rig's root bone, or regenerate the database.

The playhead is seeded with the first frame the feature set marks valid, so playback starts from a
usable pose.

Note that after either failure the stage stays half-initialised and will throw if ticked — the guard
prevents a wrong result, not a crash.

## Known gaps

Several things here are wired but not finished. Documenting them honestly matters more than making
the design sound complete.

- **`mmSearch.Dispose()` is never called.** The stage overrides no `OnDestroy`, despite
  `MoSynthStage` documenting that as the only teardown hook and the search's `Dispose` documenting
  itself as releasing native memory. In practice this is bounded because the search result array uses
  `Allocator.Domain`.
- **`UpdateFeatureWeights` is never called** — it carries a `// TODO call from editor`. So the
  `responsiveness` and `quality` inspector sliders have **no runtime effect**; weights come only from
  the serialized per-float list, copied verbatim at `Init`.

  If it is ever wired up, note the ordering hazard baked into it: it reads one weight per feature
  *definition* from the head of **the very array it then fills in per float**. So it must snapshot
  those source values before its first write, or it would start consuming values it had already
  overwritten. Trajectory weights are scaled by `responsiveness`, pose weights by `quality`.
- **The `featureWeights` list has two conflicting interpretations.** Its documentation says one entry
  per feature *definition*, while `OnValidate` resizes it to the per-*float* feature size and `Init`
  copies it element-for-element. Both readings coexist in the code.
- **The tag mask is allocated all-true and never written.** The hook for tag queries ("only walking
  frames") exists; nothing populates it. See [search backends](search-backends.md).
- Foot-contact properties on the stage have private setters that are never assigned, and the
  root/adjustment surface throws — see
  [the pipeline's unimplemented block](../animation-tools/synthesis-pipeline.md).

## The database asset

`MotionMatchingData` is a ScriptableObject: the authored *recipe* — clips, skeleton, contact
configuration, and the trajectory and pose feature definitions — plus lazy accessors that import or
deserialize the baked databases.

It implements [`IPoseSetSource`](../animation-tools/pose-database.md), exposing exactly the subset the
pose pipeline reads so that a consumer wanting only a pose database need not be a full Motion
Matching asset.

`MaximumFramesPrediction` is the longest lookahead any trajectory feature needs; poses closer than
that to the end of a clip cannot be used for prediction.

`jointsLocalForward` is derived from the rig's rest pose, on the assumption that the exported FBX is a
T-pose facing Unity forward — with arms taking the character's *right* axis as their forward, since
arms point sideways in a T-pose. See [the simulation frame](../animation-tools/simulation-frame.md).

## Source map

| Concern | File |
| --- | --- |
| The stage | `Assets/MotionMatching/Runtime/Core/MotionMatchingStage.cs` |
| The database asset | `Assets/MotionMatching/Runtime/Unity/MotionMatchingData.cs` |
| Reaching the database from a control input | `Assets/MotionMatching/Runtime/Core/MotionSynthesisComponentExtensions.cs` |

**Tests.** None. No test in the repository touches `MotionMatchingStage`. The entire
`MotionMatching.Tests` assembly is one file, `MmTestData`, which provides fixtures and asserts
nothing.
