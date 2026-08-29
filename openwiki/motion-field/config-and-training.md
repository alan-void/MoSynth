---
type: Operations Guide
title: Config, training and the staleness contract
description: The end-to-end chain from clips to a trained value function, and the two flags that decide whether an artefact still matches the config that will run it.
tags: [config, training, staleness, editor, operations]
sources:
  - id: openwiki-source-8d73966f36fbc957efe02484
    resource: repo://Assets/MotionField/Editor/MotionFieldConfigEditor.cs
  - id: openwiki-source-45348d10be6243abe134ddac
    resource: repo://Assets/MotionField/MotionFieldConfig.cs
  - id: openwiki-source-13742752b942a8c72fc71381
    resource: repo://Assets/MotionField/MotionFieldStage.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# Config, training and the staleness contract

`MotionFieldConfig` is the single asset holding everything a motion field needs: which animations
form the database, and the hyperparameters used to train and run the value function over it.

It implements [`IPoseSetSource`](../animation-tools/pose-database.md) so it can produce its own
`.mmpose` **without being a `MotionMatchingData`**, and it deliberately carries none of the
trajectory or pose feature machinery — a motion field searches on full-body pose and velocity, so a
Motion Matching feature set has no meaning here. Its `MaximumFramesPrediction` is 0: no trajectory
features, so every pose is usable.

## The artefact chain

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    C["MotionFieldConfig<br/>clips · skeleton · hyperparameters"]
    C -->|Generate Pose Database| P[".mmpose"]
    P -->|Train Motion Field| V[".mffield.npz"]
    P -->|Compute UMAP Embedding| E[".mfembed.npz"]
    V --> S["MotionFieldStage.Init"]
    E --> S
    P --> S
    S --> T["per-tick policy calls"]
```

All three live in `StreamingAssets/MotionFields/<config name>/` — a folder of its own rather than
`MMDatabases`, which the Motion Matching inspector owns and regenerates.

Every step runs **in-process through PythonNET, synchronously on the main thread**. Training takes
roughly 20 s on a GPU for a ~7.8k-state database.

## The staleness contract

This is the part that decides whether the character moves correctly, and it is stated in four places
in the source. Here it is once.

**There are two flags, not one.**

| Flag | Records | Cleared by |
| --- | --- | --- |
| `hasPoseDatabase` | the `.mmpose` matches the config as it now stands | any config edit |
| `hasTrained` | the `.mffield.npz` matches the config as it now stands | any config edit, **and regenerating the database** |

Regenerating the database always drags the value function with it, because **extraction renumbers
every state the value function indexes**. There is no path that clears the database flag without also
clearing the trained flag.

### Why it deliberately over-flags

The editor's change check cannot tell *which* field moved, so **any edit clears the flags**. That is a
choice, and the reasoning is explicit:

> Over-flagging costs a rebuild; under-flagging costs a character that quietly moves worse than it
> should.

The failure being avoided is nasty precisely because it is silent. A value function trained against a
re-extracted `.mmpose` **addresses the wrong poses** — it reads plausible numbers off the wrong rows,
and the character merely moves badly rather than visibly failing.

Where staleness *is* knowable more precisely, it is used: editing a bone weight changes the metric but
not the extraction, so it clears only `hasTrained`.

### Two details that look like bugs and are not

**Both flags start `true`.** A config saved before the flag existed was, by definition, current when
it was last generated — so it should not demand a pointless rebuild the first time it is opened. And
a config that has never been generated has no `.mmpose` at all, because **every reader takes the
file's absence, not the flag, as the answer**.

**The change check is scoped to the fields, not the whole inspector.** Unity's global changed flag is
also set by a button press, so a blanket check at the end of the method would wipe `hasTrained` the
instant the Train button set it.

`hasTrained` is set **only on the success path**: a failed training run leaves the old value function
on disk, and it is no less stale than it was a moment ago.

### The gap

**The UMAP embedding participates in none of this.** There is no `hasEmbedding` flag; the only guard
anywhere is a state-count comparison, and the editor's mitigation is prose in a help box. See
[the pose manifold](pose-manifold-embedding.md) for what that leaves open.

The runtime stage also checks only `hasTrained`, never `hasPoseDatabase`.

## Bone weights

Per-joint emphasis on the similarity metric, multiplying on top of the global position and velocity
weights. Zero removes a joint from matching entirely.

**Keyed by joint name, not index.** The metric sums over joints in depth-first order, which is not
guaranteed to survive a skeleton being re-extracted with a different hierarchy — resolving by name
means a moved joint takes its weight with it rather than silently applying it to whatever now sits at
that index.

The list is **sparse and hidden from the default inspector**, because as a raw list it is a couple of
dozen nameless elements with no indication of which bone each one is. The custom editor draws every
joint of the database as an indented hierarchy row instead, and reads **the config's own pose
skeleton** rather than the source clips' — a clip can be authored against a differently shaped rig,
so its joint list would not line up.

Only entries differing from 1 are stored; an entry returning to neutral is dropped so the asset does
not accumulate rows that say nothing. Orphan entries naming joints the current skeleton lacks are
surfaced rather than silently kept, because keeping them would make the summary lie.

## Hyperparameters worth understanding

Most are ordinary knobs. Three carry real arguments.

**`posWeight` / `velWeight`** — the defaults put **roughly three quarters** of the distance on joint
positions. That ratio matters: a velocity-dominated metric matches states that merely *move* alike
with pose ignored, **which is how the field wanders into idle poses mid-stride**. The velocity weight
is small (0.06) because velocities are in m/s and their raw magnitudes run several times the position
half's metres.

**`locomotionFactor`** — a reward bonus for landing in moving states. This is not optional tuning: at
0, **every reward is ≤ 0, so an idle pose that faces the goal scores a perfect zero forever and the
policy freezes into it.** Changing it requires a retrain.

**`kNeighbors`** — neighbours considered per step, and *also* the number of candidate actions, since
each action emphasises one neighbour.

`tugRatio` is the per-step pull back toward the nearest database state: it stops the field drifting
into regions it has no data for, and too high just replays the database. `gamma` at 0.99 gives roughly
a 100-frame planning horizon.

## Running it

The inspector has five sections and three buttons that do work.

**Generate Pose Database** is gated on the same validation the importer runs, so a config with no
skeleton — or one its clips do not line up with — cannot be generated into a database that would fail
to load. On success it sets `hasPoseDatabase` and clears `hasTrained`.

**Train Motion Field** locks assembly reloads for the duration, because the interpreter runs in this
process and a domain reload mid-call takes the editor down with it. It is **disabled in play mode**,
since it would hold the Python GIL for seconds at a time and stall the running stage. A progress
callback is marshalled into Python as a callable; because training is synchronous on the main thread,
it fires on the main thread and can touch the editor UI directly.

**Compute UMAP Embedding** has the same execution model and is independent of training — the value
function does not need it, and deleting it only turns the visualizer off.

There is also a one-time **Import** that copies the contact threshold and contact bones from another
pose-set source. It is a copy, not a link: later edits to the source do not follow. Because a
`SkeletonBone` names a Transform rather than a bare string, a copied name is resolved against this
config's own rig immediately rather than carried across unresolved.

## Source map

| Concern | File |
| --- | --- |
| The asset, paths, flags, hyperparameters | `Assets/MotionField/MotionFieldConfig.cs` |
| Authoring, the three buttons, bone-weight drawer | `Assets/MotionField/Editor/MotionFieldConfigEditor.cs` |
| Training | `Python/motion_field_trainer.py` — see [value function training](../python/value-function-training.md) |
| The `.npz` format | `Python/motion_field_io.py` |

**Tests.** None.

## What is not recorded anywhere

The `.mffield.npz` persists only the heading grid, `gamma` and `k_neighbors`. It records **nothing**
about `locomotionFactor`, `posWeight`, `velWeight`, `tugRatio` or the bone weights — yet the runtime
re-applies the locomotion bonus and re-computes the metric using the config's current values. Change
one without retraining and the runtime's scoring silently disagrees with the fitted value function.

`hasTrained` is the only thing standing between you and that, which is why it over-flags.
