---
type: Architecture Guide
title: The pose manifold and its visualizer
description: Projecting the motion field to 3D so a stalled policy can be watched rather than inferred — why the cloud is poses and the edges are velocity.
tags: [visualization, umap, debugging, motion-field]
sources:
  - id: openwiki-source-13742752b942a8c72fc71381
    resource: repo://Assets/MotionField/MotionFieldStage.cs
  - id: openwiki-source-cb7ea97444208042d30c5cbd
    resource: repo://Assets/MotionField/MotionFieldVisualizer.cs
  - id: openwiki-source-dcb665e7eaa7b9c0f9aaf867
    resource: repo://Python/motion_field_embedding.py
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# The pose manifold and its visualizer

The motion field is a cloud of motion states in a **138-dimensional** similarity space — 46 weighted
joint position/velocity triples. Nothing about a stalled policy is visible in that space directly, so
it is projected to 3D once, offline, and drawn in the scene.

That turns *"the character froze"* into a picture of **where in the field it froze**.

This page covers the whole chain, because splitting it would put the two halves of one mechanism on
different pages: the config knob, the Python projection, the C# loader, and the visualizer that draws
it.

## The cloud is poses. The edges are velocity.

This is the idea to internalise before reading the picture.

Under the default feature mode the projection covers **joint positions only**. So the cloud is a
**pose manifold**: one pose held at two different speeds lands in *one place* rather than two, and how
far apart two points sit means how differently the body is posed — and nothing else.

Velocity does not disappear from the picture. It **changes form**.

Because a state's velocity is defined as the difference to the next frame, a state's one-frame
lookahead *is* the next state. In a position-only projection the velocity of state *i* therefore
points exactly at the point for state *i+1* — which is the frame-adjacency edge already being drawn.

> **The cloud is the set of poses; the edges are the flow over them.**

The alternative was tried and rejected. Embedding the *whole* similarity feature lays out a phase
space, where one pose held at two speeds lands in two places and nothing on screen distinguishes
"a different pose" from "the same pose, moving differently" — which is what made the first version of
this cloud hard to read.

`PositionAndVelocity` is still available as a config option, and switching to it silently invalidates
the "velocity is an edge" reading. Nothing at draw time detects which mode produced the file.

## The projection uses the field's own metric

The field's k-NN metric is a **sum of per-joint L2 norms** — a mixed norm, not the flat L2 that UMAP
would otherwise assume. So the neighbour graph is built here with the field's metric and handed to
UMAP precomputed.

Embedding under the wrong metric would draw neighbourhoods **the field does not actually see**, which
would make the picture actively misleading rather than merely approximate.

One consequence follows from `precomputed_knn`: UMAP cannot `transform()` a novel point, so the live
pose cannot be projected exactly. Unity instead places it at the **similarity-weighted barycenter of
its neighbours' embeddings** — which costs nothing and is the same interpolation the field already
uses to read values.

Raw UMAP output has no meaningful units or origin, so it is centred and scaled into a box before
being placed in the scene.

## Reading the overlay

On top of the static cloud, the visualizer draws what the policy is doing right now.

| Colour | Meaning |
| --- | --- |
| **yellow** | the live pose, with a yellow edge ahead and a trail behind |
| **cyan** | its nearest state |
| **magenta** | the tug target — the neighbour the *chosen action* emphasises |
| **green** | the rest of the neighbourhood, size and brightness following similarity weight |

**A policy that stalls shows up immediately: the trail stops sweeping the cloud and collapses to a
knot.** That is the diagnostic the whole thing exists for.

### Three things that will mislead you otherwise

1. **The tug is routinely not the nearest state.** The tug is the neighbour the chosen action
   emphasises, and actions are chosen on **value**, not distance. When the two do coincide the tug
   colour wins at the nearest marker's larger size, so they can always be told apart.
2. **The live pose's edge shows flow, not choice.** It is built from the plain similarity weights, so
   it shows where the *neighbourhood* flows — not which action the policy picked. The magenta marker
   carries that.
3. **UMAP preserves neighbourhood structure, not distance.** Two states adjacent in the real metric
   can land far apart on screen. Judge closeness by the markers, never by the gap.

### The live pose's velocity edge

Every point in the cloud shows its velocity as the edge to the next frame of its clip. The live pose
cannot: **it is not a database state, so it has no successor to link to.**

Its neighbours each do have one. Averaging where they go next — under the same weights that placed the
live marker itself — puts the head of the edge where the field's blended step lands. Point to point,
on the same terms as every other edge in the picture.

The lookahead is one frame by default, matching every other edge. A longer one exists because one
frame is only **about 1.8% of the cloud's extent**, which at a large point scale is no longer than the
marker the edge leaves. Walking the successor chain keeps every head on a real projected state, and
the same count applies to every pose — **so a longer edge still never means a faster one.**

Successors are derived by inverting the edge list rather than assuming the next state is the next
index, which respects clip boundaries for free.

Edges are drawn as a gradient — darkened at the tail, brightened at the head — rather than with
arrowheads: it costs two vertices instead of a cone, and unlike a flat arrow it never disappears when
viewed edge-on. The live pose's edge also gets a head marker, because seen end-on a line has no
visible direction at all and this is the one edge whose direction is the whole point.

## Failure is always "draw nothing"

The embedding is **optional debug data**, so every failure path leaves it null and lets synthesis
carry on — the visualizer simply draws nothing. The Python loader returns `None` rather than raising
on any problem, because it is called inside `Py.GIL()` where an exception would surface as an opaque
managed error.

`MotionFieldConfig` says the same from the other side: the file is safe to delete.

Unity passes its own log callback in specifically so rejection reasons reach the Editor console —
Python's default `print` goes to a stdout PythonNET does not forward, which would make a stale
embedding look like a silent no-op.

The visualizer also disables itself with a warning when the stage is missing or `collectDebugData` is
off, because without that it is a silent no-op, **which reads as "the tool is broken" rather than
"the switch that feeds it is off"**.

## The length-collision hazard

> **The state count is the only thing checked. Nothing hashes the database.**

So an embedding fitted on a *different* database of the same length is loaded and drawn as if it were
current. Every row of the embedding is a row of the pose database, so one loaded against a
re-extracted `.mmpose` addresses the wrong states.

This is structurally the same failure as a stale value function — but unlike that one, **there is no
flag to catch it**. `hasTrained` has no embedding equivalent; see
[the staleness contract](config-and-training.md).

**Recompute the embedding after every Generate Pose Database.**

## Source map

| Concern | File |
| --- | --- |
| Drawing, overlay, trail, velocity edge | `Assets/MotionField/MotionFieldVisualizer.cs` |
| Loading and rejection | `Assets/MotionField/MotionFieldStage.cs` (`LoadEmbedding`) |
| The knobs | `Assets/MotionField/MotionFieldConfig.cs` (`umapFeatures`, `umapNeighbors`, `umapMinDist`, `umapComponents`, `umapSeed`) |
| The projection | `Python/motion_field_embedding.py` |

Add the visualizer to the same GameObject as the `MotionSynthesisComponent` whose stage list contains
a `MotionFieldStage`, and enable `collectDebugData` on that stage.

Colour modes: `Speed` (default) ramps red to cyan by root speed and **makes dead zones obvious**;
`StateIndex` cycles hue with database position; `Uniform` leaves all the signal to the overlay.

The UMAP seed is fixed at 42 because UMAP is stochastic and a fixed seed keeps the picture stable
between rebuilds — so a change on screen means a change in the data.

**Tests.** None.
