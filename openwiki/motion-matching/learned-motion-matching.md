---
type: Reference
title: Learned Motion Matching (parked experiment)
description: A self-contained neural reimplementation that is not part of the live pipeline, kept for reference — what it does and why it will not run as committed.
tags: [experiment, barracuda, legacy, status]
sources:
  - id: openwiki-source-92d29d96fc61a044af5a807c
    resource: repo://Assets/Scripts/Learned%20Motion%20Matching/Scripts/Gameplay.cs
  - id: openwiki-source-322ec7da662c5b191cff726f
    resource: repo://Assets/Scripts/Learned%20Motion%20Matching/Scripts/MotionMatching.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# Learned Motion Matching (parked experiment)

> **This is not part of the synthesis pipeline.** It shares no type with the live architecture — no
> `MoSynthStage`, no `MotionSynthesisComponent`, no `PoseBuffer`, no `Skeleton`, no `PoseSet`. It
> drives Transform local position and rotation directly on a rig it discovers through an `Animator`.
> Nothing anywhere in the repository references it.

Five files under `Assets/Scripts/Learned Motion Matching/Scripts/`, ~1,386 lines, namespace `MM`. All
of them date to March 2026; every other file described in this wiki was touched in August.

## Read the name carefully

`MM.MotionMatching` is **a different type in a different assembly** from the `MotionMatching`
namespace the rest of this section documents. The folder has **no asmdef**, so it compiles into
`Assembly-CSharp` — which is also why the rest of the wiki's taxonomy, built on asmdef boundaries,
does not otherwise reach it.

It is also:

- **The repository's only `Unity.Barracuda` consumer.** `AGENTS.md` lists Barracuda as a key
  dependency, and that is true *only* of this isolated track.
- **Dependent on `UnityEditor` from runtime code**, reached from `Awake` via
  `AnimationUtility.GetAnimationClips` and `AssetDatabase.GetAssetPath`. So it **could not be included
  in a player build as it stands** — Editor-only by accident of its dependencies rather than by
  design.

## What it does

It is an implementation of Holden et al.'s *Learned Motion Matching*: replace the database and its
brute-force search with three small neural networks.

| Network | Role | Tensor |
| --- | --- | --- |
| **projector** | stands in for the nearest-neighbour lookup — query to nearest point in latent space | 24 → 56 |
| **stepper** | autoregressive velocity predictor, advancing the latent state | 56 → 56 |
| **decompressor** | reconstructs a full pose from the latent state | 56 → joints × 15 |

The 56-float latent is `x ‖ z` — **24 query feature floats plus 32 latent floats** — and every
normalise/denormalise loop branches on that boundary.

The 24-float query is assembled as three trajectory points (9), hips velocity (3), then right and
left foot position and velocity (12).

Per frame: run the projector every `projectorFreq` frames (default 20) to re-seed the latent state,
run the stepper every frame to integrate it, run the decompressor every frame to produce a pose. Each
joint gets 15 floats — 3 position plus a 6D rotation representation, Gram–Schmidt orthonormalised
into a rotation matrix and converted to a quaternion.

The data pipeline is Editor-only, driven by an "Extract data from animator" inspector button, and
writes plain text files next to the prefab.

## The three toggles are ablation switches

`enableProjector`, `enableStepper` and `enableDecompressor` are not leftovers. With the projector off,
the latent is filled directly from stored data rather than from the user; with the stepper off, the
query comes from stored data too. So the chain can be run decompressor-only, decompressor + stepper,
or complete — which is the experiment.

## Why it will not run as committed

`MM.MotionMatching` carries **no `[Serializable]` attribute**, while the driver holds it as a plain
`public MM.MotionMatching mm` field. So it will not serialize, and it will be `null` when the driver
calls `mm.Build(gameObject)` at `Awake`.

That is a static reading of the code, not a tested one. But combined with the March timestamps and
the total absence of references, treat the entry point as broken and verify against the source before
assuming any of it is live.

## Other divergences from the rest of the project

- Uses the **legacy `Input.GetAxisRaw`**, not the Input System asset everything else uses.
- Hardcodes a **60 Hz step in three separate places** rather than reading a frame time, with a
  commented-out frame limiter at the top of the update.
- Numerous magic numbers with no derivation: 24, 56, joints × 15, `projectorFreq = 20`.
- Debug input is live in the main loop — a key press advances the chosen clip.
- All three models read their output tensor by the hardcoded name `"y"`.

## Related artefact

`External/lmm-v0.3.0.zip` sits alongside the other vendored payloads.

## Source map

| File | Lines | Role |
| --- | --- | --- |
| `MotionMatching.cs` | 1,205 | the whole implementation — data pipeline, inference, pose reconstruction |
| `Gameplay.cs` | ~90 | the only MonoBehaviour; drives it, exposes the three toggles |
| `MMInput.cs` | ~23 | serialized input smoothing parameters |
| `MMInspector.cs` | ~24 | the extract-data button |
| `AnimationData.cs` | ~60 | an unrelated EditorWindow dumping curve bindings |
