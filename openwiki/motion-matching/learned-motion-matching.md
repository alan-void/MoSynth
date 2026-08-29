---
type: Reference
title: Learned motion matching
description: What the repository provides toward replacing the database search with learned networks, and the parked Barracuda experiment that was removed to make room for it.
tags: [status, roadmap, neural, training-data]
sources:
  - id: openwiki-source-13742752b942a8c72fc71381
    resource: repo://Assets/MotionField/MotionFieldStage.cs
  - id: openwiki-source-5d4680328f7da95f88649f31
    resource: repo://Packages/manifest.json
  - id: openwiki-source-b7a0fd38fcd115e75814f58b
    resource: repo://Python/feature_set_importer.py
  - id: openwiki-source-58d35cd9c30979b2ff43e031
    resource: repo://Python/training_data.py
verified:
  - by: openwiki/0.3.3
    at: 2026-08-29T23:23:48.822Z
---

# Learned motion matching

Holden et al.'s *Learned Motion Matching* replaces the animation database and its brute-force
search with three small networks:

| Network | Role |
| --- | --- |
| **projector** | stands in for the nearest-neighbour lookup — query vector to the nearest database point, in latent space |
| **stepper** | advances the latent state autoregressively, so the search does not run every frame |
| **decompressor** | reconstructs a full pose from the latent state |

**None of them exists in this repository.** What does exist is everything they would be trained
from, which is what this page is about. The parts shared with a phase-functioned network are on
[neural synthesis readiness](../animation-tools/neural-synthesis.md); this page is the
LMM-specific half.

## What is already here

The query the projector maps from is the motion matching **feature vector**, unchanged — the same
floats [the search](matching-stage.md) compares today. The pose the decompressor reconstructs is
every joint of a database frame in the character frame. Both come out of the generated database:

- `Assets/StreamingAssets/MMDatabases/<name>/<name>.mmfeatures` holds the feature vectors, their
  normalisation statistics, and a schema block naming every feature and its width. It is readable
  without Unity — see [on-disk formats](../animation-tools/on-disk-formats.md) and
  `Python/feature_set_importer.py`.
- `Python/training_data.py` assembles the rest: every joint's position, rotation, velocity and
  angular velocity **in the character frame**, the frame's own travel and turn rate, the foot
  contacts, and a gait phase reconstructed from them. One `.npz` per database.

None of that is packed into either network's input tensor, deliberately. A decompressor target and
a PFNN input want the same quantities in different arrangements, and the arrangement is a property
of the model, not of the database.

Two things the feature layer gained specifically for this:

- **A trajectory feature may sample the past**, through a negative prediction frame. The classic
  matching query and a PFNN-style trajectory window both need samples either side of the query
  frame. See [feature vectors](feature-vectors.md).
- **A `.mmfeatures` refuses to be read against a configuration it was not written for**, rather
  than being read off the wrong offsets. Training on a stale database is the quiet failure that
  costs the most.

## What is missing

- The three networks, and their training loops.
- Inference in Unity. `com.unity.barracuda` 3.0.2 is still in the manifest but nothing consumes it,
  and it is a deprecated package on Unity 6.
- A `MoSynthStage` to run them. The seam is ready — see
  [the synthesis pipeline](../animation-tools/synthesis-pipeline.md) — and `MotionFieldStage` is the
  worked example of a stage whose model lives in Python behind PythonNET.

## The removed experiment

Five files under `Assets/Scripts/Learned Motion Matching/Scripts/`, about 1,386 lines in namespace
`MM`, were a self-contained neural reimplementation dating to March 2026. They were **deleted**, not
merely parked. Recover them from git history if the reference is wanted.

The reasons, in order of weight:

1. **`MM.MotionMatching` is a different type in a different assembly from the `MotionMatching`
   namespace the rest of this section documents.** For anyone implementing the real thing, that
   collision is a trap rather than a reference.
2. It **called `UnityEditor` from runtime code** in `Assembly-CSharp` — `AnimationUtility` and
   `AssetDatabase`, reached from `Awake` — so it could not be included in a player build.
3. It **could not run as committed**: `MM.MotionMatching` carried no `[Serializable]`, while its
   driver held it as a plain public field, so it would have been `null` when `Awake` called
   `mm.Build`.
4. It shared no type with the live architecture — no `MoSynthStage`, `MotionSynthesisComponent`,
   `PoseBuffer`, `Skeleton` or `PoseSet` — and nothing anywhere referenced it.

For the record, its shape was: a 24-float query (three trajectory points, hips velocity, then each
foot's position and velocity), a 32-float latent, a 56-float `x ‖ z` state, and a decompressor
emitting 15 floats per joint (3 position plus a 6D rotation, Gram–Schmidt orthonormalised). The
three `enable*` toggles on the driver were ablation switches, not leftovers: with the projector off
the latent came from stored data, with the stepper off so did the query, so the chain could be run
decompressor-only, decompressor + stepper, or complete.

`External/lmm-v0.3.0.zip` sits alongside the other vendored payloads and was left in place.
