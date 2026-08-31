---
type: Quickstart
title: MoSynth
description: A Unity motion-synthesis project combining motion matching with a neural motion field — what the pieces are, and where to look for what.
tags: [overview, architecture, navigation]
sources:
  - id: openwiki-source-c47354627dcbea1099ddbb20
    resource: repo://Assets/AnimationTools/Runtime/AnimationTools.asmdef
  - id: openwiki-source-c9ec04c146f7125c2e260739
    resource: repo://Assets/AnimationTools/Runtime/Core/MoSynthStage.cs
  - id: openwiki-source-08ea4bc02364c8785bdd8d8f
    resource: repo://Assets/AnimationTools/Runtime/Core/MotionSynthesisComponent.cs
  - id: openwiki-source-90731679b476c9b5e27f4d74
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseSerializer.cs
  - id: openwiki-source-5c5de1c38fc74b789911f4ea
    resource: repo://Assets/MotionField/MotionField.asmdef
  - id: openwiki-source-839acd5f9c92d76d722dddc3
    resource: repo://Assets/MotionField/PythonRuntime.cs
  - id: openwiki-source-9608fa457c4aae6b1f8f1abe
    resource: repo://Assets/MotionMatching/Runtime/Core/Burst/TagOperationsBurst.cs
  - id: openwiki-source-52cb746a5d4a6e2a9f9396cd
    resource: repo://Assets/MotionMatching/Runtime/Features/FeatureSerializer.cs
  - id: openwiki-source-495f29e2d14a2354f51c18da
    resource: repo://Assets/MotionMatching/Runtime/IK/TwoJointIK.cs
  - id: openwiki-source-84760d31adde69871d6065c9
    resource: repo://Assets/MotionMatching/Runtime/MotionMatching.asmdef
  - id: openwiki-source-322ec7da662c5b191cff726f
    resource: repo://Assets/Scripts/Learned%20Motion%20Matching/Scripts/MotionMatching.cs
  - id: openwiki-source-1110a5319fdf997c0acf8f29
    resource: repo://Python/pose_set_importer.py
generated: {by: "claude-code", at: "2026-08-30T16:03:03.865Z"}
---

# MoSynth

MoSynth synthesizes character locomotion. It is a Unity project (6000.4.4) with a CPython side
reached through PythonNET.

The one structural idea worth learning first:

> **`AnimationTools` is a synthesis-agnostic substrate. Motion matching and the neural motion field
> are two interchangeable methods that both plug into it through the `MoSynthStage` seam.**

A character is one `MotionSynthesisComponent` owning the skeleton, the pose layout and the frame
loop, running an ordered list of stages that rewrite a pose in place. Swap the stage, swap the
synthesis method. A character can run either, both, or neither.

`MotionMatching` and `MotionField` are **sibling assemblies** — both reference `AnimationTools`,
neither references the other. Anything both need lives in `AnimationTools`, and that constraint
explains several placement decisions that otherwise look arbitrary.

## The domains

### [`animation-tools/`](animation-tools/synthesis-pipeline.md) — the substrate

| Page | Covers |
| --- | --- |
| [The synthesis pipeline](animation-tools/synthesis-pipeline.md) | `MoSynthStage`, the tick loop, the extension seams |
| [The channel and layout system](animation-tools/channel-layout-system.md) | typed float buffers, handles, hashing, the rotation-rate rule |
| [Pose buffers and layouts](animation-tools/pose-buffers.md) | how a pose is laid out over a skeleton; blending; FK input rules |
| [Skeletons and rig binding](animation-tools/skeletons-and-rig-binding.md) | bone identity and ordering; asset rig vs scene rig |
| [The simulation frame](animation-tools/simulation-frame.md) | deriving a character frame; **facing vs. direction of travel** |
| [Animation sources](animation-tools/animation-sources.md) | clips, baking, BVH import |
| [The pose database](animation-tools/pose-database.md) | `PoseSet`, extraction, contacts, `IPoseSetSource` |
| [Neural synthesis readiness](animation-tools/neural-synthesis.md) | what a learned method needs from the database, and which pieces exist |
| [On-disk formats](animation-tools/on-disk-formats.md) | the four StreamingAssets artefacts and the byte convention |
| [Motion recording](animation-tools/motion-recording.md) | capturing a run to a numpy-readable file |
| [Path following metrics](animation-tools/path-following-metrics.md) | windowed spline projection; trajectory, heading, speed, laps |
| [Benchmarking](animation-tools/benchmarking.md) | the Editor-driven sweep and what its numbers mean |

### [`motion-matching/`](motion-matching/matching-stage.md) — synthesis by search

| Page | Covers |
| --- | --- |
| [The matching stage](motion-matching/matching-stage.md) | playback and search clocks, the jump gate |
| [Feature vectors and query masking](motion-matching/feature-vectors.md) | layout, normalisation, **weight-masking inactive channels** |
| [Search backends](motion-matching/search-backends.md) | the seam, the BVH, the unwired tag mechanism |
| [Control inputs](motion-matching/control-inputs.md) | the simulation-object model; which inputs actually work |
| [Spline pose keypoints](motion-matching/spline-pose-keypoints.md) | a path that carries authored poses and facing |
| [Inertialization](motion-matching/inertialization.md) | smoothing jumps by decaying offsets |
| [Debug visualizers](motion-matching/debug-visualizers.md) | looking at a database, a pose set, or the pipeline's output |
| [Learned motion matching](motion-matching/learned-motion-matching.md) | what the database already provides toward it, and what is still missing |

### [`motion-field/`](motion-field/motion-field-stage.md) — synthesis by learned value

| Page | Covers |
| --- | --- |
| [The motion field stage](motion-field/motion-field-stage.md) | the four policies, why a value function beats greedy |
| [Reaching Python](motion-field/python-interop.md) | the embedded interpreter and the ZeroMQ alternative |
| [Config, training and staleness](motion-field/config-and-training.md) | the artefact chain and the two flags |
| [The pose manifold](motion-field/pose-manifold-embedding.md) | projecting the field to 3D to watch a policy |

### [`pfnn/`](pfnn/pfnn-stage.md) — synthesis by learned pose

| Page | Covers |
| --- | --- |
| [The PFNN stage](pfnn/pfnn-stage.md) | what the network is shown each tick, and how a prediction becomes movement |
| [Training and checkpoints](pfnn/training-and-checkpoints.md) | which bones it covers, how the vectors are packed, what the checkpoint holds |

### [`python/`](python/pose-data-bridge.md) — the CPython runtime

| Page | Covers |
| --- | --- |
| [The pose data model](python/pose-data-bridge.md) | the packed array layout and its algebra |
| [The Unity call surface](python/unity-call-surface.md) | what crosses the boundary each tick |
| [Motion field policies](python/motion-field-policies.md) | the similarity metric, actions, the policies |
| [Value function training](python/value-function-training.md) | fitted value iteration and the `.npz` artefact |

## Where to look for what

| If you are… | Start at | Then |
| --- | --- | --- |
| adding a synthesis method | [the pipeline](animation-tools/synthesis-pipeline.md) | [matching stage](motion-matching/matching-stage.md) as a worked example |
| debugging odd frame choices | [feature vectors](motion-matching/feature-vectors.md) | [debug visualizers](motion-matching/debug-visualizers.md) — feature errors are visible nowhere else |
| chasing a character facing the wrong way | [the simulation frame](animation-tools/simulation-frame.md) | [spline pose keypoints](motion-matching/spline-pose-keypoints.md) |
| hitting bone-binding errors | [skeletons and rig binding](animation-tools/skeletons-and-rig-binding.md) | [animation sources](animation-tools/animation-sources.md) |
| adding a channel to a buffer | [the channel and layout system](animation-tools/channel-layout-system.md) | [pose buffers](animation-tools/pose-buffers.md) |
| reading a database from Python | [on-disk formats](animation-tools/on-disk-formats.md) | [the pose data model](python/pose-data-bridge.md) |
| finding a character that moves badly after a config edit | [config and staleness](motion-field/config-and-training.md) | [the pose manifold](motion-field/pose-manifold-embedding.md) |
| measuring one method against another | [benchmarking](animation-tools/benchmarking.md) | [path following metrics](animation-tools/path-following-metrics.md) |
| touching the C#/Python boundary | [reaching Python](motion-field/python-interop.md) | [the Unity call surface](python/unity-call-surface.md) |

## Entry points

| Thing | Where |
| --- | --- |
| Character | `MotionSynthesisComponent` + a `[SerializeReference]` list of `MoSynthStage` |
| Matching database | `MotionMatchingData` (ScriptableObject) → `StreamingAssets/MMDatabases/<name>/` |
| Motion field config | `MotionFieldConfig` (ScriptableObject) → `StreamingAssets/MotionFields/<name>/` |
| Python modules | `Python/`, added to `sys.path` by `PythonRuntime` |
| Benchmark sweep | `MoSynth > Benchmark > Run Sweep`, or `Tools/run-benchmark.ps1` |

## Conventions that will bite you

- **Quaternions are xyzw** everywhere — in the buffers, on disk, and in Python. Much of the
  quaternion literature assumes wxyz.
- **Rotational rates are never quaternions.** Angular velocity and blend offsets are scaled-angle-axis
  `float3`s. See [why](animation-tools/channel-layout-system.md).
- **A `Skeleton` must point at an asset rig, not a scene rig.** Violating this corrupts FK silently,
  and the only guard is editor-only.
- **`AnimationTools.SkeletonBone` collides with `UnityEngine.SkeletonBone`.** Files outside the
  `AnimationTools*` namespaces need a `using` alias.
- **Facing is not direction of travel.** They coincide only for controllers with no independent notion
  of facing.
- **No format is versioned.** Staleness is caught by validating content. Generated databases are
  always regenerable — regenerate rather than preserve.

## Known rough edges

This project is research code and several areas are wired but inert. They are documented on their
pages rather than hidden:

- The **pose-adjustment and feature-read-back surface** on `MotionSynthesisComponent` throws, and its
  four `Root*` properties are never assigned — so callers silently read the world origin.
- The **crowd and collision control inputs** are not usable as shipped.
- The **tag-query mechanism** is complete and never invoked.
- **`TwoJointIK`** has no call sites; foot planting is not wired up.
- `responsiveness` and `quality` on the matching stage currently have no runtime effect.

Synthesis methods here are **not finalised**, so recorded benchmark numbers compare methods against
each other and are not targets to defend.

## Related documents

`AGENTS.md` at the repository root is the orientation document for coding agents — project overview,
code style, tech stack, directory map, common tasks. This wiki extends it rather than duplicating it.
`.junie/plans/` holds design plans in a Requirements → Technical Design → Testing → Delivery format.
