---
type: Agent Reference
title: PFNN packing and boundary hazards
description: Exact vector layouts, the invariants a PFNN sample must satisfy, the C#/Python marshalling that works, and the traps found while building it.
tags: [pfnn, agents, python-interop, training-data]
generated: {by: "claude-code", at: "2026-09-21T19:17:12.006Z"}
---

# PFNN packing and boundary hazards

Operational detail for working on the PFNN side. The human-facing account is
[training and checkpoints](../../pfnn/training-and-checkpoints.md).

## Module map

| Module | Owns |
| --- | --- |
| `Python/pfnn/dataset.py` | bone selection, `PfnnSpec`, input/output layouts, `build_vectors` |
| `Python/pfnn/model.py` | `catmull_rom_weights`, `PhaseFunctionedNetwork`, `resolve_device` |
| `Python/pfnn/io.py` | `.pfnn.npz` save/load. **numpy only — never import torch here** |
| `Python/pfnn/trainer.py` | `train(...)`, the Unity entry point, plus a CLI |
| `Python/pfnn/runtime.py` | `PfnnPolicy` (stateless), `rollout` |
| `Python/pfnn/agreement.py` | `compare(...)`, driven by the Editor menu item |

`pfnn.io` stays numpy-only on purpose: a caller that wants to know what a checkpoint contains should
not have to load a deep learning framework to find out. It mirrors `motion_field.io` exactly.

## Layouts

Read them from `PfnnSpec.input_layout()` / `output_layout()` rather than hardcoding offsets;
`pfnn.dataset.block(vectors, layout, name)` slices a named block. Blocks tile the vector with no
gaps, which a unit test asserts.

Input, in order: `trajectory_positions` (2·T), `trajectory_directions` (2·T), `joint_positions`
(3·B), `joint_velocities` (3·B), `contacts` (2).

Output, in order: `joint_rotations_6d` (6·B), `joint_velocities` (3·B), `root_height` (1),
`root_delta` (3), `phase_delta` (1), `contacts` (2).

Defaults on the demo rig: T = 13 (offsets −30…30 step 5), B = 24 → **198 in, 223 out**, 696,188
parameters at 256 hidden units.

## Invariants a sample must satisfy

- Query frame `i` and target `i + 1` are in the **same clip**. `usable_queries` enforces it.
- `phase_rate[i] != 0`. `TrainingSet.usable()` does **not** check this — it only looks at the
  matching feature vector. A clip with fewer than two footfalls has no measurable cycle, and
  training on it teaches the model a phase that means nothing.
- The bone selection is **closed under parent**. `select_bones` raises otherwise. The root can never
  be excluded.
- An excluded name the skeleton does not have raises, rather than being ignored. This caught a real
  mistake during development: a shell pipeline that left `\r` on each name produced 60 "unknown
  bone" errors instead of silently training an 85-bone model.

## Verified numbers

Against `Assets/StreamingAssets/MMDatabases/MotionMatchingData` (7839 frames, 85 bones, 30 fps):

- `|root_delta.xz|` reproduces `root_velocity * frame_time` to **7.4e-09**.
- `root_delta.dyaw` reproduces `root_yaw_rate * frame_time` to **1.5e-08**, correlation 1.0.
- The offset-0 trajectory sample is **exactly** the origin with facing `(0, 1)`.
- The root bone's `x` and `z` in the character frame are **exactly** 0.
- 6D → quaternion round-trips to **6e-08**.

If any of these move, the frame transform or the packing has changed and the model must be retrained.

## The C#/Python boundary

Verified working, both directions:

| Direction | Shape |
| --- | --- |
| C# → Python | `float[]` passed positionally; `np.asarray(x, dtype=np.float32)` converts it |
| Python → C# | plain `list[float]` cast as `(float[])result[n]`; `list[str]` as `(string[])`; `list[int]` as `(int[])` |

Measured at **1.1 ms per `step()`** on CPU including marshalling, and about 32 KB of garbage per
tick from the `dynamic` call. `PfnnPolicy` is stateless by design — phase, joints and trajectory
history live in C#, so a character can be restarted without reaching across the boundary.

Forward kinematics is the caller's job, not the policy's: the network predicts rotations, and
turning those into positions needs the rig's rest offsets. `PfnnStage` does it with `SkeletonData`;
`pfnn.runtime.rollout` recovers the offsets from the first frame's character-space pose, because a
`TrainingSet` carries no rest transforms.

## Traps

- **`dphase` can come back negative** on out-of-distribution input, even though every training target
  was non-negative. `PfnnStage` clamps it. Do not remove the clamp — a negative phase step runs the
  gait backwards and nothing later recovers.
- **Dropout convention.** The paper quotes 0.7 as a *retention* rate; PyTorch's `nn.Dropout` takes a
  *drop* probability. The config field is the PyTorch sense, so the paper's setting is 0.3.
- **`np.savez` with `allow_pickle=False`** requires bone names as a `<U` array, not an object array.
  `training.training_data.save_npz` uses object arrays and therefore needs `allow_pickle=True` on load;
  `pfnn.io` deliberately does not.
- **Validation must be a contiguous tail.** A random split leaks near-duplicate frames into training
  and reports a validation loss that measures nothing.
- **`Does.Not.Contain(int)`** in NUnit binds the string-substring overload and fails to compile
  against a `List<int>`. Use `Has.No.Member(...)`.
- **`AnimationTools.SplineFold`** is named to avoid colliding with `UnityEngine.Splines.SplinePath`.
  The first attempt used `SplinePath` and produced `CS0104` ambiguity in both spline control inputs —
  the same class of collision as `AnimationTools.SkeletonBone`.

## Running the suites

```bash
python -m unittest discover -s Python/tests -t Python/tests
```

`test_pfnn_model.py` skips its network tests when torch is absent, so the folder keeps its stated
property of needing no venv extras. `test_pfnn_dataset.py` needs only numpy and scipy and reuses
`test_training_data.py`'s rig fixtures.

On the C# side the edit-mode runner filters by assembly name in
`Assets/AnimationTools/Tests/Editor/TestResultDump.cs`. **A new test assembly must be added to that
list or its tests silently do not run** — this was how `Pfnn.Tests` first appeared to pass with an
unchanged count. See [the verification loop](../tooling/verification-loop.md).
