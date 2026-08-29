---
type: Architecture Guide
title: Neural synthesis readiness
description: What a learned motion model needs from this repository before it can be trained or run, which of those pieces exist, and the failure modes the shared definitions are there to prevent.
tags: [neural, pfnn, learned-motion-matching, training-data, roadmap]
sources:
  - id: openwiki-source-992698fe95f5805c47be92d1
    resource: repo://Assets/AnimationTools/Runtime/Pose/CharacterSpacePose.cs
  - id: openwiki-source-a0893a8625fe95808c0bf2c1
    resource: repo://Python/gait_phase.py
  - id: openwiki-source-030d30d689203655d06f8a6b
    resource: repo://Python/tests/test_gait_phase.py
  - id: openwiki-source-44aeec7103fbb90f34a34626
    resource: repo://Python/tests/test_training_data.py
  - id: openwiki-source-58d35cd9c30979b2ff43e031
    resource: repo://Python/training_data.py
verified:
  - by: openwiki/0.3.3
    at: 2026-08-29T23:35:22.182Z
---

# Neural synthesis readiness

Two learned methods are the stated direction for this project: a **phase-functioned neural network**
(PFNN) and **learned motion matching** (LMM). Neither exists here. This page is about the layer
underneath both — what a learned method needs from a motion database, and which of those pieces are
in place.

The methods differ in their inputs, but they want the same *quantities*:

| | PFNN | Learned motion matching |
| --- | --- | --- |
| **Input** | a trajectory window spanning past and future, the previous frame's joints, a gait phase | the matching feature vector, as the query, plus a latent |
| **Output** | the next pose, the root delta, a phase increment | a full pose reconstructed from the latent |
| **Needs from the database** | trajectory, joints in the character frame, root deltas, contacts, phase | feature vectors, joints in the character frame, root deltas, contacts |

## The one failure mode worth designing against

A model is trained on arrays produced by one piece of code and run on arrays produced by another. If
those two disagree about the reference frame, the units, or the rate convention, **nothing throws**.
The network simply produces bad motion, and the cause is invisible from the symptom.

So each of these quantities has exactly one named definition per side, and the two sides' test suites
are written against the same properties:

| Quantity | Training (Python) | Inference (C#) |
| --- | --- | --- |
| Joints in the character frame | `training_data.build_training_set` | `CharacterSpacePose.Extract` |
| The character frame itself | `simulation_frame.derive_frames` | [`SimulationFrame`](simulation-frame.md) |
| The pose read off a live rig | — | `RigPoseReader.Read` |
| Matching feature vectors | `feature_set_importer.read_feature_set` | `FeatureSerializer` |

The rates are the one place the two are not identical by construction: Python differences
consecutive frame-local poses of a stored database, the C# composes the instantaneous rate implied by
a pose's own velocity channels. Those channels are themselves finite differences over the same
timestep, so the two agree to first order and exactly for motion that is rigid within the frame.

## What is in place

**A training set from a generated database.** `Python/training_data.py` reads a `.mmpose` (and the
`.mmfeatures` beside it) and writes one `.npz` holding, per frame: every joint's position, rotation,
velocity and angular velocity in the character frame; the frame's own travel and turn rate; the foot
contacts; and a gait phase. It builds neither network's input tensor — the packing is a property of
the model, not of the database — though `TrainingSet.pose_vector` offers one for a decompressor
target.

**Gait phase.** Not stored anywhere, and reconstructed from the foot contacts by
`Python/gait_phase.py`: π per footfall, linear between, per clip. A clip with fewer than two
footfalls is reported as having *no measurable cycle* rather than being given a made-up one, which is
what `phase_rate == 0` marks. See [on-disk formats](on-disk-formats.md) for where the contacts come
from.

**A trajectory window.** `TrainingSet.trajectory_window(offsets)` samples where the character was
and will be, around every frame, in that frame's own space — the input a phase-functioned network is
organised around. It is the one thing a model needs that the frame-local arrays cannot supply, since
the frame transform is precisely what removes it, so the world frame origin and heading are kept for
exactly this. Offsets are clamped inside the query frame's own clip rather than wrapped or dropped,
which keeps every frame usable instead of discarding the ends of every clip.

Two independent checks agree on it: one frame ahead, divided by the timestep, reproduces
`root_velocity` to float precision, and the divergence between travel and facing over 20 frames comes
out at a mean of 41° with a median of 13.5° — matching the separately measured property that
[facing is not the direction of travel](simulation-frame.md) on this database.

**Query horizons that reach into the past.** A trajectory feature's prediction frames may be
negative. Both the classic matching query and a PFNN-style window need samples either side of the
query frame; see [feature vectors](../motion-matching/feature-vectors.md) and
[control inputs](../motion-matching/control-inputs.md) for how each control input answers a past
horizon.

**A database that refuses to be misread.** Neither `.mmpose` nor `.mmfeatures` carries a version
byte, by decision. Each instead carries a block describing what it is — a skeleton block and a
feature schema block — that the reader checks before using the file. Training on a stale database is
the quiet failure that costs the most.

**A pipeline seam.** [`MoSynthStage`](synthesis-pipeline.md) is where a model would run, and
[`MotionFieldStage`](../motion-field/motion-field-stage.md) is the worked example of a stage whose
model lives in Python behind PythonNET. To make the character move, a stage writes the frame's own
motion into bone 0's velocity channels through `SimulationFrame.RecomposeRootVelocity`; the component
reads it straight back out through `ComputeVelocity`.

## What is missing

- **The models**, and their training loops.
- **Inference inside Unity.** `com.unity.barracuda` 3.0.2 is still in the manifest, nothing consumes
  it, and it is deprecated on Unity 6. Running the model in Python behind PythonNET, as the motion
  field does, is the path that needs no new dependency.
- **Writing a predicted pose back.** There is no shared inverse of `CharacterSpacePose.Extract`;
  `MotionFieldStage` writes its pose channel by channel from the Python arrays. A second consumer is
  the point at which that should become one named thing.
- **A cross-language agreement test.** The two definitions are tested against the same properties,
  but nothing compares actual numbers produced by both.
