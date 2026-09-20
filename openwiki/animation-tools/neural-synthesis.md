---
type: Architecture Guide
title: Neural synthesis readiness
description: What a learned motion model needs from this repository before it can be trained or run, which of those pieces exist, and the failure modes the shared definitions are there to prevent.
tags: [neural, pfnn, learned-motion-matching, training-data, roadmap]
sources:
  - id: openwiki-source-1ac38bf24a91b1612cc25f91
    resource: repo://Assets/AnimationTools/Runtime/Animation/GaitPhaseComponent.cs
  - id: openwiki-source-992698fe95f5805c47be92d1
    resource: repo://Assets/AnimationTools/Runtime/Pose/CharacterSpacePose.cs
  - id: openwiki-source-bc9dba9080266f6f57eee14f
    resource: repo://Assets/AnimationTools/Tests/Editor/CharacterSpacePoseApplyTests.cs
  - id: openwiki-source-e3854efddedde038431e9a6b
    resource: repo://Assets/Pfnn/Editor/PfnnAgreementCheck.cs
  - id: openwiki-source-a0893a8625fe95808c0bf2c1
    resource: repo://Python/gait_phase.py
  - id: openwiki-source-7b0683a5c5354cf2858d671c
    resource: repo://Python/pfnn_agreement.py
  - id: openwiki-source-4e55c713faf28eac207e9a7c
    resource: repo://Python/pfnn_dataset.py
  - id: openwiki-source-030d30d689203655d06f8a6b
    resource: repo://Python/tests/test_gait_phase.py
  - id: openwiki-source-44aeec7103fbb90f34a34626
    resource: repo://Python/tests/test_training_data.py
  - id: openwiki-source-58d35cd9c30979b2ff43e031
    resource: repo://Python/training_data.py
generated: {by: "claude-code", at: "2026-09-19T18:12:18.762Z"}
---

# Neural synthesis readiness

Two learned methods are the stated direction for this project: a **phase-functioned neural network**
(PFNN) and **learned motion matching** (LMM). Both now exist — see [the PFNN stage](../pfnn/pfnn-stage.md)
and [training and checkpoints](../pfnn/training-and-checkpoints.md), and
[learned motion matching](../motion-matching/learned-motion-matching.md), whose first phase is
implemented. This page is about the layer underneath both — what a learned method needs from a
motion database, and which of those pieces are in place.

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
| A predicted pose written back | — | `CharacterSpacePose.Apply` |
| The character frame itself | `simulation_frame.derive_frames` | [`SimulationFrame`](simulation-frame.md) |
| The pose read off a live rig | — | `RigPoseReader.Read` |
| Matching feature vectors | `feature_set_importer.read_feature_set` | `FeatureSerializer` |
| The two-axis rotation form | `training_data.rotations_to_6d` / `rotations_from_6d` | `PfnnStage.RotationFrom6D` |

The rates are the one place the two are not identical by construction: Python differences
consecutive frame-local poses of a stored database, the C# composes the instantaneous rate implied by
a pose's own velocity channels. Those channels are themselves finite differences over the same
timestep, so the two agree to first order and exactly for motion that is rigid within the frame.

## What is in place

**A training set from a generated database.** `Python/training_data.py` reads a `.mmpose` (and the
`.mmfeatures` beside it) and writes one `.npz` holding, per frame: every joint's position, rotation,
velocity and angular velocity in the character frame; the frame's own travel and turn rate; the foot
contacts; and a gait phase. It builds neither network's input tensor — the packing is a property of
the model, not of the database — though `TrainingSet.pose_vector` offers one canonical flat
packing. LMM does not use it: it predicts joint-*local* rotations and derives the character-space
pose by forward kinematics, so it packs its own.

**Gait phase, decided once and carried.** `AnimationTools.GaitPhase` turns a clip's footfalls into a
phase — π per footfall, linear between, per clip — and the bake writes the result into the `.mmpose`
beside the poses. Python reads it rather than deriving it. That is the point: a clip's footfalls are
*authored*, corrected by hand where detection was wrong, and a second implementation working from
the baked contacts would quietly disagree with what the editor drew. It is the same argument that
put the skeleton in that file.

**Phase you can look at and correct.** A clip carries its footfalls as a
[clip component](animation-sources.md), detected from the clip itself and editable, with a timeline
in the inspector showing the phase, the contacts behind it, and the ways the phase is known to go
wrong. Both the timeline and the database now run on one measurement, `GaitMeasure`, so what an
author corrects against is what a model trains on. The defect the tool still makes visible is missed
contacts, which cluster where root speed exceeds 1.6 m/s and make the phase jump a whole cycle
instead of half.

**Standing.** A stretch with no footfalls in it is not automatically unusable, because standing and
a missed contact leave the same hole in the anchors. How fast the character was travelling separates
them, frame by frame: below `standingSpeed` a frame sweeps its phase at a fixed period, and above it
the phase is held at a rate of zero — the *no measurable cycle* sentinel that drops the frame.

Frame by frame rather than over the stretch, because the two are almost never the same stretch. The
run-up to a clip's first footfall contains the stand *and* the acceleration out of it, and asking
whether all of it was slow answers no. See
[animation-sources](animation-sources.md#gait-phase) for what that cost on the
Edinburgh set.

Sweeping standing frames rather than dropping them is what Holden et al. do, and the reason is not
that the phase means anything there. It is that a model shown the whole cycle against a stationary
trajectory learns that its output does not depend on phase in that region, and therefore stands still
instead of paddling its legs when asked to stop. The alternative — extrapolating the neighbouring
walking rate outward — invents gait: on the untrimmed `walk1_subject1` clip the first footfall is at
frame 132, and 4.4 s of standing was once given 2.87 cycles the character never walked.

We are not taking the paper's other two answers. Its one-hot gait label is [deliberately
absent](../pfnn/training-and-checkpoints.md) — the trajectory window already shows a stationary
character as a window of zeros — and its runtime damping of the phase advance is unnecessary if the
network has learned to predict a small increment there itself.

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

**A pipeline seam.** [`MoSynthStage`](synthesis-pipeline.md) is where a model runs;
[`PfnnStage`](../pfnn/pfnn-stage.md) and [`MotionFieldStage`](../motion-field/motion-field-stage.md)
are both stages whose model lives in Python behind PythonNET. To make the character move, a stage writes the frame's own
motion into bone 0's velocity channels through `SimulationFrame.RecomposeRootVelocity`; the component
reads it straight back out through `ComputeVelocity`.

## What was missing, and what filled it

The four gaps this page originally listed have been closed by the PFNN work, except for the one
noted below:

- **The models and their training loops.** PFNN has both — `Python/pfnn_dataset.py` packs the
  vectors, `pfnn_model.py` is the network, `pfnn_trainer.py` fits it, `pfnn_io.py` stores it. LMM
  still has neither.
- **Inference inside Unity.** `PfnnStage` runs the model in Python behind PythonNET, which is the
  path that needed no new dependency. `com.unity.barracuda` 3.0.2 is still in the manifest, still
  unused, and still deprecated on Unity 6.
- **Writing a predicted pose back.** `CharacterSpacePose.Apply` is now the named inverse of
  `Extract`: it takes a pose measured in a character frame and writes it into a `PoseBuffer`, with
  bones below the root keeping their rest offsets so a prediction cannot stretch one. An edit-mode
  test asserts the round trip. `MotionFieldStage` still writes its pose channel by channel and could
  be moved onto it.
- **A cross-language agreement test.** `MoSynth/Pfnn/Check Training Agreement` compares the numbers
  the two definitions actually produce. It is a diagnostic rather than a unit test, because it needs
  a generated database and a live interpreter. Measured on the demo database: positions agree to
  1.8×10⁻⁶ m, rotations to 0.079° at worst and 0.0026° on average, with the residual concentrated
  at the deepest joints — float32-versus-float64 accumulation down the chain, not a disagreement
  about the definition.

Still missing: a shared home for the two-axis rotation conversion, which now exists three times —
in numpy in `training_data`, in torch in `lmm_fk`, and in C# in both `PfnnStage` and `LmmStage`.
