---
type: Architecture Guide
title: The PFNN stage
description: Driving a character from a phase-functioned network evaluated in Python — what the model is shown each tick, where the trajectory comes from, and how a predicted pose becomes movement.
tags: [pfnn, neural, synthesis-pipeline, python-interop, control-input]
sources:
  - id: openwiki-source-08ea4bc02364c8785bdd8d8f
    resource: repo://Assets/AnimationTools/Runtime/Core/MotionSynthesisComponent.cs
  - id: openwiki-source-fed6ec6af0a6135c6cbeddce
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/MotionMatchingControlInput.cs
  - id: openwiki-source-9ee07a0665219c02876c6ec8
    resource: repo://Assets/Pfnn/PfnnControlInput.cs
  - id: openwiki-source-a1ce493914db289593686473
    resource: repo://Assets/Pfnn/PfnnStage.cs
  - id: openwiki-source-b34ac78afb2047f4f15a0e53
    resource: repo://Assets/Pfnn/PfnnTrajectory.cs
  - id: openwiki-source-5d4680328f7da95f88649f31
    resource: repo://Packages/manifest.json
  - id: openwiki-source-a0893a8625fe95808c0bf2c1
    resource: repo://Python/gait_phase.py
  - id: openwiki-source-734ba38f801134e792f0e890
    resource: repo://Python/pfnn_runtime.py
generated: {by: "claude-code", at: "2026-08-31T20:31:56.222Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-30T16:03:03.865Z
---

# The PFNN stage

A **phase-functioned neural network** (Holden et al. 2017) replaces the animation database with a
small learned model. Instead of searching for the frame that best matches a query, it is handed a
description of where the character is going and produces the next pose directly.

`PfnnStage` is that model wired into the [synthesis pipeline](../animation-tools/synthesis-pipeline.md).
It runs the network in Python behind PythonNET, the same route the
[motion field](../motion-field/motion-field-stage.md) takes and the one the
[neural readiness](../animation-tools/neural-synthesis.md) page recommends, because it needs no new
Unity dependency.

## What the network sees

Every tick the stage assembles the three things the model was trained on:

- **A trajectory window** — where the character was and will be over roughly a second either side of
  now, as thirteen samples of ground position and facing, in the character's own frame.
- **The joints as they stand** — position and velocity for each predicted bone, again in the
  character frame.
- **The gait phase** — one angle saying where in the walk cycle the character is.

and reads back the next pose, the root delta that carries the character there, a phase increment,
and the foot contacts.

The phase is what makes the model small enough to be worth having. Rather than one network that has
to work out from the pose alone whether the left foot is about to land, the weights are themselves a
function of phase — a cubic spline through four control points, wrapped so that the network at the
end of a cycle *is* the network at the start. See [training and checkpoints](training-and-checkpoints.md)
for how that is built and stored.

## The trajectory has two halves, and they come from different places

During training both halves of the window are the same thing: the path the character really took,
sampled either side of the query frame. At runtime only the future is a wish.

So the stage keeps its own short history of where the character has been (`PfnnTrajectory`, a ring
buffer of ground position and heading), and fills the past half from that. Only the future half
comes from the control input. Filling the past from the controller too would tell the network the
character had already been following the requested path — a situation the training data never
contains, and one the model would answer confidently and wrongly.

Near the start of a run the history is shorter than the window. It is clamped to the oldest sample
held rather than extrapolated, which is what a character standing still would have produced, and it
matches how `TrainingSet.trajectory_window` answers an offset that would leave a clip.

## Control inputs

`PfnnControlInput` is a base of its own rather than a reuse of `MotionMatchingControlInput`, because
every one of those resolves its horizons out of a `MotionMatchingData` in `Start` — so none of them
works on a character that has no motion matching stage. Its whole contract is one method: where
should the character be, and which way should it face, this many frames from now.

Two implementations ship:

- **`PfnnSplineControlInput`** samples a path at each of the horizons the network asks for. It
  advances on its own clock rather than tracking where the character actually is, so drift shows up
  as measurable [path-following error](../animation-tools/path-following-metrics.md) instead of
  being steered out — the same choice `SplineControlInput` makes, and what makes both usable as
  measurement tools. It implements `IMotionSynthesisSplineControlInput`, so it is ready to be
  driven by the [benchmark](../animation-tools/benchmarking.md) harness, though PFNN is deliberately
  not registered as a sweep method yet — the model is still being tuned, and a number from an
  untuned model would only invite being compared against.
- **`PfnnDirectionControlInput`** takes a stick or WASD direction. A spring smooths the requested
  velocity into something a body could do, and running that same spring on with no new input is the
  predicted path — the construction `DirectionControlInput` documents, over the shared `Spring`.

## How a prediction becomes movement

The network outputs rotations, not positions. That is a deliberate departure from the paper, which
predicts joint positions and reconciles them with IK: `PoseBuffer` is rotation-based, there is no IK
solver in this repository, and a rotation-space prediction cannot stretch a bone. Bones the model
does not predict are held at their rest rotation.

From those rotations the stage runs forward kinematics — one pass, since bones are stored
depth-first — to get the character-frame positions, which are also what the next tick feeds back in.
The pose is then written through `CharacterSpacePose.Apply`, the named inverse of the `Extract` that
measured the training data.

The character travels because the predicted root delta is written into bone 0's velocity channels,
which `MotionSynthesisComponent` reads straight back out to advance the Transform. The stage never
touches `transform` itself. It writes the pose in a frame at the origin, because the component
re-anchors bone 0 against the frame the pose implies before accumulating the travel — an absolute
frame here would be applied twice.

**The phase only ever advances.** `gait_phase` builds phase as a monotone unwrapped angle, so every
training target was non-negative; a negative prediction is the network extrapolating outside what it
was shown. One was observed on a deliberately out-of-distribution input, and letting it through
would run the gait backwards, which no number of later frames recovers from. The stage clamps it.

## Failure is loud, once

A checkpoint records the names of the bones it predicts. At `Init` the stage maps those onto the
character's rig and refuses to run if a name is missing or the order does not match, because a model
fed a differently shaped pose does not throw — it just moves badly. If anything does throw during a
tick, the stage logs it once and disables itself rather than repeating the same failure every frame.

The stage also warns when the synthesis frame rate disagrees with the rate the model was trained at.
It steps the network once per tick, so a mismatch leaves the character travelling at the right speed
with its limbs moving at the wrong one.

## What it costs

Measured on the demo database, a 24-bone model with 256 hidden units, running inference on CPU:
about **1.1 ms per step** across the PythonNET boundary, and roughly 32 KB of garbage per tick from
the `dynamic` marshalling. One recorded lap of the Circle path put the median `Apply` at 4.2 ms
against a 33 ms budget with a 95th percentile of 83 ms — comfortable in the median, with the tail
and the allocation being what would need attention before this ran in a build. Those figures are a
single ad-hoc run, not a standing measurement: PFNN is not a registered benchmark method.

## Where the pieces are

| | |
| --- | --- |
| `Assets/Pfnn/PfnnStage.cs` | the stage: init, the per-tick loop, the pose write |
| `Assets/Pfnn/PfnnTrajectory.cs` | the history ring and the ground-plane frame transform |
| `Assets/Pfnn/PfnnControlInput.cs` | the steering contract, plus the spline and direction inputs |
| `Assets/Pfnn/PfnnBoneSelection.cs` | binding a checkpoint's bone names to a live rig |
| `Assets/Pfnn/PfnnCharacter.prefab` | a rig with the stage and a spline input wired up, to press play on |
| `Python/pfnn_runtime.py` | the stateless policy, and an offline rollout for judging a checkpoint |
