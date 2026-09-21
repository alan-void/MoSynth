---
type: Architecture Guide
title: The PFNN stage
description: Driving a character from a phase-functioned network evaluated in Python — what the model is shown each tick, where the trajectory comes from, and how a predicted pose becomes movement.
tags: [pfnn, neural, synthesis-pipeline, python-interop, control-input]
sources:
  - id: openwiki-source-0b35dfb45ac00c6c6916b1e0
    resource: repo://Assets/AnimationTools/Runtime/Animation/GaitPhase.cs
  - id: openwiki-source-a6b9fae1157f771557413c57
    resource: repo://Assets/AnimationTools/Runtime/ControlInput/MotionSynthesisControlInput.cs
  - id: openwiki-source-b106a4622123233262d982ae
    resource: repo://Assets/AnimationTools/Runtime/ControlInput/TrajectorySteering.cs
  - id: openwiki-source-21b6f7706116d71ad97ea47a
    resource: repo://Assets/AnimationTools/Runtime/ControlInput/UserInput.cs
  - id: openwiki-source-08ea4bc02364c8785bdd8d8f
    resource: repo://Assets/AnimationTools/Runtime/Core/MotionSynthesisComponent.cs
  - id: openwiki-source-dd6e345dbf04a74d30240514
    resource: repo://Assets/AnimationTools/Runtime/Utils/Spring.cs
  - id: openwiki-source-435bcda353047bbe94bff7d8
    resource: repo://Assets/AnimationTools/Tests/Editor/TrajectorySteeringTests.cs
  - id: openwiki-source-fed6ec6af0a6135c6cbeddce
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/MotionMatchingControlInput.cs
  - id: openwiki-source-9ee07a0665219c02876c6ec8
    resource: repo://Assets/Pfnn/PfnnControlInput.cs
  - id: openwiki-source-aa1128735bb9d950655cd046
    resource: repo://Assets/Pfnn/PfnnDirectionControlInput.cs
  - id: openwiki-source-22292ab2d3a290dd1ef7ed79
    resource: repo://Assets/Pfnn/PfnnSplineControlInput.cs
  - id: openwiki-source-a1ce493914db289593686473
    resource: repo://Assets/Pfnn/PfnnStage.cs
  - id: openwiki-source-b34ac78afb2047f4f15a0e53
    resource: repo://Assets/Pfnn/PfnnTrajectory.cs
  - id: openwiki-source-5d4680328f7da95f88649f31
    resource: repo://Packages/manifest.json
  - id: openwiki-source-734ba38f801134e792f0e890
    resource: repo://Python/pfnn_runtime.py
generated: {by: "claude-code", at: "2026-09-21T19:17:12.006Z"}
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
the foot contacts, and the future trajectory it expects to follow.

The phase is what makes the model small enough to be worth having. Rather than one network that has
to work out from the pose alone whether the left foot is about to land, the weights are themselves a
function of phase — a cubic spline through four control points, wrapped so that the network at the
end of a cycle *is* the network at the start. See [training and checkpoints](training-and-checkpoints.md)
for how that is built and stored.

## The trajectory comes from three places, not two

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

### The wish alone is not a path, so the model's own answer is blended into it

That leaves a mismatch the past half does not have. In training the future half is *the path the
character went on to take*, so it is always something the past half could lead into. A request is
under no such obligation: a stick thrown from forward to backward asks for a future that no body
could reach from the present, and the model answers an input like that by extrapolating — which is
what a violent pose change on a fast reversal actually is.

So the network predicts its own future trajectory as part of the output, and the stage blends that
prediction with the request before packing the window. The blend is graded by horizon
(`TrajectorySteering.HorizonBlend`): beside the character the prediction wins, because that is where
a physically impossible sample does the damage, and at the one-second horizon the request wins
outright, because that is what the character is actually being steered toward. `trajectoryFeedback`
turns the whole thing down to nothing, which is the A/B; `feedbackFalloff` moves where the handover
happens.

The prediction describes the window of the frame the character is *about* to be in — see
[what a sample is](training-and-checkpoints.md#what-a-sample-is) — so it is exactly the future half
the next tick needs, with nothing to shift or interpolate. The stage keeps it in world space rather
than in the frame it was predicted for, because the character's real next frame can differ from the
one the root delta described: `RootFollowStage` moves it, and the first tick suppresses the rate
channels. Re-reading the actual frame each tick corrects for that instead of letting it accumulate.

This is also why `maxRequestAngle` is a guard rather than the answer. The cone keeps a request on
ground the model has seen; the blend keeps the whole window on paths a body could follow, which is
the stronger statement.

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
  There is a second, quieter spring on the request itself, for the reason below.

### A one-second lookahead makes a stepped request visible

A keyboard hands the controller a step: the requested direction goes from nothing to a metre a
second, or turns ninety degrees, between one frame and the next. The near half of the window barely
notices, because those samples are dominated by the velocity the character already has. The far end
is another matter — a sample a full second ahead has converged onto the request, so it inherits the
step whole. Measured on the demo character, a ninety-degree change moved the +30 sample **1.05 m in
a single 30 Hz frame** while the samples beside the character did not move at all. What that looks
like on screen is the tail of the trajectory whipping round a stationary head, and what the network
is handed is a path no continuous body could take.

The fix is not to smooth the drawing, and not to smooth the window inside the stage. It is to
rate-limit the *request*, in the one input whose request is a step. A first-order damper on the
requested velocity — `steeringHalfLife`, a quarter of a second by default — spreads the same change
over about three half-lives instead of one frame, bringing the worst single-frame movement of the
far horizon down to **9 cm**. The near/far asymmetry survives for free: the samples close to the
character stay governed by its actual velocity, the distant ones by a request that now moves
continuously.

Smoothing inside `PfnnStage` would have been the tempting shared answer, and it is wrong here. It
would lag `PfnnSplineControlInput` too, and that input is deliberately open-loop so that drift
against the path is *measurable* rather than steered out. A spline whose samples come off a curve
the character is advancing along has no step to damp in the first place.

Two related discontinuities were fixed with it, both of which look identical on screen:

- The velocity was driven by a **position** spring handed the requested velocity as its position
  goal. That construction has a fixed point set by the step size rather than by the request: it
  settled at **3.26 m/s for a `maxSpeed` of 1**, and drifted to 3.43 m/s at 144 fps. The character
  was asking a network trained on a 1.10 m/s walk for a sprint, at a speed that changed with the
  frame rate. `Spring.CharacterPositionUpdate` — the controller written for a velocity goal, and the
  one `DirectionControlInput` already uses — settles on exactly the requested speed at every frame
  rate.
- The facing was **assigned** from the direction of travel rather than damped toward it, despite a
  `facingHalfLife` field and a comment describing a spring that was not there. On a reversal the
  velocity passes through the stopped deadzone and its direction inverts in one frame, taking the
  facing and every direction sample built from it along. Damped, a 170-degree reversal turns over
  about a second and a half at under ten degrees a frame.

`TrajectorySteering` holds all three as pure functions so the continuity can be measured rather
than judged by eye; `TrajectorySteeringTests` asserts the bounds, and asserts the undamped case still teleports,
so the bounds cannot pass vacuously.

### A control input is passive, and that used to be a quiet trap

Nothing in `PfnnDirectionControlInput` polls input. `SetMovementDirection` is called for it from
outside — by the subscription `MotionSynthesisControlInput` makes to `UserInput`'s move event while
the input is enabled. That subscription is what closed the trap: the connection used to be a
persistent `UnityEvent` listener authored in the scene, and forgetting to add one was invisible. An
undriven input still answers every horizon with a well-formed *stand exactly where you are, facing
where you already face*, so the stage's own "nothing steering" branch never fires and there is
nothing to log. The character stands there, or walks off on whatever the network makes of an input it
never saw in training, and every component involved reports itself healthy.

The warning is kept anyway, because a scene with no `UserInput` in it at all produces exactly that
silence: the component warns — once, five seconds in — when `SetMovementDirection` has never been
called at all. Not when the direction is zero: asking a character to stop is legitimate and must stay silent.
The distinction being drawn is *nobody is driving me*, which is a wiring mistake, against *I have
been told to stand still*, which is not.

The facing is seeded from the character's own transform for the same class of reason. Starting it at
world +z, as a plain field initialiser does, tells a character pointing anywhere else that its
requested heading is a turn — on frame one, before any input arrives.

The input source has a matching trap of its own. `UserInput` used to subscribe only to the input
action's `performed` edge, so releasing the keys left the last direction latched — the character
walked on, and every change of input was a step from one held direction to another rather than a
return through rest. It now publishes `canceled` as well, which is what makes releasing the keys
mean stop. Every direction control input in the scene reads that same static event.

### The gizmo draws the window, not the wish

`PfnnControlInput` draws the trajectory once for every input, from the packed floats the stage last
handed the network, taken back out of the character frame they were packed into. It used to be drawn
per subclass by re-asking the control input for its desired samples, which meant the history half was
invisible and the frame conversion could not be checked by eye — a picture that would have stayed
reassuringly correct even if the packing were wrong. Past samples are drawn cool, future samples
warm, so what the model is actually being asked is legible from the scene view.

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
touches `transform` itself.

That indirection is also why the **first** tick needs care. Per-bone angular velocities are
differenced against the previous pose, and at seed time the only previous pose is the rig standing
at rest — so differencing the first prediction against it produces a whole-pose delta divided by one
frame time. Bone 0's share of that is not a velocity at all, but the component cannot know it and
integrates it onto the Transform as a tick of root motion. The stage suppresses the rate channels
for that one tick instead. It writes the pose in a frame at the origin, because the component
re-anchors bone 0 against the frame the pose implies before accumulating the travel — an absolute
frame here would be applied twice.

**The phase only ever advances.** `GaitPhase` builds phase as a monotone unwrapped angle, so every
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
| `Assets/Pfnn/PfnnControlInput.cs` | the steering contract and the window gizmo, plus the spline and direction inputs |
| `Assets/AnimationTools/Runtime/ControlInput/TrajectorySteering.cs` | the request damper and the horizon predictions, as pure functions |
| `Assets/Pfnn/PfnnBoneSelection.cs` | binding a checkpoint's bone names to a live rig |
| `Assets/Pfnn/PfnnCharacter.prefab` | a rig with the stage and a spline input wired up, to press play on |
| `Python/pfnn/runtime.py` | the stateless policy, and an offline rollout for judging a checkpoint |
