---
type: Architecture Guide
title: Root following
description: Making the character land on something else that decides where it should be, why a correction goes through bone 0's velocity rather than the Transform, and the foot locking that is still missing.
tags: [root-motion, control-input, stage, capsule, drift]
sources:
  - id: openwiki-source-721e7f001dd1252e3e9fe0e4
    resource: repo://Assets/AnimationTools/Runtime/Benchmark/MotionQualityMetricsCalculator.cs
  - id: openwiki-source-20e38e100e36c5379b2eb8ec
    resource: repo://Assets/AnimationTools/Runtime/ControlInput/IFrameTarget.cs
  - id: openwiki-source-2cab5ee24d405d8240d13641
    resource: repo://Assets/AnimationTools/Runtime/Pose/SimulationFrame.cs
  - id: openwiki-source-5161104d3b44ffc92c1fae2a
    resource: repo://Assets/AnimationTools/Runtime/Stages/RootFollow.cs
  - id: openwiki-source-959ade3d157f2f3700dee33f
    resource: repo://Assets/AnimationTools/Runtime/Stages/RootFollowStage.cs
  - id: openwiki-source-dd6e345dbf04a74d30240514
    resource: repo://Assets/AnimationTools/Runtime/Utils/Spring.cs
generated: {by: "claude-code", at: "2026-09-06T12:33:15.598Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-09-06T12:33:15.598Z
---

# Root following

A synthesized character normally goes wherever the animation takes it. `RootFollowStage` is for when
something else is the authority on where it should be — a capsule a gameplay controller drives, a
point travelling along a path — and the character has to be there rather than near there.

It is an ordinary [`MoSynthStage`](synthesis-pipeline.md), and it is one of three answers to a single
question that motion synthesis keeps asking in different forms.

## Three roles, easily confused

| Role | The question | Who answers |
| --- | --- | --- |
| Intent | what does the player or the path want? | the control input |
| Trajectory origin | what state does the prediction start from? | the control input |
| Root reconciliation | is the character pulled onto something? | a stage, or nothing |

Only the first is obviously the control input's job, and for a long time all three were fused into
it. Separating them is what makes the following table possible, and the combinations are not
interchangeable:

| Setup | Input | Stage | Behaviour |
| --- | --- | --- | --- |
| Free simulation object | `DirectionControlInput` | none | the object leads, the character follows it through the search and never quite arrives |
| Anchored | `AnchoredDirectionControlInput` | none | the trajectory starts at the character, so there is nothing to lag behind |
| Capsule authority | any input | `RootFollowStage` | the character is placed on the target every tick; the animation supplies the pose, not the position |

**Anchored and capsule authority must not be combined.** An anchored input has no object of its own
for the stage to follow, so pointing the stage at it is either a no-op or a feedback loop.

## The drift the anchored input removes

`DirectionControlInput` integrates a position of its own and the search chases it. Nothing closes the
loop, so any difference between that position and the character accumulates: the query says "you are
two metres behind where you should be" and keeps saying it, and the character reads as permanently
catching up. That is the behaviour, not a bug in the springs.

[`AnchoredDirectionControlInput`](../motion-matching/control-inputs.md) runs the same springs and
keeps only their velocity state. Every tick it re-reads the character's own frame and predicts
forward from there, so there is no position that can drift. The trade is real and worth stating: an
anchored input describes motion, never location, so it cannot be used to place a character or to
measure path-following error. `SplineControlInput` stays open-loop for exactly that reason.

**Rejected: seeding the springs from the character's measured velocity.** It looks more honest than
carrying spring state, and it is worse. The rate read back off the rig at 30 Hz carries the hips'
sway and the clip's own noise, which would enter the query directly and then influence the search
that produced it.

**Rejected: keeping a facing state, as the PFNN input does.** The database bakes its Direction
channels as *facing*, and the character's facing is whatever yaw rate the matched clip integrated —
see [the simulation frame](simulation-frame.md). A stored facing would be a second source of truth
that the clip can contradict, which is the same defect as the free simulation object in smaller
form. Each horizon's facing is therefore derived from the character's actual yaw.

## How a correction reaches the character

The obvious implementation — write the character's Transform — is the wrong one. The component
advances the character by integrating the frame velocity the pose carries, so a Transform written
beside that integration is a correction nothing else can see, blend, or limit.

`RootFollowStage` instead writes the correction *as* that velocity, on bone 0, through
`SimulationFrame.DecomposeRootVelocity` and `RecomposeRootVelocity`. That is the same route
[`Inertialization`](../motion-matching/inertialization.md) writes its blended root through, and the
consequence is that a correction is just more root motion: it can be damped, capped, and switched
off without anything downstream knowing.

The arithmetic lives in `RootFollow.Solve`, which is pure so it can be checked against the
integration it is written for:

```
animated  = RotY(ownerYaw) · frameVelocity · dt        // what the pose alone moves this tick
residual  = (target − owner) − animated                 // what is left over
c         = DampAdjustmentImplicit(residual, positionHalfLife, dt)
            clamped to maxCorrectionSpeed · dt
corrected = frameVelocity + RotY(−ownerYaw) · c / dt
```

Facing follows the same shape against the yaw, taken the short way round so a character just short of
−180° and a target just past it are twenty degrees apart rather than most of a turn.

Two knobs, and they do different things:

- **`positionHalfLife` / `rotationHalfLife`** decide how much of the gap closes. **Zero closes all of
  it**, which is the no-lag mode and needs no special case — the damper's exponential is already
  indistinguishable from 1 at a zero half-life.
- **`maxCorrectionSpeed` / `maxCorrectionYawRate`** cap the correction *on its own*, so it hides
  inside the animation's own travel instead of reading as a slide. Zero is uncapped.

## What it will not do

**It follows in the plane only.** The character frame is ground-projected by construction, so nothing
routed through bone 0 can move the character vertically. A target above or below the character is
followed in x and z and ignored in y. Height following would have to write the Transform directly,
which is why it is not in yet.

**Order matters.** Place it after `Inertialization`. The blend smooths the pose stream, and running it
afterwards would smooth the correction away with it.

**`rootPositionsMask` off makes it inert**, since the component then never integrates anything. The
stage says so at startup rather than silently doing nothing.

**Snapping to a target authored elsewhere teleports the character.** In the shipped prefab the
capsule and the character start about two metres apart, so `alignTargetOnInit` moves the target onto
the character at startup. It is skipped when the target answers `IFrameTarget`, because such a target
decides its own position.

## IFrameTarget

A target's Transform is not always where the target is. A spline follower's point travels along its
path while its object stays where it was dropped, so following the Transform would follow nothing.
`IFrameTarget` lets a target answer for itself, and `RootFollowStage` prefers it over the Transform
whenever the component implements it.

This is not a nicety: every method in the benchmark sweep is spline-driven, so without it root
following could not be measured at all.

## Planned: foot locking

Placing the character exactly makes feet slide. The correction moves the root while the animation's
legs carry on doing what the clip said, and the difference shows up on the ground. The
[benchmark's `footskatePerMeter`](benchmarking.md) is what that costs, in numbers.

The fix is a second stage after this one, not yet written. The design, so the slot is not
re-litigated:

- **Latch the ankle, not the toe.** The ankle is the IK end effector, and with the foot's world
  rotation preserved the toe is rigidly attached to it, so it stays put too — and the footskate
  metric, which measures the toe, then measures the same thing the stage is controlling.
- **Latch on the rising edge of contact**, carrying any still-decaying offset into the new lock so a
  re-latch mid-release does not pop.
- **Release on a distance threshold as well as on contact ending.** A snap-mode follow can drag the
  body further than a leg reaches; the leg has to let go rather than hyper-extend.
- **Decay the residual on release** with the usual spring, so the foot returns to the animation
  rather than jumping to it.
- **Ask `MotionSynthesisComponent.ComputeAppliedFrame` where the pose will be rendered.** A foot is
  planted in the world, but the pose is in its own clip space and the character's Transform has not
  been advanced yet when a stage runs. That helper exists for this, and going through it is what
  stops the stage's idea of the frame from drifting from the apply path's.

`TwoJointIK` already has the closed-form solver, unused, and would need a pose-space overload — it
currently works on scene Transforms, which a stage does not have.

## Source map

| Concern | File |
| --- | --- |
| The correction arithmetic | `Assets/AnimationTools/Runtime/Stages/RootFollow.cs` |
| The stage | `Assets/AnimationTools/Runtime/Stages/RootFollowStage.cs` |
| Target seam | `Assets/AnimationTools/Runtime/ControlInput/IFrameTarget.cs` |
| Anchored input | `Assets/MotionMatching/Runtime/CharacterController/AnchoredDirectionControlInput.cs` |
| Shared steering maths | `Assets/AnimationTools/Runtime/ControlInput/TrajectorySteering.cs` |
| Frame integration | `Assets/AnimationTools/Runtime/Pose/SimulationFrame.cs` (`Advance`, `Yaw`, `SignedYawDelta`) |

**Tests.** `RootFollowTests` pins the property that matters — that integrating the corrected velocity
lands on the target — rather than the numbers themselves, plus the caps, the yaw wrap, and the
guarantee that height is never touched. `TrajectorySteeringTests` covers the steering maths both this
and the PFNN input share.
