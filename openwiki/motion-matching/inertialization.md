---
type: Architecture Guide
title: Inertialization
description: Smoothing pose jumps by decaying offsets rather than crossfading, and why the root has to be blended in frame space.
tags: [blending, inertialization, ik, stage]
sources:
  - id: openwiki-source-2cab5ee24d405d8240d13641
    resource: repo://Assets/AnimationTools/Runtime/Pose/SimulationFrame.cs
  - id: openwiki-source-495f29e2d14a2354f51c18da
    resource: repo://Assets/MotionMatching/Runtime/IK/TwoJointIK.cs
  - id: openwiki-source-33d9396a1c6cf0745b22c6d1
    resource: repo://Assets/MotionMatching/Runtime/Inertialization/Inertialization.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# Inertialization

`Inertialization` is a [`MoSynthStage`](../animation-tools/synthesis-pipeline.md) that smooths the
pose jumps upstream stages produce. It owns the blend; it does not decide when to blend.

Place it **after** whatever produces jumps. In the shipped prefab the stage list is exactly
`MotionMatchingStage` then `Inertialization`.

## Offsets and decay, not crossfade

The usual way to blend two poses is a crossfade: keep both streams alive and interpolate between them
for the duration of the transition.

Inertialization does something else. When the pose changes, **the target changes instantly** — but an
offset is computed that takes you from the current pose to the target one, and that offset is then
decayed toward zero with a spring.

Working on offsets rather than poses is why this blend is the one used here:

- **Only one pose stream is needed.** An upstream stage can jump anywhere in the database without
  keeping the abandoned animation alive to be blended out.
- **When nothing has jumped the offset is zero**, so continuous animation passes through untouched —
  no lag, no filtering, nothing to tune away.

The decay uses `Spring.DecaySpringDamperImplicit` — the "goal is zero" specialisation of the implicit
damper family. It is critically damped and unconditionally stable, so it does not blow up at a large
`deltaTime`. The single tunable is `halfLife`, defaulting to 0.1 s.

## It keys off PoseDiscontinuity

Offsets are re-anchored **only on a tick where a stage raised
`MotionSynthesisComponent.PoseDiscontinuity`**. Between jumps they simply decay.

That flag is the whole coupling. A stage that jumps without raising it **will not be smoothed** —
which is why the flag's contract is stated from both sides, here and on `MoSynthStage.Apply`.

Two subtleties in how the re-anchor is computed:

- **The offset is measured from this stage's own previous output**, not from the previous input pose.
  That is what makes a jump *during* an ongoing blend fold the remaining offset into the new one
  rather than restarting from zero.
- **Each side of the jump is read against the frame it derives for itself**, so the two poses are
  compared as two characters each standing at their own origin.

## Why the root is special

Bone 0 carries the root's **world** position and rotation. Blending it directly would smear the
character across the world whenever the target pose comes from a different part of a clip — the
offset would be a real distance across the map, decayed over a tenth of a second.

So bone 0 is blended in the space of the [simulation frame](../animation-tools/simulation-frame.md)
each pose derives: frame-local position, rotation and rates, written back through the *target* pose's
own frame. Bones 1 and up are parent-local already and blend as they are.

The velocity channels have to follow the position back through the frame, because the synthesis
component **re-derives the frame's own motion from them** when it advances the character. That is
what `SimulationFrame.DecomposeRootVelocity` / `RecomposeRootVelocity` exist for, and
`Inertialization` is their only production consumer.

Rotation offsets are quaternion-multiplicative and position offsets additive, with
`MathExtensions.Abs` picking the shortest arc — see
[the rotation-rate rule](../animation-tools/channel-layout-system.md).

## The legacy region

A `#region OLD_IMPL` block holds state and entry points from the pre-stage API, where a caller drove
transitions explicitly rather than the stage's `Apply` doing it from `PoseDiscontinuity`.

**Every entry point in it is uncalled.** Only the shared maths at the bottom of the file —
`InertializeJointTransition` and `InertializeJointUpdate` — is live, reached from `Apply`.

The region is worth knowing about for one reason: its `PoseTransition` sets up the root blend in
**world space**, which is exactly the mistake the frame-local design above exists to avoid. It is a
preserved example of the rejected approach, not a second supported path.

There is also a latent hazard there: the old `Update()` writes into an array that only the
`Inertialization(int jointCount)` constructor allocates, never `Init`. A stage constructed the
`[SerializeReference]` way would null-ref if that path were ever called.

## TwoJointIK

`TwoJointIK` is analytic two-bone inverse kinematics — closed form rather than iterative. Two stages:
*extension*, where the cosine rule gives the interior angles that make the chain exactly as long as
the distance to the target; then *aiming*, where the root joint rotates once more to swing onto the
target direction. An unreachable target is pulled in to the limit rather than failing.

Its `forward` argument is required and not obvious: without it the problem is ambiguous, because the
chain can hinge anywhere on a circle around the root-to-target axis. `forward` is what picks a
forward-bending knee.

The reachable distance is held an epsilon short at **both** ends, since at full extension and at full
fold the cosine rule is degenerate. Every `acos` argument is clamped.

> **It has zero call sites.** The only occurrence of the identifier in the repository is its own
> declaration. Foot planting is not wired up anywhere — which is consistent with toe-floor
> penetration correction being one of the two effects
> [dropped when the pipeline moved to stages](../animation-tools/synthesis-pipeline.md).

## Source map

| Concern | File |
| --- | --- |
| The stage, offsets, frame-space root blend | `Assets/MotionMatching/Runtime/Inertialization/Inertialization.cs` |
| Two-bone IK (unused) | `Assets/MotionMatching/Runtime/IK/TwoJointIK.cs` |
| Spring family | `Assets/MotionMatching/Runtime/Utils/Spring.cs` |

**Tests.** None directly. The frame-space machinery it depends on is covered by `SimulationFrameTests`,
including the decompose/recompose inverse and the tangential-velocity term.
