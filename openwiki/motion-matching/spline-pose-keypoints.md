---
type: Architecture Guide
title: Spline pose keypoints
description: A path that carries authored poses — how facing is stored as a yaw offset from the tangent, and how bone constraints switch on only near a keypoint.
tags: [spline, keypoints, facing, constraints]
sources:
  - id: openwiki-source-2e05de48a5866222c3fc595d
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/SplinePoseKeypoint.cs
  - id: openwiki-source-23c39c78df6491e3d9dfdcb1
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/SplinePoseKeypointControlInput.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# Spline pose keypoints

A `SplinePoseKeypoint` turns a `SkeletonAnimation` into two things at once: a spline fitted to the
clip's root trajectory, and a set of **keypoints** recording which pose the clip held at intervals
along it.

`SplinePoseKeypointControlInput` is what consumes it — a
[`SplineControlInput`](control-inputs.md) whose path carries poses, so that bone trajectory channels
can switch on with a real target when the character approaches a keypoint.

This is the one place in the codebase where a path carries **facing** as well as position, and
consequently where the facing/travel distinction has to be handled exactly right.

## Facing is a yaw offset, not a direction

Each keypoint stores its clip frame, its normalized spline parameter, and the character's facing
there. A bone's constraint target is that frame's bone position expressed relative to the keypoint's
own simulation frame, re-anchored onto the spline's frame at that parameter.

**Facing is stored explicitly rather than derived from the path tangent, because the two are not the
same thing.** The bone offsets are expressed in the clip's simulation frame, which is facing-aligned.
So re-anchoring a strafing or backpedalling pose against the *direction of travel* would place the
feet rotated about the character — the constraint would ask for a foot position that belongs to a
different body orientation entirely.

And it is stored as a **signed yaw offset from the tangent**, not as a bare world direction:

```csharp
public float facingYawOffset;   // signed yaw in radians from the path tangent to the facing
```

So a keypoint reads as *"30° off the heading here"*. Re-anchor the same clip onto a differently
oriented path and a 30° strafe stays a 30° strafe relative to the new heading, rather than becoming a
strafe in some fixed world direction.

That relative encoding is what makes the GameObject's transform requirement load-bearing: it **must
be unscaled and upright**. The offset is measured against the *local* tangent at build time and
applied against the *world* tangent at query time, and only an upright unscaled transform makes those
two comparable.

## Both artefacts are generated together and unserialized

The path and the keypoints are built from the clip in **one pass** and held `[NonSerialized]`, rather
than the path living in a `SplineContainer` and the keypoints in the scene.

Two reasons, and the second is a scar:

- They cannot describe different curves, because they are produced together from one source.
- **There is no editable copy for anyone to drag out of agreement.** An emptied container used to
  send `speed / length` to infinity and a NaN parameter into every evaluation downstream.

The consequence is that the path *is* exactly the clip's root trajectory. To place it differently,
move or rotate the GameObject.

Because `OnValidate` does not fire for edits made to the animation **asset**, the cache fingerprint
includes the clip's frame count rather than trusting the callback alone. A failed build caches an
empty result rather than retrying, so a repainting gizmo does not re-attempt it every frame.

## How the curve is fitted

Root positions are sampled per clip frame through the
[simulation frame](../animation-tools/simulation-frame.md), then simplified with Douglas–Peucker to
choose knots.

**Knot density is independent of keypoint spacing.** Knots follow the path's *shape* — clustering
through turns, thinning out on straights — while keypoints keep their own frame cadence.

The tolerance is a **knot-placement control, not a hard error bound**. The retained samples become
AutoSmooth knots and the curve then bows *through* them, so the fitted spline deviates somewhat more
than the tolerance. Larger means a sparser, more editable curve that tracks the original more
loosely.

### Keypoints are placed by arc length

Keypoints are placed by interpolating between their bracketing knots' distances, **not** by
projecting them onto the curve.

Nearest-point projection is ambiguous wherever a path crosses itself — the same problem
[the spline projector](../animation-tools/path-following-metrics.md) exists to solve. Here it is
avoidable entirely: the correspondence is already known, because the curve passes exactly through the
knots. The result is monotonic by construction.

## Switching constraints on

When a prediction horizon lands within `activationWindowFrames` (default 2) of a keypoint, the bone
trajectory channels switch on with that keypoint's bone position as the query target.

Away from keypoints they stay off, the stage **weight-masks them**, and the search behaves like plain
spline following. That masking mechanism — and why an inactive channel must not be filled with a
plausible value — is on [feature vectors](feature-vectors.md).

The tolerance is expressed in database frames of travel and converted to a normalized parameter using
the input's own speed, so it means the same thing on paths of different lengths.

## Two deliberate divergences

**Character space is computed by hand.** The keypoint input converts a world bone target into
character space with an explicit yaw-only rotation, a ground-projected origin, and **absolute height
retained** — rather than `InverseTransformPoint`, which would subtract the transform's Y and apply its
scale. Every other `GetTrajectoryFeature` in the codebase uses `InverseTransformPoint`; this one must
not.

**The `SplineContainer` property is always null and refuses assignment out loud.**
`PathFollowingMetric` and the benchmark driver push a container into *every* spline input they find,
and that must not look like it changed where this one goes. The inspector writes the inherited field
directly rather than going through the property, so `OnValidate` clears it too — otherwise an
assigned container would sit there looking as though it did something.

## Safety rails

A pose lookup **answers nothing rather than solving** when the clip's bake and the rig disagree on
bone count. The two are invalidated by different things, so they can end up describing different
skeletons, and letting the FK pass index one by the other's count would throw during a gizmo repaint.

The whole skeleton is solved in a single forward pass rather than per-bone parent-chain walks, which
is what makes drawing a posed skeleton affordable at all. Only one skeleton gizmo is drawn at a time,
selected by index — a skeleton at every keypoint would be thousands of gizmo lines per repaint.

Yaw interpolation between keypoints takes the short way round, and handles a closed spline's final
span wrapping past 1.

## Limitations

- `GetBoneTrajectoryFeature` handles only `Position`. A bone `Direction` channel always returns
  false, i.e. stays masked off.
- Keypoint lookup is a linear scan per query, called per prediction horizon per frame.
- The keypoint animation is assumed to use the same rig as the motion-matching data, so a channel's
  bone resolves against the keypoint skeleton.

## Source map

| Concern | File |
| --- | --- |
| Path fitting, keypoints, facing, gizmos | `Assets/MotionMatching/Runtime/CharacterController/SplinePoseKeypoint.cs` |
| The control input | `Assets/MotionMatching/Runtime/CharacterController/SplinePoseKeypointControlInput.cs` |

**Tests.** None.

For the general facing-versus-travel rule and its precedence order, see
[the simulation frame](../animation-tools/simulation-frame.md).
