---
type: Architecture Guide
title: The simulation frame, facing and travel
description: How a character frame is derived from a pose rather than stored, and why facing and direction of travel are different quantities that must not be substituted for one another.
tags: [simulation-frame, facing, spline, conventions]
sources:
  - id: openwiki-source-fdcd1ba2cc82f452d31d0aa6
    resource: repo://Assets/AnimationTools/Editor/Benchmark/SynthesisBenchmarkDriver.cs
  - id: openwiki-source-d6aed5b3fb13baa7a3775151
    resource: repo://Assets/AnimationTools/Runtime/Evaluation/SplinePathDirection.cs
  - id: openwiki-source-9cb39479f08cf519691c343e
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseSet.cs
  - id: openwiki-source-2cab5ee24d405d8240d13641
    resource: repo://Assets/AnimationTools/Runtime/Pose/SimulationFrame.cs
  - id: openwiki-source-54f745ff7f6016dea0150c17
    resource: repo://Assets/AnimationTools/Runtime/Skeleton/Skeleton.cs
  - id: openwiki-source-cd6059ad9d963983d95938af
    resource: repo://Assets/AnimationTools/Tests/Editor/SimulationFrameTests.cs
  - id: openwiki-source-5c5de1c38fc74b789911f4ea
    resource: repo://Assets/MotionField/MotionField.asmdef
  - id: openwiki-source-1fb0416756719f48922c424e
    resource: repo://Assets/MotionField/MotionFieldSplineControlInput.cs
  - id: openwiki-source-5a2c119cb47a6d7fc76496cc
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/SplineControlInput.cs
  - id: openwiki-source-23c39c78df6491e3d9dfdcb1
    resource: repo://Assets/MotionMatching/Runtime/CharacterController/SplinePoseKeypointControlInput.cs
  - id: openwiki-source-84760d31adde69871d6065c9
    resource: repo://Assets/MotionMatching/Runtime/MotionMatching.asmdef
  - id: openwiki-source-68bd8cb3d0139ef5eb77babd
    resource: repo://Assets/MotionMatching/Runtime/Unity/MotionMatchingData.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# The simulation frame, facing and travel

Given a pose, two questions have to be answerable: **where is this character** and **which way is it
facing**. The simulation frame answers both.

It is the ground-projected, yaw-only frame a character controller works in. Position is the reference
bone's world position flattened onto the XZ plane; rotation is the yaw that aims at that bone's
flattened forward axis.

`SimulationFrame` is a static class of pure functions. It stores nothing, allocates nothing, and owns
no state. `SimulationFrameDef` is the two-field configuration — a reference bone index and a
bone-local forward axis — that parameterises the derivation.

## Derived, never stored

The frame is computed on demand and never written down. Two things follow:

- **It is yaw-only by construction.** A stored rotation could be authored with pitch or roll in it. A
  derived one cannot.
- **It cannot drift out of sync with the pose it came from.** A stored frame would be a second source
  of truth for something the pose already fully determines; any edit, retarget, re-anchor or
  interpolation of the pose would leave it stale with nothing detecting the disagreement. Deriving
  makes desynchronisation *impossible* rather than merely unlikely.

`PoseSet.SimulationFrame` is a computed property, recomputed on every access.
`MotionSynthesisComponent.SimulationFrame` is assigned once from `SimulationFrameDef.Default`.

### Idempotence is what makes re-anchoring work

Expressing a pose's bone 0 frame-locally yields a pose whose own derived frame is the identity.

That property is the entire basis of the runtime. It is what lets a pose from anywhere in a database
be re-anchored under an arbitrary world transform and re-derived without loss — which is exactly what
[the pipeline](synthesis-pipeline.md) does every tick when it places bone 0 relative to the frame the
pose implies and then advances the character's Transform by that frame's own velocity.

## Why the forward axis is read from the rest pose

A rig's bone roll is arbitrary. The bone that means "hips" may have its local +Z pointing up the
spine, out to the side, or backwards, and nothing in an FBX constrains it. Two rigs that animate
identically can disagree completely about what a given bone's local +Z means.

So the forward axis is **a property of the rest pose, not a constant**. `Skeleton.RestLocalAxis`
converts the stable character-space fact — forward is +Z — into the bone-local axis that happens to
realise it on this particular rig:

```csharp
ForwardAxisLocal = math.mul(math.inverse(skeleton.RestCharacterRotation(bone)), math.forward());
```

The anchoring assumption is that the exported FBX rig is a **T-pose facing Unity forward**, so
character forward is +Z and character right is +X. Arms are the interesting exception: they point
sideways in a T-pose, so `MotionMatchingData` gives them the character's *right* axis as their
forward instead — the same mechanism generalised beyond the frame.

The Python side derives the identical axis rather than reading it from the file. See
[the pose data bridge](../python/pose-data-bridge.md).

## No configurable reference bone

Every production construction goes through `SimulationFrameDef.Default`, which fixes the reference
bone to the skeleton root and the forward axis to that bone's rest forward.

There used to be an inspector-exposed reference bone and forward axis on the config assets. Removing
that authoring surface is what guarantees the C# runtime, the serialized database and the Python
pipeline cannot disagree about the frame — three places that otherwise can. The same reasoning
governs the file format: `.mmpose` writes the skeleton block but **not** the frame, because both
halves derive it from bone 0's rest rotation, which the bone entries already carry.

The type itself remains fully general and still supports an arbitrary reference bone — the tests
exercise `ReferenceBoneIndex = 2`. It is the *authoring surface* that was removed, not the
capability. Do not delete the fields as vestigial.

## Facing is not direction of travel

This is the distinction that causes real bugs, and the reason this page exists as a single home for
an argument the source states in five separate places across three assemblies.

**Facing** is where the character's body points — what the frame's yaw encodes. **Travel** is where
its centre of mass is heading — what a path tangent gives. A character can strafe, backpedal, or turn
on the spot, and in all three the two diverge.

The rule the codebase follows:

> **The control input is the authority on facing. `SplinePathDirection` is the authority on travel.**
> They coincide only for controllers that have no independent notion of facing.

The precedence order, as the benchmark driver implements it when placing a character at the start of
a path:

1. `input.GetWorldInitDirection()` — the control input decides.
2. `SplinePathDirection.WorldStartDirection(container)` — the path's own start direction.
3. The character frame's existing rotation.

A path carrying its own facing therefore spawns the character strafing or backpedalling when that is
what it was authored to do; the path's start direction is only the fallback. A zero-length result
means "no opinion", and callers keep the rotation they have rather than snapping to something
arbitrary.

### Where they legitimately coincide

A pure path follower has no notion of facing beyond its path, so travel direction is the honest
answer — and the character's Transform forward is arbitrary anyway for something spawned onto a path
it was never authored against. Both `SplineControlInput` and `MotionFieldSplineControlInput` say so
explicitly; pure pursuit steers toward the path and has nothing else to offer.

### Where substituting one for the other breaks

The database bakes its Direction channels as the character's **facing**. Nothing downstream steers
the character toward the controller — the pipeline integrates the matched clip's own yaw rate — so
the direction query is the *only* influence a control input has over which way the character ends up
facing.

Feed it direction of travel and it fights the foot constraints on any pose whose facing differs from
its heading. The search is being asked to match a foot position *and* a direction simultaneously; if
the direction channel carries travel while the database carries facing, then on any strafing or
backpedalling pose the two objectives pull toward different clips and the search satisfies neither.

[Spline pose keypoints](../motion-matching/spline-pose-keypoints.md) is where this is handled
properly: facing is stored per keypoint as a signed yaw *offset* from the tangent, so a 30° strafe
stays a 30° strafe relative to whatever heading the path has at that point.

## SplinePathDirection

`SplinePathDirection.WorldStartDirection` answers "which world direction does this path set off in".
It lives in `AnimationTools` rather than beside either caller for a concrete reason: `MotionMatching`
and `MotionField` are **sibling assemblies with no reference between them**, and `AnimationTools` is
the only common dependency where a helper visible to both spline-following control inputs can live.
Three consumers span all three assemblies.

It is also where a Unity Splines quirk is absorbed. A knot authored with a **linear tangent evaluates
to a zero tangent**, so a path with square corners has no analytic direction at its start —
`EvaluateTangent(0f)` simply returns nothing usable. The fallback takes a short forward chord,
capped at both 0.1 m absolute and 5% of the spline's length:

```csharp
step = math.min(0.1f, length * 0.05f) / length;   // metres → normalized parameter
```

The two-term cap is doing real work: 0.1 m keeps the chord local on a long path, and 5% keeps it from
swallowing a short one. Dividing by length converts metres into normalized parameter, which is valid
because Unity's normalized spline parameters are uniform in arc length — the same fact
[spline projection](path-following-metrics.md) relies on.

The chord "agrees with the tangent everywhere the tangent exists", which is what makes it safe to
apply unconditionally rather than only on knots known to be linear.

Note that `SplinePathDirection` lives in `Runtime/Evaluation/` alongside the projector and the
metrics, not next to `SimulationFrame.cs`. It is grouped here by concept — the facing/travel
distinction — not by folder.

## Rates, and the finite-step contract

`ComputeVelocity` is defined in the **finite-step form the extraction pipeline stores**: the
reference bone's channels are advanced by one step of `dt` and the frame is re-derived. So a pose
holding one-frame finite differences reproduces the neighbouring frame's simulation frame exactly.

It returns ground-plane velocity of the frame origin in frame space, and a signed yaw rate in
radians per second.

`DecomposeRootVelocity` splits bone 0's world velocity into the part the frame already carries —
including the tangential velocity the frame's yaw imparts at bone 0's offset — and the leftover.
`RecomposeRootVelocity` is its exact inverse. [Inertialization](../motion-matching/inertialization.md)
needs the pair because velocities have to follow the position back through the frame or the component
re-derives the wrong motion; nothing about the split is specific to bone 0, and `CharacterSpacePose`
applies it to every bone.

## The whole pose in the frame

`CharacterSpacePose.Extract` measures **every** bone inside the frame the pose implies: position,
rotation, and both rates, in one forward pass. Where the character stands and which way it faces are
exactly what a locomotion model has to be invariant to, so this is the shape a network sees a pose
in, as opposed to how one is stored.

It is deliberately the C# counterpart of `Python/training_data.py`, and the reason to have one place
for it is that the failure mode is silent: a mismatch in frame, units or rate convention between the
data a model was trained on and the data it is given at inference does not throw, it just makes the
network wrong. The two sides are not bit-identical on the rates — Python differences consecutive
frame-local poses of a stored database, the C# composes the instantaneous rate implied by the pose's
own velocity channels — but those channels are themselves finite differences over the same timestep,
so the two agree to first order and exactly for motion that is rigid within the frame. The test
suites on both sides are written against the same properties.

## Degenerate inputs

Every step has a defined answer rather than a NaN:

- A reference bone aimed straight up or down leaves nothing to flatten, so the character-forward axis
  stands in. `LookRotationSafe` and `normalizesafe` are used throughout.
- `WorldStartDirection` returns `float3.zero` as an explicit **no-answer sentinel** for a null
  container, a spline with fewer than two knots, a sub-millimetre spline, or a chord that is still
  degenerate. Callers must test `lengthsq > 0` and fall back explicitly rather than propagating.

One unstated assumption worth knowing: ground projection **discards the y coordinate outright**
rather than raycasting. "Ground-projected" means the XZ plane at y = 0 in character space — a flat
world. Nothing in the source argues for or against this; it is simply what the code does.

## A naming collision

`MotionSynthesisComponent` has a property `SimulationFrame` of type `SimulationFrameDef`, which
shadows the static class `AnimationTools.SimulationFrame`. Every call site inside that file therefore
reads `AnimationTools.SimulationFrame.Compute(...)`. This is a second, milder version of the
[`SkeletonBone` collision](skeletons-and-rig-binding.md), and it will bite anyone adding a call site.

## Source map

| Concern | File |
| --- | --- |
| Frame derivation, rates, frame-local conversions | `Assets/AnimationTools/Runtime/Pose/SimulationFrame.cs` |
| Path start direction | `Assets/AnimationTools/Runtime/Evaluation/SplinePathDirection.cs` |
| Rest-pose forward axis | `Assets/AnimationTools/Runtime/Skeleton/Skeleton.cs` (`RestLocalAxis`) |

**Tests.** `SimulationFrameTests` pins the derivation itself: the ground projection and yaw
extraction (asserting the frame rotation is yaw-only), a non-root reference bone, the finite-step
velocity contract, the idempotence property, the `ToFrameLocal`/`FromFrameLocal` round trip, and the
decompose/recompose inverse including the tangential term.

**Untested:** `SplinePathDirection` has no test file at all. The chord fallback and its four constants
are unexercised, despite two neighbouring test files constructing exactly the `TangentMode.Linear`
splines that trigger it.
