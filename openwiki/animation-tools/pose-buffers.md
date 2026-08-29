---
type: Architecture Guide
title: Pose buffers and layouts
description: How a pose is laid out over a skeleton, why layouts are shared between the database and the runtime, and what blending a pose actually does.
tags: [pose, layout, buffers, fk]
sources:
  - id: openwiki-source-73b9f8f96de21e087ab1ffec
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseBuffer.cs
  - id: openwiki-source-d9cd1d1e23d37a8aa1befd56
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseLayout.cs
  - id: openwiki-source-18bc30084905c30f9e8f0f7f
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseLayoutBuilder.cs
  - id: openwiki-source-2cab5ee24d405d8240d13641
    resource: repo://Assets/AnimationTools/Runtime/Pose/SimulationFrame.cs
  - id: openwiki-source-bdd63f48a3fa038e3b180e9a
    resource: repo://Assets/AnimationTools/Runtime/Recording/EvaluationRecorderChannels.cs
  - id: openwiki-source-ee3e422427edb1b8213450f8
    resource: repo://Assets/AnimationTools/Runtime/Skeleton/SkeletonData.cs
  - id: openwiki-source-476673d2ea24f943f53d8959
    resource: repo://Assets/AnimationTools/Tests/Editor/PoseBufferTests.cs
  - id: openwiki-source-0fa9b7feabda51f6c0405790
    resource: repo://Assets/AnimationTools/Tests/Editor/PoseFkTests.cs
  - id: openwiki-source-8a561b6b9ce01438599024a5
    resource: repo://Assets/AnimationTools/Tests/Editor/PoseLayoutTests.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# Pose buffers and layouts

A `PoseBuffer` is a [`StateBuffer`](channel-layout-system.md) plus a `PoseLayoutData` — the named
section offsets that let pose consumers reach positions, rotations, velocities and contacts without
binding a handle for each one.

```csharp
pose.Positions[i]          // float3, bone i
pose.Rotations[i]          // quaternion, bone i
pose.Velocities[i]
pose.AngularVelocities[i]
```

Those are `NativeSlice` reinterpretations over the same flat float array, so they alias rather than
copy. Only `Rotations` is guarded: it asserts a stride-4 (quaternion) representation first, because
it is the one section whose stride can legally differ.

`PoseBuffer` is a view struct with the same ownership rules as `StateBuffer`, and it converts
implicitly to one — which is what lets `RecorderChannel.Sample` and every other generic consumer take
the narrower type without an adapter.

## Element index equals bone index

`PoseLayout.CreateFullPose` emits one position and one rotation channel per bone, in skeleton
depth-first order, so **element index equals bone index**. Optional velocity and angular-velocity
channels are laid out the same way.

This alignment is what makes `pose.Positions[i]` mean "bone i" with no lookup, and every FK loop in
the project assumes it.

## The storage convention

A pose is stored in the clip's own space:

- **Bone 0** is the rig's real root, carrying world position and rotation.
- **Bones 1 and up** carry rest offsets and parent-local rotations.
- **Velocity channels** are plain finite differences of those same channels — so bone 0's velocity is
  world-space too.

Non-root bones contribute rotation only. Their positions stay at the rest offsets seeded at
initialisation, and that is precisely what keeps bone lengths fixed: FK reconstructs every non-root
position from the rest pose and the rotations.

One caveat that will otherwise mislead you: **`ChannelSpace` is descriptive metadata and is never
enforced.** Every channel the builder creates is declared `ParentLocal`, *including bone 0's*, which
in fact holds world data. The space enum is folded into the content hash and does nothing else. Treat
"bone 0 is world, bones 1+ are parent-local" as a convention upheld by callers, not as something the
layout expresses or checks.

## The layout is the file's self-description

A `PoseLayout` is tied to a skeleton, and it *is* the pose file's self-description: a serialized pose
asset stores enough of it to reconstruct the layout without external context. That matters because a
deserialized database has no ScriptableObject to consult — see
[on-disk formats](on-disk-formats.md).

It also means structurally equal skeletons from different sources **share one cached layout
instance**. That sharing is not an optimisation; it is what makes `PoseBuffer.CopyFrom` work between
a `PoseSet`'s frames and a live component's frames at all, since `CopyFrom` requires equal layout
hashes.

Both sides go through `PoseLayoutBuilder.Build` over the same `Skeleton` for exactly this reason.

### Two hazards the cache handles

**A destroyed rig.** A skeleton is a Transform tree, so a cached entry's rig can be destroyed out from
under it by a reimport or a scene unload. Such an entry reports zero bones and can never match again,
so it is evicted rather than left to block its slot forever.

**A hash collision.** If two structurally different skeletons collide, the incumbent is *not* evicted
— the fresh layout is simply left uncached. Overwriting would break the other skeleton's
shared-instance guarantee, which is the whole point of the cache. The colliding layout loses the
cache benefit and stays correct.

Note that the cache is keyed on the hash but membership is always confirmed with
`Skeleton.StructurallyEqual`. The hash alone is never trusted for cache identity, even though the
handle system does treat equal hashes as identical offsets.

## One rotation representation per layout

`Build` rejects a layout mixing quaternion and rotation-vector channels. A single `RotationStride`
follows from that — 4 for quaternions, 3 otherwise.

The reason is mechanical: a per-channel representation would make the Rotation section
non-uniform-stride, which would destroy `PoseBuffer.Rotations` as a reinterpreted slice.

## What blending actually does

`PoseBuffer.Lerp` is not a single interpolation. It applies a different rule per section:

| Section | Rule |
| --- | --- |
| Positions, scales, velocities, angular velocities | componentwise `math.lerp` |
| Rotations (stride 4) | hemisphere fix (`if dot < 0, negate`) then lerp and `normalizesafe` |
| Rotations (stride 3) | componentwise |
| Bools | **step** at `t = 0.5` |
| Extra tail | linear |

A bool is not linearly interpolable — 0.5 is neither planted nor not — so it steps. Extra sections
are floats of unknown meaning, so linear is the only defensible default.

The hemisphere fix matters for the same reason `MathExtensions.Abs` exists: without it, blending
identity against a negated identity cancels out instead of staying near identity. `PoseBufferTests`
pins that case specifically.

## Contacts, and why the bone choice is constrained

`PoseLayoutBuilder.Build` produces the full pose channel set plus two `BoolChannel`s carrying
`ChannelUsage.Contact`, and hands back handles for them.

The contact host bone for each side **must be a pure function of the skeleton's bone names**, never of
the skeleton's source. If the database side picked contact bones from the source asset's
configuration while the runtime side picked them by name, the two would produce different channel
identities for the same rig, therefore different layout hashes, therefore a failing `CopyFrom` — the
exact thing the shared cache exists to prevent.

Resolution goes through `BoneNameConventions.TryFindContactBone`, preferring toes and falling back to
feet. When a rig has neither, the fallbacks are bone `0` for the left and `BoneCount - 1` for the
right — and if both sides resolve to the *same* bone, the builder forces them apart, because the
layout rejects duplicate channel identities and a skeleton missing one side must still end up with
two distinct hosts.

The same rule is restated from the consumer side by `ContactBoneWorldPositionChannel`, which measures
footskate: the position and the contact flag have to name the same bone, so it goes through
`BoneNameConventions` too rather than trusting whoever filled in the inspector.

`BuildFullPoseChannels` exposes the channel set without the contact bools, for consumers that append
their own channels before calling `PoseLayout.Build` themselves. `ContactVisualizerStage` is the one
that does.

## Forward kinematics

FK lives on `SkeletonData` and exploits the depth-first invariant — every bone's parent has a lower
index — to run as a **single forward pass** rather than walking each bone's parent chain.

It is strict about its input, and the strictness is argued:

- A pose and a skeleton disagreeing on bone count **throws**, with both counts named, rather than
  asserting. Every loop is bounded by the skeleton but indexes the pose, so the mismatch would
  otherwise surface as an `IndexOutOfRangeException` from deep inside `NativeSlice` with nothing to
  say which two things disagreed — once per repaint, from an asset preview. An assert would log and
  then run off the end anyway.
- Rotations must be quaternions.
- **Scale channels are rejected outright** — "PoseFK v1 assumes unit scale". So of the six built-in
  channel kinds, `ScaleChannel` is one the FK path actively refuses.

There is an acknowledged tradeoff in that throw: the interpolated message forfeits the Burst
compatibility `SkeletonData` advertises, because Burst cannot compile string formatting. Nothing
Burst-compiles FK today — the project's `[BurstCompile]` code all lives under
`MotionMatching/Runtime/Core/Burst/` and none of it calls in — so the cost is real but currently
unexercised.

## Sequences

`PoseSequence` re-types `StateSequence`'s frame views as `PoseBuffer`. It **hides** rather than
overrides `GetFrame` and `Layout`, deliberately: a caller holding a `StateSequence`-typed reference
gets plain `StateBuffer` frames, with no surprise covariance and no virtual dispatch on a per-frame
call.

Ownership rules are the base class's — never dispose a frame view, and only dispose sequences whose
storage was allocated there.

## Source map

| Concern | File |
| --- | --- |
| Typed access, `Lerp` | `Assets/AnimationTools/Runtime/Pose/PoseBuffer.cs` |
| Sections, cache, validation, `CreateFullPose` | `Assets/AnimationTools/Runtime/Pose/PoseLayout.cs` |
| The canonical MM channel set, contacts | `Assets/AnimationTools/Runtime/Pose/PoseLayoutBuilder.cs` |
| Frame stacking | `Assets/AnimationTools/Runtime/Pose/PoseSequence.cs` |
| The six channel kinds | `Assets/AnimationTools/Runtime/Pose/BuiltInChannels.cs` |
| FK | `Assets/AnimationTools/Runtime/Skeleton/SkeletonData.cs` |

**Tests.** `PoseBufferTests` (round trips, slice aliasing, the three `Lerp` rules, the view-struct
regression pin), `PoseLayoutTests` (hand-computed offsets, validation, the cache returning the same
reference, the extra tail), `PoseSequenceTests` (frame views, disjointness, length validation),
`PoseFkTests` (FK against hand-computed chains, and the bone-count diagnostic).

For the mechanics underneath — handles, hashing, sections, the rotation-rate rule — see
[the channel and layout system](channel-layout-system.md). For how a frame becomes a character
standing somewhere in the world, see [the simulation frame](simulation-frame.md).
