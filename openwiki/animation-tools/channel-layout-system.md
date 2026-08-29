---
type: Architecture Guide
title: The channel and layout system
description: The typed float-buffer substrate every pose, feature vector and recording is built on, and the rules that keep buffers from being read with the wrong offsets.
tags: [buffers, layout, channels, substrate]
sources:
  - id: openwiki-source-09c1f59e86c3dcb1b1132e7c
    resource: repo://Assets/AnimationTools/Runtime/Pose/ChannelDescriptor.cs
  - id: openwiki-source-cf7e23a7b11d7ae34bf1a21e
    resource: repo://Assets/AnimationTools/Runtime/Pose/ChannelHandle.cs
  - id: openwiki-source-2d7a23c484e1299ec007b15d
    resource: repo://Assets/AnimationTools/Runtime/Pose/ChannelTypes.cs
  - id: openwiki-source-d9cd1d1e23d37a8aa1befd56
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseLayout.cs
  - id: openwiki-source-7ddc01d0cb0c121ee9c59c46
    resource: repo://Assets/AnimationTools/Runtime/State/StateBuffer.cs
  - id: openwiki-source-398fe8adec82d64ba2c20301
    resource: repo://Assets/AnimationTools/Runtime/State/StateBufferLayout.cs
  - id: openwiki-source-874efac03900828f7adaceb6
    resource: repo://Assets/AnimationTools/Runtime/State/StateSequence.cs
  - id: openwiki-source-88a8de4f2cf61b8c5c264070
    resource: repo://Assets/AnimationTools/Runtime/Utils/MathExtensions.cs
  - id: openwiki-source-476673d2ea24f943f53d8959
    resource: repo://Assets/AnimationTools/Tests/Editor/PoseBufferTests.cs
  - id: openwiki-source-8a561b6b9ce01438599024a5
    resource: repo://Assets/AnimationTools/Tests/Editor/PoseLayoutTests.cs
  - id: openwiki-source-201928bc2575c57e0c52ed1a
    resource: repo://Assets/AnimationTools/Tests/Editor/StateBufferLayoutTests.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# The channel and layout system

Underneath poses, motion-matching feature vectors and recordings sits one mechanism: a flat
`NativeArray<float>` plus a layout that says which floats mean what.

`StateBufferLayout` maps a `ChannelDescriptor` to a float offset. `StateBuffer` reads and writes
through `ChannelHandle`s bound against that layout. `StateSequence` stacks N frames of one layout
back to back. Three separate subsystems build on this — [pose buffers](pose-buffers.md),
[recordings](motion-recording.md) and [matching features](../motion-matching/feature-vectors.md) —
and it is the reason a pose frame and a feature vector can share the same access machinery without
sharing any shape.

## Buffers are views

`StateBuffer` is a **view struct**, exactly like `NativeArray<T>` itself. Copying one copies the
view, not the data — every copy aliases the same underlying array.

Ownership follows the usual Native\* rule: whoever calls `Allocate` calls `Dispose`.
Component-lifetime buffers use `Allocator.Persistent` and dispose in `OnDestroy`; short-lived scratch
buffers use `Allocator.Temp`. `Allocate` always clears memory, so a fresh buffer is zeroed by
contract.

This aliasing is load-bearing rather than incidental — a helper taking a `PoseBuffer` *by value* can
still write contacts that the caller sees. `PoseBufferTests` pins it explicitly as a regression test.

## Handles, and why they carry a hash

A `ChannelHandle` stores an **absolute float offset** into the buffer, not a section-local element
index. Reading through one therefore costs no layout arithmetic at all.

That speed has a hazard attached: a handle bound against one layout would happily address arbitrary
floats in a buffer built from another. So every handle also carries the `LayoutHash` it was bound
against, and every accessor asserts the match.

`LayoutHash` is claimed as the exact equivalence class, not an approximation. It covers every
channel's content and — for pose layouts — the skeleton, so **equal hashes mean identical offsets**.
Because `CopyFrom` asserts on the same hash, handle validity and copy compatibility are the same
predicate by construction rather than by coincidence.

The four per-access assertions are `[Conditional("UNITY_ASSERTIONS")]`, so they compile out of
release builds. The safety is a development-time guarantee, not a runtime one.

`FloatOffset` is exposed publicly despite the above, because per-dimension side tables —
normalisation statistics, per-float weights — have to be indexed positionally. Reading buffer *data*
should still go through `StateBuffer`, which checks the handle.

## Two hashes, deliberately

`ChannelDescriptor` carries two different hashes, and the split is the subtlest idea here.

| Hash | Answers | Covers | Feeds |
| --- | --- | --- | --- |
| `GetHashCode` | "which channel is this?" | identity only — concrete type, bone, usage | the layout's offset lookup |
| `GetContentHash` | "is this the same channel, meaning the same thing?" | identity **plus** space and rotation representation | `StateBufferLayout.LayoutHash` |

Identity deliberately excludes descriptive metadata so that **binding never has to restate it**. You
present a `PositionChannel(boneId)` and get the handle; you do not have to remember whether that
channel was declared parent-local or world.

The consequence is worth stating plainly: **two layouts differing only in a channel's space address
identically, yet are not interchangeable.** The offsets match, so a naive design would let you copy
one into the other; the content hash is what refuses.

Neither choice works alone. Folding space into identity would force every caller to restate it.
Using content for identity would let layouts that differ in meaning but not in shape compare as
copy-compatible.

## Section keys are constants, not an enum

```csharp
public static class ChannelSections
{
    public const int Position        = 0;
    public const int Rotation        = 1;
    public const int Scale           = 2;
    public const int Velocity        = 3;
    public const int AngularVelocity = 4;
    public const int Bool            = 5;
}
```

These are `const int` rather than an enum's ordinals on purpose. Section order is a buffer-layout
contract. An enum's numeric values follow source declaration order, so someone tidying the enum into
alphabetical order would silently reorder the sections of every buffer in the project — **with no
compile error and no runtime complaint**, just pose data landing at different offsets.

Channel kinds outside this set must use a key greater than `Bool`, which keeps them in a contiguous
tail after the built-in sections. `PoseLayout.Build` throws when a non-built-in channel claims a
reserved key.

## How a layout is built

The constructor validates, groups, then assigns:

1. Reject nulls, duplicate identities and negative section keys — all `ArgumentException`.
2. Group channels into a `SortedDictionary` keyed by section, which realises "ascending section key
   order", with the list inside each group preserving declaration order.
3. Walk one cursor through the groups, assigning each channel `new ChannelHandle(cursor, channel.FloatCount, layoutHash)`
   and advancing by that channel's own width.

So: **sections in ascending key order, declaration order preserved within a section, and each channel
occupying its own width.** A section is only uniform-stride when its channels happen to be.

Note that `Channels` is exposed in *declaration* order while sections are laid out in *key* order, so
`Channels[i]` does not correspond to increasing offset. `GetChannel(sectionKey, elementIndex)` is the
offset-ordered accessor.

`layoutHash` is a constructor parameter supplied by the subclass, not computed by the base class,
because what a layout's identity must cover is subclass-specific. A `PoseLayout`'s identity has to
include the skeleton — which the base class cannot see.

Binding allocates a probe descriptor per call, so it is meant for setup, not per-frame use. The width
comes from the *stored* channel, which is what lets a probe differing only in descriptive metadata
still bind correctly.

## Rotational rates are not quaternions

This is the rule that reaches furthest outside this page.

Quaternions cannot be added or scaled meaningfully. Half a quaternion is not half the rotation; a sum
of two is not a composition. So anything treating rotation as a **rate** or an **offset** — angular
velocity, inertialization, spring damping — works in scaled-angle-axis form instead: a plain `float3`
of axis times angle in radians, which does behave like a vector.

`MathExtensions.Log` and `Exp` convert between the two forms, and everything else builds on them.
The same rule is encoded in the type system: `AngularVelocityChannel.Representation` is a read-only
expression returning `RotationVector`, not a settable default, because a quaternion has no meaningful
rate form at all.

Two details worth keeping:

- **`Abs` picks the short way round.** `q` and `-q` are the same rotation, but only one of them is
  the short arc. `Abs` negates all four components when `w < 0`, so a difference of two rotations
  comes back as 10 degrees rather than 350.
- **`Log` and `Exp` both branch at small angles** (epsilon `1e-8`), avoiding `acos` and the division
  by length near identity where both are ill-conditioned — and where the branches agree anyway.

The Python side mirrors this deliberately, and for a measured reason: round-tripping a rad/s rate
through a quaternion aliases every joint turning faster than π rad/s, which is about **2.6% of the
project's dataset**, of which the majority come back pointing the opposite way. See
[the pose data bridge](../python/pose-data-bridge.md).

## Usage, and explicit root motion

`ChannelUsage` disambiguates multiple channels of the same kind on the same bone —
`Default`, `RootMotion`, `Contact`.

Its point is that root motion becomes an **explicit, discoverable channel declaration** rather than
an implicit "bone index 0 is the root" convention. An implicit rule is undiscoverable and
unenforceable; a declared usage shows up in the layout and can be bound against.

## Sequences

`StateSequence` holds N frames of one layout in a single array. `GetFrame` hands out a `StateBuffer`
that is a sub-array **view** into that storage.

Two ownership rules follow, and both are easy to get wrong:

- **Never dispose a frame view.** Disposing a sub-array throws.
- **Never dispose a sequence built over storage you did not allocate.** A sequence wrapping an
  `Allocator.Domain` array — which `FeatureSet` builds — outlives the sequence, and disposing it
  there would throw.

A zero-frame sequence is illegal: the constructor requires the data length to be a positive multiple
of the layout width.

## Booleans

A bool is one float, written as `0f` or `1f` and read back with a `> 0.5f` threshold.

That keeps a buffer a single homogeneous `float` array, which is what makes flat `CopyFrom`,
`NativeSlice` reinterpretation and single-call numpy reads possible. See
[on-disk formats](on-disk-formats.md) for the payoff on the Python side.

## Source map

| Concern | File |
| --- | --- |
| Buffer access and assertions | `Assets/AnimationTools/Runtime/State/StateBuffer.cs` |
| Offsets, sections, validation | `Assets/AnimationTools/Runtime/State/StateBufferLayout.cs` |
| Frame stacking | `Assets/AnimationTools/Runtime/State/StateSequence.cs` |
| Identity and content hashing | `Assets/AnimationTools/Runtime/Pose/ChannelDescriptor.cs` |
| Handle | `Assets/AnimationTools/Runtime/Pose/ChannelHandle.cs` |
| Section keys, spaces, representations | `Assets/AnimationTools/Runtime/Pose/ChannelTypes.cs` |
| Rotation maths | `Assets/AnimationTools/Runtime/Utils/MathExtensions.cs` |

**Tests.** `StateBufferLayoutTests` covers offset assignment, bind semantics, section ranges and
duplicate rejection. The two cases that pin the ideas on this page most directly live in
`PoseLayoutTests` instead: `LayoutHash_ChannelsDifferingOnlyInSpace_ProduceDifferentHash` (the
two-hash design) and `Handle_UsedOnBufferFromDifferentLayout_TripsLayoutAssert` (the handle check).

## Downstream

[Pose buffers](pose-buffers.md), [recordings](motion-recording.md) and
[matching features](../motion-matching/feature-vectors.md) all build on this page and defer to it for
layout mechanics. [Inertialization](../motion-matching/inertialization.md) and the springs in
[control inputs](../motion-matching/control-inputs.md) depend on the scaled-angle-axis rule above.
