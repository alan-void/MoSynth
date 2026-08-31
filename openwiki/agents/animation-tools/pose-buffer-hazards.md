---
type: Agent Reference
title: Editing pose code without breaking it
description: The compile errors, ownership rules and unit conventions that catch every first attempt at touching a PoseBuffer, a PoseLayout or a Skeleton.
tags: [agents, animation-tools, pose, skeleton, native-collections]
sources:
  - id: openwiki-source-73b9f8f96de21e087ab1ffec
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseBuffer.cs
  - id: openwiki-source-d9cd1d1e23d37a8aa1befd56
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseLayout.cs
  - id: openwiki-source-18bc30084905c30f9e8f0f7f
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseLayoutBuilder.cs
  - id: openwiki-source-3407e50b1558ced033634667
    resource: repo://Assets/AnimationTools/Tests/Editor/TestSkeletons.cs
  - id: openwiki-source-5c14bcca1cea98bfb51c8d09
    resource: repo://Assets/MotionMatching/Tests/Editor/MmTestData.cs
generated: {by: "claude-code", at: "2026-08-29T23:23:48.822Z"}
---

# Editing pose code without breaking it

Concept-level background is on [pose buffers and layouts](../../animation-tools/pose-buffers.md) and
[skeletons and rig binding](../../animation-tools/skeletons-and-rig-binding.md). This page is the
operational half: the things that go wrong the first time.

## You cannot assign through `pose.Positions[i]`

`PoseBuffer.Positions`, `.Rotations`, `.Velocities` and `.AngularVelocities` are **properties
returning a `NativeSlice<T>` by value**, so an indexed write through one is
`error CS1612: Cannot modify the return value ... because it is not a variable`.

```csharp
pose.Positions[0] = position;              // CS1612

var positions = pose.Positions;            // take the slice into a local first
positions[0] = position;
```

Reads are fine. In test code where the writes are scattered, small `SetPosition(bone, value)` helpers
that take the slice into a local are less noise than repeating it.

## Never Dispose a frame you were handed

`PoseSet.GetPoseBuffer` and `StateSequence.GetFrame` return **views** — `NativeArray.GetSubArray`
slices over storage someone else owns. Disposing a sub-array throws.

`PoseSet`'s own storage is `Allocator.Domain`: never disposed, freed on domain unload, and shared
because `MotionMatchingData` caches one pose set. `PoseSet.Dispose` deliberately releases only the
tags. Do not "fix" that.

Ownership follows the usual Native\* rule everywhere else: whoever called `Allocate` disposes.
Component-lifetime buffers use `Allocator.Persistent` and dispose in `OnDestroy`; scratch uses
`Allocator.Temp`.

## Two different `Layout` types

| Buffer | `.Layout` type | Has `LayoutHash` |
| --- | --- | --- |
| `StateBuffer` | `StateBufferLayoutData` | yes |
| `PoseBuffer` | `PoseLayoutData` | no |

`PoseLayoutData` carries the named section offsets instead. If you want to compare two pose layouts,
compare the `PoseLayout` objects — and note that `PoseLayout.Build` caches per (skeleton content
hash, channel set), so two builds over the same skeleton hand back **the same instance**. That cache
is what lets `PoseBuffer.CopyFrom` work between a `PoseSet` frame and a `MotionSynthesisComponent`
buffer, since both go through `PoseLayoutBuilder.Build`.

Consequence for tests: calling `PoseLayoutBuilder.Build(skeleton, out var contacts)` a second time is
a cheap way to recover the contact handles for a buffer you already allocated.

## A skeleton is real GameObjects

`Skeleton` is a Transform tree, so every fixture that builds one creates actual GameObjects.
`TestSkeletons.Build`/`CreateChain3`/`CreateBranch4` and `MmTestData.BuildSkeleton` all do.

**Every suite that builds one must call `DestroyAll()` from `[TearDown]`**, which also calls
`Skeleton.InvalidateAll()` to drop the derived-data caches keyed on the now-dead roots. Skipping it
leaks GameObjects into the test scene and poisons the layout cache for later tests.

## Units in the velocity channels

Angular velocity channels are **rotation vectors in radians per second**, taken the short way round —
not Euler angles, not degrees, not a per-tick delta. `SkeletonData.CharacterSpaceVelocity`
cross-multiplies that channel with a bone offset, so a wrong unit here produces plausible numbers
rather than an error.

Linear velocity channels are metres per second. Both are plain finite differences of the
corresponding pose channels, which is why bone 0's velocity is in world space like its position while
every other bone's is parent-local.

Use `MathExtensions.AngularVelocity(from, to, dt)`, or
`QuaternionToScaledAngleAxis(Abs(mul(to, inverse(from)))) * inverseDt` when you need to guard a zero
timestep without dividing.

## Bone indices and ids are not the same number

Bone **ids** follow the index + 1 convention, with 0 meaning unset. `Skeleton.GetBoneId(i)` is what a
`ChannelDescriptor` takes; `Skeleton.IndexOfId(id)` goes back. Channel constructors take ids, and
array indices are indices — mixing them is off by one and silently addresses the wrong bone.

`SkeletonData.ParentIndices[i] < i` always holds, because bones are stored in depth-first order. That
invariant is what lets forward kinematics and rate composition be a single forward loop rather than a
walk per bone; rely on it rather than re-deriving it.
