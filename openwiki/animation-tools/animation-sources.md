---
type: Architecture Guide
title: Animation sources and clip baking
description: How clips and BVH files become poses, why a clip is accepted on one animated bone, and what baking does and does not capture.
tags: [clips, baking, bvh, import]
sources:
  - id: openwiki-source-3de15f937f7e1dfc53c7ec56
    resource: repo://Assets/AnimationTools/Editor/CreateAnnotatedClipMenu.cs
  - id: openwiki-source-eaff015da326d13eeef3a663
    resource: repo://Assets/AnimationTools/Editor/Importers/BvhImporter.cs
  - id: openwiki-source-2e0d28f6a1ee99e6e464a576
    resource: repo://Assets/AnimationTools/Runtime/Animation/AnimationClipBaker.cs
  - id: openwiki-source-81d1a220d914a0173e6778bb
    resource: repo://Assets/AnimationTools/Runtime/Animation/AnnotatedAnimationClip.cs
  - id: openwiki-source-e1ad0ab569ae74b451c3418f
    resource: repo://Assets/AnimationTools/Runtime/Animation/SkeletonAnimation.cs
  - id: openwiki-source-3d5c835733aa13f4eeed0f83
    resource: repo://Assets/AnimationTools/Tests/Editor/AnimationClipBakerTests.cs
  - id: openwiki-source-de96c5aeacc3125110ef502b
    resource: repo://Assets/Scripts/BioVisionHierarchyToAnimClip.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# Animation sources and clip baking

This is where animation data enters the project: an `AnimationClip` plus the `Skeleton` it is sampled
against, baked into the flat float pose format. The layer stops at `PoseSequence` — it knows nothing
about databases, features or serialization.

| Type | Role |
| --- | --- |
| `SkeletonAnimation` | pairs a clip with its skeleton; bakes lazily into a `PoseSequence` |
| `AnnotatedAnimationClip` | adds a `[startFrame, endFrame)` slice and a tag struct |
| `AnimationClipBaker` | validates and bakes; the only code that samples a clip |
| `BvhImporter` | a `ScriptedImporter` turning `.bvh` into a rig plus a clip sub-asset |

## The acceptance rule

`AnimationClipBaker.TryValidateClip` exists because of a silent failure. `AnimationClip.SampleAnimation`
matches the clip's curve paths against whatever hierarchy it is handed, and **paths that do not
resolve are silently ignored** — so a clip authored for a different rig produces no error at all,
just a figure frozen in its rest pose. Breaking that silence is the entire point of validation.

The rule it applies is deliberately permissive:

> A clip is accepted as soon as it animates **one** bone, not all of them.

Rigs routinely leave leaf bones uncurved. The project's own clips animate **72 of 85 bones**, the
other 13 being the FBX `_end` markers. Demanding full coverage would reject every asset in the
repository.

`AnimationClipBakerTests` keeps that number as an explicit regression guard against over-strictness.
A separate test covers the divergence case that produced the original console spam: a rig whose bones
changed since the skeleton was built, where the message reports both counts because the bake itself
bails with a bare null and cannot say why.

Curve bindings are only readable in the Editor, so **in a player build validation returns `true`
unconditionally**.

## What baking captures

- **Only the root bone's translation** is baked per frame. Every other bone's position stays at its
  skeleton rest offset, because FK reconstructs non-root positions from the rest pose and the
  parent-local rotations.
- **Humanoid (muscle-curve) clips are unsupported** — they produce no transform motion on a plain
  hierarchy. This is caught at asset-creation time with a dialog telling you to set the rig to
  Generic, rather than failing later in the bake.
- **Baking is runtime-legal**, using `AnimationClip.SampleAnimation`, so player-build pose extraction
  keeps working. Only validation is editor-only.

The rig a clip is sampled against is **derived rather than stored**: it is the skeleton root's
topmost ancestor. An importer writes curve paths relative to the asset's main object, and a second
serialized field could disagree with the skeleton where a derived value cannot. Because the skeleton
may be rooted part-way down the rig, `Bake` records the sibling-index path from rig to skeleton root
and replays it on a temporary clone.

That clone is flagged `HideAndDontSave` and destroyed in a `finally`. The cleanup is worth noting:
`instance` is a `Transform`, and Unity refuses to destroy a Transform component — so getting it wrong
leaks the whole rig.

A non-unit local scale produces a one-time warning: baked FK is rigid and will be incorrect.

## Lazy baking and its cache key

`SkeletonAnimation` bakes on first access to `PoseSequence`, not at import.

Validation is memoized, and the key is more than it looks:

```csharp
HashCode.Combine(clip, skeleton.ContentHash, Skeleton.CollectTransformsDfs(skeleton.Root).Count)
```

Reading curve bindings walks every curve in the clip — **730 of them for a walk cycle** — and
inspectors reach validation on every repaint, once per clip in a config's list. So it has to be
cached.

The two non-obvious parts:

- Keying on the clip *and* the bone tree, rather than relying on `ClearRuntimeCaches`, because a rig
  reimport does not trigger that callback.
- Including the **live bone count**, because `Skeleton.ContentHash` is derived from the skeleton's own
  cached bone list and therefore cannot move in the one case the check most needs to catch: a rig
  whose bones changed underneath a skeleton that has not noticed yet.

`AnnotatedAnimationClip` shadows `FrameCount` and `GetFrame` with its slice window, clamped
defensively because a serialized `endFrame` can exceed the clip's frame count until `OnValidate`
re-runs.

## Creating an annotated clip

`Assets/Create/MotionMatching/Annotated Clip From Selection` is the one place the root bone is
**guessed**. The asset stores the result, so a rig the heuristic cannot read has to fail here rather
than later with an empty skeleton.

The heuristic: first descendant whose name ends with `Hips` (case-insensitive), else follow a
single-child chain down from the rig's first child. Failing both, a dialog explains what it looked
for.

The menu also warns when the source model uses keyframe reduction, since setting Anim. Compression to
Off improves bake fidelity.

## BVH import

`BvhImporter` is a `ScriptedImporter` producing a rig GameObject as the main object and an
`AnimationClip` as a sub-asset — the same shape as a small FBX.

The details that matter:

- **The bone hierarchy sits under a container GameObject** because Unity renames an imported asset's
  main object to the file name, which would clobber the root bone's name and break name-based
  skeleton binding. The bones come along as children of that container; adding one separately would
  register a second root for the same asset.
- **The HIERARCHY root `OFFSET` is discarded.** Root position always comes from motion channel 0, so
  the root bone gets an identity rest transform.
- **Handedness.** BVH is right-handed and Unity is left-handed, so Z is negated on import.
- **Channel tolerance.** The root must have exactly six channels; non-root joints may have three or
  six, and when six the leading XYZ-position triple is safely ignored. Per-channel axis order is
  captured explicitly.
- `EnsureQuaternionContinuity` runs after the curves are built, to remove double-cover
  discontinuities between keys.
- Default unit scale is `0.01` — centimetres to metres.

### There is a second, incompatible BVH converter

`Assets/Scripts/BioVisionHierarchyToAnimClip.cs` is an unrelated BVH → `.anim` converter reached from
`Assets/Convert BVH to AnimationClip`. It is **not** a ScriptedImporter, produces only a standalone
clip with no rig object, and — critically — **uses a different handedness convention**: it flips X for
position and mirrors rotation, where `BvhImporter` negates Z.

The two paths are not interchangeable. Treat this one as legacy. It also sits outside the
`AnimationTools` assembly while declaring `namespace AnimationTools`, and is a `MonoBehaviour` with an
unguarded `using UnityEditor`.

`BvhVisualiser` (in the MotionMatching assembly) is a gizmo consumer of `SkeletonAnimation` that
rebuilds a bone hierarchy and drives it from baked frames — a direct demonstration of the
"only root translation is baked" contract, since it sets bone 0's local position and every bone's
local rotation and nothing else. It skips gizmos for bones named `End Site`, a name produced by the
*other* converter and never by `BvhImporter`; that branch is likely vestigial.

## Source map

| Concern | File |
| --- | --- |
| Validation and baking | `Assets/AnimationTools/Runtime/Animation/AnimationClipBaker.cs` |
| Clip + skeleton pairing, lazy bake | `Assets/AnimationTools/Runtime/Animation/SkeletonAnimation.cs` |
| Slice window and tags | `Assets/AnimationTools/Runtime/Animation/AnnotatedAnimationClip.cs` |
| BVH import | `Assets/AnimationTools/Editor/Importers/BvhImporter.cs` |
| Asset creation | `Assets/AnimationTools/Editor/CreateAnnotatedClipMenu.cs` |
| Legacy BVH converter | `Assets/Scripts/BioVisionHierarchyToAnimClip.cs` |

**Tests.** `AnimationClipBakerTests` covers acceptance on full and partial coverage, rejection of a
foreign rig, both halves of the rig-relative path convention, the both-counts diagnostic, and the
"nothing to check" case.

Downstream, baked frames feed [the pose database](pose-database.md).
