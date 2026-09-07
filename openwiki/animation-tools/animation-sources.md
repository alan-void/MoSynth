---
type: Architecture Guide
title: Animation sources and clip baking
description: How clips and BVH files become poses, why a clip is accepted on one animated bone, and what baking does and does not capture.
tags: [clips, baking, bvh, import]
sources:
  - id: openwiki-source-22146543a8998d684667efbc
    resource: repo://Assets/AnimationTools/Editor/AnnotatedClipFactory.cs
  - id: openwiki-source-610664f4e17fffc5dfb7d14c
    resource: repo://Assets/AnimationTools/Editor/BatchAnnotatedClipWindow.cs
  - id: openwiki-source-3de15f937f7e1dfc53c7ec56
    resource: repo://Assets/AnimationTools/Editor/CreateAnnotatedClipMenu.cs
  - id: openwiki-source-eaff015da326d13eeef3a663
    resource: repo://Assets/AnimationTools/Editor/Importers/BvhImporter.cs
  - id: openwiki-source-2e0d28f6a1ee99e6e464a576
    resource: repo://Assets/AnimationTools/Runtime/Animation/AnimationClipBaker.cs
  - id: openwiki-source-d9244dc06fc73f4e2315c8ed
    resource: repo://Assets/AnimationTools/Runtime/Animation/AnimationClipComponent.cs
  - id: openwiki-source-81d1a220d914a0173e6778bb
    resource: repo://Assets/AnimationTools/Runtime/Animation/AnnotatedAnimationClip.cs
  - id: openwiki-source-0b35dfb45ac00c6c6916b1e0
    resource: repo://Assets/AnimationTools/Runtime/Animation/GaitPhase.cs
  - id: openwiki-source-1ac38bf24a91b1612cc25f91
    resource: repo://Assets/AnimationTools/Runtime/Animation/GaitPhaseComponent.cs
  - id: openwiki-source-e1ad0ab569ae74b451c3418f
    resource: repo://Assets/AnimationTools/Runtime/Animation/SkeletonAnimation.cs
  - id: openwiki-source-3448451e765fedad3f169713
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseExtractor.cs
  - id: openwiki-source-9cb39479f08cf519691c343e
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseSet.cs
  - id: openwiki-source-3d5c835733aa13f4eeed0f83
    resource: repo://Assets/AnimationTools/Tests/Editor/AnimationClipBakerTests.cs
  - id: openwiki-source-593d3333db07e2608e4170c1
    resource: repo://Assets/AnimationTools/Tests/Editor/GaitPhaseTests.cs
  - id: openwiki-source-5cd85933c91658849ca6377d
    resource: repo://Assets/Scripts/Editor/BioVisionHierarchyToAnimClip.cs
generated: {by: "claude-code", at: "2026-09-02T22:53:10.494Z"}
---

# Animation sources and clip baking

This is where animation data enters the project: an `AnimationClip` plus the `Skeleton` it is sampled
against, baked into the flat float pose format. The layer stops at `PoseSequence` — it knows nothing
about databases, features or serialization.

> Clips now normally arrive already retargeted onto one shared rig, rather than each BVH bringing its
> own skeleton. See [the retargeting pipeline](retargeting-pipeline.md) for how a BVH becomes an FBX
> take before it reaches any of this; `BvhImporter` below remains the direct, single-skeleton route.

| Type | Role |
| --- | --- |
| `SkeletonAnimation` | pairs a clip with its skeleton; bakes lazily into a `PoseSequence` |
| `AnnotatedAnimationClip` | adds a `[startFrame, endFrame)` slice and a list of clip components |
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

`Assets/Create/MoSynth/Annotated Clip From Selection` is the one place the root bone is
**guessed**. The asset stores the result, so a rig the heuristic cannot read has to fail here rather
than later with an empty skeleton.

The heuristic: first descendant whose name ends with `Hips` (case-insensitive), else follow a
single-child chain down from the rig's first child. Failing both, a dialog explains what it looked
for.

The menu also warns when the source model uses keyframe reduction, since setting Anim. Compression to
Off improves bake fidelity.

### A whole model at a time

That menu takes one clip per invocation and writes beside the source model, which is the wrong
shape for a retargeted capture session: those arrive as hundreds of takes inside a single FBX, under
a source tree that is not where the assets belong. `MoSynth/Animation/Create Annotated Clips From
Model...` takes a model, a case-insensitive name filter and an output folder, and writes one asset
per matching take. Preview lists what matched and its total frame count before anything is written.

Both routes share `AnnotatedClipFactory`, so the root-bone guess and the humanoid refusal are the
same code and cannot drift apart.

Two properties make a batch safe to re-run over a folder it already filled:

- **Each take maps to a fixed path, and the asset there is updated rather than replaced.** Configs
  reference their clips by GUID; delete-and-recreate would mint new ones and silently empty every
  list pointing at them.
- **A `GaitPhaseComponent` is added only to clips that have none.** Detection overwrites the
  footfall list, so re-running must not touch a clip whose anchors someone has corrected. To
  re-detect deliberately, use the Detect Footfalls button in [the clip editor](clip-editor.md).

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

`Assets/Scripts/Editor/BioVisionHierarchyToAnimClip.cs` is an unrelated BVH → `.anim` converter
reached from `Assets/Convert BVH to AnimationClip`. It is **not** a ScriptedImporter, produces only a
standalone clip with no rig object, and — critically — **uses a different handedness convention**: it
flips X for position and mirrors rotation, where `BvhImporter` negates Z.

The two paths are not interchangeable. Treat this one as legacy. It sits outside the `AnimationTools`
assembly while declaring `namespace AnimationTools`; it used to be a `MonoBehaviour` in the runtime
assembly with an unguarded `using UnityEditor`, which is why it now lives in an `Editor` folder as a
static class.

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
| Slice window and components | `Assets/AnimationTools/Runtime/Animation/AnnotatedAnimationClip.cs` |
| The component base | `Assets/AnimationTools/Runtime/Animation/AnimationClipComponent.cs` |
| Gait phase | `Assets/AnimationTools/Runtime/Animation/GaitPhase.cs`, `GaitPhaseComponent.cs` |
| BVH import | `Assets/AnimationTools/Editor/Importers/BvhImporter.cs` |
| Asset creation | `Assets/AnimationTools/Editor/CreateAnnotatedClipMenu.cs` |
| Legacy BVH converter | `Assets/Scripts/Editor/BioVisionHierarchyToAnimClip.cs` |

**Tests.** `AnimationClipBakerTests` covers acceptance on full and partial coverage, rejection of a
foreign rig, both halves of the rig-relative path convention, the both-counts diagnostic, and the
"nothing to check" case.

## Clip components

A clip carries a `[SerializeReference]` list of **components** — annotation about the clip that the
clip's own curves cannot supply. This is the same seam `MoSynthStage` and `BenchmarkOverride` use, so
a component is a plain `[Serializable]` class with a parameterless constructor, it gets a type
dropdown in the inspector for free, and subclasses may live in any assembly that references
`AnimationTools`.

The list replaces a mechanism that never worked. `AnnotatedAnimationClip` declared a `Tag` struct but
had no field of that type, `PoseSet.AddTag` was private with no callers, and the `.mmpose` tag block
was therefore always empty — the format could *read* tags that nothing could ever *write*. Tags are
the natural second component; the dead code is left in place until one exists.

One hazard worth knowing before adding a component type: a managed reference is keyed in YAML by its
**(class, namespace, assembly)** triple. Renaming the type, moving its namespace, or moving its file
into another assembly orphans every authored instance and drops the data on the next save.
`[FormerlySerializedAs]` renames fields, not types; `[MovedFrom]` is the right tool and is not yet
used anywhere in this repository.

### Gait phase

The first component. It stores a clip's **footfalls** — which frame each foot was planted on — and
the settings used to find them. [Gait phase](neural-synthesis.md) is what a phase-functioned network
is organised around, and it was once reconstructed only at training time, in Python, from contacts
baked with one global threshold. Nothing could see it and nothing could correct it.

Anchors are stored rather than a baked phase curve: a few hundred markers say the same thing as
thousands of floats, they are the direct input to the phase rule, and they are what a person would
correct.

`GaitPhase` turns anchors into phase — half a cycle per alternating footfall, a whole cycle when the
same foot falls twice running (which is what a missed contact looks like), linear between, zeroed on
a right footfall — and the pose-database bake writes the result into the `.mmpose`. **This is now
the only implementation of the rule.** It used to be mirrored in Python, and the two had drifted:
the Python side extrapolated the neighbouring walking rate outward past the first and last anchor
"so a clip has no flat ends", which invented gait — on the untrimmed `walk1_subject1` clip the first
real footfall is at frame 132, and the 4.4 s of standing before it was given 2.87 complete cycles
the character never walked. That is what a model trained to cycle its legs while standing was
learning from.

A stretch with no anchors is now answered by how fast the character was travelling, since standing
and a missed contact leave the same hole. Standing sweeps the phase at a fixed period so a model can
learn a stationary pose across the whole cycle; moving holds the phase at a rate of zero, which
drops the frame.

Detection settings live on the clip rather than on a database config, because one threshold cannot
serve two clips at different speeds: `walk1_subject5` travels at 1.27 m/s against
`walk1_subject1`'s 0.675 m/s. The bake reads those same settings, and both the timeline and the
database measure contacts through one routine, `GaitMeasure` — so what an author corrects a clip
against is what the database records. That was not true before: the bake composed per-bone velocity
channels while the editor differenced character-space positions, and the two disagreed badly, the
bake reporting a toe planted 0.22/0.17 of the time on `walk1_subject5` where the editor reported
0.40/0.36. Differencing is the more direct measurement of whether a toe moved, so it is the one that
survived.

Neither defect is inherent to the detector, and the Bandai-Namco `walk_normal` set is the
counter-example worth knowing about. Its 24 takes walk continuously from the first frame — 0.5% of
frames are stationary, so there is no standing intro to invent gait across — and they sit in a tight
speed band of 1.05–1.25 m/s, which is the case where one global threshold genuinely is enough. At
the same 0.15 m/s, detection finds 372 anchors over 6186 frames with **no** same-foot-twice span at
all, on a half-cycle of 17 frames. When a clip set reports nothing like that, suspect the import
before the threshold — a rig whose FK has collapsed reports zero anchors, not bad ones. See
[the retargeting pipeline](retargeting-pipeline.md).

**The clip-side and bake-side contact measurements do not agree, and the gap is unexplained.** A
clip's baked poses carry no velocity channels, so the component differences consecutive
character-space toe positions, where `PoseExtractor.ExtractPoseContacts` composes the per-bone
velocity channels. At the same 0.15 m/s threshold the component reports a foot planted far more
often — duty 0.40/0.36 against 0.22/0.17 on `walk1_subject5`, and 0.57/0.56 against 0.51/0.49 on
`walk1_subject1` — and finds half as many missed contacts. Differencing the composed position is the
more direct measure of whether a toe moved, so this is a reason to distrust the bake-time contacts;
why the composition inflates the speed has not been established.

Anchors are authored and corrected in [the annotated clip editor](clip-editor.md), which draws the
phase, the contacts it was read from, and the anchors themselves against a preview of the motion.

Downstream, baked frames feed [the pose database](pose-database.md). The footfalls do **not** reach
it yet: the database still derives its own contacts, and Python still reconstructs phase from them.
Feeding the anchors through is the step that would change what a network trains on.
