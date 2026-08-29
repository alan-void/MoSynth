---
type: Architecture Guide
title: Skeletons and rig binding
description: Bone identity and ordering for the whole project, why a skeleton must point at an asset rig, and how an asset skeleton binds to a live scene rig.
tags: [skeleton, bones, rig, binding, editor]
sources:
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-9e699dfd183a370e93283673
    resource: repo://Assets/AnimationTools/Editor/SkeletonDrawer.cs
  - id: openwiki-source-e1ad0ab569ae74b451c3418f
    resource: repo://Assets/AnimationTools/Runtime/Animation/SkeletonAnimation.cs
  - id: openwiki-source-08ea4bc02364c8785bdd8d8f
    resource: repo://Assets/AnimationTools/Runtime/Core/MotionSynthesisComponent.cs
  - id: openwiki-source-88c3b2718587d20ac4831d81
    resource: repo://Assets/AnimationTools/Runtime/Skeleton/BoneNameConventions.cs
  - id: openwiki-source-54f745ff7f6016dea0150c17
    resource: repo://Assets/AnimationTools/Runtime/Skeleton/Skeleton.cs
  - id: openwiki-source-98b90f0fe54f83ab272ff785
    resource: repo://Assets/AnimationTools/Runtime/Skeleton/SkeletonBone.cs
  - id: openwiki-source-be8fb964478d0fdd009e441a
    resource: repo://Assets/AnimationTools/Runtime/Skeleton/SkeletonBoneOverrides.cs
  - id: openwiki-source-ee3e422427edb1b8213450f8
    resource: repo://Assets/AnimationTools/Runtime/Skeleton/SkeletonData.cs
  - id: openwiki-source-c86b2c47fdb0f198af8469af
    resource: repo://Assets/AnimationTools/Tests/Editor/SkeletonTests.cs
  - id: openwiki-source-45348d10be6243abe134ddac
    resource: repo://Assets/MotionField/MotionFieldConfig.cs
  - id: openwiki-source-68bd8cb3d0139ef5eb77babd
    resource: repo://Assets/MotionMatching/Runtime/Unity/MotionMatchingData.cs
  - id: openwiki-source-de61724f7be1d1531c682f36
    resource: repo://Assets/Scripts/BoneUIElements.cs
  - id: openwiki-source-057bf19a9464594dde1397c6
    resource: repo://Assets/Scripts/SkeletonUIManager.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# Skeletons and rig binding

`Skeleton` is not a data structure that stores bones. It is **a single serialized `Transform`
reference plus a derived, cached preorder depth-first walk of that Transform's subtree**.

That walk is the project-wide bone order. Pose layouts, FK, feature extraction and the `.mmpose`
format all index against it.

The area splits cleanly in two:

- `Skeleton`, `SkeletonBone`, `SkeletonData` describe an **asset rig at its rest pose** — read-only,
  authoritative for structure and rest offsets.
- `SkeletonBoneOverrides` binds that description to a **live scene rig** that actually gets posed.

## The invariants

- Bone 0 is always the root; its parent index is −1.
- `ParentIndices[i] < i` for every bone. `Build` throws if a bone at index > 0 has no parent inside
  the walk.
- Bone **ids** are the index plus one, so **0 unambiguously means unset**. `IndexOfId` returns −1 for
  id 0 or any id past the end.

The parent-precedes-child invariant is not decoration: it is what lets FK run as a single forward
pass instead of walking each bone's parent chain.

## A skeleton must point at an asset rig

The rest pose is read live off the Transforms. So a `Skeleton` built over a **scene** rig that
something animates reports whatever pose it is currently animated to *as its rest pose* — which
silently corrupts FK.

Nothing downstream can detect this. The numbers are all well-formed; they are simply the wrong rest
pose, and every FK result shifts. There is no error, no warning, no visible failure mode.

The pipeline keeps the two apart deliberately: `MotionSynthesisComponent.skeleton` comes from an
asset, and the scene rig it drives is bound to that skeleton through `characterRig`.

**The guard is thinner than it looks.** The only automatic check anywhere is
`SkeletonAnimation.TryValidate`'s `IsPersistent` test, and it is inside `#if UNITY_EDITOR`. There is
no runtime check at all, and `MotionSynthesisComponent`'s own skeleton field has no equivalent. The
practical defence is the drawer refusing scene drops — which a script assignment or a prefab-variant
edit bypasses entirely.

## Nothing but bones under the root

Everything under `Root` becomes a bone. A mesh node, an attachment point or an IK helper parented
beneath the skeleton root **shifts every bone index after it**, silently invalidating features,
contacts and any serialized database.

This is unenforced. It is a rig-authoring constraint, and the failure is quiet.

## Caching is per root, not per instance

`SkeletonBone` stores its `Skeleton` **by value**, not by reference. A skeleton is only a root
Transform, so a bone costs two object references.

The consequence of Unity serialization is that an asset with several bone-reference fields ends up
holding several distinct `Skeleton` instances over one rig. So the derived bone walk is cached in a
static map **keyed by root Transform**, not per instance — which means those instances share one
walk, one parent-index array, and one `Allocator.Domain` `SkeletonData`.

Per-instance caching would pay N walks, N parent arrays and N Domain allocations that are never
freed. `SkeletonTests` pins the sharing with a `ReferenceEquals`-style assertion on the native array.

`SkeletonData`'s arrays are `Allocator.Domain`: they live for the domain's lifetime, are cleared
automatically on reload, and **callers must never dispose them**.

## Invalidation is manual

Neither the cached bone list nor `ContentHash` notices a rig changing underneath a live skeleton —
`ContentHash` is derived from the cached list, so it cannot move in the one case that most needs
catching. And `PoseLayout`'s cache is keyed on that hash.

So after a reimport or a bone-tree edit, `Skeleton.Invalidate()` (or `InvalidateAll()`) has to be
called explicitly. Test fixtures do it in teardown, which is the only place the static cache's
lifetime hazard is visible.

## Comparison

| Method | Compares | Ignores |
| --- | --- | --- |
| `ContentHash` | each bone's name and parent index | rest pose |
| `StructurallyEqual` | bone count, every name, every parent index | rest pose |
| `MatchesFrom(startIndex, other)` | whether `other` is *embedded* in this skeleton from that bone | rest pose |

`StructurallyEqual` deliberately ignores the rest pose, so two rigs with identical structure but
different bone lengths compare equal. `ContentHash` is a hint — equal hashes mean *very likely*
identical, and `StructurallyEqual` is what confirms.

`MatchesFrom` shifts parent indices by the offset, so a rig nested under extra ancestors — an
`armature` node, say — can be compared against the bare hierarchy. `MatchesFrom(0)` reduces to plain
structural equality.

## Binding to a live rig

`SkeletonBoneOverrides.Bind` resolves each skeleton bone against the live hierarchy under `Root`, by
**name**, with per-bone overrides as the escape hatch.

Name-based binding looks like a violation of the project's own rule that bone identity is a Transform
reference rather than a string. It is not: the asset rig and the scene rig are **different object
graphs sharing no Transforms at all**. A name is the only identity that survives the crossing. The
same reasoning governs `SkeletonBone.ResolveIndex`, which tries Transform identity first and falls
back to an exact name match — for a contact bone picked on one rig and resolved against another that
is structurally the same.

Failure semantics are split deliberately by layer:

- **`SkeletonBoneOverrides`** logs an error, leaves the slot null, and carries on — the caller decides
  how severe that is.
- **`MotionSynthesisComponent`** counts the nulls and disables itself.
- **Duplicate names** under the root only produce a *warning*, because the first depth-first match
  silently wins. This is reported, not solved.

## The name collision with Unity

`AnimationTools.SkeletonBone` collides with the built-in `UnityEngine.SkeletonBone`.

The codebase copes by **aliasing rather than renaming**: nine files outside the `AnimationTools*`
namespaces carry

```csharp
using SkeletonBone = AnimationTools.SkeletonBone;
```

Any new file outside those namespaces that names the type needs the same line.

## Bone-name heuristics are a fallback

`BoneNameConventions` recognises the shipped BVH naming convention — `Hips`, `LeftUpLeg`, `LeftFoot`,
`LeftToe`, `LeftShoulder`, `LeftArm`, `LeftForeArm`, `LeftHand` and so on.

These are best-effort fallbacks for rigs that follow that convention. They are explicitly **not a
substitute for an explicit `SkeletonBone` reference** where one is available. Contact search prefers
toes and falls back to feet, case-insensitively; `ForeArm` matching via the `Arm` token is
intentional.

The one place the heuristic is *mandatory* rather than a fallback is contact-bone selection in
[pose layouts](pose-buffers.md), where the choice has to be a pure function of bone names on both
sides of the database boundary.

## Editor tooling, and the bugs it encodes

The bone pickers exist because of a specific Unity limitation, and their implementations encode
several hard-won details.

**Why there is a dropdown at all.** An imported model exposes only its *top-level children* as
sub-assets — for a typical FBX, the mesh node and the topmost bone. Any bone deeper than that is
reachable from neither the Project window nor the object picker, so a plain object field could not
express "this skeleton starts at `Model:Hips`".

**Why drag-and-drop is handled by hand.** `EditorGUI.ObjectField` normalises a dragged sub-asset up
to its model's main asset — which is why dropping the exposed root node used to assign the whole FBX.
The drawer intercepts `DragUpdated`/`DragPerform` *before* the field draws and reads
`DragAndDrop.objectReferences` to get what the user actually dragged.

**Why the skeleton dropdown has no "(none)".** Being unset means something for a `SkeletonBone` — an
unset root-motion bone falls back to the skeleton root, unset contact bones fall back to the name
conventions. A skeleton with no root is simply unusable, so the entry is left out.

**The PPtr trap.** Reading a PPtr off a nested `Skeleton` (a Generic property) does not throw — it
makes Unity log "type is not a supported pptr value" from native code, once per GUI event, forever.
The bone drawer's type check exists to avoid that.

**Missing is not null.** A serialized reference whose target no longer resolves is *missing*, which
Unity deliberately reports as non-null so that touching it throws a diagnostic. Inspectors reach
these properties every repaint, so the drawers probe inside `try`/`catch (MissingReferenceException)`
rather than null-checking.

**The preview cache is keyed on content, not identity.** Root identity alone misses a rig whose bones
changed underneath it, which used to leave the editor's cache and the asset's baked `PoseSequence`
describing two different skeletons — with the FK pass indexing one by the other's bone count.

`SkeletonRigBuilder` materialises a skeleton as plain GameObjects at rest pose. It is deliberately
runtime-only, using no `UnityEditor` APIs, so a skeleton can be instantiated at build time with no
imported rig asset behind it. `RigFromMmDataMenu` is a thin migration wrapper over it, for getting a
rig to assign to feature and contact fields when no asset provides one.

## Two things in this area that are not part of it

**`MotionMatchingDocumentation.cs`** is 13 lines that open
`https://jlpm22.github.io/motionmatching-docs/` in a browser. It contains no reference to skeletons or
bones; it is the only in-editor docs entry point and it points off-project.

**The `Assets/Scripts` skeleton-UI trio** — `SkeletonUIManager`, `BoneSelectorWindow`,
`SkeletonUIManagerEditor` — is an unrelated legacy system. It lives in the global namespace, models
bones as `{ id, displayName, Rect, Transform }` nodes on a 2D draggable canvas, and references
neither `AnimationTools.Skeleton` nor `AnimationTools.SkeletonBone`. `BoneUIElements.cs` is entirely
commented out. Do not read it as part of the rig-binding story.

## Source map

| Concern | File |
| --- | --- |
| Bone tree, ordering, caching, comparison | `Assets/AnimationTools/Runtime/Skeleton/Skeleton.cs` |
| Single bone reference | `Assets/AnimationTools/Runtime/Skeleton/SkeletonBone.cs` |
| Asset-to-scene binding | `Assets/AnimationTools/Runtime/Skeleton/SkeletonBoneOverrides.cs` |
| Burst mirror and FK | `Assets/AnimationTools/Runtime/Skeleton/SkeletonData.cs` |
| Name heuristics | `Assets/AnimationTools/Runtime/Skeleton/BoneNameConventions.cs` |
| Runtime rig construction | `Assets/AnimationTools/Runtime/Skeleton/SkeletonRigBuilder.cs` |
| Pickers and drawers | `Assets/AnimationTools/Editor/SkeletonDrawer.cs`, `SkeletonBoneDrawer.cs`, `BonePopup.cs` |

**Tests.** `SkeletonTests` covers 22 cases including the preorder order, the parent-precedes-child
invariant, the bone-id convention, per-root `SkeletonData` sharing, `StructurallyEqual` ignoring the
rest pose, the five `MatchesFrom` cases, and the cross-rig name fallback. Fixtures live in
`TestSkeletons`, which builds real GameObjects and must be torn down with `DestroyAll`.
