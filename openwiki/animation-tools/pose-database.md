---
type: Architecture Guide
title: The pose database
description: How clips become a searchable set of poses, and why the extraction contract was narrowed to an interface that a motion field can satisfy.
tags: [pose-set, extraction, contacts, interface]
sources:
  - id: openwiki-source-00b42bf2c141d2bea6637e2a
    resource: repo://Assets/AnimationTools/Runtime/IPoseSetSource.cs
  - id: openwiki-source-3448451e765fedad3f169713
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseExtractor.cs
  - id: openwiki-source-9cb39479f08cf519691c343e
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseSet.cs
  - id: openwiki-source-eedea5a32a3cded8ecfd1876
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseSetImporter.cs
  - id: openwiki-source-45348d10be6243abe134ddac
    resource: repo://Assets/MotionField/MotionFieldConfig.cs
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# The pose database

A `PoseSet` is the in-memory pose database: flat `PoseBuffer` frames in one `PoseSequence` over a
skeleton that is exactly the clips' own skeleton. `PoseExtractor` fills it from clips.
`IPoseSetSource` is the contract an asset must satisfy to own one.

No file I/O happens here — that is [on-disk formats](on-disk-formats.md) — and no feature vectors,
which are motion-matching only.

## Why IPoseSetSource exists

`PoseSet`, `PoseExtractor` and `PoseSerializer` used to take a `MotionMatchingData` directly. That
meant **anything wanting a pose database also had to be a full Motion Matching asset** — trajectory
features, pose features, skeleton map and all — even though only a handful of those members ever
reach the pose path.

The interface is exactly that handful:

```csharp
string  name { get; }                      // doubles as folder and file base name
Skeleton Skeleton { get; }
List<AnnotatedAnimationClip> AnimationClips { get; }
float   ContactVelocityThreshold { get; }
string  LeftContactBoneName { get; }
string  RightContactBoneName { get; }
string  GetAssetPath();
```

It lives in the **AnimationTools** assembly rather than in MotionMatching because it is generic
pose-database infrastructure, not specific to any one synthesis technique. That placement is what
lets `MotionFieldConfig` produce its own `.mmpose` without being a `MotionMatchingData` — and it
deliberately carries none of the feature machinery, because a motion field searches on full-body pose
and velocity, so a Motion Matching feature set has no meaning for it.

Two implementers: `MotionMatchingData` and `MotionFieldConfig`.

Note that `MaximumFramesPrediction` is **not** on the interface. It lives on the implementers, and
`MotionFieldConfig` returns 0 — no trajectory features, so every pose is usable, where a Motion
Matching database must discard the tail frames a lookahead query would read past.

## What a stored pose is

A pose is stored **exactly as the clip bakes it**: bone 0's world position and rotation, rest offsets
and parent-local rotations below it. Nothing is prepended and nothing is reparented, because the
character frame is [derived from the stored pose on demand](simulation-frame.md) rather than baked
into storage.

The pose skeleton and each clip's skeleton are the same bone tree. A mismatch therefore means the
clip belongs to a different rig, and `PoseSetImporter.Import` **skips that clip** rather than failing
the whole extraction.

## Extraction

```
IPoseSetSource
  └─ PoseSetImporter.TryValidate     skeleton set? ≥1 clip? each clip valid and structurally equal?
  └─ PoseSetImporter.Import
       └─ PoseSet.SetSkeleton        builds the layout, yields the two contact handles
       └─ per clip: PoseExtractor.Extract
            ├─ resolve contact bones
            ├─ pose pass
            ├─ velocity pass  (forward differences)
            ├─ contact pass   (needs velocities already present)
            └─ SmoothContacts (median filter)
```

**One frame short, deliberately.** Extraction registers `nFrames - 1` poses. The final clip frame is
materialised only into an `Allocator.Temp` pose, so that the last *stored* frame can be given a
forward-difference velocity. Without it the last frame of every clip would have no rate. The Python
side reconstructs the same extra frame for the same reason — see
[the pose data bridge](../python/pose-data-bridge.md).

This off-by-one is real behaviour but is not argued anywhere in the source.

## Foot contacts

Contact is detected by thresholding the contact bone's **character-space** velocity against
`ContactVelocityThreshold`, with no ground-distance test at all — there is a standing TODO to
consider one. The result is then median-filtered with a window radius of 6 frames (a 13-frame window)
to remove short spurious contact and release regions.

Bone resolution runs a three-step fallback:

1. An explicitly configured bone name, if set.
2. `BoneNameConventions.TryFindContactBone` — toes, then feet.
3. **Bone index 0**, a compatibility fallback preserving the legacy behaviour of defaulting to the
   root joint on failure.

Step 3 is worth flagging: it will silently measure contact on the hips rather than a foot.

## Memory and lifetime

Pose storage uses `Allocator.Domain` and is never disposed — an outgrown buffer is simply abandoned
to the domain after its contents are copied over. `PoseSet.Dispose` therefore releases **only the tag
arrays**, because pose sets are shared between consumers.

Frames obtained from a set are **views aliasing its storage**. Never dispose one.

Validation never logs, because inspectors call it every repaint; extraction logs instead, and returns
a null pose set on failure so the caller reports through the inspector rather than the console.

All clips in one set must share a frame rate, asserted on both the extraction and deserialization
paths.

## Two things that are not what the docs say

**`PoseSet.EndClip` does not exist.** Two doc comments direct you to it — "write new ones through
`BeginClip`/`EndClip`", and "Finish with `EndClip` so tag ranges resolve" — but there is no such
method anywhere in the codebase. The real write paths are `BeginClip` and `AppendRawFrames`, the
latter used by deserialization because it appends frames with no clip or tag bookkeeping.

**Tags are never populated during extraction.** `PoseSet.AddTag` is private with no callers, and
`PoseExtractor` never reads `AnnotatedAnimationClip.Tag`. So a freshly extracted set always reports
zero tags; the only live tag path is `AddTagDeserialized`, meaning tags can be read back from a
`.mmpose` but nothing currently writes them from clip annotations.

That connects to a second dead end downstream: the motion-matching search's tag mask exists, is
allocated all-true, and is never written. See
[search backends](../motion-matching/search-backends.md).

## A naming trap

`PoseSet.AnimationClip` is a nested struct of `{ Start, End, FrameTime }` — it shadows
`UnityEngine.AnimationClip`, and `PoseSet`'s internal clip list holds *the struct*, not the Unity
type. Easy to misdocument and easy to misread.

## Source map

| Concern | File |
| --- | --- |
| The extraction contract | `Assets/AnimationTools/Runtime/IPoseSetSource.cs` |
| Storage, clips, tags, layout | `Assets/AnimationTools/Runtime/Pose/PoseSet.cs` |
| Clip → poses, velocities, contacts | `Assets/AnimationTools/Runtime/Pose/PoseExtractor.cs` |
| Validation and import orchestration | `Assets/AnimationTools/Runtime/Pose/PoseSetImporter.cs` |

**Tests.** There is no dedicated `PoseSetTests` or `PoseExtractorTests`. `PoseSet` is exercised
indirectly by `PoseSerializerTests`, which builds one through `SetSkeleton`/`BeginClip` and asserts
every channel round-trips, and by `MmTestData`'s demo-database loader.

The serialized form, and the second consumer that reads it, are covered in
[on-disk formats](on-disk-formats.md).
