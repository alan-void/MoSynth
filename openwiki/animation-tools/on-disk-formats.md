---
type: Reference
title: On-disk formats
description: The four artefacts MoSynth ships to StreamingAssets, the byte convention that makes them readable from numpy, and how staleness is detected without a version header.
tags: [formats, serialization, streamingassets, python-interop]
sources:
  - id: openwiki-source-d9cd1d1e23d37a8aa1befd56
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseLayout.cs
  - id: openwiki-source-90731679b476c9b5e27f4d74
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseSerializer.cs
  - id: openwiki-source-eedea5a32a3cded8ecfd1876
    resource: repo://Assets/AnimationTools/Runtime/Pose/PoseSetImporter.cs
  - id: openwiki-source-e574d54dea1fdc6eed27d71c
    resource: repo://Assets/AnimationTools/Runtime/Recording/BuiltInRecorderChannels.cs
  - id: openwiki-source-5b29043ebc821d9f8710ab37
    resource: repo://Assets/AnimationTools/Runtime/Recording/RecordingManifest.cs
  - id: openwiki-source-ade074f70f19c4861912eb28
    resource: repo://Assets/AnimationTools/Runtime/Utils/BinarySerializerExtensions.cs
  - id: openwiki-source-85d42e7c9b2a85600801d398
    resource: repo://Assets/AnimationTools/Tests/Editor/PoseSerializerTests.cs
  - id: openwiki-source-45348d10be6243abe134ddac
    resource: repo://Assets/MotionField/MotionFieldConfig.cs
  - id: openwiki-source-52cb746a5d4a6e2a9f9396cd
    resource: repo://Assets/MotionMatching/Runtime/Features/FeatureSerializer.cs
  - id: openwiki-source-68bd8cb3d0139ef5eb77babd
    resource: repo://Assets/MotionMatching/Runtime/Unity/MotionMatchingData.cs
  - id: openwiki-source-3e08b9fa4adc34050a83a175
    resource: repo://Assets/MotionMatching/Tests/Editor/FeatureSerializerTests.cs
  - id: openwiki-source-93b271bb01f2503c507c34ed
    resource: repo://Python/binary_reading.py
  - id: openwiki-source-b7a0fd38fcd115e75814f58b
    resource: repo://Python/feature_set_importer.py
  - id: openwiki-source-1110a5319fdf997c0acf8f29
    resource: repo://Python/pose_set_importer.py
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-29T23:38:46.103Z
---

# On-disk formats

MoSynth writes four distinct artefacts into `StreamingAssets`, produced and consumed across three C#
assemblies plus CPython.

| Artefact | Location | Written by | Read by |
| --- | --- | --- | --- |
| `.mmpose` | `MMDatabases/<name>/`, `MotionFields/<name>/` | `PoseSerializer` | `PoseSetImporter`, `Python/pose_set_importer.py` |
| `.mmfeatures` | `MMDatabases/<name>/` | `FeatureSerializer` | `MotionMatchingData` via `FeatureSerializer.Deserialize` |
| `.mffield.npz` | `MotionFields/<name>/` | `Python/motion_field_io.py` | `MotionFieldStage` |
| `.mfembed.npz` | `MotionFields/<name>/` | `Python/motion_field_embedding.py` | `MotionFieldStage.LoadEmbedding` → visualizer |

Both `.npz` files ship with the player. The `.mfembed.npz` is **optional debug data** — safe to
delete, and the visualizer simply draws nothing without it.

Plus one format that is not in StreamingAssets: a [motion recording](motion-recording.md) is a raw
`.bin` next to a JSON manifest.

## The naming rule

Generated artefacts live in a folder per source asset:

```
StreamingAssets/<domain>/<asset-name>/<asset-name>.<ext>
```

The asset's name doubles as both the folder name and the file base name. Two domains exist:
`MMDatabases` for `MotionMatchingData`, and `MotionFields` for `MotionFieldConfig` — kept separate
deliberately, because the Motion Matching inspector owns and regenerates its own folder.

When no serialized `.mmpose` is found, the importer warns, extracts at runtime, and — in the Editor
only — writes the result back, so the fallback is paid for once.

## The byte convention

> Components in order, little-endian float32, **no length prefix and no padding**. Arrays are their
> elements back to back, which is why the read helpers must be told a length. Quaternions are
> **xyzw** throughout.

That plain layout is not incidental. It is precisely what lets the Python side read these files with
one `numpy.fromfile` and a reshape:

```python
np.frombuffer(f.read(n_joints * 12), dtype=np.float32).reshape(n_joints, 3)
```

The xyzw ordering matters at both ends: C# writes `value.x/y/z/w` and reads back into
`new quaternion(x, y, z, w)`; Python unpacks `'<4f'` and hands it to `scipy`'s `Rotation.from_quat`,
which uses the same convention. Much of the quaternion literature assumes wxyz, so this is a real
trap for anyone porting a formula in.

### The one exception

Strings break the no-length-prefix rule, because C# `BinaryWriter` prefixes them with a **LEB128
7-bit-encoded length**. The Python importer decodes it by hand:

```python
count = 0; shift = 0
while True:
    b = f.read(1)[0]
    count |= (b & 0x7F) << shift
    shift += 7
    if not (b & 0x80): break
```

This applies to bone names and tag names. It is imposed by `BinaryWriter`, not chosen by this
codebase.

## `.mmpose` layout

1. **Skeleton block** — `uint boneCount`, then per bone: name (LEB128 string), `uint parentIndex`,
   `float3` rest local position, `quaternion` rest local rotation.
2. `uint numberClips`, then per clip: `uint Start`, `uint End`, `float FrameTime`.
3. `uint numberPoses`, `uint boneCount` *(again)*, `uint numberTags`.
4. Per pose: positions, rotations, velocities, angular velocities (all `boneCount` long), then
   `uint` left contact and `uint` right contact. That is `boneCount * 13 * 4 + 8` bytes per pose.
5. Per tag: name, `uint numberRanges`, then range pairs.

Bone 0's parent index of −1 is written as an unsigned `0xFFFFFFFF` and read back as signed −1 by C#.
Python reads the same bytes directly as a signed int — the same value, spelled two ways.

### Why the skeleton block is in this file

C# does not need it: a config carries its own `Skeleton` and reads rest pose live off the Transforms.
**The Python half has no ScriptableObject to read**, and needs joint names for the bone-weight table,
parent indices for FK, and rest offsets for root-space positions.

Keeping it in the same file as the poses is what stops the two drifting apart — **which is exactly
what a separate `.mmskeleton` file used to allow.** That rejected alternative is named explicitly on
both sides of the boundary.

The simulation frame is *not* written. Both halves derive it from bone 0's rest rotation, which the
bone entries already carry. See [the simulation frame](simulation-frame.md).

## No version header, and what replaces it

**Nothing in any MoSynth binary format carries a magic number or a version.** Staleness is caught by
validating content instead — and for `.mmpose`, the skeleton block *is* what stands in for a version.

`ReadAndCheckSkeleton` compares the block against the live skeleton and rejects three ways, each
naming the offending bone and telling you to regenerate:

- differing bone count,
- differing bone name at an index,
- **differing parent index** — caught even when the names and the count both match.

That third case is the one a bone count alone would miss: the same bones, reparented. `PoseSerializerTests`
covers it under a name that says so, and the test file's own doc comment states the design outright.

The redundant second bone count in the pose header has a narrower job: the skeleton block has already
been reconciled against the live asset, so a disagreement between the file's two counts can only mean
a **truncated or corrupt write**. Truncation past that point — a header promising more poses than the
file holds — is caught by the pose reads themselves, which refuse the file and name the pose they ran
out inside.

One dead end worth knowing: `PoseLayout` declares `public const int PoseFormatVersion = 1`, and it has
**zero usages anywhere**. `PoseSerializer` writes no version. It is a leftover constant, not a format
version — do not read it as one.

## `.mmfeatures`

Kept separate from `.mmpose` on purpose: features are one way of *indexing* poses for search, and
changing them should not force re-extracting the poses.

1. `uint numberFeatureVectors`, `uint featureSize` (floats per vector), `uint numberTrajectoryFeatures`,
   `uint numberPoseFeatures`.
2. **Schema block** — per feature in vector order (trajectory first, then pose): name (LEB128 string),
   `uint floatsPerPrediction`, `uint predictionCount`. A pose feature is always `3 × 1`.
3. Per float of a vector: `float mean`, `float standardDeviation`.
4. Per vector: `uint valid`, then `featureSize` floats.

The trajectory/pose split is in the header so **the file describes itself**. Nothing outside Unity has
a `MotionMatchingData` to read the feature configuration from, and anything training on this database
has to know which floats are trajectory and which are pose.

### Its version substitute is the schema block

Like `.mmpose` it carries no version, and the schema block is what stands in for one.
`FeatureSerializer.Deserialize` reads it *before allocating anything off the header* and refuses the
file — with a message naming what disagreed — on a differing feature count, a differing vector width,
or a differing name, width or horizon count at any position. Anything thrown while reading is caught
and reported the same way, because a file written for another configuration gets its string lengths
and array sizes from the wrong offsets and dies long before it runs out of bytes.

`MotionMatchingData.GetOrImportFeatureSet` answers a refusal by re-extracting and rewriting, so a
stale file costs one extraction rather than a database of plausible numbers.

This replaces a `Debug.Assert`-only check, which was stripped in release players and left a
`.mmfeatures` from a different configuration silently misread in a build.

## The `.npz` pair

`.mffield.npz` holds the trained value function plus the three scalars the runtime cannot recover on
its own — the heading grid, `gamma`, and `k_neighbors` — because the policy has to reuse them for its
one-step lookahead to score actions the way training scored them.

It records **nothing about the run that produced it**, deliberately: whether a value function still
matches the config that will run it is decided in Unity by `MotionFieldConfig.hasTrained`, before the
file is opened. The loader takes whatever it is handed and returns `None` rather than raising, since
it runs inside `Py.GIL()` where an exception surfaces as an opaque managed error.

Since nothing records a format version there either, a file written by an older layout shows up as a
`KeyError` on the first missing key.

`.mfembed.npz` is the UMAP projection. Its only integrity check is a state count — see
[the pose manifold](../motion-field/pose-manifold-embedding.md) for the collision hazard that leaves.

For the full staleness contract and its two flags, see
[config and training](../motion-field/config-and-training.md).

## Recording manifests

A recording is a raw float32 `.bin` plus a sidecar JSON manifest that describes the data well enough
to load it with **no AnimationTools code at all**. The manifest documents the recipe itself:

```python
import json, numpy as np
m = json.load(open("recording.json"))
data = np.fromfile("recording.bin", dtype="<f4").reshape(-1, m["frameFloatCount"])
ch = {c["name"]: c for c in m["channels"]}
hips = data[:, ch["hips"]["floatOffset"] : ch["hips"]["floatOffset"] + 3]
```

Per-channel entries carry name, type, float offset and width, and the pose channel additionally
carries a flattened mirror of the `PoseLayout` including bone names — so a numpy reader can slice a
whole pose section without touching C#.

**This is the only MoSynth format carrying a `formatVersion`, and nothing reads it.**
`RecordingReader` validates `frameFloatCount` and tolerates a file shorter than the manifest claims
after a crash, but never inspects the version. Even the one versioned format is decorative in
practice — which is at least consistent with the project's stated "validate by content" stance.

The manifest's `dtype` and `endianness` fields are hardcoded initialisers rather than derived. Correct
on every shipped target, but not actually checked at write time.

## Source map

| Concern | File |
| --- | --- |
| Byte primitives | `Assets/AnimationTools/Runtime/Utils/BinarySerializerExtensions.cs` |
| `.mmpose` read/write, skeleton block | `Assets/AnimationTools/Runtime/Pose/PoseSerializer.cs` |
| Path rule, fallback import | `Assets/AnimationTools/Runtime/Pose/PoseSetImporter.cs` |
| Recording sidecar | `Assets/AnimationTools/Runtime/Recording/RecordingManifest.cs` |
| `.mmfeatures` | `Assets/MotionMatching/Runtime/Features/FeatureSerializer.cs` |
| Artefact paths | `Assets/MotionField/MotionFieldConfig.cs` |
| Python reader | `Python/pose_set_importer.py` |

**Tests.** `PoseSerializerTests` round-trips every channel with per-frame, per-bone distinct values so
a misaligned read cannot pass by coincidence, and covers all four rejection cases.
`FeatureSerializerTests` round-trips the demo database and covers the two refusals — a file written
for another feature configuration, and a truncated one. There are **no** tests for the recording round
trip, and none checking that the C# and Python `.mmpose` readers agree.
