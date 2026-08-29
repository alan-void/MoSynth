---
type: Architecture Guide
title: The pose data model
description: The packed array layout the motion field is defined over, how it is read from the exported database, and the algebra defined on it.
tags: [python, pose, numpy, conventions]
sources:
  - id: openwiki-source-18938fea788ba704e8dbb6d7
    resource: repo://Python/action_predictor.py
  - id: openwiki-source-52283f3a91111f36426b4ada
    resource: repo://Python/Animation.py
  - id: openwiki-source-556de75b4b36254c0d3e2158
    resource: repo://Python/MotionField.py
  - id: openwiki-source-1110a5319fdf997c0acf8f29
    resource: repo://Python/pose_set_importer.py
  - id: openwiki-source-4fa83394f41912d6e68884da
    resource: repo://Python/Pose.py
  - id: openwiki-source-1cb70134e6b0891ac831bc83
    resource: repo://Python/simulation_frame.py
  - id: openwiki-source-0d63de290d2cb6efbfbcddb8
    resource: repo://Python/Skeleton.py
  - id: openwiki-source-955a83f351a179da954c1a57
    resource: repo://Python/utils/quaternions.py
generated: {by: "claude-code", at: "2026-08-24T17:01:26.052Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-24T17:01:26.052Z
---

# The pose data model

Six modules turn the `.mmpose` byte stream Unity writes into the numpy arrays the motion field is
defined over, and define the algebra on those arrays.

| Module | Owns |
| --- | --- |
| `Pose.py` | the packed array convention; `Pose` (state) and `PoseDelta` (rate), with add, subtract, blend, stack |
| `Skeleton.py` | the joint tree, depth-first iteration, FK in root space |
| `Animation.py` | `PoseSet` — the whole database as flat per-frame arrays |
| `pose_set_importer.py` | the `.mmpose` reader, and derivation of the frame definition |
| `simulation_frame.py` | reconstructing the character frame per pose, and differencing it into rates |
| `utils/quaternions.py` | weighted quaternion blending |

Nothing here knows about the motion field, k-NN or torch. Everything above the importer consumes
`PoseSet`.

## The packed layout

```
row 0        [:3]  character frame position          (zero in database states)
row 1        [:3]  rig root bone position, in that frame
row 2        [ :]  character frame rotation, xyzw    (identity in database states)
row 3        [ :]  rig root bone rotation, within it
rows 4+      [ :]  remaining joints' parent-local rotations

shape (..., num_bones + 2, 4), float32
```

**This is a four-slot header, not two.** Some parameter docs describe it as "root, hips, quats",
which is true only in the sense that the quaternion block starts at row 2 and its first two entries
are the frame rotation and the root rotation. That phrasing is the single most confusing thing in the
package — trust the layout above.

`num_bones` counts the joints of the *virtual-root* skeleton, so joint 0 is the character frame,
joint 1 is the rig's bone 0, and joint *i* is rig bone *i−1*. See
[the Unity call surface](unity-call-surface.md) for why that extra joint exists.

Quaternions are **xyzw** throughout, matching what Unity writes and what `scipy`'s `Rotation` expects
— not the wxyz order much of the quaternion literature assumes.

### The `hips` naming

`Pose.py` calls slot 1 `hips` / `hipPos`. **Every consumer treats it as the rig's root bone**, not as
a Hips joint — the FK code says so in as many words.

Amusingly, in the shipped database bone 0 *is* named `Model:Hips` (85 bones total), so the legacy
name coincides with reality for this rig. It still is not what the code means, and it will not hold
for another rig.

## Rates are rotation vectors, not quaternions

`PoseDelta` uses the same slot map with 3-component payloads, and angular rates stay as **rotation
vectors**.

The reason is measured rather than stylistic. A unit quaternion can only carry a rotation below 2π,
and scipy canonicalises it further — so round-tripping a rad/s rate through a quaternion **silently
aliases every joint turning faster than π rad/s**. That is about **2.6% of this dataset**, of which
the majority come back pointing the opposite way.

Keeping rotation vectors also makes scaling exact, and makes blending correct: rotation vectors blend
linearly, which is the right operation for angular rates and needs no renormalisation.

The C# side reaches the same conclusion independently — see
[the rotation-rate rule](../animation-tools/channel-layout-system.md).

## The integration identity

```python
pose_x[i].add(PoseDelta.from_array(pose_v[i]).scaled(frame_time)) == pose_x[i + 1]
```

Root translation is root-local, so it is rotated by the frame quaternion before being applied. Joint
rotations are **left-multiplied** — `next = delta * current` — matching the C# angular-velocity
convention. Subtraction is the exact inverse, with result quaternions flipped onto `w ≥ 0` so the
short way round is taken.

## The frame is reconstructed, not read

Poses are stored exactly as the clips define them. The ground-projected, yaw-only frame the motion
field is defined in **is not stored** — it is reconstructed here.

The definition is **structural, not authored**: the reference bone is joint 0, and the forward axis is
the one that points along character forward in joint 0's rest rotation. Deriving it keeps one skeleton
for the clips, the database and the rig, and makes the frame exactly reproducible on the C# side,
which builds the same definition from the same bone entries.

Unity is y-up and left-handed, so a character faces +z. Yaw is `atan2(forward.x, forward.z)`. A
reference bone aimed straight up or down falls back to a fixed heading, matching Unity's own
`LookRotation` behaviour.

Expressing a pose in that frame **removes exactly the translation and yaw the motion field must not
see**. Applying it to an already frame-local sequence is the identity, because the reference bone's
own horizontal offset and yaw are zero by construction.

## Reading `.mmpose`

The importer decodes the format described in
[on-disk formats](../animation-tools/on-disk-formats.md) — including the LEB128 string prefix C#
`BinaryWriter` imposes, and the root's −1 parent written as `0xFFFFFFFF`.

**The skeleton block is read from inside the pose file** because the Python side has no
ScriptableObject to read the bone tree from: names key the per-bone weight table, parent indices
drive FK, and rest offsets give root-space positions. Keeping it in the same file as the poses is
what stops the two drifting apart — which a separate skeleton file previously allowed.

## Every frame has a rate, including the last of a clip

The extractor samples one frame beyond what it stores, so every stored pose has a velocity — even the
last of a clip.

The importer inverts that exactly: stepping the last pose by its own stored rates reconstructs the
frame past the end. That is what lets a derived rate exist for **every** stored frame rather than for
all but the last, and it keeps rates **aligned with the poses rather than lagging them** — rate *i*
still describes the step out of pose *i*, and at the end of a clip that step lands on the
reconstructed frame instead of on the unrelated first pose of the next clip.

The corresponding C# behaviour is on [the pose database](../animation-tools/pose-database.md).

## Global indexing

Frames are indexed **globally across the database**, so frame *i+1* does not continue frame *i* across
a clip boundary. The clip table is what maps global indices back to individual animations, and the
low-level rate function must never be fed frames from two different clips.

## Forward kinematics

`Skeleton.fk_root_space` performs FK **while ignoring the root**. Joint 0 — the character frame — is
pinned to the origin with identity rotation, and that is exactly what makes the result independent of
where the character stands and which way it faces. Joint 1 is the rig's own root bone, carrying the
translation the pose stores.

That invariance is the foundation of the similarity metric; see
[motion field policies](motion-field-policies.md).

## Blending

`utils/quaternions.blend_quaternions` uses **NLERP rather than SLERP**, because SLERP handles only two
rotations while blending a k-NN neighbourhood means combining *k* at once. The cost is a non-constant
angular rate, which does not matter when the neighbours are already close together.

Signs are aligned against the first quaternion first: q and −q are the same rotation, so without that
two near-equal neighbours can **cancel instead of reinforce**.

`Pose.blend` reshapes its weights for broadcast explicitly rather than indexing with a trailing axis,
which would align them against the wrong axes and fail on batched input. `stack` puts poses on a new
*leading* axis, preserving each one's batch shape — `concatenate` would fold the batch into the blend
axis, so blending two batches of A poses would produce one pose rather than A.

## Source map

| Concern | File |
| --- | --- |
| Packed layout, arithmetic, blending | `Python/Pose.py` |
| Joint tree and FK | `Python/Skeleton.py` |
| The database as arrays | `Python/Animation.py` |
| Binary reader, frame definition | `Python/pose_set_importer.py` |
| Frame reconstruction and rates | `Python/simulation_frame.py` |
| Quaternion blending | `Python/utils/quaternions.py` |

**Unused surface.** `Pose.lerp`, `Pose.concatenate` and `PoseDelta.concatenate` have no callers.
`lerp` additionally looks unsound for batched input — treat it as legacy rather than as a supported
operation.

**A softness worth knowing.** The importer reads the clip table's per-clip frame time into a single
scalar, so the last clip silently wins. The C# side stores it per clip and hard-asserts that they
agree.
