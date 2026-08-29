---
type: Architecture Guide
title: Retargeting BVH onto the shared target rig
description: Why source motion is retargeted in Blender before it reaches Unity, what the two stages do, and which hidden inputs decide whether the result is right.
tags: [retargeting, bvh, blender, rokoko, import]
---

# Retargeting BVH onto the shared target rig

Source motion no longer enters the project on its own skeleton. Every dataset is retargeted in
Blender onto one shared target rig **before** it reaches Unity, and only the retargeted FBX is
imported.

The reason is [the pose database](pose-database.md): `MotionMatchingData.skeleton` is a single
skeleton that every clip in the database must share. Two clips on two source skeletons cannot
go in one database, so combining LAFAN with AMASS or Bandai-Namco means normalising them onto
a common rig first. `BvhImporter` (see [animation sources](animation-sources.md)) still exists
and still produces a rig plus a clip, but it puts each BVH on its *own* skeleton, which is
exactly what this pipeline replaces.

## Two stages, and why the first one exists

| Stage | What it does |
| --- | --- |
| One | copies the raw BVH motion onto a **cleaned copy of the same skeleton** — same bone names, same bone lengths, rest pose re-oriented into a T-pose — with IK on the limbs |
| Two | Rokoko's retarget from that cleaned skeleton onto the model rig |

Stage one looks redundant and is not. A BVH rest pose puts every bone along a single axis: for
LAFAN, the whole skeleton is laid out along +X, legs and arms alike. Rokoko infers its mapping
from rest poses, so handed that directly it **produces unusable results**. The cleaned skeleton
exists purely to give stage two a rest pose it can reason about.

Stage one is constraints, not a mapping: `COPY_ROTATION` on all 22 bones, `COPY_LOCATION` on
the root, and `IK` with chain length 3 on the feet and forearms targeting the source's toe and
hand, so end effectors land correctly despite proportion differences.

## The setup is the blend

One `.blend` per source skeleton, holding the source armature, the cleaned armature with its
constraints, the model rig, and the Rokoko bone list. Every Rokoko setting — source, target,
auto-scale, pose mode, the mapping — is a **scene property**, so it is saved in the file. There
is no sidecar config, and `batch_retarget.py` finds the three armatures by role rather than by
name.

Authoring one is manual and needs judgement; `Tools/Retargeting/README.md` is the procedure.
Replaying it is not, which is what the batch script is for.

## Hidden inputs that decide correctness

Three properties of this pipeline are invisible in the file and were each found the hard way.

**Pose-bone rotation mode.** The setup's bones are `ZYX` euler, the convention BVH arrives in,
and stage one bakes `rotation_euler` curves. An armature object created with
`bpy.data.objects.new` gets **quaternion** bones, which silently ignore those curves. The
symptom is not an error: the root still translates, so the character walks the correct path
with the wrong body — 22° of mean joint-angle error against 5° for a correct run. The stage-two
source is therefore made by duplicating the cleaned skeleton and stripping its constraints,
never by synthesising an object over the armature data.

**The Rokoko bone list is cleared when you assign an armature.** Both
`rsl_retargeting_armature_source` and `..._target` carry an `update=clear_bone_list` callback,
and every clip has to reassign the source. The mapping is read once at startup and written back
before each retarget.

**The raw import is not the setup's source rest pose.** A fresh `import_anim.bvh` of a LAFAN
file differs from the blend's source armature by a rigid transform — a half turn about Z plus
the root's horizontal offset — and matches exactly once that is applied. Rather than replicate
it, the batch transplants only the *action* onto the setup's existing source skeleton, which
keeps every clip in the frame the setup was tuned in and makes clip start positions consistent.

A useful non-input: the current frame. Rokoko's `copy_rest_pose` applies a pose as the rest
pose, which looks frame-dependent, but it unlinks the action first, so results are identical at
any frame.

## Verifying a change to this pipeline

Curve-level equality against a known-good action is the sharpest check — stage one and stage
two each reproduce the reference to 6e-06 and 0.0 respectively.

For a change where exact equality is not expected, compare **joint angles** (knee, elbow,
shoulder, spine) between the stage-one source and the retargeted result. Joint angles are
geometric, so they survive the two rigs' differing bone-axis conventions, where a direct
comparison of bone directions does not: both a good and a bad retarget sit near 90° there,
separated only by spread. On LAFAN `walk1_subject1` a correct run scores **5.3° mean joint-angle
error**, which is the inherent source-to-model rig difference and not pipeline error.

On the Unity side, an exported FBX imports as a **Generic** rig with no importer setup, and every
take validates with **72 of 85 bones animated** — the same figure `AnimationClipBakerTests` guards,
the other 13 being the FBX `_end` leaves. Note that Unity defaults the model importer to
`KeyframeReduction`, which stacks a second lossy pass on top of the exporter's `--simplify`; set
Anim. Compression to Off if both are not wanted.

## Source map

| Concern | File |
| --- | --- |
| Batch engine | `Tools/Retargeting/batch_retarget.py` |
| Setup authoring procedure | `Tools/Retargeting/README.md` |
| Unity launcher | `Assets/AnimationTools/Editor/Retargeting/RetargetBatchWindow.cs` |
| Per-dataset run settings | `Assets/AnimationTools/Editor/Retargeting/RetargetBatchSettings.cs` |

Downstream, each take becomes an `AnnotatedAnimationClip`; see
[animation sources](animation-sources.md).
