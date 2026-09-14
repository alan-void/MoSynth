---
type: Architecture Guide
title: Retargeting BVH onto the shared target rig
description: Why source motion is retargeted in Blender before it reaches Unity, what the two stages do, and which hidden inputs decide whether the result is right.
tags: [retargeting, bvh, blender, rokoko, import]
sources:
  - id: openwiki-source-2e0d28f6a1ee99e6e464a576
    resource: repo://Assets/AnimationTools/Runtime/Animation/AnimationClipBaker.cs
  - id: openwiki-source-e1ad0ab569ae74b451c3418f
    resource: repo://Assets/AnimationTools/Runtime/Animation/SkeletonAnimation.cs
  - id: openwiki-source-54f745ff7f6016dea0150c17
    resource: repo://Assets/AnimationTools/Runtime/Skeleton/Skeleton.cs
  - id: openwiki-source-98b90f0fe54f83ab272ff785
    resource: repo://Assets/AnimationTools/Runtime/Skeleton/SkeletonBone.cs
  - id: openwiki-source-be8fb964478d0fdd009e441a
    resource: repo://Assets/AnimationTools/Runtime/Skeleton/SkeletonBoneOverrides.cs
  - id: openwiki-source-3c35eadfeff4e039a271f8ae
    resource: repo://Tools/Retargeting/batch_retarget.py
  - id: openwiki-source-31ba5a7a051e33f0aaa4981b
    resource: repo://Tools/Retargeting/make_lafan_corrected_setup.py
  - id: openwiki-source-c6bd9da2c59d3b0501214cc4
    resource: repo://Tools/Retargeting/README.md
  - id: openwiki-source-0077e64b8f7c29bf1001e97a
    resource: repo://Tools/Retargeting/run_batch_all.py
generated: {by: "claude-code", at: "2026-09-07T21:17:41.812Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-09-07T21:17:41.812Z
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
| One | copies the raw BVH motion onto a **cleaned copy of the same skeleton** — same bone names, same bone lengths, rest pose re-oriented into a T-pose — with IK on the limbs where the proportions call for it |
| Two | Rokoko's retarget from that cleaned skeleton onto the model rig |

Stage one looks redundant and is not. A BVH rest pose puts every bone along a single axis: for
LAFAN, the whole skeleton is laid out along +X, legs and arms alike. Rokoko infers its mapping
from rest poses, so handed that directly it **produces unusable results**. The cleaned skeleton
exists purely to give stage two a rest pose it can reason about.

Stage one is constraints, not a mapping: `COPY_ROTATION` on the mapped bones and
`COPY_LOCATION` on the root. Whether it also needs `IK` depends on the dataset. LAFAN's setup
carries `IK` with chain length 3 on the feet and forearms, targeting the source's toe and hand.
Bandai-Namco's does not, and should not: cleaning kept the actor's every bone length to 3.4e-07 m,
so the two chains reach exactly as far, and every BVH joint but `Hips` has a constant translation.
Copying rotations already reproduces the source exactly, and an IK solver on top could only add
noise to an exact result. Reach for `IK` when cleaning had to *change* a chain's length, not by
default.

Because the cleaned rest is the calibration pose, what it is *set to* is the whole design decision.
Build it as **the actor's own skeleton, posed into a T-pose**: each bone's *direction* comes from
the model rig, its *roll* is measured (below), and each joint then sits at the source's own offset
re-expressed through the corrected parent frame. A child's offset in its parent's local frame does
not change when the parent is posed, so driving placement from the source's offsets makes the rest
a genuine pose of the actor — and under stage one's constraints every joint lands exactly on the
source's, to 0.0 mm.

**Do not copy the model's roll.** Taking direction *and* roll from the model is the tempting
shortcut and it is wrong, because the two rigs disagree about which local axis is sideways. The
Bandai actor's shoulder sits 4.98 cm along the chest's local **Z**; the model's sits along local
**X**. Copy the model's roll and the actor's shoulder offset is laid along the model's *forward*
axis, so the retargeted chest comes out **yawed by exactly 90°** — hips correct, torso square to
the direction of travel.

Measure the roll instead of assuming it. Both rigs state where sideways is, in their own child
offsets, so for each bone take the twist about its axis that carries the source's local child
directions onto the model's, and average over the children (a child sitting straight down the bone
says nothing and is skipped). On this dataset every bone comes out at 0° except the chest, which
comes out at exactly 90°.

**No bone-frame comparison can see this.** Rokoko builds its helper bones at the *target's* rest and
reads only the cleaned bone's rotation, so a rig-to-rig comparison of bone frames reads a flawless
0.000° while the character's torso is square. Frame equality only means something once the two rigs
share an axis convention. The check that works is geometric: limb directions expressed in a body
frame, or the shoulder-to-hip line angle computed inside each rig and then compared. Compare the
shoulder lines themselves across rigs in world space and you measure the rigid transform between
them, not the torso.

Two smaller traps sit alongside it. A BVH's zero-length `End Site` joints (`Head`, hands, toes)
import with an arbitrary bone direction — Bandai-Namco's `Head` points *forward* at rest where the
model's points up — so their direction must come from the model. And the finished rest should be
dropped so its lowest joint stands on the model's floor, which is what makes Rokoko's auto-scale
compare like with like.

## The setup is the blend

One `.blend` per source skeleton, holding the source armature, the cleaned armature with its
constraints, the model rig, and the Rokoko bone list. Every Rokoko setting — source, target,
auto-scale, pose mode, the mapping — is a **scene property**, so it is saved in the file. There
is no sidecar config, and `batch_retarget.py` finds the three armatures by role rather than by
name.

Authoring one is manual and needs judgement; `Tools/Retargeting/README.md` is the procedure.
Replaying it is not, which is what the batch script is for.

## Replaying a setup over thousands of clips

`batch_retarget.py` retargets a list of clips into one FBX. That does not stretch to a whole
dataset, for two reasons that share a cause: every retargeted action stays parked on an NLA strip
until the single export at the end, and Rokoko's rest-pose drift accumulates on the model rig with
every clip. Memory grows with the run, and past 1e-4 m of drift the run aborts — its own error
message says to split the batch up. **Both reset only when the process exits**, which is why
`run_batch_all.py` runs one Blender process per shard rather than one long run. Measured on
Bandai-Namco dataset-2, a ~220-clip shard ends at about 5.2e-05 m of drift, half the budget, so
the 250-clip default cap is not arbitrary.

Shards follow the dataset's own motion labels, so an output file is named after what is in it
rather than by an index. Because each shard is a separate process writing a separate FBX, a run is
resumable — an existing output is skipped — and shards run in parallel.

**How a file name maps to a label is a named scheme, chosen with `--shard-by`, not inferred.**
Bandai-Namco's `dataset-2_walk_normal_001` and LAFAN's `walk1_subject1` both separate fields with
underscores and cannot be told apart from a name alone, so sniffing would be a guess. The `action`
scheme strips the trailing take number from the first token, putting `walk1` through `walk4` in one
shard, and ignores the performer — every LAFAN subject was retargeted onto a single skeleton before
the dataset was released, so the subject says nothing about the motion and would otherwise give
every clip a shard of its own.

**The clip cap is a proxy for frames, and the two datasets are twenty times apart.** A Bandai clip
averages ~133 frames; a LAFAN clip averages 6,450, and the dataset is 496,672 frames over 77 clips.
The 250-clip default that puts a Bandai shard at ~1 GB has to drop to 10 for LAFAN to keep a shard's
working set comparable — which yields 17 shards peaking at 59,613 frames, about twice a Bandai
shard. Drift is a non-issue at that size: a 9-clip LAFAN shard ends around 2e-06 m, two orders of
magnitude inside the budget.

Three things a run at this scale needs that a single run does not:

- **A clip failure must not discard the run.** The export happens only at the end, so without
  `--continue-on-error` one bad clip late in a batch loses every clip before it. With it the clip
  is logged and skipped, the FBX is still written, and the run exits 2 rather than 0.
- **A manifest.** Stdout is the only other record of which take came from which BVH, and it does
  not survive the run. It also carries each take's starting root position, which is what makes the
  next point observable across a whole dataset instead of one clip at a time.
- **Rest offsets that vary from clip to clip.** A BVH stores each joint's rest `OFFSET` beside
  absolute position channels, and the batch deliberately transplants only the *action* onto the
  setup's source skeleton. A clip whose rest offset differs from the setup's is therefore re-framed
  by exactly that difference. On LAFAN this is the intended normalisation. On Bandai-Namco
  dataset-2 it is a trap: the same actor has **three Hips offsets spanning 8.8 m**, while the
  positions actually recorded all sit within centimetres of the origin, so two thirds of the clips
  would land metres from where they were captured. `--keep-capture-position` compensates the
  location curves; it is off by default so LAFAN's behaviour is untouched.

**One of those two reasons is Rokoko's, and a setup can be built that does not have it.**
`--script direct` runs `direct_retarget.py` instead, which needs a setup carrying its own helper
rig and bakes the model straight through it. Nothing applies or un-applies an object transform, so
the drift has no mechanism: the full Edinburgh dataset — 1,855 clips over 9 shards — ended at
`0.00e+00 m`, where a comparable Bandai shard ends at 5.2e-05. The shard cap still earns its keep
there, but only for memory. See *A live rig, and no Rokoko* below.

**The bone-length guard measures only the bones the retarget uses.** A BVH root's "length" is the
offset down to the first real joint, which records where the actor stood in the capture volume, not
anything about the actor. Measuring it rejected 2,058 of dataset-2's own 2,902 clips as belonging
to another subject — the same three Hips offsets, seen through a different check.

## The two setups today

Both datasets retarget onto the same corrected 22-bone rig, which is what lets their clips
share one `MotionMatchingData`: its `skeleton` is a single skeleton, and until this was true the
datasets' bone names differed by the `Model:` prefix alone before the extra bones were counted.

| Dataset | Setup blend |
| --- | --- |
| LAFAN | `Assets/LFS/Retargeting/lafan_bvh_to_lafan_corrected.blend` |
| Bandai-Namco | `Assets/LFS/Retargeting/bandai-namco/bandai_namco_retarget_to_corrected_lafan_claude.blend` |

LAFAN reaches the corrected rig in **one** Rokoko pass from the BVH. Going through the old
`Model:*` rig first and correcting afterwards was rejected: it costs a second Rokoko pass whose
source is the badly-oriented rig the correction exists to replace, and the two passes' errors
compound. The retired `Assets/LFS/Retargeting/retargeting.blend` is kept unmodified as the record
of how its cleaned skeleton was authored.

Nothing about the LAFAN setup needed authoring by hand, because no judgement was left in it. The
part that needs judgement — the cleaned skeleton whose rest pose is stage two's calibration pose —
already existed in the retired blend, and the corrected rig already existed as a proven export
configuration in the Bandai setup. Only stage two's target changes, so
`Tools/Retargeting/make_lafan_corrected_setup.py` composes the new blend from the two: it drops the
old model rig and any stale per-clip proxy, empties the file of actions, appends `Lafan_corrected`
with its `Mesh` child, re-points the Rokoko pointers, and rewrites the 22 bone-list rows by
stripping the `Model:` namespace from each target name. Re-run it if either input moves.

Three things that composition has to get right, each of which fails quietly otherwise:

- **The corrected rig must come from the Bandai setup, not from `lafan_rig_correction.blend`.**
  Both hold the same 22-bone armature at the same transform, but only the Bandai copy carries the
  `T-Pose` NLA track. The correction blend has no animation data at all, and
  `order_rest_take_first` is a hard error without that track.
- **Pre-existing actions have to be removed, not purged.** Several are held alive by a fake user
  and survive `orphans_purge`, and a surviving action named `T-Pose` renames the appended one to
  `T-Pose.001`, which then exports as a take under that name.
- **Appending the rig also brings whatever action it was last posed with** — on the Bandai setup,
  one of its own retarget results.

Because the corrected rig *is* the LAFAN skeleton re-oriented — `LeftUpLeg` 0.435 m and `LeftLeg`
0.42372 m against the BVHs' 43.500 cm and 42.372 cm — LAFAN's retarget is near-identity in
proportion, and auto-scaling has nothing to correct. It is set off regardless, matching Bandai.

Bandai-Namco is the setup where source and model proportions genuinely differ. LAFAN's
`source_skeleton` has bone lengths identical to its model rig — the capture was done on that rig —
so its IK never compensated for size, only for the rest-pose re-orientation. The Bandai actor is
shorter: torso ×1.47, arms ×1.25, legs ×1.12, hip width ×1.64 against the corrected rig. What
saves the result is that the two differences cancel where it matters — the actor's pelvis sits
higher above the thigh joint, so hip-to-ankle comes out 0.866 m against the model's 0.858 m, and
the feet land.

**Its auto-scale is deliberately off.** Rokoko's auto-scale keys off the rest hip-to-toe span
(0.864 m against 0.920 m) and would scale the source by 1.066, which lifts the retargeted hips to
0.938 m — higher than the model's legs reach, so the feet float. With it off the hips ride between
0.873 m and 0.919 m over a walk and the toes sit on the floor to within half a centimetre. The
setting is a scene property, so it travels in the blend and nothing on the command line restores
it; changing it silently rescales every clip a later batch produces.

One `.bvh` skeleton very nearly covers **both** Bandai datasets: across all 3,077 files the joint
names, hierarchy and offsets are identical apart from the `Hips` placement. Two qualifications
matter in practice.

The `Hips` placement is **not constant** — 9 values across the two datasets, 3 within dataset-2
alone, up to 8.8 m apart. It is capture-volume placement rather than anatomy, and it is what
`--keep-capture-position` exists for.

And **11 of dataset-1's 175 files carry a genuinely different skeleton**: they give `Hand_L/R`,
`Toes_L/R` and `Head` real 5–10 cm `End Site` offsets where the other 3,066 files have degenerate
zeros. Blender makes an `End Site` the parent bone's tail, so those leaves import 0.05–0.10 m long
instead of as 0.001 m stubs — 50 to 100 times over the guard's 1 mm tolerance, so they are rejected.
They are all `_normal_` takes, and they include both `call` clips and the only `respond` clip, so
skipping them costs those two motions entirely. Retargeting dataset-1 needs a decision about them
first; dataset-2 is unaffected.

Its bone map is 21 rows. `Chest` maps to `Spine2`, which leaves the model's `Spine1` unmapped and
riding at rest, so the spine bends in two places instead of three; the source's `joint_Root` is
static at the origin and is unmapped too.

### Edinburgh, where the source had no rotations at all

The Edinburgh Locomotion database ships no motion files. It is two `.npz` arrays of 21 joint
*positions* per frame, with no rotations, no hierarchy and no skeleton, so before any of this
pipeline can touch it a skeleton has to be solved out of the point cloud.
`Tools/Retargeting/edinburgh_npz_to_bvh.py` does that, and the interesting part is what it changes
downstream: because the converter authors the BVH, it also chooses the rest pose and the bone
names, and it chooses both to make the retarget easy. The rest is written as a real T-pose, and the
joints carry the target rig's own names.

That removes the reason stage one exists. There is no degenerate rest to clean, so the cleaned
skeleton is a plain duplicate wired through with `COPY_ROTATION` and no IK, and the Rokoko map is
21 identity rows. `Tools/Retargeting/make_edinburgh_setup.py` composes the blend from one converted
clip plus the shared rig, then verifies it by re-opening the saved file and resolving it with
`batch_retarget`'s own code rather than a reimplementation that could drift.

**A joint whose children move independently cannot be one BVH joint.** The hip joints and the
clavicles are rotating segments in this source, not fixed offsets — the distance across the hip
joints swings by 7.9 cm — while every parent-child bone length is constant to the last digit. Each
of those four therefore gets a bone of its own, sitting on its parent at a zero offset and carrying
the rotation that aims it. Modelling them cut the error between the written BVH and the source
point cloud from 2.0 cm to 0.9 mm, and it is what lets the clavicles drive the target's
`LeftShoulder`/`RightShoulder` instead of freezing them. Only `Spine2` is left riding at rest.

**A rest pose made of canonical axes is a claim about the actor, and it was wrong.** Writing the
BVH's rest as a clean T-pose is right for the limbs — arms out and legs down is what a T-pose
*means*, and the target rig agrees. It is wrong for the torso, and worst of all for the root. The
Edinburgh actor's pelvis-up axis, from his hip joint to his lower back, sits about 17 degrees ahead
of vertical; his spine carries its own curve on top. Declaring those straight made a rest pose no
human stands in, and since a rotation retarget applies *delta-from-source-rest* onto *target-rest*,
the entire difference was added to every frame. The character came out hunched, chest thrust
forward, head jutting — the symptom looked like a broken spine and was in fact a broken rest.

The measurement that settles it is torso pitch, the lean of hips-to-neck away from vertical.
LAFAN's rig rests at about 2 degrees. The retargeted Edinburgh character sat at **19.3 degrees**;
deriving the spine's rest directions from the actor's mean pose barely moved it (24.2), because the
root was carrying the lean. Deriving the root's too brought it to **7.3 degrees**.

The root is the awkward one: every other bone derives its rest against its parent, and the root has
none, which is exactly why it was left canonical and why it was the last thing suspected. Measuring
it against the world vertical and the hip line gives a frame independent of which way the character
faces, and so a mean worth taking. Note what this does *not* say: the source's own torso pitch is
21.6 degrees and should not transfer, because it describes where this marker set puts its neck
joint, not how the actor stood.

**The actor's lean is real, and it still has to be scaled down.** Edinburgh's hip-to-neck axis
reads about 2.4 times the lean his body actually had — checked against where his head sits over his
ankles, the one indicator on this skeleton that involves no pelvis marker: at a hip-to-neck pitch of
15–25 degrees his head is 7.7 degrees forward, at 25–40 it is 13.7. The marker set samples the spine
differently from the target rig, so copying that axis verbatim puts a 20–30 degree hunch on a
character whose body leaned 8–14.

**A scale, not an offset.** Subtracting a fixed angle is the obvious move and cannot work: tuned for
a level walk it tips a strongly-leaning clip backwards, and tuned for that clip it hunches the walk.
Two rounds of this went past before the shape of the error was clear — it is a gain. Scaling each
frame's bend by one constant fixes both at once and keeps every clip's own lean in proportion; one
factor covers the dataset, and nothing about it is per clip.

It belongs in the setup blend, not the BVH, which stays exactly faithful to the point cloud. And
almost all of it is in **one bone**: the hip-to-lower-back segment reads 24.0 degrees where
hip-to-neck reads 23.7, so the spine above it is nearly straight and damping only the spine bones
changed nothing measurable. That segment is the root, which also carries the character's heading, so
its rotation is split — a bone points along its own local Y, which here is the pelvis-up axis, so
copying Y whole keeps the turns and damping X and Z brings only the lean down. Torso pitch lands at
6–8 degrees against LAFAN's 2-degree rest, heading survives intact (a clip that turns 341 degrees
retargets to 340), and the limbs come out slightly *better* than before.

**A segment whose roll nothing measures should inherit its parent's.** The clavicles and the
hip-joint segments each reach a single marker, which fixes where they point and says nothing about
how they are rolled — so whatever roll they get is a convention. Borrowing a neighbouring direction
for it (the clavicles took the head's, the hip joints the pelvis-up axis) looked reasonable and let
each segment twist against the very body part it is rigidly attached to, because its *parent's* roll
was pinned to something else again. Measured on the source: the clavicles drifted 28 to 35 degrees
against the chest over a single clip and the hip joints 38 against the pelvis, symmetric and
opposite left to right — the signature of a reference that belongs to a different body part. Built
instead by carrying the parent's own roll axis onto the segment, the drift is 0.0 degrees by
construction. Note what this is not: a fix for noise, but a choice about which convention is
honest when the data is silent.

Two things that make the change safe to land on top of hand-authored work. Joint positions are
untouched, because roll never moves a joint — the forward-kinematics error stayed at 0.0881 cm to
the digit. And the `OFFSET` block is *invariant* under a roll convention change: the measured frame
and the rest frame both rotate by the same amount about the bone, so `rest @ mean_local` cancels it
out. The regenerated files declared a byte-identical skeleton, which is what let an authored rest
pose carry straight over.

**Roll is where every visible artefact came from**, and they looked like one bug with three
different causes. A bone direction fixes two of three degrees of freedom; the third needs a second
measured direction, and no joint position can supply it — roll about a bone leaves every joint
exactly where it was. So `--verify` reports the worst frame-to-frame roll step beside the position
error, and that is the number to watch: position error stayed at 0.9 mm while the worst roll step
came down from 163° to 13.6°.

- **A straight limb has no bend plane**, so the measurement must fade out as the joint straightens.
  Holding the last good normal instead looks reasonable and is the worst option: it freezes a
  *world-space* direction while the body keeps turning, so the held vector drifts further the longer
  the limb stays straight, and the whole difference lands in the one frame the measurement resumes
  on. A knee that sat just under the threshold for eight frames jumped 18° in a single frame.
- **A foot's roll is pinned by its shin, not by the knee's bend plane.** Nothing keeps a noisy knee
  normal off the foot's own axis, and when it lands there the frame is built from two nearly
  parallel vectors: a foot swung through 163° in one frame while its perpendicular component
  collapsed to 0.19. The ankle hinges perpendicular to the shin, which is never parallel to the foot.
- **What a limb falls back on must follow the limb.** A body-fixed axis is a fair guess for a thigh,
  which stays near its rest direction, and badly wrong for an arm: during a stride the arms hang
  about 90° from the T-pose, so a body-fixed reference drags their roll a quarter turn off. Carrying
  the rest reference onto the bone's current direction by the shortest rotation — its zero-twist
  pose — is what fixed it, and only then did tuning become monotonic instead of blowing up at half
  the settings tried.

**Splitting a bone's rotation per axis puts you at the mercy of an euler order.** Damping the
root's lean while keeping its heading means enabling some axes of a `COPY_ROTATION` and not others,
and Blender does that by decomposing to euler -- which gimbal-locks on its *middle* axis. The BVH's
bones are `ZYX`, whose middle axis is Y, the very one carrying the heading, so a clip that turned
far enough snapped the damped axes through 94 degrees as its yaw crossed 90. Setting that one bone
to `YXZ` puts the middle on X, where a pelvis would have to pitch 90 degrees to reach it. The source
BVH's own `Hips` never moved more than 12.6 degrees between frames while the retargeted one moved
93.9: whenever a retarget is jumpier than its source, the setup is adding it.

**Measure the whole rotation, not the part you suspect.** The converter reported worst-roll-step for
a while, and it read a clean 13.6 degrees on a dataset where a hand was visibly snapping through 125
-- because most of that was swing, not twist. A check aimed at the failure you have in mind will
confirm it is absent and tell you nothing about the one you have not thought of.

Two things that look like the obvious fix and are not. Snapping a reversed normal into the
fallback's hemisphere steps by 180° every time a noisy normal crosses perpendicular, which made the
worst case three times worse; fading it out instead is continuous, and also removes the blend's one
unstable case of two opposed vectors cancelling to no length at all. And measuring roll by rolling a
seed vector reads a bone's *swing* as twist whenever the swing is large — the check needs a
swing-twist decomposition.

Which way round a knee or an elbow bends at rest is anatomy rather than data, so the converter
states it and then checks it against the dataset: knees must lean backwards and elbows forwards. A
wrong sign there moves nothing by a millimetre and simply delivers a limb inside out.

**Expect worse foot contact than a rotational source.** The released root trajectory already slides
a planted toe 5.5 cm/s on its own, and retargeting onto an actor of different proportions roughly
doubles it. Measured the same way, a retargeted Edinburgh clip slides 10.8 cm/s against LAFAN's
1.9 cm/s — LAFAN being near-identity because its capture rig *is* the corrected rig. Decimating
60 Hz to 30 Hz costs nothing here (5.6 against 5.4 cm/s); the gap is the data and the proportions.

### A live rig, and no Rokoko

Rokoko retargets by building a helper bone per mapped bone — parented to the source bone, sitting
at the target bone's rest — baking the target through them, and deleting the lot. Blender evaluates
those helpers as `source_world @ source_rest^-1 @ helper_rest`, which is the rotation delta the
retarget is defined as. **Keeping the helpers instead of deleting them makes the whole chain live**,
and because Blender reads the rest matrices fresh on every evaluation, editing the cleaned rest pose
in Edit Mode moves the target character immediately. That is what `make_edinburgh_setup.py --live`
builds, and it is the only way the one thing in a setup that must be judged by eye can be judged
while it is being changed.

The same rig then turns out to be a complete retarget engine. `direct_retarget.py` bakes the model
once through it, replacing stage one's bake, the constraint-free proxy and Rokoko's
duplicate-and-bake — roughly 3x faster, with the drift gone rather than merely small, and with no
dependency on the addon. Against a real Rokoko bake of the same clip and rest pose the two agree to
**0.016 degrees mean, 0.079 max**. The part that matters more than the speed: the blend used to
author a rest pose runs the same code as the batch, so a preview cannot disagree with the output.

It reads the bone map off the model's own constraints rather than from a stored list, and finds the
rig by looking for the `RT_*` helpers rather than by name, so the wiring *is* the mapping. A setup
without helpers is refused outright instead of retargeted badly.

**An unmapped target bone is the one thing constraints do not protect.** Takes already stacked on
the model have to be muted before each clip: a constraint overrides an action underneath it, so a
*mapped* bone is safe, but the target's `Spine2` has no constraint and would be posed by whatever
the previous clips left on the NLA stack — and the whole upper body hangs off it. The error
compounds clip by clip rather than showing up at once.

**Standing the character on the floor belongs in the bridge, not the target rig.** The retargeted
character floats, because the source actor's hip-to-toe chain is 109.6 cm against the target's
113.9. The fix is a helper bone, `RT_ground`, parented to the bridge's `Hips` and used as the Copy
Location target for the model's `Hips`, so the offset is a single authored number and the shared
target rig is never touched. Two details make it behave as a world-space dial: it rests along `+Y`
with zero roll, which makes its local axes the world's, and it does not inherit rotation, so the
offset stays vertical instead of tipping with the pelvis. `use_offset` on the constraint is a
different mechanism and the wrong one — it adds the bone's own 93 cm rest height.

It corrects a constant, and the residue is not constant: across 24 clips spanning the dataset the
lowest toe averaged +0.2 mm with the offset set, but individual clips ranged from −5.2 cm to
+4.7 cm. That spread is in the capture, and closing it is a footlock problem rather than a rest-pose
one.

## Hidden inputs that decide correctness

Five properties of this pipeline are invisible in the file and were each found the hard way.

**Where the exporter puts unit scaling.** Blender's FBX exporter has to reconcile a metre scene
with a file format that defaults to centimetres, and `apply_scale_options` decides where the
factor of 100 goes. `FBX_SCALE_NONE` puts it on **each object's transform**, so the armature node
is written with a local scale of 100 and its bones keep metre-sized offsets. The composite is
visually correct and Unity renders the rig fine — but `Skeleton` reads rest offsets straight off
the Transforms and its FK is **rigid**, so it drops that scale. Every bone's rest offset then
reads 100× too small while the root's animated curve arrives in true metres, and forward
kinematics collapses the entire skeleton onto the root: on a walk clip the toe sat 9 mm below the
hip instead of 0.9 m.

Nothing fails. Contact detection simply never fires, every clip reports zero footfalls, and gait
phase is empty. The only signal is one line from `AnimationClipBaker`:
`"Root" in rig "..." has non-unit local scale; baked FK is rigid and will be incorrect.` Treat
that warning as fatal, not advisory.

`FBX_SCALE_UNITS` sends the factor to the file header instead and leaves every object at unit
scale, which is what the rig invariant requires — nothing but bones under the skeleton root, and
those bones at identity scale.

The fix does not reach back into output already on disk. Everything exported before it carries the
collapsed scale and is silently useless downstream, and re-exporting is the only remedy — the setup
blend and each shard's clip list survive in `.batch/`, so it costs a re-run rather than re-authoring
a retarget.

**Read the FBX records to tell a fixed export from a broken one; a Blender re-import cannot.** A
good file has `UnitScaleFactor` 100 with every node at unit scale; a broken one has
`UnitScaleFactor` 1 with the armature node at `Lcl Scaling` 100. Blender's importer normalises that
away, so the file looks clean on re-import either way. The same read shows which take the file leads
with, which is the second fix below.

Every Bandai shard has now been re-exported under both fixes, and LAFAN was retargeted onto the
corrected rig under them from the start. The cost of the Bandai re-export was nothing but compute:
none of the stale shards was referenced by any asset, so no clip, config or prefab had to be
re-pointed. That was luck of timing rather than a property of the pipeline — see the file-ID churn
below.

One file is deliberately not covered by that. `bandai_walk-turn-left_normal_005.fbx` was exported
by hand from a single clip rather than by the driver, so a `--force` run does not reach it; it is
unreferenced, and its one take now also lives in `bandai_dataset-2_walk-turn-left.fbx`. It is
redundant rather than pending.

**Re-exporting churns the FBX's internal file IDs.** The `.meta` survives, so the asset GUID is
stable and nothing loses track of the *file* — but every reference to a bone Transform or an
`AnimationClip` *inside* it breaks, and Unity surfaces that only as
`"Skeleton has no root Transform assigned"` from whatever asks for the rest pose next. After the
Bandai re-export, 95 assets had to be re-pointed by name: 93 `AnnotatedAnimationClip` assets (each
one's `clip`, `skeleton.root` and `rootMotionBone`), the `PfnnBandaiWalk` config's `skeleton.root`,
and the character prefab's `MotionSynthesisComponent.skeleton.root`. Budget for that as part of any
re-export, and re-run `TryValidate` over the clips afterwards.

**Which take the FBX leads with.** Unity poses an imported model's hierarchy from the **first
take's first frame**, not from the file's bind pose. `Skeleton` then reads its rest pose live off
those Transforms, so whichever take comes first *is* the rig's rest pose for everything downstream.
Lead with a locomotion take and the rest pose becomes a mid-stride capture.

The damage lands on `RestLocalAxis(0, forward())`, which derives the character frame's forward axis
from that pose, so every frame in the database ends up measured in a frame rotated away from the
character's actual facing. Both halves of the pipeline derive it identically from the same file, so
training, the offline rollout, the loss curve and `Check Training Agreement` all agree with each
other and pass. It becomes visible only where a pose meets a rig that was not part of that
agreement: `MotionSynthesisComponent` writes bone 0's rotation onto the live character rig, which
sits at a real bind, and the hips jump the difference on the first tick.

Measured on `bandai_walk_normal` before the fix, the imported `Hips` read `(169.7, -6.5, -179.7)` in
xyz-euler at `(0.011, 0.932, 5.433)` — the 5.4 m being where the actor stood — against the rig's
actual bind of `(-20.67, 0, 0)` at `(0, 0.932, -0.043)`, a difference of **173.6°**. The stored rest
matched database frame 12072 to 0.002° mean over all 27 bones, and that frame is the first frame of
take `dataset-2_walk_normal_001`.

`order_rest_take_first` moves the `T-Pose` track to the bottom of the NLA stack immediately before
export, because `push_to_nla` appends a track per clip and would otherwise bury the rest track the
blend was saved with. A setup with no such track is now a hard error rather than a silently posed
bind.

One trap while diagnosing this: the FBX's **bind pose is correct either way**, because Blender
writes it from the armature's rest data. Re-importing the file into Blender therefore shows a clean
T-pose and hides the problem completely. Unity's importer is what ignores it, so check the imported
hierarchy in Unity rather than the file.

**Pose-bone rotation mode.** The setup's bones are euler in whatever order the BVH declares —
`ZYX` for LAFAN, `ZXY` for Bandai-Namco — and stage one bakes `rotation_euler` curves. Which is
why the cleaned skeleton is made by *duplicating* the source: the mode comes along, and a
mismatched one would be silent. An armature object created with
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
separated only by spread. On LAFAN `walk1_subject1` onto the corrected rig a correct run scores
**3.24° mean joint-angle error**, against 5.3° for the same clip onto the retired `Model:*` rig.
The residual is not uniform and should not be: spine 0.12° and neck 0.27° against ankles 5.5° and
elbows 4.5°. The large ones are exactly the chains stage one drives with IK, which is where the
cleaned skeleton is allowed to differ from the source — so the shape of the residual is itself the
evidence, not just its size.

**Comparing bone frames between the two rigs proves nothing.** It is the obvious check and it
certified a 90°-yawed torso as flawless. Use geometric checks, which survive the rigs' differing
bone-axis conventions:

| Check | What it catches | Reading it |
| --- | --- | --- |
| joint angles (knee, elbow, ankle, spine) vs the source | the retarget actually following the motion | 3.91° mean on Bandai-Namco dataset-2; 3.24° for LAFAN onto the corrected rig |
| limb directions in a body frame built from the hips | any limb or torso rotated in the body | a difference that is **constant** across frames is a rest convention, not an error |
| shoulder-to-hip line angle, computed **inside each rig** and then compared | torso yaw specifically | source 13.849° vs retargeted 13.772° on LAFAN `walk1_subject1` |
| cleaned rest, left/right joint pairs mirrored | a rest that is not a T-pose | 0.0000 m asymmetry |
| constrained cleaned joints vs the source's | stage one drifting | 0.0 mm on every bone |
| hip height above the lowest toe **joint**, source vs retargeted | the character floating, sinking, or rescaled | 88.0 cm vs 86.8 cm on LAFAN `walk1_subject1` |

**Two of those rows are written the way they are because the obvious version measures something
else.** A raw `import_anim.bvh` sits a half turn about Z from the setup's source rest, so comparing
the two rigs' shoulder lines in world space reads that rigid transform — about 167° on a clip whose
torso is in fact square. Computing the shoulder-to-hip angle within each rig and comparing the two
scalars removes it. Likewise a BVH root offset records where the actor stood in the capture volume,
not a ground plane, so absolute toe height against z=0 is meaningless for the source; hip height
above the toe is the comparable quantity. Measure the toe **head** and not the bone's tail: LAFAN's
`End Site` joints are zero-length, so the source's toe tail is a direction Blender invented, and
including it turns a 1.3 cm agreement into a spurious 4.8 cm one.

The distinction in row two is what makes the check usable across rigs. On
`dataset-2_walk-turn-left_normal_005` the current Bandai setup reads 0.000° at both knees and
0.05° at the elbows, but a fixed 16.3° at the ankles — mean equal to worst on every bone, which is
the signature of the two rigs disagreeing about a rest pose rather than of the motion being wrong.
The feet still land, which is the check that settles it.

**Do not verify a change by comparing against an FBX exported earlier.** The setup blend is the
input, and it is edited by hand between runs; an output from a previous state of it will differ by
whatever changed — a scale factor if auto-scale was toggled, a constant rotation if the cleaned
rest was rebuilt — and none of that says anything about the change under test. Measure against the
source BVH instead, which does not move.

For contrast, the raw bone-frame comparison now reads **90° at the chest** and 0° elsewhere —
that difference *is* the roll correction, and seeing it there is the sign the setup is right, not
wrong.

On the Unity side, an exported FBX imports as a **Generic** rig with no importer setup. Both
datasets now produce the same shape: 27 bones with **22 animated**, the other five being the FBX
`_end` leaves. The 72-of-85 figure that `AnimationClipBakerTests` cites belonged to the retired
`Model:*` rig.
Note that Unity defaults the model importer to `KeyframeReduction`, which stacks a second lossy
pass on top of the exporter's `--simplify`; set Anim. Compression to Off if both are not wanted.

**Check the imported scale before trusting anything downstream.** Select the imported rig and
confirm the armature node's local scale is 1 and that a leg chain spans roughly the character's
height. The cheapest end-to-end check is to add a `GaitPhaseComponent` to one annotated clip and
press Detect: a correct import on a walk gives an alternating anchor every ~17 frames, and a
collapsed one gives none at all.

Every retargeted FBX — from either dataset — and the corrected rig blend import as the same
27-bone tree under `Hips`: the 22 animated bones plus the five `_end` leaves the exporter
synthesises. That match is what lets a character take its live rig from the blend while its
skeleton comes from the FBX the network was trained on; `SkeletonBoneOverrides` binds them by name, and
`MotionSynthesisComponent` disables itself outright if even one bone fails to bind, so a rig
missing the leaves would be a hard stop rather than a degraded mode.

## Source map

| Concern | File |
| --- | --- |
| Batch engine, one FBX per run | `Tools/Retargeting/batch_retarget.py` |
| The same, through a setup's own helper rig instead of Rokoko | `Tools/Retargeting/direct_retarget.py` |
| Whole-dataset driver: sharding, parallelism, resume, manifest | `Tools/Retargeting/run_batch_all.py` |
| Setup authoring procedure | `Tools/Retargeting/README.md` |
| Composing the LAFAN corrected setup from existing blends | `Tools/Retargeting/make_lafan_corrected_setup.py` |
| Solving BVH out of the Edinburgh point clouds | `Tools/Retargeting/edinburgh_npz_to_bvh.py` |
| Composing the Edinburgh setup from one converted clip | `Tools/Retargeting/make_edinburgh_setup.py` |
| Overlaying a solved skeleton on the points it came from | `Tools/Retargeting/edinburgh_solve_preview.py` |
| Unity launcher | `Assets/AnimationTools/Editor/Retargeting/RetargetBatchWindow.cs` |
| Per-dataset run settings | `Assets/AnimationTools/Editor/Retargeting/RetargetBatchSettings.cs` |

Downstream, each take becomes an `AnnotatedAnimationClip`; see
[animation sources](animation-sources.md).
