---
type: Architecture Guide
title: Retargeting BVH onto the shared target rig
description: Why source motion is retargeted in Blender before it reaches Unity, what the two stages do, and which hidden inputs decide whether the result is right.
tags: [retargeting, bvh, blender, rokoko, import]
sources:
  - id: openwiki-source-4b4c7626aee592b0133aadec
    resource: repo://Assets/AnimationTools/Editor/Retargeting/RetargetBatchSettings.cs
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
  - id: openwiki-source-c6bd9da2c59d3b0501214cc4
    resource: repo://Tools/Retargeting/README.md
  - id: openwiki-source-0077e64b8f7c29bf1001e97a
    resource: repo://Tools/Retargeting/run_batch_all.py
generated: {by: "claude-code", at: "2026-09-03T11:06:29.275Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-09-03T11:06:29.275Z
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
frame, or simply the angle between the two rigs' shoulder lines.

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

**The bone-length guard measures only the bones the retarget uses.** A BVH root's "length" is the
offset down to the first real joint, which records where the actor stood in the capture volume, not
anything about the actor. Measuring it rejected 2,058 of dataset-2's own 2,902 clips as belonging
to another subject — the same three Hips offsets, seen through a different check.

## The two setups today

`Assets/LFS/Retargeting/retargeting.blend` covers LAFAN and
`Assets/LFS/Retargeting/bandai-namco/bandai_namco_retarget_to_corrected_lafan_claude.blend`
covers Bandai-Namco. They target **different model rigs**, and that is a migration in progress
rather than an oversight: LAFAN retargets onto the 85-bone `Model:*` rig, whose bones all point
along −Y in world and carry no anatomical direction, while Bandai-Namco retargets onto the
corrected 22-bone rig (`Root`, rest identical to `Skeleton` in `lafan_rig_correction.blend`)
whose bones are properly oriented. The corrected rig supersedes the old one; until LAFAN is
re-retargeted onto it, clips from the two datasets **cannot go in one `MotionMatchingData`**,
because its `skeleton` is a single skeleton and the bone names differ by the `Model:` prefix
alone before the extra 50 bones are counted.

Bandai-Namco is also the first setup where source and model proportions genuinely differ. LAFAN's
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
a retarget. Two selections have been re-exported: `bandai_walk_normal` (24 takes) and
`bandai_walk_turn_normal` (69 takes, the `normal` style of both turn directions, ~60 s). Both were
then exported again for the rest-pose fix below, so they now carry 25 and 70 takes — the extra one
being the `T-Pose` reference. Every other Bandai shard predates both fixes.

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
separated only by spread. On LAFAN `walk1_subject1` a correct run scores **5.3° mean joint-angle
error**, which is the inherent source-to-model rig difference and not pipeline error.

**Comparing bone frames between the two rigs proves nothing.** It is the obvious check and it
certified a 90°-yawed torso as flawless. Use geometric checks, which survive the rigs' differing
bone-axis conventions:

| Check | What it catches | Reading it |
| --- | --- | --- |
| joint angles (knee, elbow, ankle, spine) vs the source | the retarget actually following the motion | 3.91° mean on Bandai-Namco dataset-2, against LAFAN's 5.3° for a correct run |
| limb directions in a body frame built from the hips | any limb or torso rotated in the body | a difference that is **constant** across frames is a rest convention, not an error |
| angle between the two rigs' shoulder lines | torso yaw specifically | 0.000° every frame |
| cleaned rest, left/right joint pairs mirrored | a rest that is not a T-pose | 0.0000 m asymmetry |
| constrained cleaned joints vs the source's | stage one drifting | 0.0 mm on every bone |
| lowest toe height over the clip | the character floating or sinking | −0.5 cm to +1.2 cm on a walk |

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

On the Unity side, an exported FBX imports as a **Generic** rig with no importer setup. The LAFAN
takes validate with **72 of 85 bones animated** — the figure `AnimationClipBakerTests` guards, the
other 13 being the FBX `_end` leaves. The corrected rig is smaller, so a Bandai FBX carries 27
bones with 22 animated, and that test's expectation belongs to the old rig until LAFAN moves over.
Note that Unity defaults the model importer to `KeyframeReduction`, which stacks a second lossy
pass on top of the exporter's `--simplify`; set Anim. Compression to Off if both are not wanted.

**Check the imported scale before trusting anything downstream.** Select the imported rig and
confirm the armature node's local scale is 1 and that a leg chain spans roughly the character's
height. The cheapest end-to-end check is to add a `GaitPhaseComponent` to one annotated clip and
press Detect: a correct import on a walk gives an alternating anchor every ~17 frames, and a
collapsed one gives none at all.

Both `bandai_walk_normal.fbx` and the corrected rig blend now import as the same 27-bone tree
under `Hips` — the 22 animated bones plus the five `_end` leaves the exporter synthesises. That
match is what lets a character take its live rig from the blend while its skeleton comes from the
FBX the network was trained on; `SkeletonBoneOverrides` binds them by name, and
`MotionSynthesisComponent` disables itself outright if even one bone fails to bind, so a rig
missing the leaves would be a hard stop rather than a degraded mode.

## Source map

| Concern | File |
| --- | --- |
| Batch engine, one FBX per run | `Tools/Retargeting/batch_retarget.py` |
| Whole-dataset driver: sharding, parallelism, resume, manifest | `Tools/Retargeting/run_batch_all.py` |
| Setup authoring procedure | `Tools/Retargeting/README.md` |
| Unity launcher | `Assets/AnimationTools/Editor/Retargeting/RetargetBatchWindow.cs` |
| Per-dataset run settings | `Assets/AnimationTools/Editor/Retargeting/RetargetBatchSettings.cs` |

Downstream, each take becomes an `AnnotatedAnimationClip`; see
[animation sources](animation-sources.md).
