# Batch BVH retargeting

Source motion is retargeted onto the shared MoSynth target rig **in Blender**, before it
reaches Unity. `batch_retarget.py` replays a retarget setup you authored once by hand over
every BVH belonging to it, and exports one FBX carrying each clip as a take.

Run it from Unity — `MoSynth/Retargeting/Run Batch…` — or directly:

```
blender --background <setup>.blend --python Tools/Retargeting/batch_retarget.py -- \
    --bvh-dir Assets/LFS/Animation/lafan1/bvh \
    --pattern "*_subject1.bvh" \
    --out     Assets/LFS/Animation/lafan1/retargeted/lafan1_subject1.fbx
```

`--dry-run` resolves the setup and lists the clips without doing any work; `--limit N` stops
after N clips. **Never pass `--factory-startup`** — the Rokoko addon supplies stage two. The
blend is only ever read.

## Retargeting a whole dataset

One `batch_retarget.py` run holds every clip's action in memory until it exports, and Rokoko's
rest-pose drift accumulates across it, so a few thousand clips do not belong in one run. Both
reset only when the process exits. `run_batch_all.py` therefore splits a dataset into shards and
runs one Blender process per shard:

```
python Tools/Retargeting/run_batch_all.py \
    --setup   Assets/LFS/Retargeting/bandai-namco/bandai_namco_retarget_to_corrected_lafan_claude.blend \
    --bvh-dir External/LFS/Bandai-Namco-Research-Motiondataset/dataset/Bandai-Namco-Research-Motiondataset-2/data \
    --out-dir Assets/LFS/Animation/bandai-namco/retargeted \
    --keep-capture-position --simplify 1 --workers 6
```

Shards follow the dataset's own motion labels, with any motion over `--max-per-file` (default 250)
split into numbered parts, so an output file is named after what is in it. Start with `--dry-run`
to see the split.

**`--script` picks which retarget engine each shard runs.** The default `batch` is
`batch_retarget.py`, the Rokoko path. `direct` is `direct_retarget.py`, for a setup that carries
its own helper rig — see *Retargeting without Rokoko* below. Both take the same command line, so
nothing else about a run changes.

**`--shard-by` picks how a file name is read**, because two datasets that both separate fields with
underscores cannot be told apart from a name alone:

| Scheme | Convention | `walk1_subject1.bvh` / `dataset-2_walk_normal_001.bvh` becomes |
| --- | --- | --- |
| `dataset-motion` (default) | Bandai-Namco's `<dataset>_<motion>_<style>_<index>` | `bandai_dataset-2_walk` |
| `action` | LAFAN's `<action><take>_subject<n>` | `lafan_walk` |

Under `action` the trailing take number is stripped from the first token, so `walk1` … `walk4` all
land in one shard, and the performer is ignored — every LAFAN subject was retargeted onto one
skeleton before release, so it says nothing about the motion and would otherwise put each clip in
a shard of its own. `--only` matches the motion label either way.

LAFAN's 77 clips are long (496,672 frames, mean 6,450), so the cap is what keeps a shard's peak
memory near Bandai's rather than the clip count:

```
python Tools/Retargeting/run_batch_all.py \
    --setup   Assets/LFS/Retargeting/lafan_bvh_to_lafan_corrected.blend \
    --bvh-dir Assets/LFS/Animation/lafan1/bvh \
    --out-dir Assets/LFS/Animation/LafanCorrected \
    --prefix lafan --shard-by action --max-per-file 10 --simplify 1 --workers 4
```

That gives 17 shards peaking at 59,613 frames, about twice a Bandai shard. No
`--keep-capture-position`: on LAFAN the rest-offset re-framing is the intended normalisation.

Edinburgh's 1855 clips are short and uniform, and its file names carry no motion label, so the
default scheme reads them as one motion and splits it on the clip cap alone:

```
python Tools/Retargeting/run_batch_all.py     --setup   Assets/LFS/Retargeting/edinburgh_bvh_to_lafan_corrected.blend     --bvh-dir Assets/LFS/Animation/Edinburgh/bvh     --out-dir Assets/LFS/Animation/LafanCorrected/edinburgh     --prefix "" --max-per-file 250 --simplify 1 --workers 4
```

That gives 9 shards — eight of `edin_locomotion` and one of `edin_test` — peaking at 27,600
frames, half a LAFAN shard. `--prefix ""` is deliberate: the converter already names its files
`edin_<split>_<index>`, so the default `bandai` prefix would only repeat itself.

- **Resumable**: a shard whose FBX already exists is skipped, so a run interrupted at shard 12
  costs only the shards it had not reached. `--force` rebuilds regardless.
- **Resume is the wrong tool for replacing output that is already there.** The predicate is only
  whether the shard's FBX exists, so re-running to fix bad output reports nothing to do and skips
  exactly the files that need replacing. Use `--only <labels> --force` to name the shards to
  rebuild, or delete their FBX files first. A shard writes its FBX only at the very end, so an
  interrupted shard leaves the old file in place and looks finished.
- **`--workers`** is how many Blender processes run at once; each holds roughly a gigabyte.
- **`--only walk,run`** limits the run to named motions, and **`--limit N`** takes only the first
  N clips of each shard. Together they make a smoke run.
- Everything that is not an FBX — the clip lists, per-shard logs and manifests, and the merged
  `manifest.json` — goes in a `.batch/` folder beside the output. The leading dot is what keeps
  Unity from importing any of it.

The merged manifest records every take with its source BVH, frame range and starting root
position, plus anything that was skipped. It is the only record of which take came from which
file once the run is over.

### Two flags worth understanding

`--simplify` sets FBX keyframe reduction, passed to every shard. On dataset-2, `1` costs about
0.2 degrees and takes the output from roughly 1.7 GB to 0.45 GB. Unity's model importer then
defaults to `KeyframeReduction` on top of that, so set Anim. Compression to Off if one lossy pass
is enough.

`--keep-capture-position` matters when a dataset varies the rest offset from clip to clip. A BVH
stores each joint's rest `OFFSET` beside absolute position channels, and the batch transplants
only the action, so a clip whose rest offset differs from the setup's is re-framed by that
difference. Bandai-Namco dataset-2 gives the same actor three Hips offsets up to 8.8 m apart
while the positions it actually recorded all sit within centimetres of the origin — so without
this flag two thirds of its clips end up metres from where they were captured. LAFAN does not
need it, and it is off by default so that pipeline is unchanged.

### When a clip fails

`--continue-on-error` (which the driver always passes) logs the clip, carries on, and still
exports; the run exits 2 instead of 0 and names the skipped clips in the manifest. Without it a
single bad clip late in a run discards everything before it, since the export happens only at the
end. Exit 1 still means the setup itself is unusable and nothing was written.

## What a setup blend must contain

One blend per source skeleton. The script finds the pieces by role, not by name, so only the
wiring matters:

| Role | How it is found | What it is |
| --- | --- | --- |
| Source skeleton | whatever the cleaned skeleton's constraints point at | the raw BVH rest pose, as `import_anim.bvh` produces it |
| Cleaned skeleton | the one armature carrying bone constraints | same bone names and lengths, rest pose re-oriented into a usable T-pose, constrained to the source |
| Model rig | Rokoko's target armature (a scene property) | the shared MoSynth target rig |

Plus a Rokoko bone list mapping cleaned-skeleton bones to model bones. All the Rokoko
settings — source, target, auto-scale, pose mode and the bone list — are **scene properties,
so they are saved in the blend**; nothing is duplicated into a config file.

## When the source has no rotations

A dataset may ship joint *positions* and nothing else — the Edinburgh Locomotion database is two
`.npz` files of 21 points per frame. `edinburgh_npz_to_bvh.py` solves that into BVH, and because it
authors the file it also picks the rest pose and the bone names, which changes what the setup blend
has to do:

- The rest pose is written as a genuine T-pose with the target rig's own bone names, so stage one
  has nothing to clean. Its cleaned skeleton is a plain duplicate wired through with
  `COPY_ROTATION` and no IK, and the Rokoko map is 21 identity rows.
- `make_edinburgh_setup.py` composes that blend from one converted BVH plus the shared rig, and
  verifies it by re-opening the saved file and resolving it with `batch_retarget`'s own code.
- **Every file the converter writes must declare the same skeleton**, since
  `check_skeleton_matches` rejects a millimetre of disagreement. It fits one skeleton across every
  clip of every split in a single run, which is why both `.npz` are passed at once.
- **The rest pose must be the actor's own posture, not canonical axes.** A clean T-pose is right
  for the limbs and wrong for the torso and the root: this actor's pelvis-up axis sits 17 degrees
  ahead of vertical, and calling it vertical added that lean to every retargeted frame. Torso pitch
  went from 19.3 degrees to 7.3 against the target rig's 2-degree rest once the root and spine took
  their rest directions from the data. Limb rows in `SKELETON` keep a canonical `rest_dir`; torso
  rows set it to `None`.
- **The spine bend is scaled down in the setup blend, by `SPINE_LEAN_SCALE`.** Edinburgh's
  hip-to-neck axis reads about 2.4x the actor's real lean, so copying it verbatim hunches the
  character. Nearly all of it sits in the root segment, which also carries heading, so that bone's
  rotation is split: local Y (the pelvis-up axis) copied whole to keep the turns, X and Z damped.
  A scale rather than an offset, because the lean varies per clip and one subtracted angle can only
  ever suit one of them.
- **The root's euler order is `YXZ`, deliberately.** Damping its lean per axis makes Blender
  decompose to euler, and an euler order gimbal-locks on its middle axis; the BVH's `ZYX` puts that
  on Y, the heading axis, so a turning clip snapped 94 degrees. If a retarget is ever jumpier than
  the BVH it came from, the setup is adding it -- compare the two before touching the converter.
- **Clavicles and hip joints inherit their parent's roll** (`PARENT_ROLL` in `SKELETON`). Each
  reaches one marker, so its roll is a convention rather than a measurement; borrowing a
  neighbour's direction let them twist 28-38 degrees against the body part they hang off. This
  changes roll only, so joint positions and the `OFFSET` block are untouched -- which is what lets
  a regenerated dataset drop under a hand-authored rest pose without invalidating it.
- **Watch the movement, not the joint positions.** Positions fix a bone's direction and say nothing
  about its roll, so `--verify` reports the worst frame-to-frame *whole-bone* step beside the
  position error, with how much of it was twist. Every visible artefact in this conversion was a roll artefact, and none of them moved a
  joint by a millimetre. `edinburgh_solve_preview.py` builds a blend with the solved skeleton
  overlaid on the source points, with bone axes shown, for judging one by eye.
- Expect worse foot contact than a rotational source. Edinburgh's own released root trajectory
  already slides a planted toe 5.5 cm/s, and retargeting onto an actor of different proportions
  roughly doubles it: 10.8 cm/s against LAFAN's 1.9 cm/s measured the same way.

## Authoring a rest pose with live feedback

The cleaned skeleton's rest pose is the calibration pose, and it is the one thing in a setup that
has to be judged by eye. `make_edinburgh_setup.py --live --no-damping --clips <bvh...>` builds a
blend for exactly that:

```
blender --background --python Tools/Retargeting/make_edinburgh_setup.py -- \
    --bvh Assets/LFS/Animation/Edinburgh/bvh/edin_locomotion_0000.bvh \
    --no-damping --live --clips <a few bvh> \
    --out Assets/LFS/Retargeting/edinburgh_rest_authoring.blend
```

Edit `edinburgh_cleaned`'s rest in Edit Mode and the character follows immediately. The rig is
Rokoko's own: one helper bone per mapped bone, sitting at the model bone's rest but parented to the
cleaned bone, so Blender evaluates `cleaned_world @ cleaned_rest^-1 @ helper_rest` -- the delta
Rokoko bakes. Rokoko builds this, bakes through it and deletes it; keeping it makes the retarget
live, and because the rest matrices are read fresh every evaluation, an edit shows up at once.

- **Helpers live in a hidden `Retarget` bone collection and are disconnected.** Both matter. A
  connected child moves with its parent in Edit Mode, so a helper dragged along by the bone being
  edited would show no change at all.
- **Switch clips** by unmuting a different NLA track on `edinburgh_source`, and note that the
  batch mutes them all before a run: an assigned action evaluates on top of the NLA stack rather
  than instead of it, so a take left parked here would blend into every clip of the run.
- **`RT_ground` is where the character's height above the floor is set.** It is a helper bone
  parented to the bridge's `Hips` that the model's `Hips` copies its location from, so the whole
  character rides on it and the target rig is never edited. It rests along `+Y` with zero roll, so
  its local axes are the world's and its Z field reads in world metres, and it does not inherit
  rotation, so the offset stays vertical instead of tipping with the pelvis. `use_offset` on the
  Copy Location constraint is not the same knob and does not work: it adds the bone's own 93 cm
  rest height.
- **A live blend is a run's setup, not just an editing aid.** `direct_retarget.py` retargets
  straight through this rig, so a preview cannot disagree with the output.
- Rokoko cannot be trusted to retarget from a constraint-driven rig in the first place: with
  `use_pose = REST` it clears a bone's pose by zeroing `rotation_quaternion`, which a constraint
  re-applies on the next evaluation, so it builds its helpers against whatever frame the playhead
  is on. That is why the batch bakes stage one and strips the constraints before stage two.

## Retargeting without Rokoko

A setup built with `--live` already holds the rig Rokoko would build and throw away, so
`direct_retarget.py` runs the whole chain as one bake:

```
blender --background Assets/LFS/Retargeting/edinburgh_rest_authoring.blend \
    --python Tools/Retargeting/direct_retarget.py -- \
    --bvh-dir Assets/LFS/Animation/Edinburgh/bvh --pattern "edin_test_*.bvh" \
    --out Assets/LFS/Animation/LafanCorrected/edinburgh/edin_test.fbx
```

Same command line and the same FBX as `batch_retarget.py`, which it imports as a library for
everything except the retarget itself. It finds the rig structurally — the one armature carrying
`RT_*` helper bones — and reads the bone map off the model's own constraints, so the wiring *is*
the mapping and the two cannot drift apart. A setup with no helper rig is refused rather than
retargeted badly; `batch_retarget.py` still owns those.

What it buys:

- **The rest-pose drift is gone, not merely small.** Rokoko applies and un-applies the target's
  object transform on every clip, which is identity only in exact arithmetic; that is the drift the
  guard watches and the reason shard size is capped. With no such mechanism here, 1855 clips across
  9 shards ended at `0.00e+00 m`, against about `5.2e-05` per 220-clip Bandai shard.
- **Roughly 3x faster**, because one bake of the model replaces stage one's bake, the
  constraint-free proxy, and Rokoko's own duplicate-and-bake.
- **The authoring blend and the batch run the same code**, so a rest pose judged by eye is the rest
  pose that ships.
- Nothing depends on the addon being installed.

Measured against a real Rokoko bake of the same clip and rest pose, the two agree to **0.016
degrees mean, 0.079 max**.

**Takes already on the model are muted before each clip.** Constraints override an action
underneath them, so a stacked take cannot disturb a *mapped* bone — but an unmapped one (here the
target's `Spine2`) has no constraint, so it would be posed by whatever the previous clips left on
the stack, and everything hanging off it moves with it. The error compounds clip by clip.

## Setting one up for a new dataset

1. Import one representative BVH with the same settings the batch uses (`global_scale=0.01`,
   `rotate_mode="NATIVE"`, `axis_forward="Z"`, `axis_up="Y"`). This armature is the source
   skeleton, and every later clip's action is transplanted onto its rest pose.
2. Duplicate it — never build an object over the armature data, see the rotation-mode note in
   the wiki — and **clean the rest pose** into a T-pose *of the actor*: each bone's direction from
   the model rig, its roll measured, and each joint at the source's own offset re-expressed
   through the corrected parent frame. This is the load-bearing step: Rokoko retargets by the
   source's world rotation delta from its own rest, so this rest *is* the calibration pose, and a
   raw BVH rest with the limbs along a single axis gives unusable results.

   Do **not** copy the model's roll along with its direction. The two rigs disagree about which
   local axis is sideways, and copying it yaws the retargeted torso by that angle — 90° for
   Bandai-Namco. Measure it instead: the twist about each bone that carries the source's local
   child directions onto the model's, averaged over the children.

   Verify geometrically, never by comparing bone frames between the rigs — that comparison reads
   a perfect 0.000° on a 90°-yawed torso. Check limb directions in a body frame, mirror every
   left/right rest pair, and confirm the constrained cleaned joints land on the source's.
3. Constrain the cleaned skeleton to the source: `COPY_ROTATION` on every mapped bone and
   `COPY_LOCATION` on the root. Add `IK` on the limbs (chain length 3 on the feet and forearms,
   targeting the source's toe and hand) only if cleaning changed the chain lengths, as LAFAN's
   setup does; where the cleaned skeleton is a rotation-only copy, rotations alone already
   reproduce the source exactly and IK can only add solver noise.
4. Under Rokoko > Retargeting, set the cleaned skeleton as source and the model rig as target,
   build the bone list and **check every row**.
5. Save. Do not bake anything by hand — the script does that per clip.

The script refuses to run on a blend that fails any of this, naming what it expected.

## What a run does per clip

1. Imports the BVH and keeps only its action, transplanting it onto the source skeleton. The
   setup's rest pose is authoritative; a raw import differs from it by a rigid transform.
2. Bakes the constrained cleaned skeleton (stage one).
3. Duplicates the cleaned skeleton, strips its constraints, and runs Rokoko onto the model rig
   (stage two).
4. Pushes the result to an NLA track named after the clip.

`direct_retarget.py` collapses 2 and 3 into a single bake of the model through the live rig.

Then it exports every track as an FBX take, keeping the setup's `T-Pose` track as a bind-pose
reference.

## Notes

- A BVH whose bone names or bone lengths do not match the setup is rejected: it belongs to
  another subject or dataset and needs its own blend.
- The model rig's rest pose is checked for drift after every clip. Rokoko applies and
  un-applies the target's object transform on each retarget, which is only identity in exact
  arithmetic.
- `--simplify` sets FBX keyframe reduction. It defaults to 0 (every key kept). On a
  7840-frame LAFAN clip, 1.0 costs about 0.2 degrees of accuracy and saves roughly two thirds
  of the file size.
- Expect around 2.5 minutes and ~35 MB per 7840-frame clip.

Downstream, the exported takes become `AnnotatedAnimationClip` assets; see
`openwiki/animation-tools/animation-sources.md`.
