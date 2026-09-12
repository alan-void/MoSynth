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
