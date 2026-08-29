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

1. Import one representative BVH. This armature is the source skeleton.
2. Duplicate it and **clean the rest pose**: keep the bone lengths, re-orient the bones into a
   T-pose. This is the load-bearing step — Rokoko produces unusable results from a raw BVH
   rest pose, where the limbs run along a single axis.
3. Constrain the cleaned skeleton to the source: `COPY_ROTATION` on every bone,
   `COPY_LOCATION` on the root, and `IK` on the limbs (in the LAFAN setup, chain length 3 on
   the feet and forearms, targeting the source's toe and hand) so end effectors land right
   despite proportion differences.
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
