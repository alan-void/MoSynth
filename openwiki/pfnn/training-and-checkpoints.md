---
type: Architecture Guide
title: Training a PFNN, and what a checkpoint holds
description: From a generated pose database to a trained phase-functioned network — which bones it covers and who decides, how the input and output vectors are packed, and why the checkpoint stores names rather than indices.
tags: [pfnn, neural, training, on-disk-formats, config]
sources:
  - id: openwiki-source-2cab5ee24d405d8240d13641
    resource: repo://Assets/AnimationTools/Runtime/Pose/SimulationFrame.cs
  - id: openwiki-source-e3854efddedde038431e9a6b
    resource: repo://Assets/Pfnn/Editor/PfnnAgreementCheck.cs
  - id: openwiki-source-bd27f9f1ae47d93c1df22ffd
    resource: repo://Assets/Pfnn/Editor/PfnnConfigEditor.cs
  - id: openwiki-source-d1d9910b321c79bf0cb42361
    resource: repo://Assets/Pfnn/Editor/PfnnDefaultBoneSelection.cs
  - id: openwiki-source-864bdf0b3b6700cc57dad145
    resource: repo://Assets/Pfnn/PfnnBoneSelection.cs
  - id: openwiki-source-4b00342520efef49db73b087
    resource: repo://Assets/Pfnn/PfnnConfig.cs
  - id: openwiki-source-a1ce493914db289593686473
    resource: repo://Assets/Pfnn/PfnnStage.cs
  - id: openwiki-source-4e55c713faf28eac207e9a7c
    resource: repo://Python/pfnn_dataset.py
  - id: openwiki-source-9da779382ccd33c31524e98d
    resource: repo://Python/pfnn_io.py
  - id: openwiki-source-48577afa854f12ec4dcdd35c
    resource: repo://Python/pfnn_model.py
  - id: openwiki-source-28e2d896c07a4fa8c47e273d
    resource: repo://Python/pfnn_trainer.py
  - id: openwiki-source-1110a5319fdf997c0acf8f29
    resource: repo://Python/pose_set_importer.py
  - id: openwiki-source-33ee0478665cfbc83218ffbe
    resource: repo://Python/tests/test_pfnn_model.py
  - id: openwiki-source-58d35cd9c30979b2ff43e031
    resource: repo://Python/training_data.py
generated: {by: "claude-code", at: "2026-09-03T11:06:29.275Z"}
---

# Training a PFNN, and what a checkpoint holds

A `PfnnConfig` owns the whole chain: the clips, the bones the model covers, the shape of the
network, and the hyperparameters it is fitted with. It generates its own pose database rather than
borrowing a `MotionMatchingData`, for the reason `MotionFieldConfig` does — a synthesis method should
not have to be a Motion Matching asset to have poses.

```
clips ──► Generate Pose Database ──► .mmpose
                                        │
                                        ▼
                         training_data.build_training_set
                                        │
                              pfnn_dataset.build_vectors
                                        │
                                        ▼
                            Train PFNN ──► .pfnn.npz
```

Two flags on the config say whether each artefact still matches it. Any inspector edit clears both;
regenerating the database clears the trained flag, because it changes the frames the network was
fitted to. The change check cannot tell which field moved, so it over-flags — a needless retrain
costs a minute, while a model quietly running against a config it was not trained for costs a
character that moves worse than it should for no visible reason.

## Which bones the model predicts

The selection is **authored on the config**, as a sparse list of excluded bone names drawn by the
inspector as the skeleton hierarchy with a toggle per bone. Absent from the list means predicted, so
the list stays short, and it is keyed by name rather than index so a joint that moves in the
hierarchy takes its setting with it. This is the same shape and treatment `MotionFieldConfig` gives
its per-bone weights.

It is data, not a rule in the code. A name heuristic would be invisible, unauditable per rig, and
silently wrong on a rig that names things differently. Two footer buttons supply the useful defaults
in one click — **Exclude Fingers And Leaves** and **Predict All** — and the heuristic behind the
first lives only in that button, never in the training or inference path. Pressing it writes the
answer into the asset, where it can be read and corrected.

**Excluding a bone excludes its whole subtree.** That is not a convenience: the network predicts
rotations, and a rotation needs its parent's frame to be applied in, so a kept bone whose parent is
dropped has nowhere to go. The editor's toggle enforces it, and `select_bones` refuses a selection
that is not closed under parent rather than guessing what was meant. A name the skeleton does not
have is likewise an error, not a silent no-op: a misspelt exclusion would quietly train a different
model than the config describes, and the first symptom would be a checkpoint that no longer loads.

On the demo rig the default keeps **24 of 85 bones** — the legs, spine, arms to the hand, and
neck — holding back 58 finger joints and 3 leaf tips. Fingers barely move in locomotion, so a model
that predicts them spends most of its capacity learning constants.

## What a sample is

For a query frame `i` that has a successor in the same clip, the input describes frame `i` and the
target describes frame `i + 1`.

| Input | floats |
| --- | --- |
| trajectory positions, `(x, z)` per window sample | 2·T |
| trajectory directions, unit facings per sample | 2·T |
| joint positions in the character frame | 3·B |
| joint velocities in the character frame | 3·B |
| foot contacts | 2 |

| Target | floats |
| --- | --- |
| joint rotations, in the two-axis 6D form | 6·B |
| joint velocities | 3·B |
| root height | 1 |
| root delta `dx, dz, dyaw`, in frame `i`'s space | 3 |
| phase increment | 1 |
| foot contacts | 2 |

With the default window — thirteen samples from −30 to +30 frames — and 24 bones, that is **198 in,
223 out**.

Three details worth keeping:

- **The root delta is read out of the trajectory sampler**, not re-derived from `root_velocity`.
  `trajectory_window([1])` already answers "where is the character one frame from now, in this
  frame's space", and it is the definition two independent checks were run against. One definition
  of where the character goes next is better than two that agree today.
- **Joint positions are not a target.** They come back from forward kinematics over the predicted
  rotations, which removes 72 outputs and guarantees the pose respects the rig's bone lengths.
- **Frames with no measurable gait cycle are dropped.** `TrainingSet.usable()` checks only the
  matching feature vector; a phase-functioned network additionally needs `phase_rate != 0`, which is
  how [gait phase](../animation-tools/neural-synthesis.md) marks a stretch it could not find a stride
  in. Note what that now excludes and what it does not: a stretch with no footfalls that the
  character *moved* through is a missed contact and is dropped, while one the character *stood*
  through carries a real rate and is kept, so the network sees stationary poses across the whole
  cycle. Standing is trained; unexplained is not.
- **There is no gait label.** The paper feeds a one-hot vector — stand, walk, jog, crouch, jump —
  authored per clip. Nothing here is labelled that way. The contacts the phase was built from carry
  the part of that signal a locomotion model can use, and a stationary character is already legible
  in the input as a trajectory window of zeros. The one-hot earns its place mainly when *walk* and
  *jog* have to be told apart at the same speed, which is not something this dataset models.

## The network

Three layers, ELU activations, dropout between them. Each layer holds four sets of weights, and a
cubic Catmull-Rom spline over the gait phase decides how much of each to use — wrapped, so phase 0
and phase 2π give the same network rather than merely a similar one.

**The outputs are blended, not the weights.** Because the spline is a linear combination of its
control points and a layer is linear in its weights,

```
(Σ wₖ Aₖ) x + Σ wₖ bₖ  ==  Σ wₖ (Aₖ x + bₖ)
```

The right-hand side is four dense matmuls and a weighted sum. The left has to build a distinct
weight matrix for every item in the batch first — millions of floats per sample for a wide layer,
and what makes naive PFNN implementations slow. A unit test asserts the two agree.

The default hidden width is **256, not the paper's 512**. On a database of a few thousand frames,
512 is 1.9 M parameters against 7.8 k samples; 256 is 0.70 M. The paper trained on hours of motion.
Dropout is expressed as PyTorch's *drop* probability, so the paper's quoted 0.7 retention appears
here as 0.3.

Validation holds out a **contiguous tail**, not a random subset. Neighbouring frames of an animation
are nearly the same pose, so a random split puts near-duplicates of the validation set into training
and reports a number that measures nothing.

## The checkpoint

`<name>.pfnn.npz` holds the parameters, the normalisation the vectors were packed with, and the
description of that packing — the bone names and the trajectory horizons — because a model fed a
differently shaped input does not throw.

**Bone names, never bone indices.** An index is only meaningful against one database, so a stored
one goes stale the moment a rig gains a joint; a name is checked against the live skeleton at load,
which is the same thing `.mmpose` does with its skeleton block. The stage additionally refuses a
checkpoint whose bones resolve *out of order* against the rig, since matching names on a different
hierarchy is exactly the case nothing else would catch.

The file is [unversioned](../animation-tools/on-disk-formats.md), by the project's standing
decision, and loads with `allow_pickle=False` — reading a checkpoint should never mean executing
what is inside it. A missing or unreadable one returns `None` with a logged reason rather than
raising, because callers run inside `Py.GIL()` from Unity, where an exception arrives as an opaque
managed error.

## More than one config

A `PfnnConfig` derives both its `.mmpose` and its `.pfnn.npz` from its own asset name, under
`StreamingAssets/Pfnn/<config name>/`. Two configs over different datasets and different rigs
therefore coexist without either overwriting the other's artefacts, which is how a new dataset gets
trained and compared without destroying the checkpoint you already have.

The one thing that changes once a second config exists: `MoSynth/Pfnn/Check Training Agreement`
auto-picks a config only when the project holds exactly one, so from then on it wants one selected
in the Project window.

`PfnnBandaiWalk` is the second: 93 Bandai-Namco takes on the corrected rig — 24 `walk_normal` plus
69 `walk-turn-left`/`walk-turn-right` — giving 17043 poses, 22 predicted bones once the five `_end`
leaves are excluded, 186 inputs to 205 outputs, 665k parameters. It fits in about 270 s on a GPU to
a train/validation loss of 0.108/0.065, and its rollout holds 1.07 m/s against the database's 1.10.

Note the shape of the frame budget with many short clips rather than one long one: of 17043 poses,
16950 have a successor within their own clip, and on this dataset every one of those also had a
measurable gait cycle — none was dropped for `phase_rate == 0`. That count predates the standing
rule and the move of phase into the database, so it will have shifted; re-measure it rather than
quoting it. Clip boundaries are respected
everywhere that matters — `trajectory_window` clamps its offsets to the containing clip, so a window
near an edge repeats its last real sample instead of reading across a cut.

## The dataset has to contain the motion you want

The turns were not an enrichment; they were a correction. The first `PfnnBandaiWalk` held only the
24 `walk_normal` takes, and those walk in straight lines: measured as net heading change from a
clip's first frame to its last, they average **−2.4°**. The root-delta target's yaw component is
read out of the trajectory sampler one frame ahead, so across that entire dataset it carried
essentially no turning signal. The resulting model walked convincingly and could not turn — which is
not a tuning failure, and no amount of hidden units or epochs would have addressed it.

The 69 turn takes average about **+96°** (left) and **−101°** (right), which is the signal that was
missing. Beware the obvious diagnostic here: per-frame yaw *rate* does **not** separate the two sets
— every clip in the combined database averages over 20°/s, because the pelvis swings back and forth
with each stride and that oscillation swamps the path's actual curvature. Net heading change over a
whole clip is the measure that distinguishes a straight walk from a turn.

Two other things came free with the larger set. Validation loss now sits *below* training loss,
which is the expected direction when dropout is active during training only; on the 5524-sample
walk-only set it sat above, the signature of a model with more capacity than data. And the
validation split is a contiguous **tail**, so appending 69 turn clips to 24 walk clips would have
made the held-out set pure turning and the number meaningless — the config's clip list is shuffled
with a fixed seed so the tail is representative.

## Checking the two sides still agree

`MoSynth/Pfnn/Check Training Agreement` runs the same frames of the same database through
`CharacterSpacePose.Extract` in C# and `training_data.build_training_set` in Python and reports how
far apart they are. It is a diagnostic rather than a test, because it needs a generated database and
a working interpreter — neither of which the edit-mode suites may assume.

On the demo database it reports positions agreeing to **1.8 × 10⁻⁶ m** and rotations to **0.079°**
at worst, 0.0026° on average; on the shallower corrected rig, 1.2 × 10⁻⁶ m and 0.074°. The worst
bones are at depth 10 and 12 of a maximum 13, which is what
identifies the residual as float32-versus-float64 accumulation down the chain rather than a
disagreement about the definition — a real one would not care how deep a joint sits. That is why the
report names the depth. Rates are deliberately not compared: the two sides differ there by
construction, and the difference is already documented.

What it cannot catch is the two sides agreeing on the *wrong* frame. Both derive the character
frame's forward axis from the same rest pose in the same `.mmpose`, so a rig exported holding a
frame of motion rather than its bind pose yields a rotated frame that both halves reproduce exactly:
the check passes at 10⁻⁶ m while every pose in the database is measured against the wrong facing.
The symptom shows up only in Unity, where the pose is written onto a live rig that does sit at a real
bind. See [skeletons and rig binding](../animation-tools/skeletons-and-rig-binding.md).

## Running it outside Unity

Both halves work from a shell, which is the quickest way to judge a model before wiring a character:

```bash
python Python/pfnn_trainer.py <database-folder> <name> --out out.pfnn.npz --epochs 150 \
    --exclude Model:LeftHandIndex1 Model:LeftHandIndex2 ...

python Python/pfnn_runtime.py out.pfnn.npz --database <database-folder> --name <name> --rollout 300
```

The rollout runs the model against its own predictions for 300 frames, driven by the database's own
path. It is the cheapest honest test of a checkpoint: a model that has learned nothing still scores
a plausible per-frame loss, because a pose barely changes in a thirtieth of a second, but it cannot
survive being fed its own output for ten seconds.
