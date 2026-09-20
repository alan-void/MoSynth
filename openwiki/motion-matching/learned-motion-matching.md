---
type: Reference
title: Learned motion matching
description: How MoSynth replaces the database search with learned networks, the staging that makes each replacement measurable, and the parked Barracuda experiment that was removed to make room for it.
tags: [neural, training-data, synthesis-method, roadmap]
sources:
  - id: openwiki-source-6ae3fca0d1cd346c5b593ebf
    resource: repo://Assets/Lmm/LmmConfig.cs
  - id: openwiki-source-b6e201bd6bd5f97d52502753
    resource: repo://Assets/Lmm/LmmStage.cs
  - id: openwiki-source-66118a33a3b24581a4e80f0b
    resource: repo://Assets/MotionMatching/Runtime/Core/IMotionMatchingDataProvider.cs
  - id: openwiki-source-a683fec09e82e53751b26426
    resource: repo://Assets/MotionMatching/Runtime/Core/MotionMatchingQuery.cs
  - id: openwiki-source-f84c8bceda0edfaac6926af8
    resource: repo://Assets/MotionMatching/Runtime/Core/MotionMatchingStage.cs
  - id: openwiki-source-1388d8aff87ba11326b6f618
    resource: repo://Assets/MotionMatching/Runtime/Core/MotionSynthesisComponentExtensions.cs
  - id: openwiki-source-a1ce493914db289593686473
    resource: repo://Assets/Pfnn/PfnnStage.cs
  - id: openwiki-source-5d4680328f7da95f88649f31
    resource: repo://Packages/manifest.json
  - id: openwiki-source-b7a0fd38fcd115e75814f58b
    resource: repo://Python/feature_set_importer.py
  - id: openwiki-source-339aaba5b079decffa731275
    resource: repo://Python/lmm_dataset.py
  - id: openwiki-source-db8dcca3e2e1d55acdef19c4
    resource: repo://Python/lmm_fk.py
  - id: openwiki-source-587e1d3447a096780933e80b
    resource: repo://Python/lmm_io.py
  - id: openwiki-source-b8a5877cdf623adbefc700df
    resource: repo://Python/lmm_model.py
  - id: openwiki-source-787f833ac52b3ea674bfc67b
    resource: repo://Python/lmm_runtime.py
  - id: openwiki-source-be9fd23c32d4b00057034c1a
    resource: repo://Python/lmm_trainer.py
  - id: openwiki-source-58d35cd9c30979b2ff43e031
    resource: repo://Python/training_data.py
generated: {by: "claude-code", at: "2026-09-19T18:12:18.762Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-09-19T18:12:18.762Z
---

# Learned motion matching

Holden et al.'s *Learned Motion Matching* replaces the animation database and its brute-force
search with three small networks:

| Network | Role |
| --- | --- |
| **decompressor** | reconstructs a full pose from the latent state |
| **stepper** | advances the latent state autoregressively, so the search does not run every frame |
| **projector** | stands in for the nearest-neighbour lookup — query vector to the nearest database point, in latent space |

A fourth, the **compressor**, exists only at training time: it produces the latents the other three
are defined over, and is kept in the checkpoint for diagnostics.

The method arrives in three phases, and **phase A is implemented**: the compressor and decompressor
are trained, and `LmmStage` runs the decompressor. The stepper and the projector are not written
yet, and `LmmStage` refuses a mode that would need them.

The parts shared with a phase-functioned network are on
[neural synthesis readiness](../animation-tools/neural-synthesis.md); this page is the LMM-specific
half.

## The idea, and why the staging matters

Motion matching is two things: a **search** that decides which frame the character should be
playing, and a **database** that says what that frame looks like. LMM replaces both with networks,
but replacing them at once makes the result impossible to account for — a character that moves badly
could be failing at either job.

So each phase replaces one piece and keeps the rest, and the ablations stay in the shipped code as
`LmmMode` rather than being removed once the next phase lands:

| Mode | Search | Playback between searches | Pose |
| --- | --- | --- | --- |
| `DecompressorOnly` | the real database search | walks the database | **decompressor** |
| `Stepper` | the real database search | **stepper** | decompressor |
| `Full` | **projector** | stepper | decompressor |

In `DecompressorOnly` the character is playing exactly the frames `MotionMatchingStage` would play,
against the same database, using the same query built by the same code. The only difference is where
the pose comes from. That is a measurement of reconstruction quality and nothing else.

The point of keeping the modes is that LMM's headline benefit — dropping the database — is only
*realised* by shipping `Full`. Until then the runtime holds both the database and the latents, so
the memory saving is something the ablations let you **measure**, not something the stage delivers.

## The four vectors

Everything turns on how `X`, `Y`, `Q` and `Z` relate.

- **`X` is the matching feature vector**, read straight out of the `.mmfeatures` the classic matcher
  searches — 33 floats on the shipping `Lafan_corrected` database. It is not rebuilt for LMM, and
  that is deliberate: the method is meant to replace the *search over* that query, so a learned
  matcher answering a differently shaped question would not be comparable to the matcher it
  replaces.
- **`Y` is the pose as a rig is posed**: every predicted bone's **joint-local** 6D rotation, its
  linear and angular rate, the root's height, the character frame's own travel and turn, and the two
  foot contacts. That is `12 × bones + 7` floats — 331 over the 27-bone rig.
- **`Q` is the same pose after forward kinematics**: where every joint ended up in the character
  frame, as a position and a 6D rotation, `9 × bones` — 243 floats. **Nothing predicts `Q`.** It is
  derived from `Y` on both sides of the loss, and that derivation is the point: it makes an error
  deep in a chain cost what it is worth rather than what one joint angle is worth.
- **`Z` is a 32-float latent**, which the decompressor takes *beside* `X` to produce `Y`.

`Y` carries no joint positions: they are forward kinematics of what it does carry, so predicting
them as well would let the network describe a pose its own rotations cannot reach.

Two things here depart from the paper's `Y`, both forced by what this runtime consumes. It carries
**no joint-local translations** — the rig has fixed bone lengths and `CharacterSpacePose.Apply`
writes the rest offsets regardless, so a predicted translation would be discarded at the one place
it could be used. And its **rate channels stay in the character frame** rather than being expressed
joint-locally, because that is the convention `Apply` reads and what `training_data` already
derives.

### The latent spans two spaces, not two frames

The compressor takes `[Y Q]` — one frame, measured two ways. That is Algorithm 1 of the paper
(`Z ← C([Y Q]ᵀ)`), and the paper's reason is that the compressor "was able to copy features directly
to the latent space if it found them useful".

This corrects an earlier reading on this page. The reference implementation's compressor input is
exactly twice its pose width, which looked like two consecutive frames; the vendored archive settles
it the other way by shipping `YtxyData.txt` and `QtxyData.txt` side by side, differing in 1,105 of
1,155 entries at the same frame. **So nothing about the latent is temporal by construction.** What
leaves it steppable is the velocity regulariser in the loss, and that is now the only thing doing
that job — which is why it is measured directly rather than assumed.

The pair rule survives for a different reason: three terms of the loss are differences across
adjacent frames. **A frame has a latent only when its successor is in the same clip.** The last frame
of every clip has none, and a training pair additionally loses the second-to-last. Differencing
across a clip boundary would score a jump cut as motion.

## Training

One stage in phase A: the compressor and decompressor, fitted jointly, under the paper's
**Algorithm 1**.

```
Q  ← FK(Y)                     Z  ← C([Y Q])
Ŷ  ← D([X Z])                  Q̃  ← FK(Ŷ)

L_loc  = w_loc  · |Y − Ŷ|₁                L_chr  = w_chr  · |Q − Q̃|₁
L_lvel = w_lvel · |ΔY/δt − ΔŶ/δt|₁        L_cvel = w_cvel · |ΔQ/δt − ΔQ̃/δt|₁
L_sreg = 0.1 · mean|Z|   L_lreg = 0.1 · mean(Z²)   L_vreg = 0.01 · mean|ΔZ/δt|
```

Four things in there are load-bearing.

**Every pose is scored twice, in two spaces.** This is what the paper is for: a naive per-joint loss
gives "jittery, low quality motion", because it cannot see that a small error at the hip is a large
one at the hand. `L_chr` is the term that can, and it needs forward kinematics *inside* the training
step — which is what `lmm_fk` provides, differentiably, in the same definition the offline report
and `LmmStage` pose with.

**L1, not mean squared error.** MSE regresses to the mean of the plausible poses at a given
`(X, Z)`, and the mean of two plausible poses is generally not a pose. L1 regresses to the median,
which is. This is a deliberate departure from `pfnn_trainer`'s weighted MSE, and it is the
difference between a crisp pose and a mushy one.

**The loss is computed on denormalised quantities**, in metres, radians and seconds. That is the
only reading under which the weights mean anything, because they trade a metre of position error
against a radian of angle:

| term | weights |
| --- | --- |
| `L_loc` | rotations 10.0 · velocities 10.0 · angular velocities 1.25 · root height 75.0 · root velocity 2.0 · root yaw rate 2.0 · contacts 2.0 |
| `L_chr` | positions 15.0 · rotations 5.0 |
| `L_lvel` | Δrotations 1.75 |
| `L_cvel` | Δpositions 2.0 · Δrotations 0.75 |

**The paper states none of these numbers** — only "roughly equal weight to all pose-based losses and
a small weighting to regularization losses". They come from the author's released training code.
`w_vreg` looks tiny beside the others because it multiplies a *per-second* derivative: a factor of
sixty against the per-frame delta it would be easy to write instead.

**The velocity regulariser is what makes phase B possible at all.** Nothing else makes `Z` continuous
in time, and without it phase A looks perfect while the stepper is unlearnable — a failure you
discover a phase late.

### Normalisation: once for `X`, per-block for `Y` and `Q`

`X` is fed to the decompressor **unnormalised**. Unity's `FeatureSet` already subtracts a mean and
divides by a per-feature standard deviation when it bakes the `.mmfeatures`, which is the paper's
own recipe — so a second standardisation here would divide out a spread that has already been
divided out. The checkpoint still carries `x_mean` and `x_std`; the trainer writes an identity into
them, and the runtime applies whatever it is given.

`Y` and `Q` get a mean per float but **one standard deviation per block**. Dividing a 3-vector by
three different numbers warps it: the axes stop being comparable, an L1 error means something
different per axis, and the near-isotropy of a joint's motion is destroyed on the way in. The paper
scales "each feature (e.g. left foot position)" by a single number, and this follows it at block
granularity.

### Optimiser

AdamW with `amsgrad`, learning rate `1e-3`, weight decay `1e-3`, multiplied by 0.99 every 1,000
steps. Counted in **steps rather than epochs**, because that is the paper's schedule and because an
epoch means something different on every database size. Training also takes a wall-clock limit: a
run that has to fit a budget should be cut by the clock rather than by guessing an iteration count.

The parameters written to the checkpoint are the ones that scored best on the held-out tail, not the
last ones — on a small database the held-out curve turns back up well before the end.

One departure from the paper: batch **256** on a GPU rather than its 32 on a CPU. The paper's own
note is that its small networks made CPU training more efficient, which stops holding at this batch
size.

**Do not raise it further to buy throughput.** The step is bound by how many CUDA kernels it
launches rather than by their size, so a microbenchmark makes a bigger batch look free — and it is
not. Run at a fixed 400 s wall clock on Edinburgh, batch 1024 was about 17% slower per step and
reached a *worse* held-out loss than 256 (2.7956 against 2.7402) despite seeing four times the
samples. More optimiser steps beat bigger ones here, which is the direction the paper's batch of 32
already points.

### Whether the latent is carrying anything

Three numbers get reported and only one of them is a measurement of the thing that matters. The
other two are here because they look like measurements of it, and are not.

#### The one that counts: can the latent's step be predicted?

Phase B's stepper advances `(X, Z)` one frame at a time, so what it needs is for `Z' − Z` to be a
*function* of `(X, Z)`. `latent_step_predictability` asks exactly that: the held-out R² of a ridge
regression from `(X, Z)` to the next step. It is the **linear lower bound** on what a 512×2 network
could learn — a nonlinear stepper will do better, but not arbitrarily better.

Measured on Edinburgh: **R² = +0.173.** A linear model explains 17% of the step's variance. That is
a weak foundation, and it is the honest advance warning about phase B.

This is the only one of the three a latent cannot flatter by going quiet, because R² is scored
against that latent's own variance — shrink the steps and the target shrinks with them.

#### The two that mislead

**Lag-1 autocorrelation** rewards the *worse* model. An undertrained latent carries less
information, so it moves less, so it scores higher. Measured here: the 2,000-iteration model on the
small database scored 0.965; the fully trained Edinburgh model scored 0.931 and is better by every
reconstruction measure. An earlier revision of this page gated on 0.95 and would have preferred the
undertrained one.

**`latent_extrapolation_share`** — how a linear extrapolation of the last two latents scores against
holding the current one — is better, and still a quiescence metric. The sweep below caught it
outright: at `w_vreg` 0.2 the share falls to 97%, which *passes* the obvious "under 100%" reading,
on the worst model of the three by both held-out loss and predictability. It improves because the
latent slows down, not because it becomes more followable.

**The `D(X‖0)` / `D(X‖Z)` ablation ratio** is a diagnostic, not a pass mark either. `X` holds both
feet, their velocities and the hips, so a decompressor given `X` alone already places most of the
body; the ratio mostly measures how informative the query is. An earlier revision gated on 3×.

#### So `w_vreg` stays at the paper's value

The three latent regularisers are `0.1·mean|Z|`, `0.1·mean(Z²)` and `w_vreg·mean|ΔZ/δt|`. The first
two both shrink `|Z|`, which also shrinks the absolute step the third penalises, while the
*relative* step is penalised by none of them — so on paper the equilibrium looks data-dependent in
a way the other weights are not, and `w_vreg` looks like the dial for steppability.

Measured, it is not:

| `w_vreg` | held-out loss | extrapolation | step size | **predictability** |
| --- | --- | --- | --- | --- |
| 0.01 (the release's) | **2.7402** | 108% | 18.7% | **+0.173** |
| 0.05 | 2.8992 | 102% | 15.2% | +0.167 |
| 0.2 | 3.0279 | 97% | 11.4% | +0.155 |

Monotone, and in the wrong direction on both numbers that matter: raising it buys a slower latent,
not a more predictable one, and pays for it in reconstruction. Each arm is 27,000 iterations on
Edinburgh at batch 256 — matched on iterations rather than on wall clock, because a run sharing the
machine reaches fewer steps in the same seconds and the loss column is then measuring the machine.

It remains a parameter on `LmmConfig` — the paper gives no number for it, so 0.01 is the release's
choice rather than a constant — but there is no evidence for moving it on this data.

### Usable frames

Frames come from `TrainingSet.usable()` — matching-feature validity, nothing more.
`pfnn_dataset.usable_queries` looks interchangeable and is not: its `phase_rate != 0` filter discards
every clip with no measurable gait cycle, which a phase-functioned network needs and a learned
matcher wants kept. Idling is motion a matcher has to be able to produce.

## The networks

All plain stacks of `Linear` and an activation, and that is a constraint rather than an observation:
the reason to prefer a learned matcher over a database search is that a few small networks run
anywhere, which stops being true the moment something in the forward pass cannot be exported to
ONNX. A test asserts that every leaf module is one of `Linear`, `ELU` or `ReLU`, so a `torch.where`
added in good faith cannot quietly close off the Unity-native inference path.

| Network | In → Out | Hidden | Activation | Params |
| --- | --- | --- | --- | --- |
| Compressor | `Y‖Q` (574) → 32 | 516 × 3 | ELU | 1.05 M over the shipping rig |
| Decompressor | `X‖Z` (65) → `Y` (331) | **512 × 1** | ReLU | |
| Stepper (phase B) | 65 → 65 | 512 × 2 | ReLU | |
| Projector (phase C) | 33 → 65 | **512 × 4** | ReLU | |

The depths are the paper's Table 1 and the widths are read off the reference implementation's
shipped ONNX graphs, whose parameter counts reproduce the file sizes exactly — measured, not
recalled. The two agree except that the paper rounds both hidden widths to 512.

**The decompressor is shallow and the projector is deep, which looks backwards and is not.** Decoding
a latent into a pose is a smooth map and wants width; approximating a nearest-neighbour lookup is
piecewise constant over thousands of pieces and wants depth.

The mixed activations are the paper's and the reference's alike. An earlier version of this page
used ELU throughout, on the grounds that the mixture had no stated justification and matching
`pfnn_model` gave the repository one story; that has been reverted in favour of following the paper.

## Authoring and artefacts

`LmmConfig` (`Assets/Create/MoSynth/Learned Motion Matching Config`) **owns no pose database.** It
points at a `MotionMatchingData` and reads both halves of that asset's generated output. That is the
whole basis of the comparison, and it is why `LmmConfig` is not an `IPoseSetSource` the way
`PfnnConfig` and `MotionFieldConfig` are.

It also owns the **search weights**, which is the opposite of `MotionMatchingStage`, where they live
on the stage. Training has to see them: the phase C projector learns to approximate a
nearest-neighbour lookup under a particular metric, and one fitted against uniform weights would
approximate a search nobody runs. They go into the checkpoint and `LmmStage` refuses a checkpoint
whose weights no longer match — checked from phase A, so the answer is never "it worked until we
turned the projector on".

The artefact is one file, `StreamingAssets/Lmm/<name>/<name>.lmm.npz`: every network's parameters,
the normalisation, the baked latents, and a full description of the packing. Unversioned, per the
project's [standing decision](../animation-tools/on-disk-formats.md) — the content is the check.

**The contents are the same at every phase; what changes is which arrays are populated.** A
`stages_trained` array records how far training got, and the stage refuses a mode whose networks are
missing, naming the one it wanted. Writing the file with only the autoencoder **removes** any stepper
and projector, because both were fitted against a latent space that has just been replaced; that is
enforced in the writer rather than left to callers to remember.

## Runtime

`LmmStage` takes `MotionMatchingStage`'s slot, so the pipeline stays
`LmmStage → Inertialization → RootFollowStage`.

**No new control inputs.** Every `MotionMatchingControlInput` reaches the database through
`MotionSynthesisComponent.GetMmData()`, and none of them touches the stage itself — so that lookup
matches on `IMotionMatchingDataProvider` rather than on `MotionMatchingStage`, and both stages
implement it. The spline, direction and crowd inputs drive a learned matcher unchanged. (This was
PFNN's problem: it had no `MotionMatchingData` at all.)

The trajectory half of the query is built by `MotionMatchingQuery.FillTrajectory`, shared by both
stages. Sharing it is the point rather than a convenience — a second implementation, even one that
started as a faithful copy, would make the comparison measure the difference between two query
builders as much as the difference between two matchers. The **pose** half is the one place the two
genuinely differ: the classic matcher reads it from the frame it is playing, a learned one carries it
in the state its networks advance. In `DecompressorOnly` those are the same floats.

### Three details that are easy to get wrong

**The rotations are joint-local, and the rig is posed by composing them down the hierarchy.** This is
the same map `lmm_fk.forward_kinematics` runs inside the training loss, so the character-space error
that was optimised is the error the character exhibits. The only translation the model predicts is
the root's height — its other two coordinates are what the character frame transform removed, so
they are zero, and every other bone sits at its rest offset. A reconstructed pose therefore cannot
violate a bone length.

**The root's motion is already a per-second rate.** It goes into `CharacterSpacePose.Apply`
undivided. `PfnnStage` divides by the frame time at the same point because its network predicts a
per-frame delta; doing that here would make the character travel at the database frame rate times the
right speed, and it is the most likely first-run bug.

**The latent table stays in Python.** In `DecompressorOnly` both `X` and `Z` are functions of the
frame being played, so the stage passes a frame index and the lookup happens on the far side, inside
the boundary crossing it was already making. Marshalling a two-hundred-thousand-row table at startup
would cost far more than a lookup that is free in Python. Phase B changes this, because a stepped
latent is no longer one of the baked rows.

### The accept rule, and the discontinuity flag

The rule is `MotionMatchingStage`'s — score the state you hold, ask for something better, take it
only if offered — with one substitution. `minFrameSwitchDistance` suppresses a jump to a nearby
*frame*, which is meaningless once the pose is reconstructed rather than indexed and has no meaning
at all in later modes. It becomes a relative **`acceptanceRatio`**, which says the same thing in the
only terms all three modes share.

`raisePoseDiscontinuity` is a **stage** field, not a config one, so a `BenchmarkOverride` can turn it
off without dirtying a ScriptableObject. The risk it exists against is the flattering one: a learned
matcher has no frame jumps by construction, so it is tempting never to raise the flag — and then it
is silently smoother than the classic matcher on exactly the transitions both of them take, and the
comparison is corrupted in its favour.

### Frames with no latent

Playback can step onto the one frame per clip that has no latent. The stage holds the previous
latent for that frame rather than disturbing the search cadence, which happens exactly where a clip
crossing is already raising a discontinuity. The search itself is masked to latent-carrying frames.

## Verifying a checkpoint

The training loss is a weighted sum over blocks in mixed units: comparable between runs, and
comparable to nothing else. It cannot say whether a foot is a centimetre or a hand's breadth out of
place.

`MoSynth/Lmm/Report Reconstruction Error` answers that, in metres. Joint positions come from
**forward kinematics of the predicted rotations** — which is the only place they can come from, since
that is all the decompressor predicts, and it is also exactly what the stage poses with. The number
measured is therefore the number the character exhibits.

### Two numbers, and they answer different questions

The report scores either the whole database or the **held-out tail** the trainer validated on, and
the gap between them is the point rather than an inconvenience.

| | measured on Edinburgh, 27 bones |
| --- | --- |
| whole database (≈90% training frames) | **0.80 cm** mean joint error, 1.17 cm std, worst bone 1.41 cm |
| held-out tail | **2.04 cm** mean, 2.09 cm std, worst bone 6.16 cm |

**The first is what `DecompressorOnly` actually exhibits.** In that mode the search returns a
*database* frame, and the stage decompresses that frame's own `X` and its baked `Z` — a pair the
decompressor saw in training. So phase A's runtime quality is the training-set number.

**The second is what phase B and C will exhibit**, because a stepper and a projector synthesise
`(X, Z)` pairs that are not database rows. A 2.6× gap between the two is the honest advance warning
that the later phases have less headroom than phase A suggests.

For scale, the paper reports 1.4 cm mean with 1.1 cm std (§6.3). That is *their* accuracy on *their*
data, and importing it as a pass mark for this database would be the same category of mistake as the
3× ablation bar — so it is quoted here as scale, not as a threshold.

### What limits each number

The training-set figure is not capacity-limited: the network fits to 0.80 cm with a 512-wide, one
hidden layer decompressor. More iterations, a wider network or a bigger latent would all be solving
a problem that is not the bottleneck.

The held-out figure is limited by **how much motion there actually is**. Of Edinburgh's 218,365
frames, only 108,265 carry a valid matching feature vector — the set is 1,835 short clips averaging
about 119 frames, and a trajectory horizon of a second at 60 fps invalidates roughly the last 60
frames of each. That leaves 104,595 training pairs, about 29 minutes of motion. The lever is more
data, not more training; LAFAN1 and Bandai-Namco sit on the same rig and their clips are long, so
they would add disproportionately many *valid* frames.

### The other check

`MoSynth/Lmm/Check Training Agreement` is the other half, borrowed whole from PFNN: it compares
`CharacterSpacePose.Extract` against `training_data.build_training_set` on the same frames. A model
is trained on arrays produced by one and run on arrays produced by the other, and a disagreement
about the frame, the units or the rate convention does not throw — it just makes the network wrong.
It matters more now than it did, because forward kinematics sits on both sides of the boundary:
`lmm_fk` derives the character space the loss is written in, and `LmmStage.BuildCharacterSpacePose`
derives the one the rig is posed in.

## Known gaps

- **The stepper and the projector are not written.** `LmmStage` refuses `Stepper` and `Full`.
- **Inference is PythonNET.** `com.unity.barracuda` 3.0.2 is still in the manifest but nothing
  consumes it, and it is deprecated on Unity 6. The ONNX-exportability constraint above exists so
  that a Sentis path can be added without redesigning the networks.
- **Regenerating the database silently invalidates a checkpoint** — the latents are indexed by frame.
  The stage catches it by comparing the frame count and the feature width at `Init`, but nothing
  warns in the inspector before play mode.
- **The benchmark suite does not include an LMM arm yet**, deliberately, following the PFNN
  precedent: a method is added once it is tuned, not while it is being fitted.
- **Phase B has a weak foundation.** A linear model predicts the latent's step from `(X, Z)` with
  held-out R² of only +0.173. A nonlinear stepper will do better than that lower bound, but the
  margin is unknown, and raising `w_vreg` does not improve it. Measure a real stepper before
  assuming the gap closes.
- **Half of Edinburgh is unusable.** 108,265 of 218,365 frames carry a valid feature vector, for the
  structural reason given above. Nothing is wrong, but the database is much smaller than its frame
  count suggests.

## The removed experiment

Five files under `Assets/Scripts/Learned Motion Matching/Scripts/`, about 1,386 lines in namespace
`MM`, were a self-contained neural reimplementation dating to March 2026. They were **deleted**, not
merely parked. Recover them from git history if the reference is wanted.

The reasons, in order of weight:

1. **`MM.MotionMatching` is a different type in a different assembly from the `MotionMatching`
   namespace the rest of this section documents.** For anyone implementing the real thing, that
   collision is a trap rather than a reference.
2. It **called `UnityEditor` from runtime code** in `Assembly-CSharp` — `AnimationUtility` and
   `AssetDatabase`, reached from `Awake` — so it could not be included in a player build.
3. It **could not run as committed**: `MM.MotionMatching` carried no `[Serializable]`, while its
   driver held it as a plain public field, so it would have been `null` when `Awake` called
   `mm.Build`.
4. It shared no type with the live architecture — no `MoSynthStage`, `MotionSynthesisComponent`,
   `PoseBuffer`, `Skeleton` or `PoseSet` — and nothing anywhere referenced it.

Its shape is worth recording, because the live implementation converged on much of it independently:
a 24-float query, a 32-float latent, a 56-float `X‖Z` state, a decompressor emitting 15 floats per
joint, and three `enable*` toggles on the driver that were ablation switches rather than leftovers —
the same three modes `LmmMode` now carries.

`External/lmm-v0.3.0.zip` sits alongside the other vendored payloads and was left in place. It is a
working Unity LMM project that *consumes* ONNX, shipping trained models for all four networks, and
reading those graphs is where the widths and depths above came from. It contains **no training
code**, which is why every loss weight here comes from the author's separate code release rather
than from the archive. It does ship per-character `Database/ZData.txt`, confirming that the latents
are baked to disk and the compressor never runs at runtime.
