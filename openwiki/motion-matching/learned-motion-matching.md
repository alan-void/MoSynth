---
type: Reference
title: Learned motion matching
description: How MoSynth replaces the database search with learned networks, the staging that makes each replacement measurable, and the parked Barracuda experiment that was removed to make room for it.
tags: [neural, training-data, synthesis-method, roadmap]
sources:
  - id: openwiki-source-886205997e2f3c41abc7a5a5
    resource: repo://Assets/Lmm/Editor/LmmAgreementCheck.cs
  - id: openwiki-source-01628a2219f813961e9adea2
    resource: repo://Assets/Lmm/Editor/LmmTraining.cs
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
generated: {by: "claude-code", at: "2026-09-21T14:47:08.791Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-09-21T14:54:36.568Z
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

The method arrives in three phases and **all three are implemented**: the compressor and
decompressor are trained, the stepper is fitted against the latents they bake, and the projector is
fitted against those same latents and the authored search metric. `LmmStage` runs any of the three
modes, and each of them stays in the shipped code as a permanent ablation rather than as a step on
the way to the last.

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

## Training the autoencoder

The first stage: the compressor and decompressor, fitted jointly, under the paper's
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
a weak foundation, and it was the honest advance warning about phase B — one that
[the fitted stepper bore out](#the-stepper-is-measured-by-free-running-it-and-nothing-else-will-do),
missing its drift bar at the search cadence.

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
outright: at `w_vreg` 0.2 the share falls to 96%, which *passes* the obvious "under 100%" reading,
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
| 0.05 | 2.8645 | 102% | 14.9% | +0.162 |
| 0.2 | 2.9263 | 96% | 11.1% | +0.141 |

Monotone, and in the wrong direction on both numbers that matter: raising it buys a slower latent,
not a more predictable one, and pays for it in reconstruction. Each arm is 27,000 iterations on
Edinburgh at batch 256 from the same seed, so the three differ only in this weight — matched on
iterations rather than on wall clock, because a run sharing the machine reaches fewer steps in the
same seconds and the loss column would then be measuring the machine.

The 0.2 row is the clearest statement of why the extrapolation column is not the gate: at 96% it
passes the obvious reading of that number while being the worst of the three on both the held-out
loss and the predictability.

It remains a parameter on `LmmConfig` — the paper gives no number for it, so 0.01 is the release's
choice rather than a constant — but there is no evidence for moving it on this data.

### Usable frames

Frames come from `TrainingSet.usable()` — matching-feature validity, nothing more.
`pfnn_dataset.usable_queries` looks interchangeable and is not: its `phase_rate != 0` filter discards
every clip with no measurable gait cycle, which a phase-functioned network needs and a learned
matcher wants kept. Idling is motion a matcher has to be able to produce.

## Training the stepper

The stepper replaces walking the database between searches. It takes **exactly the vector the
decompressor takes** — the same `X‖Z`, in the same units — and answers with a **rate per second**,
which the stage integrates as `x += ẋ·dt`. Sharing the input is deliberate: the stage carries one
copy of the state and hands it to both networks, so there is no second normalisation of the same
sixty-five floats to get wrong. The rate is what makes synthesis rate and database frame rate
independent of each other, and it is the convention every other predicted motion channel in this
repository already follows.

Only the *output* is normalised, per element, by rate statistics stored in the checkpoint. At
initialisation the network's raw output is near zero, so it predicts the mean rate — which on this
data is near zero too. An untrained stepper therefore holds still rather than flying apart, which is
the right prior for a network whose job is a small correction each frame.

### The latents are an input, never a parameter

The compressor is not merely frozen while the stepper trains — **it does not run at all.** The
stepper is fitted against the latent table already baked into the checkpoint.

The failure this rules out is subtle. Train the two together and the pair has a far cheaper route to
a steppable latent than learning to step it: make the latent constant. Reconstruction would pay for
that, but not by enough, and the symptom is a phase B that looks like it worked. Taking the latents
as a fixed input removes the option rather than penalising it.

It is also why `refit_stepper` exists as a separate entry point — *Fit Stepper Only* on the config,
or `lmm_trainer.py --stepper-only`. The autoencoder is the half-hour half and the stepper is fitted
against what it already produced, so trying a different window or a longer schedule need not pay for
it again. That path reads the `.mmfeatures` alone and not the three hundred megabytes of `.mmpose`
beside it: everything else it needs is in the checkpoint, and `latent_valid` is *exactly* the frame
mask phase A derived, so there is no second definition of which frames carry a latent for the two to
disagree about.

### Unrolled, never single-step

```
x̂ = X_i ; ẑ = Z_i
for n in 1..N:  (ẋ, ż) = S(x̂‖ẑ)
                x̂ += ẋ·δt ; ẑ += ż·δt
                L += w_x·|x̂ − X_{i+n}| + w_z·|ẑ − Z_{i+n}|
                   + w_ẋ·|ẋ − Ẋ_{i+n-1}| + w_ż·|ż − Ż_{i+n-1}|
```

The state the stepper is asked to advance at frame *n* is the state **it produced** at frame *n−1*,
errors and all. A stepper fitted on true states only is one that has never seen its own mistakes,
and the whole failure mode here is error that compounds. `N` defaults to 20: the stage searches
every `searchInterval` seconds, 10/60 by default, which is ten frames of a 60 Hz database — so the
default window is two searches' worth of headroom, and it is also the released code's.

**No window crosses a clip boundary**, and that falls out of the latent rule rather than being
checked again: a frame has a latent only when its successor is in the same clip, so `N+1`
consecutive latents are `N+1` consecutive frames of one animation. Training across a cut would teach
the stepper to predict a jump, which is the search's job and not its.

The loss is divided by `N`, so its magnitude — and with it the effective learning rate — does not
change when the window does.

### The weights, and what they say

| term | weight |
| --- | --- |
| `X` | 2.0 |
| `Z` | **7.5** |
| `Ẋ` | 0.2 |
| `Ż` | 0.5 |

The author's released code again; the paper states no numbers for these either. Both halves are
divided by **one scalar spread for the whole half** before weighting — a single `X.std()` and a
single `Z.std()` over latent-carrying frames — so a feature error and a latent error are being
traded in comparable terms rather than in whatever units each happened to arrive in.

The latent terms carrying roughly four times the feature terms is the interesting part, and it
matches what the runtime does with each: the feature half is partly re-supplied by the controller at
every search, while a drifted latent is only ever corrected by the next accepted candidate. The rate
terms are small because a per-second rate is sixty times a per-frame step and so arrives already
large.

### It stops when it stops improving, and that is not the autoencoder's rule

`stepperPatience` — held-out scores without an improvement — is the stopping rule; `stepperIterations`
is only a ceiling. That is a measured decision rather than a habit. On Edinburgh:

| | autoencoder | stepper |
| --- | --- | --- |
| held-out at 1,000 iterations | 4.2023 | 5.8520 |
| **best held-out** | 2.4827 at **116,000** | 5.8247 at **2,000** |
| at 30,000 | — | 6.3748, still rising |

The autoencoder's held-out loss is *still falling* when it hits its iteration limit; the stepper's
bottoms out at 2,000 and rises monotonically from there while its training loss keeps falling from
5.73 to 3.38. Nine-tenths of a fixed 30,000-step run is therefore pure waste, and the right count is
a property of the database rather than something to guess. With patience the same parameters are
found in 220 s instead of 930 s — the best-parameter snapshot means the long run was never *wrong*,
only slow. The autoencoder has no such knob because patience would never fire.

**This is not a capacity problem.** Narrowing the stepper makes every number worse, monotonically:

| hidden | params | best iteration | held-out | X/Z drift at 10 frames |
| --- | --- | --- | --- | --- |
| **512** (the reference's) | 330 k | 2,000 | **5.8247** | **0.260 / 0.371** |
| 256 | 99 k | 4,000 | 5.9744 | 0.269 / 0.384 |
| 128 | 33 k | 4,000 | 6.1008 | 0.299 / 0.386 |
| 64 | 13 k | 9,000 | 6.3173 | 0.337 / 0.401 |

So the early overfit is the same wall phase A hit, arriving sooner. The autoencoder sees 108,265
independent frames; the stepper sees windows that overlap by nineteen frames out of twenty, so its
genuinely independent content is nearer 5,400 trajectories — and it memorises them in two thousand
steps.

The small fixture database says the same thing from the other end. `MM_LafanCorrected` is one clip,
about 2.2 minutes, and its stepper is far worse on every measure — latent drift 0.819 at the search
cadence against Edinburgh's 0.369, from a latent whose step predictability is +0.062 against +0.173.
Twenty-seven times the data moves both numbers in the same direction, which is what a data-limited
model does and is not what a capacity-limited one does.

## Training the projector

The projector is the network that replaces the search itself: hand it the query and it answers with
a state the database could have held — a feature vector and the latent beside it — so nothing is
looked up and nothing has to be resident to look it up in.

### The target is the nearest neighbour of the *noisy* query, and that is the whole method

Every training sample takes a database frame's query vector, displaces it, finds the true weighted
nearest neighbour **of the displaced vector**, and asks the projector for that neighbour's state.

Targeting the frame the noise was added to instead is the obvious mistake and it does not look like
one: the loss falls, the reconstruction is fine, and what has been fitted is a *denoiser* — a
network that undoes a perturbation. The classic matcher does not undo anything. Asked for a query
that no frame answers, it returns whichever frame answers it best, and for a query displaced any
real distance that is a different frame entirely. A denoiser put in its place returns the character
to where it already was.

The metric is the **authored** feature weights, carried in the checkpoint and expanded per float
exactly as `MotionMatchingStage.UpdateFeatureWeights` expands them. A projector fitted under a
uniform metric approximates a search nobody runs, and the comparison against the classic matcher is
then quietly measuring two different things. The stage refuses a checkpoint whose weights no longer
match the config's.

### One noise scale, and phase B is why there is not a second

Each feature is displaced by its own spread plus one, scaled by a per-sample `sigma` drawn uniformly
over `[0, projectorNoise]`. Two decisions there:

- **The floor of one.** Without it a feature that barely varies is barely displaced, and the
  projector never learns what to do when a controller asks for a value that feature never takes.
  Unity has already divided `X` by a per-feature spread when it baked the `.mmfeatures`, so the
  floor roughly doubles a scale that is already near one.
- **A different `sigma` per sample**, rather than a fixed displacement. One batch then spans a query
  a frame answers almost exactly and a query nothing answers well. A projector trained at a single
  displacement is good at that displacement and guessing everywhere else, and the runtime supplies
  every displacement — the controller can ask for anything at all.

An earlier design had **two** scales, a wide one for the trajectory half and a narrow one for the
pose half, on the argument that their runtime errors have different characters: the trajectory half
is whatever the controller wants, while the pose half is the stepper's own drifting state. The
argument is sound and the second scale is still unnecessary, because phase B measured the quantity
it was guessing at. The stepper drifts **0.258 of `X`'s spread** over the ten frames between
searches, which sits comfortably inside the range one uniform scale already covers. The measurement
retired the knob rather than setting it.

### Three terms, and the third one is the accept rule

| term | weight | what it scores |
| --- | --- | --- |
| features | 1.0 | `X̂` against the true neighbour's `X`, over `X`'s own spread |
| latent | 5.0 | `Ẑ` against that neighbour's `Z`, over `Z`'s own spread |
| distance | 0.3 | how far the answer sits from the query, against how far the true neighbour sits |

The weights are the author's released training code, as the autoencoder's and the stepper's are; the
paper states none. The latent term dominating is the same asymmetry the stepper has and more of it —
an `X` that is slightly wrong is re-supplied by the controller at the next search, while a `Z` that
is wrong is a pose the decompressor has never been asked for.

**The distance term is not a refinement.** That scalar is exactly what the stage's accept rule
compares: it takes the projection only when it is nearer the query than the state already held. A
projector that is close in `X` but systematically wrong about *how* close would bend every accept
decision on the tick path in the same direction, and nothing downstream could see it happening.

The latents are an input and never a parameter, for the reason they are in phase B: fitting them
alongside would offer a cheaper route to an easily projected latent than learning to project it.

### What it costs, and when it stops

Each iteration searches the whole database for the true neighbour of every query in the batch. That
is a matmul rather than a loop — the candidates' weighted norms are computed once and the query's
own norm is dropped, since it shifts a whole row equally and cannot change which entry is smallest —
chunked over the database so a batch against two hundred thousand candidates never needs the whole
distance matrix resident. On Edinburgh that is around 38 iterations a second, against the stepper's
hundreds, and it is the search rather than the network that costs.

`projectorPatience` is the stopping rule and `projectorIterations` only a ceiling, exactly as for the
stepper and for the same reason: where the fit stops improving is a property of the database.

On Edinburgh the fit ran 26,000 of its 30,000 iterations in 275 s before patience stopped it, over
95,787 training query frames of the 106,430 that carry a latent, and kept the parameters from
iteration 21,000. Unlike the stepper — which bottoms out at 2,000 and rises monotonically — the
projector's held-out curve is still roughly flat when patience fires, so it is not running into the
same early-overfit wall. It is not running away from it either.

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
| Stepper | `X‖Z` (65) → `d/dt X‖Z` (65) | 512 × 2 | ReLU | 330 k |
| Projector | `X` (33) → `X‖Z` (65) | **512 × 4** | ReLU | 839 k |

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

A projector may not be written **without** a stepper, for the matching reason in the other
direction: the `Full` mode runs both, so a file carrying one alone describes a mode nothing can run.

The autoencoder is the half-hour half, and both later networks are fitted against latents it has
already baked — so the inspector offers **Fit Stepper Only** and **Fit Projector Only** beside
**Train LMM**, and tuning either costs minutes rather than paying for the autoencoder again. Neither
refit clears `hasTrained`: they leave the autoencoder exactly as stale or as fresh as they found it,
and the inspector cannot tell which field an edit moved, so claiming the checkpoint is current would
be a guess.

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
would cost far more than a lookup that is free in Python. In `Stepper` mode the state is no longer
any database row, so the stage carries it and hands it back each tick.

### The `Stepper` tick

`policy.tick(x, z, dt)` advances the state and decompresses it in **one** call, returning the pose
and the state to carry into the next tick. They are one call because the stepper's answer is only
ever wanted as the decompressor's input, so splitting them would marshal sixty-five floats across
the boundary for nothing. On a search tick the accepted frame's latent is read in the same
acquisition of the GIL — the search itself is pure C# and has already run — so the tick path
acquires it once either way.

The state is advanced **before** it is decompressed, which is what `DecompressorOnly` does with its
playhead: it moves by `deltaTime` and then reads the frame it landed on. Including on a search tick,
where both modes take the accepted frame and then move on from it by `deltaTime`. The two modes
differ in where the state comes from and in nothing else.

**The controller's trajectory is never spliced into `X` between searches.** It is tempting: the
controller knows where it wants to go every frame, and eighteen of the thirty-three floats describe
exactly that. But the stepper advances all thirty-three together and the decompressor consumes them
whole, so overwriting the trajectory half each tick hands it a trajectory paired with a latent it
never saw. The controller enters through the query, on search ticks, and nowhere else.

`PoseDiscontinuity` is raised on an accepted jump and **never on a stepper tick**, which is what a
stepper tick being continuous means. There is also no clip crossing to raise it for, since nothing
is playing.

### The `Full` tick

A search tick calls `policy.project(query)` and gets back a state; every other tick is the `Stepper`
tick above, unchanged. The two modes share the tick path entirely and differ only in where a new
state comes from on a search: one is handed a database frame to restart from, the other has the
state written directly.

**Whether to take the answer is decided in C#**, under the same weights and the same squared
distance the search is given, rather than in Python. That is what keeps the accept rule one rule
across all three modes — and it is why `project` stays a separate call instead of being folded into
`tick`. It costs a second acquisition of the GIL on a search tick, a few times a second, which is
the price of the comparison staying on the side of the boundary where the classic matcher makes it.

`mmSearch` is **not initialised** in this mode. Building the acceleration structure would cost
exactly the startup time and the memory the mode exists to remove.

The `MotionMatchingData` asset is still referenced, and that is worth stating plainly rather than
claiming a saving that has not been made: the control inputs read its trajectory horizons for their
own prediction, and constructing a `FeatureSet` loads the `.mmpose` beside the `.mmfeatures`. Phase
C removes the *search*, not the dependency.

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
crossing is already raising a discontinuity. The search itself is masked to latent-carrying frames,
so this only ever arises from playback — `Stepper` mode never meets it, because it only ever reads a
latent for a frame the search returned.

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

**The second is what the later modes exhibit**, because a stepper and a projector synthesise
`(X, Z)` pairs that are not database rows. A 2.6× gap between the two was the honest advance warning
that they have less headroom than phase A suggests, and the stepper's measured 7.09 cm after a full
search interval of free running is that warning collected.

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

### The stepper is measured by free-running it, and nothing else will do

`MoSynth/Lmm/Report Stepper Drift` runs the stepper from held-out database states with nothing
correcting it, and reports two kinds of number per horizon: how far the state has wandered, in units
of each half's own spread, and what that wandering does to the character in metres after
decompressing it.

| free-run frames | `X` drift | `Z` drift | mean joint error |
| --- | --- | --- | --- |
| 5 | 0.189 | 0.298 | 4.72 cm |
| **10 — the search cadence** | **0.258** | **0.369** | **7.09 cm** |
| 20 | 0.322 | 0.433 | 8.51 cm |
| 30 | 0.439 | 0.482 | 10.83 cm |

Free-running is the only honest measurement here, and it is worth being precise about why. Score the
stepper one frame at a time from true states and it looks far better — but the failure mode of a
stepper is error that *compounds*, and a single-step score cannot see compounding by construction.
The same argument is why training unrolls.

**The 10-frame row misses the bar this plan set** — `< 0.25` on both halves — by a little on `X` and
clearly on `Z`. What it does pass is the shape test: the curve is strongly sublinear, so the stepper
is not diverging, it is settling onto a plausible trajectory that is not the database's. At the
30-frame horizon, three times the cadence it was built for, it is degraded rather than exploded and
produces no NaN.

For scale: 7.09 cm after a full search interval of free running, against 2.04 cm of held-out
reconstruction error with the true latent. So roughly three and a half times the error, and the
search pulls it back every ten frames.

The number is measured twice by two independent code paths — once inside the fit, on device tensors
against the live model, and once here, through the same `tick` the stage calls. They agree to 0.002.
That duplication is deliberate: a report that shared code with the trainer would not be testing the
runtime path, which is half of what an offline report is for.

### The projector is measured against the search it replaces, not against a loss

`MoSynth/Lmm/Report Projector Recall` displaces held-out queries at three fixed sizes and compares
each answer with the true nearest neighbour of that displaced query. Three fixed sizes rather than
one averaged draw, because the ends behave differently: a query that barely misses has its own frame
as its neighbour, and a query displaced a whole scale is somewhere the database barely reaches.

On Edinburgh, over 2,048 held-out queries:

| displacement | distance ratio | top-1% recall | latent error |
| --- | --- | --- | --- |
| 0.25 | 1.028 | 100% | 0.655 |
| 0.5 | 0.988 | 100% | 0.597 |
| 1.0 | 1.004 | 100% | 0.644 |

**Read all three together, because each one alone is misleading.**

The **distance ratio** is the mean distance the answers sit from their queries over the mean distance
the true neighbours sit. It is a ratio of means rather than a mean of ratios: a barely-displaced
query has its own frame at a distance near zero, and dividing by that reports a number in the
hundreds for a model that is a centimetre out. The plan's bar was 1.15 and all three clear it. But
0.988 is *below one*, and that is not a better search — it means the answer is not a database state
at all but a point between several, which can sit nearer the query than any real frame does.

The **top-1% recall** saturates at 100% here, and it does so for a barely-trained projector too: a
400-iteration model scored 100% at all three displacements. On a database this size the nearest one
percent is a thousand frames, and once a query is displaced any real distance the frames are all
about equally far away. It passes its 80% bar and it is not evidence of anything.

The **latent error** is the number that discriminates, and it is the one the plan's criteria did not
include. At 0.6 of the latent's own spread the projector's latent is further from the true
neighbour's than ten frames of free-running stepper drift (0.369). Part of that is the measure
over-penalising — a query does not determine a latent uniquely, and several latents decode to much
the same pose given the same `X` — but it is the weakest number phase C has, and it is the one that
would move with more motion in the database.

### The full loop has to be free-run, and nothing else will do

`MoSynth/Lmm/Report Full Loop Rollout` runs projector, stepper and decompressor together for 900
frames — thirty seconds — with nothing reading the database. Every per-frame score the three
networks produce stays plausible long after the loop as a whole has stopped producing motion, and a
model that has quietly collapsed stands still with an excellent loss while it does.

The controller is held still: each seed goes on asking for the trajectory its own frame asked for.
That is a request a character can follow indefinitely, which a replayed one cannot be — Edinburgh's
clips average about two seconds and this runs for thirty — and it is the harder ask, because nothing
in it ever pulls a drifting state back towards the data.

On Edinburgh, from 256 held-out states:

| | measured | the seeds' own frames | ratio | bar |
| --- | --- | --- | --- | --- |
| speed | 1.009 m/s | 1.084 m/s | 0.93× | within 10% |
| reach from the root | 0.928 m | 0.916 m | 1.01× | under 1.2× |
| non-finite values | none | | | none |

The projection is taken at **95%** of searches, which is worth knowing on its own: it says the loop
is genuinely being driven by the projector rather than degenerating into a long stepper rollout that
happens not to explode.

### The other check

`MoSynth/Lmm/Check Training Agreement` is the other half, borrowed whole from PFNN: it compares
`CharacterSpacePose.Extract` against `training_data.build_training_set` on the same frames. A model
is trained on arrays produced by one and run on arrays produced by the other, and a disagreement
about the frame, the units or the rate convention does not throw — it just makes the network wrong.
It matters more now than it did, because forward kinematics sits on both sides of the boundary:
`lmm_fk` derives the character space the loss is written in, and `LmmStage.BuildCharacterSpacePose`
derives the one the rig is posed in.

## Known gaps

- **The projector's latent is its weak half.** It answers at 0.99–1.03× the true nearest
  neighbour's distance, which is the search itself within measurement, but its *latent* sits about
  0.6 of the latent spread from that neighbour's — further than ten frames of stepper drift. The
  full loop survives thirty seconds regardless, so this is a quality ceiling rather than a
  stability problem, and it is the same data limit phases A and B both ran into.
- **The top-1% recall bar is not discriminating on this database.** It reads 100% for a trained
  projector and 100% for a 400-iteration one alike. It is reported because it would catch a
  projector answering from the wrong neighbourhood entirely, not because passing it means anything.
- **The `Full` mode has not been benchmarked or driven in play mode at length.** It runs, and the
  offline free-run says the loop holds together for thirty seconds; what has not been done is
  watching a character on a spline under it beside the other two modes.
- **Inference is PythonNET.** `com.unity.barracuda` 3.0.2 is still in the manifest but nothing
  consumes it, and it is deprecated on Unity 6. The ONNX-exportability constraint above exists so
  that a Sentis path can be added without redesigning the networks.
- **Regenerating the database silently invalidates a checkpoint** — the latents are indexed by frame.
  The stage catches it by comparing the frame count and the feature width at `Init`, but nothing
  warns in the inspector before play mode.
- **The benchmark suite does not include an LMM arm yet**, deliberately, following the PFNN
  precedent: a method is added once it is tuned, not while it is being fitted.
- **The stepper misses its drift bar.** At the ten-frame search cadence it drifts 0.258 of `X`'s
  spread and 0.369 of `Z`'s, against a bar of 0.25 on both, and the character stands 7.09 cm from
  where the database says. It degrades rather than diverges, and the search corrects it every ten
  frames, so `Stepper` mode is usable — but the margin phase C needs is not there yet. Phase A's
  advance warning (a linear model predicts the latent's step with held-out R² of only +0.173) was
  accurate, and the width sweep rules out the cheap explanation: this is data, not capacity.
- **What would move it is more motion, not more training.** The same conclusion phase A reached, and
  the stepper reaches it harder: its windows overlap by nineteen frames in twenty, so 62,757 windows
  carry nearer 5,400 trajectories' worth of independent content. LAFAN1 and Bandai-Namco sit on the
  same rig with longer clips.
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
