# Style transfer stage — design draft

A `MoSynthStage` that restyles the pose produced by **any** synthesis method (MM, MF, PFNN, LMM) in
real time: sad walk, old walk, proud walk, and so on. It is trained on 100STYLE, retargeted onto
Y Bot. Benchmarking is out of scope for this draft.

## 1. Scope decisions

These keep the stage method-agnostic and stop it fighting the rest of the pipeline. Everything after
them follows from them.

1. **Root motion is a toggle, `rootMotion`** (§4.7):
   - `Preserve`: the method owns the root. The stage never writes bone 0's frame velocity or yaw,
     only bone 0's height and its pitch/roll within the character frame (crouch, slouch).
     `RootFollowStage` and the path metrics are unaffected.
   - `Stylize`: the stage also restyles the root, meaning speed, lateral sway, and yaw wobble,
     written into bone 0's frame velocity and yaw. Where the character goes then depends on the
     style as well as the method.
2. **Timing stays with the method.** Stance timing comes from the input's contact flags, so the
   stage keeps the method's cadence and contact flags and changes only *how* the body moves. Styles
   whose essence is cadence or support pattern (Hop, Skip, Crawl, single-leg styles) come out weak
   or wrong. They are excluded from v1 rather than half-supported (§6).
3. **Residual, not reconstruction.** The stage outputs per-bone rotation *deltas* on top of the
   input pose. At intensity 0 it is exactly the identity, and it never has to reproduce the base
   motion, which is the part of the input that differs most between methods.
4. **Causal.** It only sees the current and past ticks. No look-ahead and no added latency.

## 2. Pipeline placement

```
<method stage> → Inertialization → StyleTransferStage → RootFollowStage → ContactLockStage
```

- **After `Inertialization`.** The stage sees a smooth stream, which shrinks the gap between method
  output and training data. It also never needs to react to `PoseDiscontinuity`. It does not set
  that flag itself, because its output is continuous by construction (§4.4).
- **Before `RootFollowStage` and `ContactLockStage`.** Neither stage's documented ordering changes.
  `ContactLockStage` plants the *restyled* feet, which absorbs the foot offsets a posture change
  introduces.
- **What it leaves alone, as `ContactLockStage` does:** velocity channels (the next tick re-reads the
  rig), contact flags (timing is preserved, decision 2), and bone positions other than bone 0's height.
  The one exception is bone 0's frame velocity under `rootMotion = Stylize`.

## 3. Data: 100STYLE onto Y Bot

### 3.1 Fetch

- Add a `hundredstyle` dataset to `Tools/Data/fetch_datasets.py` (Zenodo record 8127870,
  `100STYLE.zip`, raw BVH) with sha256 entries in `checksums.json`. Extract to
  `External/LFS/100STYLE/` and then `Assets/LFS/Animation/100style/bvh/`.
- Add a `DATA.md` section: source, version, frame count, citation (Mason, Starke, Komura, I3D 2022).
  **Verify the licence, frame rate and units from the archive itself.** None of the three is confirmed yet.

### 3.2 Retarget

- Stage one needs a new cleaned-skeleton setup for the 100STYLE BVH skeleton, authored the way
  `make_lafan_corrected_setup.py` authors LAFAN's: same bone names and lengths, rest pose
  re-oriented to a T-pose. Stage two is the existing Rokoko retarget onto Y Bot. Strip the setup
  into `Tools/Retargeting/setups/` with `strip_setup.py`.
- Shards: **one output FBX per style**, holding that style's takes (FW, BW, SW, FR, BR, SR, ID and
  the TR transitions), with take names that keep the source file names. That is 100 FBX. Then run
  `lfs_meta.py export` and `export_shard_lists.py`.
- The Y Bot memory hazard applies: create annotated clips **one FBX at a time** with
  `UnloadUnusedAssetsImmediate` in between, never as one batch over all 100.

### 3.3 Annotate in Unity

- `Create Annotated Clips From Model...` per FBX.
- A new editor step, **Tag 100STYLE Clips**, parses each take name into two whole-clip
  `AnimationTagComponent` channels: `style.<Name>` and `action.<gait>` (forward/backward/side ×
  walk/run, idle, transition). The tags are authored data from then on, so a mislabelled take is
  fixed in the clip editor, not in a parser.
- Detect Footfalls over the whole folder gives every clip a `GaitPhaseComponent`. Idle clips use
  the existing standing sweep.
- `style.Neutral` is the reference style. 100STYLE records one performer doing every style over the
  same gait set, which is what makes pairing possible (§4.2).

### 3.4 Bake

`StyleTransferConfig` (§5.2) bakes its own `.mmpose`, as `PfnnConfig` does. Gait phase and contacts
are already in that file. The one thing it lacks is **which style each clip is**, since `.mmpose`
clip entries carry only start/end. Two options:

- **(a) Sidecar, recommended for v1.** At bake time the config writes
  `<name>.style_labels.json`: one `{style, gait}` entry per clip, in database clip order. C# writes
  it next to the `.mmpose` and Python reads it. A training run checks the clip count against the
  `.mmpose` and refuses on a mismatch, which is the same content check the project already uses
  instead of version numbers.
- **(b) Revive the `.mmpose` tag block.** `AGENTS.md` keeps it dead *until something wants to read
  it*, and this is the first consumer. It is cleaner long term, because every database would carry
  its tags, but it costs a full database regeneration and touches both readers. **This is your call.**
  The rest of the design works either way.

## 4. Model

### 4.1 Decomposition

STyMo-style, in the character frame, per styled bone *b*:

```
R_out[b] = R_in[b] · slerp(I, P_s[b], w_p·g) · slerp(I, D_s[b](x_t), w_d·g)
```

- `P_s[b]` is the **posture term**: a time-invariant per-style, per-bone rotation offset, plus a
  bone-0 height offset and pitch/roll. It is fitted in closed form as the mean paired delta per
  style, with one entry per gait class when the means differ enough. No network. This is the
  slouch, the raised chin, the hands drawn in.
- `D_s[b](x_t)` is the **dynamics term**: a small MLP whose hidden layers are FiLM-modulated by a
  learned per-style embedding `e_s`. It outputs a 6D rotation per styled bone, plus bone-0
  height/pitch/roll. This is the swagger, the shuffle, the bounce.
- `w_p` and `w_d` are the user's posture and dynamics intensity, and `g ∈ [0,1]` is the gate (§4.5).
  Separate dials are the point of the split.
- **Per-region weights** multiply `w_p` and `w_d` per bone (§5.1), so you can have "sad from the
  waist up".

### 4.2 Training pairs

100STYLE has no frame-paired data, so pairs are built by phase. For each style *s* and gait *g*,
each styled frame *i* is paired with the `Neutral` frame of the same gait that best matches its
gait phase, ties broken by nearest character-frame speed. The bake-time `GaitPhase` is exact
offline, so pairing uses it. Speeds differ between the two sides of a pair, and that is fine: the
residual learns what this style does to a neutral body at this point of the cycle.

- Input: the neutral frame, which is what the stage sees at runtime. Target: the styled frame.
- Neutral→neutral pairs with `e_Neutral` train toward a zero delta, so `Neutral` is the identity
  style rather than an extra special case.
- Transition takes (TR) have no clean neutral counterpart. v1 leaves them out of pairing and keeps
  them as a held-out robustness set (§7).

### 4.3 Input vector `x_t`

Built in C# with `CharacterSpacePose.Extract`, mirrored in Python by `training/training_data.py`,
so frame, units and rate conventions agree by construction:

- local rotations (6D) of the styled bones, bone-0 height
- character-frame speed (forward, lateral) and yaw rate
- causal gait phase as (sin, cos), plus its rate (§4.4)
- a short past window, e.g. −0.1 s and −0.2 s, of speed, yaw rate and foot heights, giving the
  network acceleration and context without look-ahead

The network deliberately gets nothing method-specific: no MM feature vector, no PFNN phase, no LMM
latent.

### 4.4 Causal phase: `OnlineGaitPhase`

Every method writes contact flags (`MfConnector`, `MotionFieldStage`, `PfnnStage`, `LmmStage`, and
MM through the database), so phase can come from them for any method.

- This is a new pure, unit-tested class in `AnimationTools/Runtime/Animation/`, next to `GaitPhase`
  and with the same conventions: half a cycle per footfall, cycle zeroed on a right footfall, the
  standing sweep by ground speed.
- **Causal rule:** a debounced rising edge on a contact flag is a footfall. Between footfalls the
  phase extrapolates at the last stride's rate. At a footfall the remaining error is closed by a
  critically damped spring, not a snap, so phase stays continuous.
- **The trainer computes `x_t`'s phase with a Python port of this same causal rule** from the
  `.mmpose` contact flags, *not* from the baked phase. The network then trains on the phase it will
  see at runtime, extrapolation lag included. Add an agreement check like
  `MoSynth/Pfnn/Check Training Agreement` that runs both implementations on the same frames.

### 4.5 Gate

`g` turns the style down where the input is outside what training covered: a speed, yaw rate or
phase rate beyond the paired data, or a pose far from it. That is STyMo's stylizability gate, done
without learning:

- a novelty score: distance from the input descriptor (speed, lateral speed, yaw rate, phase rate,
  foot heights) to its nearest training neighbours, normalised per dimension, stored in the
  checkpoint
- mapped to [0,1] with a soft threshold, then smoothed with a half-life so the style never flickers

### 4.6 Losses

- rotation loss on the 6D deltas, plus an FK position loss through `lmm/fk.py` (reused, not copied)
- **foot slide**: a penalty on contact-bone velocity while the input's contact flag holds, so the
  style does not create slide for `ContactLockStage` to hide
- temporal smoothness: a penalty on the delta's frame-to-frame change
- a small magnitude regulariser, so ambiguous frames fall back toward the posture term

### 4.7 Root motion (`rootMotion = Stylize`)

**What is learned.** The paired targets already move at the styled speed, so the root residual comes
from the same pairs. In the character frame, per gait class *g*:

- **speed ratio `k_s,g`**: mean styled forward speed over mean neutral forward speed, a closed-form
  scalar like the posture term. Sad walks slower and Proud a little faster.
- **oscillatory residual**: the network gets three more outputs, the forward, lateral and yaw-rate
  residual of `v_styled − k·v_neutral` at that phase. It is close to zero-mean over a cycle, so it
  sways and wobbles without drifting.
- **facing wobble** is carried by the yaw-rate residual alone. Bone 0's frame-local yaw is zero by
  construction, since the character frame is derived from bone 0, so its delta has no yaw to keep.

The root channel has its own dial, `rootIntensity`, which scales both the speed change (as `1 + (k−1)·w`)
and the oscillation.

**Applied** as a rewrite of bone 0's frame velocity: forward speed × k, plus the residual. That is
the channel `Inertialization` and `RootFollowStage` already use, so it stays damped and visible to
the recorder.

**Keeping the feet honest.** If the root speed changes while the legs keep the method's stride,
the feet slide. Under `Stylize` the foot-slide loss (§4.6) is computed in the **world** frame,
root residual included, so the leg deltas learn the shorter or longer stride that goes with the new
speed. That relationship is in the data, since the styled targets are styled legs at styled speed.
`ContactLockStage` absorbs what is left.

**How each control input behaves:**

| Control input | Behaviour under `Stylize` |
|---|---|
| Anchored (`AnchoredDirectionControlInput`) | Works as intended. The input re-reads the character each tick and holds no position, so a slower style is simply slower. |
| Spline inputs | Work. Lap and progress tracking measure the character's own progress, so a slower style takes longer. |
| Free simulation object (`DirectionControlInput`) | The object runs ahead at the requested speed and the gap grows, the same permanent-lag problem it already has, made worse. |
| `RootFollowStage` with half-life 0 | **Cancels the root style**: it puts the character exactly on the target every tick. `Init` warns about this combination. |
| `RootFollowStage` with half-lives > 0 | Partial. Sway near stride frequency gets through, the speed change does not. This depends on tuning. |

**Proposed follow-up, needs your call.** Root styling and capsule authority compose properly only if
the follow target can carry the style. Two clean ways:

- `MotionSynthesisComponent` exposes a per-tick **root offset** (position and yaw relative to the
  target) that stages contribute to and `RootFollowStage` adds to its target. This gives sway on a
  followed character.
- An optional **`ISpeedScaled` control input** that the stage pushes `k` into, so the input itself
  asks for the styled speed and the method produces it natively. This gives slower or faster
  without fighting the target.

Neither is needed for anchored or spline inputs, so v1 ships without them.

## 5. Unity side

### 5.1 `StyleTransferStage` (new `Assets/Style/` assembly, references `AnimationTools` and `GameplayTags`)

| Field | Purpose |
|---|---|
| `config` | `StyleTransferConfig`, the checkpoint and its skeleton |
| `style` | `GameplayTagSO` under the config's style root, e.g. `style.Old` |
| `postureIntensity`, `dynamicsIntensity` | `w_p`, `w_d`, 0–2 (above 1 exaggerates) |
| `regionWeights` | sparse name-keyed per-subtree weights, drawn as the skeleton hierarchy like `PredictedBoneSelection` |
| `rootMotion` | `Preserve` / `Stylize` (§4.7) |
| `rootIntensity` | root-style dial, only read under `Stylize` |
| `styleBlendHalfLife` | crossfade time when the style changes |
| `gateEnabled`, `gateHalfLife` | §4.5 |

- `SetStyle(GameplayTagSO, float blendTime)` lets gameplay switch styles. A switch crossfades by
  running the old and new style deltas and slerping between them over the half-life, so the
  embedding is never interpolated into a style that does not exist.
- `Init` checks the config's skeleton with `Skeleton.StructurallyEqual`, like every stage backed by
  a database, and checks that the stage list has `Inertialization` before it and
  `ContactLockStage`/`RootFollowStage` after it, warning otherwise.
- `Apply`: extract the character-space pose, update `OnlineGaitPhase`, build `x_t`, one stateless
  `step` call, then compose and write rotations, bone-0 height and bone-0 pitch/roll. Yaw is removed
  from the bone-0 delta explicitly, by swing–twist about the frame's up axis, because the character
  frame is derived from bone 0. Under `Stylize`, bone 0's frame velocity is also rewritten (§4.7).
- The checkpoint always carries the root outputs, so the toggle switches at runtime with no retrain.
  Flipping it crossfades over `styleBlendHalfLife`, like a style change.

### 5.2 `StyleTransferConfig`

The `PfnnConfig` pattern: clips, skeleton, `contactBones`, styled-bone exclusion
(`PredictedBoneSelection`, fingers and leaves excluded by the existing button), style root tag,
neutral style tag, list of enabled styles, network size, training settings, checkpoint path, and
`hasTrained`. `TryValidate(out error)` follows the project pattern. The inspector has Bake and Train
buttons. Training records wall-clock time, hardware and the epoch count (the training-time rule in
`AGENTS.md`).

### 5.3 Inference

**v1 uses PythonNET with a numpy-only runtime**, the same route PFNN and LMM take: `PythonRuntime`,
one GIL call per tick, `style.runtime.step` stateless with all state in C#. The MLP is small enough
that numpy is enough, and training stays in torch.

**Later**, export to ONNX and run it in Unity's inference engine (`com.unity.ai.inference`, not
installed today; Barracuda is deprecated). A stage meant to sit on every method should not have to
cost a Python round-trip. The C#/Python split above keeps that swap local to `Apply`.

## 6. Style subset for v1

Not all 100 styles fit decisions 1–2. The plan:

1. Train on all styles whose stance timing matches neutral within a tolerance: footfall count per
   metre and stance/swing ratio against `Neutral`, measured from the gait-phase annotation.
2. Report per-style held-out error and the rate at which the gate fires. A style that only works
   with the gate closed is listed as unsupported, not shipped.
3. Cadence-defined styles (Hop, Skip, Crawl and similar) are dropped: they are not baked or paired,
   and the config's style list does not offer them.

## 7. Offline checks (not benchmarks)

- `python -m style.runtime <ckpt> --database <neutral db> --style Old`: stylise a held-out neutral
  clip and a **LAFAN** walking clip (the cross-dataset gap), and report delta magnitudes, gate
  activity, and foot slide before and after.
- Hold out the TR takes and a slice of each style's FW take for reconstruction error.
- Bandai-Namco dataset-2 is already in the project and is a stylisation dataset over shared content.
  It is a free cross-dataset sanity check of whether the style transfers to another performer's
  neutral.

## 8. Deferred

- **Timing exaggeration** (STyMo's temporal control): a phase-rate multiplier that would need the
  *method* to slow down or speed up. It belongs with a style speed preference fed to the control
  input, not in this stage.
- **Environment-conditioned styles** (leaning away from or ducking under an obstacle): extend `x_t`
  with local obstacle features. They stay body-only under decision 1. Dodges that move the root
  belong to the control input.
- In-engine inference (§5.3).

## 9. File plan

| Path | New |
|---|---|
| `Tools/Data/fetch_datasets.py`, `checksums.json`, `DATA.md` | 100STYLE entry |
| `Tools/Retargeting/setups/hundredstyle_to_*.blend`, setup script, shards | retarget |
| `Assets/AnimationTools/Runtime/Animation/OnlineGaitPhase.cs` (+ tests) | causal phase |
| `Assets/Style/StyleTransferStage.cs`, `StyleTransferConfig.cs`, `Style.asmdef` | runtime |
| `Assets/Style/Editor/` | config inspector, Tag 100STYLE Clips, agreement check |
| `Assets/Style/Tests/` | stage tests; **add the assembly to `TestResultDump`** |
| `Python/style/{dataset,model,io,trainer,runtime}.py` | pairing, network, `.style.npz`, training, inference |
| `Python/tests/test_style_*.py` | pairing, causal phase port, composition identity at intensity 0 |
| `openwiki/` | new page for the stage once it exists (via the OpenWiki workflow, not by hand) |

## Decisions

1. Style labels: **sidecar JSON** (§3.4 a).
2. Inference: **PythonNET** with a numpy runtime (§5.3).
3. Cadence-defined styles: **dropped**. §6 step 1 is the filter, and nothing in v1 reserves room for them.
4. Root motion: **a runtime toggle** (§4.7). Open: whether to add the root-offset and
   `ISpeedScaled` follow-ups so it composes with `RootFollowStage`.

## Estimates

Rough figures. Training time is scaled from the one recorded run on this machine (RTX 5060 Laptop,
8 GB): `PfnnBandaiWalk`, 665k parameters on 17k poses, about 270 s.

**Training, per run:**

- Data: 100STYLE is about 4.06M frames. Dropping the transition takes (~0.79M) and the cadence styles
  and baking at 30 fps leaves roughly **1.3–1.5M pairs**, about 80× the PFNN run.
- Model: a 3×512 FiLM MLP is ~1M parameters. The FK and world-frame slide losses dominate step
  cost, and on this project training is launch-bound, not FLOP-bound.
- **Full run: about 20–60 min.** **One-style or ten-style iteration: 2–5 min.**
- Pair construction (phase matching) is CPU preprocessing of a few minutes, cached between runs.
- Record each run's time (the training-time rule).

**Setup, wall-clock work to a first working v1:**

| Step | Effort | Unattended compute |
|---|---|---|
| Fetch script, checksums, `DATA.md` | ½ day | download, size unverified |
| Stage-one retarget setup for the 100STYLE skeleton | **½–2 days, the main risk** | — |
| Batch retarget, 810 BVH → 100 FBX | ½ day of shards | a few hours in Blender |
| Import, annotated clips (one FBX at a time), Tag 100STYLE Clips, footfalls + spot QA | 1 day | ~1 h |
| `OnlineGaitPhase` + tests + Python port + agreement check | 1–2 days | — |
| `Python/style/` dataset, model, io, trainer, runtime + tests | 3–4 days | — |
| `StyleTransferStage`, config, inspector, asmdef + tests | 3–4 days | — |
| Root-motion toggle (§4.7) | 2–3 days | — |
| Tuning: gate, losses, per-style triage | 1–2 weeks | many short runs |

**Total: about 3–5 weeks** to a v1 that restyles all four methods with both root modes. The spread
is mostly the retarget calibration and the tuning.

## 10. Implementation contract (v1, validated on motion matching)

Both halves are written against this section. Change it here first.

### 10.1 Data on disk

- `External/LFS/100STYLE/100STYLE.zip` (md5 `3cf627852fd8192024c04a8d0ef49583`, CC BY 4.0, Zenodo 8127870)
- `External/LFS/100STYLE/bvh/<Style>/<Style>_<Gait>.bvh`: 100 styles, gaits `BR BW FR FW ID SR SW TR1 TR2 TR3`,
  60 fps, centimetres, 23 joints (`Hips`, `Chest`…`Chest4`, `Neck`, `Head`, collars/arms, `*Hip/Knee/Ankle/Toe`)
- `External/LFS/100STYLE/Frame_Cuts.csv`: per style, `<GAIT>_START,<GAIT>_STOP` source frames at 60 fps, `N/A` when absent
- `External/LFS/100STYLE/Dataset_List.csv`: `Style Name,Description,Stochastic,Symmetric,Notes`
- Kept outside `Assets/` on purpose, so `BvhImporter` never imports 810 BVH

### 10.2 Tags and labels

- Tag hierarchy: `style.<StyleName>` (description from `Dataset_List.csv`) and `gait.<Code>`
  (`FW BW SW FR BR SR ID TR1 TR2 TR3`). Each clip has exactly one of each, as a whole-clip channel (one key at 0)
- `StyleTransferConfig` bakes `<name>.mmpose` plus `<name>.style_labels.json`:
  `{"clips": [{"style": "<TagName>", "gait": "<TagName>"}, ...]}`, one entry per database clip in database clip
  order. `mirrorClips` must be off. Python refuses a label count that differs from the `.mmpose` clip count

### 10.3 `OnlineGaitPhase` (C# in `AnimationTools/Runtime/Animation/`, Python port in `style/phase.py`)

Parameters and defaults: `standingSpeed = 0.1` m/s, `standingPeriod = 1.0` s (both GaitPhaseComponent's
defaults), `minStepInterval = 0.15` s, `correctionHalfLife = 0.1` s, `minRate = Tau / 2.0`,
`maxRate = Tau / 0.4` rad/s, `standTimeout = 1.0` s.

State: `phase` (internal, wrapped to [0, Tau)), `rate`, `offset`, `time`, `lastFootfallTime` (−∞),
`lastFoot` (none), `lastFootTime[2]` (−∞), `previousContact[2]`, `initialized`.

`Update(dt, leftContact, rightContact, groundSpeed)`:

1. First call: `previousContact = (left, right)`, `phase = 0`, `rate = Tau / standingPeriod`, `offset = 0`,
   `time = 0`, `initialized = true`. Report and return. A foot already down is not a footfall, as in
   `GaitPhase.FootfallsFromContacts`
2. `time += dt`, `phase = wrap(phase + rate * dt)`
3. For each foot in the order Left, Right: a footfall is `contact && !previousContact[foot]` and
   `time − lastFootTime[foot] ≥ minStepInterval`. For each footfall:
   - `target = 0` for Right and `π` for Left (the cycle is zeroed on a right footfall)
   - if `lastFootfallTime > −∞`: `interval = time − lastFootfallTime`; `cycles = 1` if `foot == lastFoot`
     else `0.5`; if `interval > 0`, `rate = clamp(cycles * Tau / interval, minRate, maxRate)`
   - `error = wrapToPi(target − phase)`; `phase = wrap(phase + error)`; `offset −= error`
   - `lastFootfallTime = time`, `lastFoot = foot`, `lastFootTime[foot] = time`

   Then `previousContact[foot] = contact` for both feet, footfall or not
4. Standing: if `groundSpeed < standingSpeed` and `time − lastFootfallTime ≥ standTimeout`,
   `rate = Tau / standingPeriod`
5. `offset *= exp(−ln 2 · dt / correctionHalfLife)`, or `offset = 0` when the half-life is ≤ 0
6. Report `Phase = wrap(phase + offset)` in [0, Tau) and `Rate = rate`

`wrap` maps to [0, Tau); `wrapToPi` maps to (−π, π]. Golden vectors: `Python/tests/fixtures/online_gait_phase_golden.json`
= `{"params": {...}, "dt": float, "inputs": [[left(0/1), right(0/1), speed], ...], "expected": [[phase, rate], ...]}`,
generated by Python and asserted by both test suites (phase compared wrap-aware, tolerance 1e-3).

### 10.4 Python runtime ↔ C# stage

`style.runtime.StylePolicy(checkpoint_path, log=print, device='cpu')`. Construction raises
`FileNotFoundError` when the checkpoint is missing or unreadable, as `PfnnPolicy` does.

| Accessor | Returns |
|---|---|
| `bone_names()` | every skeleton bone name the checkpoint was trained on, skeleton order |
| `styled_bone_names()` | the styled subset, skeleton order (bone 0 always first) |
| `style_names()` | style tag names, index = style index |
| `contact_bone_names` | attribute, `list[str]` |
| `frame_time()` | float |
| `history_offsets()` | `list[float]` seconds, ascending, e.g. `[0.1, 0.2]` |
| `descriptor_size()` | int D = 3 + n_contacts |
| `describe()` | one-line str |

`step(rotations, positions, contacts, root_velocity, root_yaw_rate, phase, phase_rate, history, style_indices)`:

- `rotations`: B×4 flat, xyzw, every bone **in the character frame**, as `CharacterSpacePose.Extract` returns them
- `positions`: B×3 flat, character frame
- `contacts`: C floats (0/1), contact slot order
- `root_velocity`: 3 floats, the character frame's own linear velocity expressed **in that frame**, m/s
- `root_yaw_rate`: float rad/s; `phase`, `phase_rate`: from `OnlineGaitPhase`
- `history`: H×D flat, the descriptors `step` returned at the ticks closest to each history offset (oldest
  available until the history fills); H = len(history_offsets)
- `style_indices`: 1 or 2 ints (2 while crossfading: previous, current)

Returns a tuple of flat lists and floats, S = len(style_indices), Bs = styled bones:

`(posture_rotations[S·Bs·4], dynamics_rotations[S·Bs·4], posture_height[S], dynamics_height[S],
speed_ratio[S], root_residual[S·4], gate, descriptor[D])`

- rotations are parent-local deltas, xyzw, composed on the right: `R_out = R_in · P · D`. Bone 0's delta
  is the same on its world rotation, and C# removes its yaw afterwards
- heights in metres, added to bone 0's position
- `root_residual` per style: `(vx, vy, vz, yaw_rate)` in the same frame-local convention as `root_velocity`
- `gate` is in [0, 1], unsmoothed
- `descriptor` = `[root_velocity forward, root_velocity lateral, root_yaw_rate, contact-bone heights...]`;
  C# stores it opaquely for `history`

### 10.5 Trainer entry point

`style.trainer.train(data_dir, db_name, out_path, neutral_style='Neutral', excluded_styles=(), excluded_bones=(),
hidden_units=512, layers=3, embed_dim=32, history_offsets=(0.1, 0.2), gait_sigma=0.35, epochs=60,
batch_size=1024, learning_rate=1e-3, weight_decay=1e-4, validation_fraction=0.1, seed=42, device='auto',
progress=None) -> dict`. The summary includes `seconds`, `device`, `samples`, `styles_kept`, and
`styles_dropped` (name → reason). CLI: `python -m style.trainer <database> <name> [--out ...]`.

## 11. First run: 12 styles over motion matching (2026-10-20)

Styles: Neutral, Old, Depressed, Elated, Angry, Drunk, Crouched, BentForward, Akimbo, ArmsBehindBack,
BigSteps, DragLeftLeg. Selected with the pipeline's `-styles` argument; the 17 other retargeted
FBX are parked in `External/LFS/100STYLE/fbx_pending/` until the full pass.

- **Retarget**: 29 of 100 styles were built before the full batch was stopped for memory. Re-running
  `run_batch_all.py` resumes; use 2 workers on this 24 GB machine. Move the parked FBX back first.
- **Contact threshold 0.5 m/s, not 0.15** (`HundredStyleClipSetup.ContactVelocityThreshold`). At 0.15
  this 60 fps capture reads Neutral as planted 30% of the walk and 2% of the run, and the cadence
  filter dropped half the styles on missed contacts. 0.5 gives 0.58 / 0.38.
- **Duty-factor tolerance 0.3** (`style.dataset.DUTY_FACTOR_TOLERANCE`): Old plants longer (0.80 against
  0.58) without being a hop or skip.
- **Memory**: each take is capped at 20 s (`MaxTakeSeconds`), excluded styles stay out of the database,
  and `PoseSetImporter` releases each clip's bake and curves after use (`SkeletonAnimation.ReleaseLoadedData`).
  The bake held 110,599 poses (376 MB).
- **Training**: 12 styles kept, 0 dropped, 22 of 65 bones styled, 775k parameters, 84,991 training
  pairs, 308 s on an RTX 5060 Laptop GPU, run from the shell (`training_log.json`). Train/val loss
  0.344 / 0.548. Training inside the Editor was killed once for memory, so prefer the shell.
- **Offline check** (Old, 600 frames): posture 12.5°, dynamics 9.4° mean per bone, gate 0.89,
  speed ratio 0.35.
- **Over motion matching** (`StyleSmokeCheck`, `Mm_YBot_Style` in `StyleTransferMM.unity`): no errors,
  pose finite, phase covers the cycle. Torso lean: unstyled 8.8°–14.5°, Old +4.4° with the gate on
  (mean gate 0.36), +13.5° with it off; Elated leans back (−9.0° ungated).
- **Open: the gate is too conservative on MM output.** It closes about two thirds of the way on
  motion matching, against 0.89 on 100STYLE's own frames. Likely causes: the MM database's
  contacts use 0.15 m/s (so the causal phase rate differs from training's), and MM speeds lie
  outside 100STYLE's neutral distribution. Recalibrate the gate on method output, or detect
  contacts for the runtime phase the same way training did.
- **Open: speed under `Stylize` with `DirectionControlInput`.** The free simulation object runs
  ahead and the search speeds up to catch it, so a slower style shows only partly (0.90 against
  1.06 m/s ungated). An anchored input shows the real effect.
- C# EditMode: 590 / 593 pass, all 33 Style tests included. The 3 failures are in
  `TimelineKeySelectionTests`, which this work did not touch.
