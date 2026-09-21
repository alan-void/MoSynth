# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, and others reading this file) when working with code in this repository.

## Project Overview

**MoSynth** is a motion synthesis system for character animation that combines motion matching with neural motion fields. It's a Unity project (v6000.4.4) that synthesizes realistic character locomotion by blending traditional motion matching with learned motion fields. The system uses a pipeline architecture where pose data flows through multiple "stages" that transform it, with interop to Python via PythonNET for neural network inference.

### Code Style
- Readability first: keep code readable, concise, clear, and easy to understand. Prefer the straightforward implementation over a clever or compact one.
- Do not solve a task with a workaround that makes the code cryptic (hidden coupling, magic values, misusing an existing mechanism because it happens to work). If the clean solution requires touching more code, touch more code.
- If a broader architectural change would allow meaningfully better code quality than working within the current structure, say so and propose it instead of silently working around the limitation — the user decides whether to take it.
- Do not write comments in generated code that depend on the context of the conversation that produced them (e.g. referencing the task, a fix, a prior approach, or "why we're changing this now"). Comments should only explain non-obvious WHY that a reader with just the code in front of them would need.
- Keep that WHY to one or two sentences. A comment is a signpost, not an essay. If the full argument needs a rejected alternative, a measurement, or a critique of a package’s internals to land, that argument belongs in the wiki under `openwiki/`, and the comment carries a one-line pointer to the page instead.
- Keep `<summary>` to a sentence or two naming what the member does and any non-obvious contract. Use `<remarks>` to point at the wiki page holding the detail, not to hold the detail. A long `<remarks>` block — especially on a private member — means the content is in the wrong place.
- When a change invalidates something the wiki asserts, update the wiki in the same pass. Stale docs are worse than no docs.
- Use `var` to declare variables unless an explicit type makes the code clearer (IDEs surface type info well).
- Naming:
  - Private instance/static fields: `_camelCase` (e.g. `_skeleton`, `_poseSet`)
  - Inspector-serialized fields (`[SerializeField]` private, or public fields that aren't `[NonSerialized]`): `camelCase` (e.g. `poseSkeleton`, `synthesisFrameRate`)
  - Public non-serialized fields (e.g. `[NonSerialized] public`): `PascalCase` (e.g. `CurrentPose`, `SkeletonTransforms`)
  - Types, methods, properties, events, constants: `PascalCase`
  - Locals and parameters: `camelCase`

### Tech Stack
- **Unity**: 6000.4.4f1 (LTS)
- **Language**: C# / Python (CPython 3.13 via PythonNET)
- **Key Dependencies**: 
  - Barracuda (3.0.2) - in the manifest for neural network inference in Unity, but nothing consumes it today; it is deprecated on Unity 6
  - Mathematics, Collections, Jobs (for performance)
  - PyThonNET - C#/Python interoperability
  - GameplayTags - hierarchical tag assets, embedded as a **git submodule** at
    `Packages/com.alanvoid.gameplaytags` rather than a manifest git URL, because it is developed
    alongside this project. A fresh clone needs `git submodule update --init` or Unity finds no
    `GameplayTags` assembly and `AnimationTools` fails to compile
  - Scipy, NumPy, PyTorch (Python side)

## Architecture

### Core Pipeline: MoSynthStage System

The motion synthesis pipeline is built on an abstract stage architecture:

```
MotionSynthesisComponent
  └── Stages[] (pipeline that transforms pose data each frame)
      ├── MotionMatchingStage (searches animation database for best pose)
      ├── MotionFieldStage (applies neural motion field transformations)
      └── Other stages (Inertialization, PoseSetVisualizer, etc.)
```

**Key classes:**
- `MoSynthStage` (abstract base): interface for all stages with `Init()`, `Apply(PoseBuffer, deltaTime)`, `OnDestroy()`
- `MotionSynthesisComponent`: main orchestrator that runs stages each frame; manages skeleton transforms and pose updates
- `PoseBuffer`: the mutable pose data structure carrying joint positions, rotations, velocities, contact states
- `Skeleton`: a bone hierarchy defined by a Transform tree — serializes as a single root Transform, and the bone list is the preorder depth-first walk from that root (bone 0 = root); index+1 convention for bone IDs (0 = unset), so id 1 is the root bone
- `SkeletonBone`: one bone of a `Skeleton` — a `Skeleton` + `Transform` pair; also the serializable type used for a standalone bone-reference field (contact bones, root motion bone)
- `SkeletonBoneOverrides`: binds a `Skeleton` (an asset rig at rest) to a different, live rig by name, with per-bone overrides; `MotionSynthesisComponent.characterRig` is one of these

### Data Flow: Pose Representation

Poses are packed into flat arrays for efficiency with Python interop:
- **Format**: `(..., num_bones + 2, 4)` array, `num_bones` counting the virtual character-frame joint `motion_field.action_predictor.with_virtual_root` adds ahead of the pose set's own (single, real) skeleton
  - Index 0: character-frame position (xyz) + padding (1 float) — zero, since the pose is already expressed in its own frame
  - Index 1: root bone position (xyz) + padding (1 float) — `core/pose.py` still calls this `hips`, but it is the skeleton's real bone 0, not a separate Hips joint
  - Indices 2+: joint rotations as quaternions (xyzw per joint) — index 2 is the frame's own (identity) rotation, index 3 the root bone's
- **Velocities**: parallel arrays using same format but represent per-second rates (scaled by `frame_time` to get one-frame deltas)
- **Python classes**: `Pose` (immutable), `PoseDelta` (velocity), `Skeleton` (FK/IK)
- `motion_field.action_predictor.get_pose_arrays` unpacks a synthesized pose back down to plain per-bone arrays (no frame slots — C# derives its own frame from bone 0) for the PythonNET boundary

### PfnnStage

A phase-functioned neural network (Holden et al. 2017) as a `MoSynthStage`, running the model in
Python behind PythonNET.

- **C# side** (`Assets/Pfnn/`): `PfnnStage` assembles the trajectory window, joints and gait phase
  each tick, calls one stateless Python `step`, converts the predicted 6D rotations, and writes the
  pose through `CharacterSpacePose.Apply`. The stage owns the state — phase, joints and the
  trajectory history ring — because a PFNN's state is a pose Unity has to write anyway
- **Python side** (`Python/pfnn/*.py`): `pfnn.dataset` packs vectors, `pfnn.model` is the network,
  `pfnn.trainer` fits it, `pfnn.io` stores it, `pfnn.runtime` runs it
- **Which bones it predicts is authored on `PfnnConfig`**, as a sparse name-keyed exclusion list
  drawn as the skeleton hierarchy. Excluding a bone excludes its subtree, because the network
  predicts rotations and a rotation needs its parent's frame. The heuristic behind the
  "Exclude Fingers And Leaves" button lives only in that button
- The past half of the trajectory window is the character's own history, which is what training
  used. The future half is the control input's wish **blended with the network's own predicted
  trajectory**, graded by horizon: near the character the prediction wins, at the one-second horizon
  the request does. A request alone can describe a path no body could follow, which training never
  contained
- Full detail: `openwiki/pfnn/`

### Steering a character: three roles

Intent, trajectory origin and root reconciliation are separate concerns, and the combination decides
how a character behaves:

- **Free simulation object** (`DirectionControlInput`): the input integrates a position of its own
  and the search chases it. Nothing closes the loop, so the character permanently lags
- **Anchored** (`AnchoredDirectionControlInput`): the trajectory starts at the character's own frame,
  re-read every tick, and only the velocity springs carry state. No position exists to drift. It
  describes motion, never location, so it cannot place a character or measure path error
- **Capsule authority** (`RootFollowStage`, in `AnimationTools`): a post-pose stage that puts the
  character on a target — a capsule, or a point on a path via `IFrameTarget`. The correction is
  written as bone 0's frame velocity, the same channel `Inertialization` writes through, so it can be
  damped and rate-limited; a half-life of 0 is exact placement. It must run **after**
  `Inertialization`, and it follows in the ground plane only
- Anchored and capsule authority **do not compose** — an anchored input has no object to follow

**One reference, and it lives on the input.** Every control input derives from
`MotionSynthesisControlInput` (`AnimationTools`), which holds the only link to the
`MotionSynthesisComponent` — authored, or found on a parent when the field is empty — and *claims*
that character in `OnEnable`. A stage never carries a control-input reference of its own: it reads
`MotionSynthesisComponent.ControlInput` and casts. A second input enabling on a claimed character
warns and disables itself, so switching between two inputs on one character means deactivating one.
An override of `OnEnable` that acquires anything must call base and then check `IsBound`.

Stick input needs no wiring either: `UserInput` publishes the move action on a static
`MoveChanged` event, and the base subscribes any input implementing
`IMotionSynthesisDirectionControlInput`. That is why `AnimationTools` references `Unity.InputSystem`
and owns `InputActions.inputactions` — an asmdef assembly cannot reference `Assembly-CSharp`, where
`UserInput` used to live.

Full detail, including the rejected alternatives: `openwiki/animation-tools/root-following.md`.
Hazards: `openwiki/agents/motion-matching/root-and-control-input-hazards.md`.

### MotionFieldStage (Current Feature Branch)

Integrates a neural motion field into the pipeline via PythonNET:

**C# side (`Assets/MotionField/MotionFieldStage.cs`):**
- Initializes Python engine, loads the `motion_field.field` and `motion_field.action_predictor` modules
- Each frame `StepPolicy()` calls one of the four Python policy methods, chosen by the stage's
  `Policy` enum. The dispatch lives in C# so the policy never crosses the boundary as a value
- Converts Python arrays back to `PoseBuffer` for Unity

**Python side (`Python/motion_field/field.py`):**
- KNN search over motion states (positions + velocities as features)
- Blends nearest neighbor poses and velocities
- Policy methods: `optimal_action()` (value function, falling back to greedy when none is loaded),
  `greedy_action()`, and the debug pair `get_next_pose()` / `get_next_pose_from_field()`
- Supporting: `get_knn()`, `get_batched_knn()`, `build_motion_states()`, `load_value_function()`

**Python setup:**
- The CPython DLL and venv paths belong to a machine, not to the project, so they live outside its assets. `PythonRuntime.EnsureInitialized` resolves them from the `MOSYNTH_PYTHON_DLL` / `MOSYNTH_PYTHON_VENV` environment variables first, then from `PythonPathSettings` — `UserSettings/MoSynthPython.json`, which is gitignored and edited under **Project Settings → MoSynth → Python** — then, for the DLL only, PythonNET's own `PYTHONNET_PYDLL`. The variables win because they are the only source that reaches a machine with no project folder to read. Both types live in `AnimationTools` (namespace `AnimationTools`), not in `MotionField`, because more than one synthesis method now needs them
- Project modules import from `PythonRuntime.ScriptsFolder` — the repository's `Python/` folder, derived from `Application.dataPath`
- Python 3.13 is required for PythonNET compatibility

### Python module layout

`Python/` is one package per subsystem — `core`, `formats`, `training`, `pfnn`, `lmm`,
`motion_field`, `debugging` — and `Python/` itself is the import root, because that is the single
folder `PythonRuntime` puts on `sys.path`. Three consequences follow, and all three are easy to
break:

- **Imports are absolute from that root**, `from core.pose import Pose`, never relative. A module
  is reached the same way whether Unity imported it or a shell did, so there is one spelling to
  keep right. Inside its own package a module is imported bare (`from lmm import dataset`);
  from outside, or when the bare name shadows a stdlib one, it is aliased (`from pfnn import io as
  pfnn_io`)
- **C# names modules by that dotted path**: `PythonRuntime.Import("pfnn.runtime")`. Moving or
  renaming a module breaks a string literal no compiler checks, so grep `PythonRuntime.Import`
  after any move
- **Scripts run with `-m`, from the `Python/` folder**: `python -m pfnn.trainer …`. Running a file
  by path (`python pfnn/trainer.py`) puts `Python/pfnn/` on `sys.path` instead of `Python/`, and
  every absolute import in the file fails. The usage examples in each module's docstring assume
  that working directory, which is why their database paths start `../Assets/`

## Key Directories & Files

```
Assets/
├── AnimationTools/
│   ├── Runtime/Core/
│   │   ├── MoSynthStage.cs          [abstract base for all stages]
│   │   └── MotionSynthesisComponent.cs [main orchestrator; binds an asset-backed Skeleton to the scene rig via SkeletonBoneOverrides]
│   ├── Runtime/Skeleton/
│   │   ├── Skeleton.cs              [Transform-tree skeleton: single root reference; bone list is the DFS walk from it]
│   │   ├── SkeletonBone.cs          [Skeleton + Transform pair; also the standalone bone-reference field type]
│   │   ├── SkeletonBoneOverrides.cs [binds a Skeleton (asset rig) to a different live rig, by name with overrides]
│   │   ├── BoneNameConventions.cs   [name heuristics for contact bones]
│   │   ├── SkeletonRigBuilder.cs    [materializes Skeleton as GameObject hierarchy]
│   │   ├── SkeletonData.cs          [unmanaged, Burst-compatible mirror of a Skeleton's hierarchy]
│   │   └── ISkeletonProvider.cs     [interface for skeleton access]
│   ├── Runtime/Animation/
│   │   ├── AnnotatedAnimationClip.cs [a clip + skeleton, a frame slice, and a clip component list]
│   │   ├── AnimationClipComponent.cs [base for polymorphic per-clip annotation]
│   │   ├── GaitPhaseComponent.cs    [footfall anchors + how they were detected]
│   │   ├── GaitPhase.cs             [footfalls -> phase; the project's only definition of it]
│   │   └── GaitMeasure.cs           [foot contacts + ground speed off a baked clip, for the editor and the bake]
│   ├── Runtime/Pose/
│   │   ├── PoseBuffer.cs            [mutable pose in motion synthesis]
│   │   ├── CharacterSpacePose.cs    [Extract/Apply: a pose measured in, and written back from, its character frame]
│   │   └── PoseFK.cs                [forward kinematics]
│   ├── Runtime/Stages/
│   │   ├── RootFollow.cs            [pure: the frame velocity that lands the character on a target]
│   │   └── RootFollowStage.cs       [places the character on a capsule or path point, via bone 0's rates]
│   ├── Runtime/ControlInput/
│   │   ├── IMotionSynthesisControlInput.cs [synthesis-agnostic steering surfaces]
│   │   ├── MotionSynthesisControlInput.cs [base for every control input: the one MSC reference, the claim, the input subscription]
│   │   ├── UserInput.cs             [publishes the move action as a static channel; InputActions.inputactions lives beside it]
│   │   ├── IFrameTarget.cs          [a target that answers for its own position, not its Transform's]
│   │   └── TrajectorySteering.cs    [request damping + horizon prediction, shared by MM and PFNN inputs]
│   ├── Runtime/Python/
│   │   ├── PythonRuntime.cs         [shared CPython bootstrap and module reloading]
│   │   └── PythonPathSettings.cs    [per-user interpreter paths in UserSettings/MoSynthPython.json]
│   ├── Editor/Python/PythonSettingsProvider.cs [Project Settings → MoSynth → Python, and Verify Setup]
│   ├── Runtime/Benchmark/           [sweep config, lap tracking, motion-quality & cost metrics]
│   ├── Runtime/Evaluation/          [PathFollowingMetric(sCalculator): in-scene A/B tool + pure metrics]
│   ├── Runtime/Recording/           [MotionRecorder, channels, manifest, reader]
│   ├── Editor/Benchmark/            [sweep driver, CLI entry point, menu, report writer]
│   ├── Editor/Preview/              [SkeletonPreview: the off-screen posed-skeleton render + overlay seam]
│   ├── Editor/ClipEditor/           [the clip editor window: axis, timeline, track seam, registry]
│   │   └── Tracks/GaitPhaseTrack.cs [the worked example track]
│   ├── Editor/
│   │   └── SkeletonBoneDrawer.cs    [Inspector dropdown for bone selection]
│   └── (other runtime & editor features)
├── MotionField/
│   ├── MotionFieldStage.cs          [stage implementing neural field]
│   ├── MfConnector.cs               [data connector for field]
│   └── MotionFieldConfig.cs         [serializable config: clips, skeleton, hyperparameters]
├── Pfnn/
│   ├── PfnnStage.cs                 [phase-functioned network stage; owns phase, joints, trajectory history]
│   ├── PfnnConfig.cs                [clips, skeleton, predicted-bone selection, network + training settings]
│   ├── PfnnBoneSelection.cs         [marshals the selection to Python; binds a checkpoint's bone names to a rig]
│   ├── PfnnTrajectory.cs            [history ring buffer and the ground-plane frame transform]
│   ├── PfnnControlInput.cs          [steering contract; + spline and direction inputs]
│   └── Editor/                      [config inspector, training, default bone selection, agreement check]
├── MotionMatching/
│   ├── Runtime/CharacterController/
│   │   ├── MotionMatchingControlInput.cs   [steering contract + the character-frame query helpers]
│   │   ├── DirectionControlInput.cs        [stick/WASD over a free simulation object]
│   │   ├── AnchoredDirectionControlInput.cs [stick/WASD anchored on the character; no drift]
│   │   └── SplineControlInput.cs           [open-loop path follower; also an IFrameTarget]
│   ├── Runtime/Core/
│   │   ├── MotionMatchingStage.cs   [database search stage]
│   │   └── ContactVisualizerStage.cs
│   ├── Runtime/Pose/                [pose set visualization]
│   ├── Runtime/Features/            [matching feature channels]
│   └── Editor/                      [tools and visualization]
├── Scenes/
│   ├── ExampleSimpleMMStages.unity  [demo scene with motion matching]
│   └── (animation/scene test assets)
└── Animation/
    ├── MotionMatching/              [animation database assets]
    └── Pfnn/                        [PfnnConfig asset]

Python/                              [one package per subsystem; see "Python module layout"]
├── core/                            [the data model every synthesis method shares]
│   ├── pose.py                      [pose data classes]
│   ├── skeleton.py                  [skeleton FK/IK]
│   ├── pose_set.py                  [PoseSet: the Python mirror of the C# pose database]
│   ├── simulation_frame.py          [character frames, FK, and per-frame rates]
│   └── quaternions.py               [quaternion utilities]
├── formats/                         [readers for the binary databases Unity bakes]
│   ├── binary_reading.py            [BinaryWriter primitives shared by the format readers]
│   ├── pose_set_importer.py         [.mmpose reader]
│   └── feature_set_importer.py      [.mmfeatures reader: matching feature vectors + schema]
├── training/                        [what PFNN and LMM both train on]
│   ├── training_data.py             [per-frame arrays a PFNN/LMM model trains on, + .npz]
│   └── neural_packing.py            [named-block vector layouts shared by both models]
├── pfnn/                            [phase-functioned neural network]
│   ├── dataset.py                   [bone selection and the PFNN input/output vector layouts]
│   ├── model.py                     [the phase function and the network it drives]
│   ├── io.py                        [.pfnn.npz checkpoint format; numpy only]
│   ├── trainer.py                   [training loop and the Unity entry point]
│   ├── runtime.py                   [stateless inference policy, + an offline rollout]
│   └── agreement.py                 [compares the C# and Python character-frame definitions]
├── lmm/                             [learned motion matching]
│   ├── dataset.py                   [pose/character vector layouts]
│   ├── fk.py                        [differentiable FK used by the training loss]
│   ├── io.py                        [.lmm.npz checkpoint format; numpy only]
│   ├── model.py                     [compressor, decompressor, stepper, projector]
│   ├── trainer.py                   [training loop and the Unity entry point]
│   └── runtime.py                   [inference policy, + an offline rollout]
├── motion_field/                    [neural motion field]
│   ├── field.py                     [neural motion field implementation]
│   ├── io.py                        [.mffield.npz value function format]
│   ├── trainer.py                   [fitted value iteration]
│   ├── embedding.py                 [UMAP projection for the debug visualizer]
│   └── action_predictor.py          [animation loading & conversion]
├── tests/                           [stdlib unittest suites; no Unity or venv extras needed]
└── debugging/                       [debug scripts, not in builds]

Packages/manifest.json               [Unity package dependencies]
ProjectSettings/ProjectVersion.txt   [Unity version: 6000.4.4f1]
```

## Common Development Tasks

### Opening the Project
```powershell
# Open in Unity (requires Unity 6 LTS installed)
& "C:\Program Files\Unity\Hub\Editor\6000.4.4f1\Editor\Unity.exe" -projectPath "E:\UnityProjects\MoSynth"

# Or open the solution in Visual Studio/Rider
start MoSynth.sln
```

### Running a Demo Scene
1. Open `Assets/Scenes/ExampleSimpleMMStages.unity` in the Unity Editor
2. Press Play in the Editor
3. Use WASD/mouse for character control (if MoSynthControlInput is configured)

### Python Environment Setup
```bash
# Create a venv anywhere (Python 3.13 is required for PythonNET compatibility)
python -m venv .anim_env
.anim_env\Scripts\activate

# Install dependencies
pip install numpy scipy torch

# Smoke-test the Python side (from the Python/ folder)
python -m unittest discover -s tests -t tests
```

Then tell the project where they are, in **Project Settings → MoSynth → Python**. That writes
`UserSettings/MoSynthPython.json`, which is gitignored, so the paths never travel to another machine.
`MoSynth/Python/Verify Setup` (also a button on that page) starts the interpreter and logs the
version and the numpy/torch it found, which is the quickest way to tell a wrong path from a missing
package.

For a build agent, or any machine with no project folder to read, the environment variables override
the settings file:

```powershell
setx MOSYNTH_PYTHON_DLL  "C:/path/to/python313.dll"
setx MOSYNTH_PYTHON_VENV "C:/path/to/.anim_env"
```

Restart Unity so it picks the variables up. Either way the interpreter is chosen once per process, so
a path changed mid-session takes effect at the next domain reload.

### Building for Distribution
The project uses standard Unity build pipeline:
1. `File → Build Settings` in Unity Editor
2. Select target platform and scenes
3. Configure player settings (scripts-only for development)

### Benchmarking the synthesis methods

A sweep runs every configured method against every path, one character at a time, and writes
`results.csv` / `results.json` plus the raw per-tick recordings.

- **Configure**: a `SynthesisBenchmarkConfig` asset (`Assets/Create/MoSynth/Synthesis Benchmark
  Config`). A "method" is a character prefab plus an optional list of `BenchmarkOverride`s, so the
  four MotionField policies are four config entries over one prefab rather than four prefabs. Paths
  are prefabs with a `SplineContainer` in a folder the config points at
  (`Assets/Benchmarks/Paths` by default) — drop a prefab in and it joins the sweep
- **First-time setup**: `MoSynth/Benchmark/Create Starter Assets` extracts the characters and
  splines already wired up in `Assets/Scenes/ExampleSplines 1.unity` into those folders and writes a
  config pointing at them; `MoSynth/Benchmark/Create Standard Paths` adds the parametric suite
  (Circle, Oval, FigureEight, SharpCorners). Both replace the open scene and are re-runnable
- **Random paths**: `MoSynth/Benchmark/Create Random Paths…` opens a window that writes a seeded
  batch into `Assets/Benchmarks/Paths/Random`, which the folder scan reaches, so they join the sweep
  with no config edit. Names carry the seed and slot (`Random_s4821_03_SharpLoop`), and geometry is a
  pure function of the two, so a report row identifies the exact curve and the same seed rebuilds it.
  Six families: each of three shapes — a closed `Loop`, an open `Line` corridor, and an open `Walk`
  that is contained by nothing — with rounded turns or with corners. The knot-count range is the dial
  that decides how convoluted a path is; the extent means overall size in every family, open paths
  being scaled to the same length as a loop of that extent. **Nothing enforces a minimum turn
  radius** — how well a method follows a demanding curve is the measurement — so the generation log
  reports the tightest turn (smooth) or sharpest corner (sharp) per path instead.
  `MoSynth/Benchmark/Delete Random Paths` clears the batch — leaving a stale one in place silently
  adds its runs to every later sweep
- **Run, visible**: `MoSynth/Benchmark/Run Sweep`, or the button on the config's inspector. It
  replaces the open scene with an empty one and owns the play session
- **Run, headless**: `Tools/run-benchmark.ps1` (`-Visible` to watch it). It must not pass `-quit`:
  the sweep runs across play-mode frames, so `SynthesisBenchmarkDriver` sets the exit code itself
- Each run spawns the character **on** the path's first point facing the tangent, waits out
  `settleTime`, then measures at least `lapsRequired` laps of the character's own progress round the
  spline — not a fixed duration, because `MotionFieldSplineControlInput` has no speed model to
  predict a lap time from. A run that cannot finish is reported with `timedOut` set
- **Open paths** finish at the far end instead: `lapsRequired` is ignored, and `completedLaps` reads
  as the fraction of the path covered rather than a lap count. Both spline inputs clamp instead of
  wrapping there — see `SplineControlInput.Fold`

Three metric families, all computed from the recording after the fact so the calculators stay pure
and unit-tested: path following (`PathFollowingMetricsCalculator`, reused unchanged), motion quality
(`MotionQualityMetricsCalculator` — footskate, root jerk, discontinuity rate), and cost
(`SynthesisCostMetrics` — per-stage `Apply()` ms and GC bytes per tick). Cost is measured wall-clock
inside the Editor: valid for comparing methods within one sweep on one machine, and nothing more.

### Testing & Validation
- **C# edit-mode suites**: `MoSynth/Tests/Run EditMode Tests` runs `AnimationTools.Tests`,
  `MotionMatching.Tests` and `Pfnn.Tests` and writes `Temp/animtools_test_results.txt`, which
  survives the domain reload a test run causes — so a scripted caller reads results from there, not
  from the console. That assembly list is hardcoded in `TestResultDump`; a new test assembly must be
  added to it or its tests silently do not run
- **Python suites**: `python -m unittest discover -s Python/tests -t Python/tests`. They use only
  numpy and scipy and build their own fixtures, so they need neither Unity nor a generated database
- **Editor play mode**: test motion synthesis visually
- **PFNN**: train from the config's inspector, or standalone —
  `python -m pfnn.trainer <database-folder> <name> --out out.pfnn.npz`. Judge a checkpoint
  before wiring a character with `python -m pfnn.runtime <checkpoint> --database <folder>
  --name <name> --rollout 300`, which runs the model against its own predictions.
  `MoSynth/Pfnn/Check Training Agreement` compares the C# and Python character-frame definitions on
  the same frames — the check that the model is run on the arrays it was trained on
- **Motion database**: `MoSynth/Database/Regenerate Motion Matching Databases` rebuilds every
  `MotionMatchingData` asset's `.mmpose` and `.mmfeatures`. Run it after any change to an extraction
  format or a feature definition — the files are unversioned, and a stale one is refused rather than
  silently misread only because each carries a schema block to check against

## Important Patterns & Conventions

### Stage Development
When adding a new `MoSynthStage`:
1. Inherit from `MoSynthStage` abstract class
2. Override `Init(MotionSynthesisComponent)` for setup (called once at startup); read `motionSynthesisComponent.Skeleton` here if the stage needs one. A stage backed by its own database (`MotionMatchingStage`, `PoseSetVisualizerStage`) validates that skeleton against the database's skeleton with `Skeleton.StructurallyEqual` and errors out on mismatch
3. Override `Apply(PoseBuffer pose, float deltaTime)` to transform pose (called every frame); return `false` to halt the pipeline
4. Implement `IDisposable.Dispose()` if holding unmanaged resources (e.g., Python state)
5. `OnDestroy()` is called automatically when the scene unloads

### Python Interop with PythonNET
- Always wrap Python calls in `using (Py.GIL()) { ... }` to acquire the Global Interpreter Lock
- Use `dynamic` types for Python objects in C#
- Python modules should be added to `sys.path` or placed in `Assets/../Python/`
- For module hot-reloading during development, enable `reloadPythonModules` on `MotionFieldStage`. It calls `PythonRuntime.InvalidateProjectModules()`, which forgets every cached module under `Python/` so the next import re-reads them all — a per-module `importlib.reload` left dependencies cached and broke on cross-module edits

### Skeleton & Bone Binding
- A `Skeleton` is a Transform tree: bone identity is a Transform reference, not a name string. The bone list is the preorder depth-first walk from the skeleton's root Transform, built lazily and cached per root. A name fallback exists only for cross-rig resolution (`SkeletonBone.ResolveIndex`), e.g. matching a contact bone picked on one rig against another that's structurally the same
- Bone indices follow the index+1 convention (bone ID = index + 1, 0 = unset); index 0 is the skeleton's real root bone, and the pose skeleton is the same tree as the clip skeleton — nothing is prepended
- `SkeletonBoneOverrides` binds a `Skeleton` (an asset rig at rest) to a different, live rig by name, with per-bone overrides for mismatched names; `SkeletonBone` is drawn by `SkeletonBoneDrawer` with a rig-aware dropdown
- **Two invariants nothing enforces at compile time:**
  1. A `Skeleton`'s rest pose is read live off its Transforms, so it must point at an ASSET rig (imported FBX or prefab), never a live scene rig — a skeleton over an animated scene rig reports the current pose as the rest pose and silently corrupts FK. `MotionSynthesisComponent` serializes its own `Skeleton` field — assigned to the rig's root bone from the FBX asset, since the field's drawer refuses scene objects — and binds it to the scene rig via `SkeletonBoneOverrides`. **An asset rig is not automatically at rest**: Unity poses an imported model's hierarchy from its *first animation take*, so an exported FBX has to lead with a rest-pose take or it imports holding a frame of motion, passes the `IsPersistent` check, and hands that frame over as the rest pose — which is what `RestLocalAxis` derives the character frame's forward axis from. `Skeleton.TryValidateRestPose` catches the part of this that is measurable (the root bone standing away from the rig origin); see `openwiki/animation-tools/retargeting-pipeline.md`
  2. Everything under a skeleton's root Transform becomes a bone, so a rig must have nothing but bones beneath its skeleton root — a mesh node, IK helper, or attachment point there would shift every index after it

### Clip components

- `AnnotatedAnimationClip` carries a `[SerializeReference] [SubclassSelector] List<AnimationClipComponent>`.
  Annotation a clip's curves cannot supply is added by declaring a component type, not by adding
  another field to the clip. Same seam as `MoSynthStage`: `[Serializable]`, parameterless ctor,
  public mutable fields, and subclasses in any assembly referencing `AnimationTools` are offered
  with no registration
- **A managed reference is keyed in YAML by its (class, namespace, assembly) triple.** Renaming a
  component type, moving its namespace, or moving its file to another assembly orphans every
  authored instance and drops the data on the next save. `[FormerlySerializedAs]` renames fields,
  not types — `[MovedFrom]` is the tool, and is not yet used anywhere here
- `GaitPhaseComponent` is the first one: a clip's footfall anchors plus the per-clip settings they
  were detected with. `GaitPhase` turns anchors into phase, and **is the only implementation of that
  rule** — `PoseExtractor` evaluates it at bake time and writes phase and its rate into the
  `.mmpose`, so Python reads it instead of reconstructing it from contacts. A stretch with no anchors
  is answered by ground speed, **one frame at a time**: a standing frame sweeps at a fixed period so
  a model can learn a stationary pose across the whole cycle, a moving one holds the phase at rate 0
  and is dropped. Judging the stretch as a whole instead discards the stand with it, because a clip
  that stands and then walks off is already accelerating before its first heel strike. Contacts
  for both the editor and the bake come from one routine, `GaitMeasure` — see
  `openwiki/animation-tools/animation-sources.md`
- **Annotation frames are clip-local, and nothing may delete annotation for falling outside the
  clip's `[startFrame, endFrame)` slice.** Both were once the other way round, and trimming a clip
  destroyed the anchors beyond the new end — permanently, on every `OnValidate`. `GaitPhase.Evaluate`
  already ignores out-of-range anchors, so the deletion protected nothing
- `AnimationTagComponent` is the second one: hierarchical `GameplayTagSO` tags over stretches of a
  clip, one channel per tag, each stored as the **boolean keyframes it flips at** rather than as
  intervals. `AnimationTagging.FindSegments` answers a `GameplayTagQuery` with
  `AnimationClipSegment`s. A query runs downhill only — `action.walk` answers a query for `action`,
  never the reverse. Full detail: `openwiki/animation-tools/clip-tags.md`
- **How a component is drawn is its own decision too.** `AnnotatedClipEditorWindow`
  (`MoSynth/Animation/Clip Editor...`) gives each component a timeline lane, an inspector and an
  optional 3D preview overlay. Declare an `AnimationClipComponentTrack` tagged
  `[ClipComponentTrack(typeof(TheComponent))]` and it is discovered with no registration, exactly
  like the components themselves; a component with no track still gets a lane and a fully editable
  inspector from `DefaultComponentTrack`. Before writing one, read
  `openwiki/agents/animation-tools/clip-editor-tracks.md` — the edits that silently drop anchors are
  listed there
- **Keyframe lanes share one Blender-style keymap**, in `Editor/ClipEditor/Keys/`: box select, `G`
  to move, `S` to scale, numbers to type an exact value, `X` to delete. `AnimationTagTrack` and
  `GaitPhaseTrack` both run on it, and a third such track should too rather than hand-rolling drags.
  Selection is keyed by `(row, frame)`, never by list index, because every edit re-sorts its list.
  Note `A` is select-all, so framing the view moved to `Home` / `.`. **Handle keys off raw `e.type`,
  never `e.GetTypeForControl`** — it drops key events unless that control owns `keyboardControl`, and
  a lane's is Passive; that alone had killed every binding in both tracks. The window is UI Toolkit,
  so a mode follows the cursor by sampling `mousePosition` on repaint rather than relying on
  `MouseMove`. `Add Component` and the lane list live in the timeline's draggable left gutter, not
  the toolbar
- The `Tag` struct, `PoseSet.AddTag` and the `.mmpose` tag block are **still** a dead subsystem, but
  no longer for want of an author: `AnimationTagComponent` is one and is deliberately not wired to
  it, because nothing downstream reads tags out of the database and doing so would cost a full
  database regeneration for dead weight. Leave the dead code alone until something wants to read it

### `SkeletonAnimation` has one source of truth
- A `SkeletonAnimation` (and its `AnnotatedAnimationClip` subclass) stores a clip and a `Skeleton`, and nothing else about where the bones are. There is no separate rig field: a second reference could disagree with the skeleton, and did — an earlier `GameObject` → `Transform` change orphaned every asset's rig
- `AnimationClipBaker` still needs the asset's main object, because `clip.SampleAnimation` matches curve paths against the hierarchy it is handed and importers write those paths relative to the main object (the FBX root, or the container `BvhImporter` puts its bones under). It derives that as `skeleton.Root.root` — the root bone's topmost ancestor — so the bake target cannot drift from the skeleton
- The root-bone guess (single child, else a `*Hips` descendant) now happens once, in `AnnotatedClipFactory`, and is written into the asset. Nothing re-derives it at access time. That factory is shared by the single-selection menu (`Assets/Create/MoSynth/Annotated Clip From Selection`) and by `MoSynth/Animation/Create Annotated Clips From Model...`, which creates one asset per matching take of a model into a chosen folder — the shape a retargeted capture session arrives in. A batch re-run updates assets in place (configs reference clips by GUID) and adds a `GaitPhaseComponent` only where there is none, so corrected anchors survive
- **Bulk footfall detection** is `DetectFootfallsMenu`: `Assets/MoSynth/Detect Footfalls` on a Project-window selection (folders included, via `SelectionMode.DeepAssets`), mirrored at `MoSynth/Animation/Detect Footfalls In Selection`. It seeds a `GaitPhaseComponent` where there is none, and **skips a clip that already holds anchors unless the prompt is answered otherwise** — nothing records that a clip was hand-corrected, so that prompt is the only guard. It writes to assets rather than through a `SerializedObject`, so it snapshots each clip with `Undo.RegisterCompleteObjectUndo` and calls `AnnotatedClipEditorWindow.RefreshOpenWindows()` afterwards
- A `Skeleton` field is drawn by `SkeletonDrawer` as a rig object field plus a root-bone dropdown, because a bone *inside* an imported rig is not reachable from the Project window or the object picker — only the asset's main object is. Drop the rig in to seed the field, then pick the actual root from the dropdown. It deliberately does not guess: every `Skeleton` — a clip's or a config's — starts at the rig's root bone. `SkeletonDrawer` and `SkeletonBoneDrawer` share their listing code via `BonePopup`
- `TryValidate` rejects a skeleton root that is not `EditorUtility.IsPersistent`, and `Skeleton.TryValidateRestPose` rejects one standing away from its rig's origin. Those two are the whole automatic defence on invariant 1 above, and neither catches a rig posed at a frame that happens to sit near the origin

### `SkeletonBone` naming collision
`AnimationTools.SkeletonBone` collides with the built-in `UnityEngine.SkeletonBone`. Any file outside the `AnimationTools*` namespaces that names the type needs `using SkeletonBone = AnimationTools.SkeletonBone;`.

### Config assets and their skeletons
- `MotionMatchingData` and `MotionFieldConfig` each carry an explicit T-pose `Skeleton` field whose root is the rig's real root bone — the pose skeleton is identical to each clip's skeleton, so nothing is prepended at load. The compatibility check is therefore `Skeleton.StructurallyEqual`, not a shifted comparison
- Neither config has a configurable simulation-frame bone anymore: `SimulationFrameDef.Default(skeleton)` always sets the reference bone to the skeleton root (bone 0) and the forward axis to that bone's rest forward (`Skeleton.RestLocalAxis(0, math.forward())`); `PoseSet.SimulationFrame` is a computed property returning this default. That reference bone is what `SimulationFrame.Compute`/`ComputeVelocity` (`Assets/AnimationTools/Runtime/Pose/SimulationFrame.cs`) derives the ground-projected, yaw-only character frame from — that frame is never stored on the pose, only computed on demand from a `SimulationFrameDef { ReferenceBoneIndex, ForwardAxisLocal }`
- Both assets expose `TryValidate(out string error)`. `GetOrImportPoseSet()`/`GetOrImportFeatureSet()` return null silently when it fails, because `OnValidate` reaches them every Inspector repaint; the custom Inspector shows the reason in a HelpBox and disables Generate. `SkeletonAnimation` follows the same pattern
- The `.mmskeleton` format is gone. C# takes the skeleton from the config's own field, but Python has no ScriptableObject to read, so `.mmpose` opens with a skeleton block — per bone: name, parent index, rest local position, rest local rotation — written by `PoseSerializer.WriteSkeleton` and read by `formats.pose_set_importer.read_skeleton`, which derives the same structural simulation frame (reference bone 0, forward axis from that bone's rest rotation) rather than reading it as separate fields. Keeping it in the same file as the poses is what stops the two drifting apart
- **Neither `.mmpose` nor `.mmfeatures` is versioned**, by decision: everything is regenerated when a format moves, so a version byte guards nothing. A stale `.mmpose` is caught instead by `ReadAndCheckSkeleton`, which compares the file's bone names and parent indices against the config's skeleton — real data validation, and it catches more than a version number would. The `.mfembed.npz` has no schema version and no database hash either: `load_embedding` checks only that the state count matches, so **recompute the UMAP embedding after every Generate Pose Database** or the visualizer will draw a stale cloud as if it were current

### Pose Data Handling
- `PoseBuffer` is mutable and used during synthesis; holds positions, rotations, velocities, and contact states
- `Pose` (Python class) is immutable; unpack with `.from_array()`, pack with `.pack()`
- Always use consistent `frame_time` when scaling velocities (stored in metadata)
- In `Pose.add()`, the character-frame velocity (`PoseDelta.rootVel`) is rotated by the frame's own rotation before applying; the root bone's velocity (`PoseDelta.hipVel`) is added directly, since it is already expressed within that frame

### File Paths
- Use `Application.dataPath` for Assets folder (e.g., `Path.Combine(Application.dataPath, "../Python")`)
- Use `Application.streamingAssetsPath` for packaged read-only data (e.g., animation databases)
- Avoid hard-coded absolute paths like `D:\iitbpg\...` and `C:\Users\...` — these break on other machines

## Subagent Delegation

Project subagents live in `.claude/agents/`. Design, review, and integration stay in the main agent; the routine subtask goes to the agent whose scope fits it.

Opus is the ceiling. The read-only and mechanical agents run on Sonnet; the two implementers run on Opus and must never be dropped below it — writing code from a spec is not a job for a cheap model. Never use Haiku for any of them.

| Agent | Model | Use it for |
|---|---|---|
| `code-locator` | sonnet | "Where is X / who calls Y / what files touch Z" — returns file:line, not analysis |
| `compile-checker` | sonnet | Verifying C# edits build; reading Unity/Rider errors |
| `python-runner` | sonnet | Running a module, test, or probe in the venv and reporting real output |
| `docs-updater` | sonnet | Mechanical doc/comment upkeep against already-established facts |
| `csharp-implementer` | **opus** | Writing a C#/Unity change from a spec that already names files and design |
| `python-implementer` | **opus** | Writing a Python-side change from a decided approach |

Implementer agents need a spec that names the files, the intended design, and the acceptance check — they will not make architecture decisions, and vague briefs produce guesswork. Keep concurrent subagents to about 5. Run the C# implementer and `compile-checker` as a pair: the implementer cannot verify its own build.

## Known Issues & TODOs

1. **Module reloading**: `MotionFieldStage.reloadPythonModules` defaults to true for development convenience; disable it for builds
2. **PoseBuffer vs Pose confusion**: dual representation exists; unify or document the split
3. **Dropped pipeline features**: inertialized hips blending and toes-floor penetration correction were lost when the pipeline moved to stages and are both still missing; the intended home for each is a `MoSynthStage` running after the pose is produced. The foot-side design, and the slot after `RootFollowStage` it goes in, are written up in `openwiki/animation-tools/root-following.md`
4. **Foot sliding under root following**: `RootFollowStage` places the character exactly, and nothing yet keeps planted feet still while it does. Expect `footskatePerMeter` to rise when it is enabled
5. **Foot-contact detection bones** are configurable on `MotionMatchingData`/`MotionFieldConfig` via `leftContactBone`/`rightContactBone` (`SkeletonBone`); empty fields fall back to name heuristics in `BoneNameConventions`

## Testing & Debugging

- **Unity Console** shows Python errors with proper stack traces (thanks to PythonNET error propagation)
- **Python print() statements** flush to Unity's Debug output when wrapped in `Py.GIL()`
- **Profiler**: Enable in Player settings to monitor stage performance; each stage's `Apply()` runs in sequence
- **Scene playback**: ExampleSimpleMMStages is the main test scene; use it to validate pose quality and blending

## Before Your First Commit

- Verify hard-coded paths are not leaking (search for `D:\`, `C:\Users\`)
- Test on a fresh Unity project if adding dependencies
- Update `Assets/MotionField/MotionFieldConfig.cs` if adding configurable parameters
- Run Python linting on any modified `.py` files (PEP8 style preferred)

<!-- OPENWIKI:START -->

## OpenWiki

`openwiki/` serves two audiences, and the split decides who owns what.

- **`openwiki/agents/` is yours.** It exists for the detailed, operational material you need to work here efficiently: exact invariants, hazards and "do NOT fix this" notes, editing and verification workflows, per-subsystem internals. Organise it as one subfolder per subsystem. Write to it freely — you do not need permission. It currently holds `agents/tooling/`, `agents/animation-tools/`, `agents/motion-matching/` and `agents/python/`; add a subfolder the first time you have something operational worth keeping about a subsystem that has none.
- **Everything else is for human readers.** Concept-first, plain language, explaining what a system is for and how it behaves before naming files and symbols. No exhaustive inventories — link to the matching `agents/` page for that depth.

Working rules:

- **Update the wiki as part of finishing a task, and commit the wiki change with the code.** A behaviour change that leaves its page stale is not finished. There is no "only when asked" restriction.
- Use the OpenWiki MCP lifecycle rather than editing blind: `openwiki_begin` at the repo root, `openwiki_inspect_claims` before materially editing an existing factual page, `openwiki_resolve_claims` for new or changed propositions, `openwiki_finish` at the end. Do not hand-edit Claims sidecars, indexes, logs, provenance, run metadata, or the scheduled workflow.
- **`openwiki_finish` rewrites the block between the `OPENWIKI:START`/`END` markers in this file with its own default text, discarding this policy.** Check `git diff AGENTS.md` after every finish and restore the section if it was replaced.
- Treat source code as authoritative. A page's unknowns and review items are verification gaps, not automatic requirements.
- Prefer the narrowest quiet validation that proves the changed behavior. Preserve complete failure output.
- These rules bind any agent asked to organise, refresh, or restructure the wiki, not just agents changing code.

It is still optional just-in-time context, not required startup reading. Full brief: [`openwiki/INSTRUCTIONS.md`](openwiki/INSTRUCTIONS.md).

<!-- OPENWIKI:END -->
