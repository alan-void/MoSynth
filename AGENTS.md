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
- **Format**: `(..., num_bones + 2, 4)` array, `num_bones` counting the virtual character-frame joint `action_predictor.with_virtual_root` adds ahead of the pose set's own (single, real) skeleton
  - Index 0: character-frame position (xyz) + padding (1 float) — zero, since the pose is already expressed in its own frame
  - Index 1: root bone position (xyz) + padding (1 float) — `Pose.py` still calls this `hips`, but it is the skeleton's real bone 0, not a separate Hips joint
  - Indices 2+: joint rotations as quaternions (xyzw per joint) — index 2 is the frame's own (identity) rotation, index 3 the root bone's
- **Velocities**: parallel arrays using same format but represent per-second rates (scaled by `frame_time` to get one-frame deltas)
- **Python classes**: `Pose` (immutable), `PoseDelta` (velocity), `Skeleton` (FK/IK)
- `action_predictor.get_pose_arrays` unpacks a synthesized pose back down to plain per-bone arrays (no frame slots — C# derives its own frame from bone 0) for the PythonNET boundary

### MotionFieldStage (Current Feature Branch)

Integrates a neural motion field into the pipeline via PythonNET:

**C# side (`Assets/MotionField/MotionFieldStage.cs`):**
- Initializes Python engine, loads `MotionField.py` and `action_predictor.py` modules
- Each frame `StepPolicy()` calls one of the four Python policy methods, chosen by the stage's
  `Policy` enum. The dispatch lives in C# so the policy never crosses the boundary as a value
- Converts Python arrays back to `PoseBuffer` for Unity

**Python side (`Python/MotionField.py`):**
- KNN search over motion states (positions + velocities as features)
- Blends nearest neighbor poses and velocities
- Policy methods: `optimal_action()` (value function, falling back to greedy when none is loaded),
  `greedy_action()`, and the debug pair `get_next_pose()` / `get_next_pose_from_field()`
- Supporting: `get_knn()`, `get_batched_knn()`, `build_motion_states()`, `load_value_function()`

**Python setup:**
- The CPython DLL and venv paths are resolved by `PythonRuntime.EnsureInitialized` from the `MOSYNTH_PYTHON_DLL` / `MOSYNTH_PYTHON_VENV` environment variables first, then the serialized `MotionFieldConfig` fields (`pythonDllPath`, `pythonVenvPath`), then — for the DLL only — PythonNET's own `PYTHONNET_PYDLL`. An interpreter's location is a property of the machine, so prefer the environment variables; the serialized fields are why a checked-in asset can name someone else's drive
- Project modules import from `PythonRuntime.ScriptsFolder` — the repository's `Python/` folder, derived from `Application.dataPath`
- Python 3.13 is required for PythonNET compatibility

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
│   ├── Runtime/Pose/
│   │   ├── PoseBuffer.cs            [mutable pose in motion synthesis]
│   │   └── PoseFK.cs                [forward kinematics]
│   ├── Runtime/Benchmark/           [sweep config, lap tracking, motion-quality & cost metrics]
│   ├── Runtime/Evaluation/          [PathFollowingMetric(sCalculator): in-scene A/B tool + pure metrics]
│   ├── Runtime/Recording/           [MotionRecorder, channels, manifest, reader]
│   ├── Editor/Benchmark/            [sweep driver, CLI entry point, menu, report writer]
│   ├── Editor/
│   │   └── SkeletonBoneDrawer.cs    [Inspector dropdown for bone selection]
│   └── (other runtime & editor features)
├── MotionField/
│   ├── MotionFieldStage.cs          [stage implementing neural field]
│   ├── MfConnector.cs               [data connector for field]
│   └── MotionFieldConfig.cs         [serializable config: clips, skeleton, hyperparameters]
├── MotionMatching/
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
    └── MotionMatching/              [animation database assets]

Python/
├── MotionField.py                   [neural motion field implementation]
├── Pose.py                          [pose data classes]
├── Skeleton.py                      [skeleton FK/IK]
├── Animation.py                     [PoseSet: the Python mirror of the C# pose database]
├── action_predictor.py              [animation loading & conversion]
├── binary_reading.py                [BinaryWriter primitives shared by the format readers]
├── pose_set_importer.py             [.mmpose reader]
├── feature_set_importer.py          [.mmfeatures reader: matching feature vectors + schema]
├── simulation_frame.py              [character frames, FK, and per-frame rates]
├── gait_phase.py                    [gait phase reconstructed from foot contacts]
├── training_data.py                 [per-frame arrays a PFNN/LMM model trains on, + .npz]
├── motion_field_io.py               [.mffield.npz value function format]
├── motion_field_trainer.py          [fitted value iteration]
├── motion_field_embedding.py        [UMAP projection for the debug visualizer]
├── tests/                           [stdlib unittest suites; no Unity or venv extras needed]
├── utils/
│   └── quaternions.py               [quaternion utilities]
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

# Test Python modules
python Python/MotionField.py
```

Then tell the project where they are, preferably per machine rather than in the asset:

```powershell
setx MOSYNTH_PYTHON_DLL  "C:/path/to/python313.dll"
setx MOSYNTH_PYTHON_VENV "C:/path/to/.anim_env"
```

Restart Unity so it picks the variables up. Failing that, the `MotionFieldConfig` asset's
`pythonDllPath` and `pythonVenvPath` fields are the fallback.

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
- **C# edit-mode suites**: `MoSynth/Tests/Run EditMode Tests` runs `AnimationTools.Tests` and
  `MotionMatching.Tests` and writes `Temp/animtools_test_results.txt`, which survives the domain
  reload a test run causes — so a scripted caller reads results from there, not from the console
- **Python suites**: `python -m unittest discover -s Python/tests -t Python/tests`. They use only
  numpy and scipy and build their own fixtures, so they need neither Unity nor a generated database
- **Editor play mode**: test motion synthesis visually
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
  1. A `Skeleton`'s rest pose is read live off its Transforms, so it must point at an ASSET rig (imported FBX or prefab), never a live scene rig — a skeleton over an animated scene rig reports the current pose as the rest pose and silently corrupts FK. `MotionSynthesisComponent` serializes its own `Skeleton` field — assigned to the rig's root bone from the FBX asset, since the field's drawer refuses scene objects — and binds it to the scene rig via `SkeletonBoneOverrides`
  2. Everything under a skeleton's root Transform becomes a bone, so a rig must have nothing but bones beneath its skeleton root — a mesh node, IK helper, or attachment point there would shift every index after it

### `SkeletonAnimation` has one source of truth
- A `SkeletonAnimation` (and its `AnnotatedAnimationClip` subclass) stores a clip and a `Skeleton`, and nothing else about where the bones are. There is no separate rig field: a second reference could disagree with the skeleton, and did — an earlier `GameObject` → `Transform` change orphaned every asset's rig
- `AnimationClipBaker` still needs the asset's main object, because `clip.SampleAnimation` matches curve paths against the hierarchy it is handed and importers write those paths relative to the main object (the FBX root, or the container `BvhImporter` puts its bones under). It derives that as `skeleton.Root.root` — the root bone's topmost ancestor — so the bake target cannot drift from the skeleton
- The root-bone guess (single child, else a `*Hips` descendant) now happens once, in `CreateAnnotatedClipMenu`, and is written into the asset. Nothing re-derives it at access time
- A `Skeleton` field is drawn by `SkeletonDrawer` as a rig object field plus a root-bone dropdown, because a bone *inside* an imported rig is not reachable from the Project window or the object picker — only the asset's main object is. Drop the rig in to seed the field, then pick the actual root from the dropdown. It deliberately does not guess: every `Skeleton` — a clip's or a config's — starts at the rig's root bone. `SkeletonDrawer` and `SkeletonBoneDrawer` share their listing code via `BonePopup`
- `TryValidate` rejects a skeleton root that is not `EditorUtility.IsPersistent`, which is the only automatic check on invariant 1 above

### `SkeletonBone` naming collision
`AnimationTools.SkeletonBone` collides with the built-in `UnityEngine.SkeletonBone`. Any file outside the `AnimationTools*` namespaces that names the type needs `using SkeletonBone = AnimationTools.SkeletonBone;`.

### Config assets and their skeletons
- `MotionMatchingData` and `MotionFieldConfig` each carry an explicit T-pose `Skeleton` field whose root is the rig's real root bone — the pose skeleton is identical to each clip's skeleton, so nothing is prepended at load. The compatibility check is therefore `Skeleton.StructurallyEqual`, not a shifted comparison
- Neither config has a configurable simulation-frame bone anymore: `SimulationFrameDef.Default(skeleton)` always sets the reference bone to the skeleton root (bone 0) and the forward axis to that bone's rest forward (`Skeleton.RestLocalAxis(0, math.forward())`); `PoseSet.SimulationFrame` is a computed property returning this default. That reference bone is what `SimulationFrame.Compute`/`ComputeVelocity` (`Assets/AnimationTools/Runtime/Pose/SimulationFrame.cs`) derives the ground-projected, yaw-only character frame from — that frame is never stored on the pose, only computed on demand from a `SimulationFrameDef { ReferenceBoneIndex, ForwardAxisLocal }`
- Both assets expose `TryValidate(out string error)`. `GetOrImportPoseSet()`/`GetOrImportFeatureSet()` return null silently when it fails, because `OnValidate` reaches them every Inspector repaint; the custom Inspector shows the reason in a HelpBox and disables Generate. `SkeletonAnimation` follows the same pattern
- The `.mmskeleton` format is gone. C# takes the skeleton from the config's own field, but Python has no ScriptableObject to read, so `.mmpose` opens with a skeleton block — per bone: name, parent index, rest local position, rest local rotation — written by `PoseSerializer.WriteSkeleton` and read by `pose_set_importer.read_skeleton`, which derives the same structural simulation frame (reference bone 0, forward axis from that bone's rest rotation) rather than reading it as separate fields. Keeping it in the same file as the poses is what stops the two drifting apart
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
3. **Dropped pipeline features**: inertialized hips blending and toes-floor penetration correction were lost when the pipeline moved to stages; the intended home for each is a `MoSynthStage` running after the pose is produced (TODO in `MotionSynthesisComponent`, ~line 315)
4. **Foot-contact detection bones** are configurable on `MotionMatchingData`/`MotionFieldConfig` via `leftContactBone`/`rightContactBone` (`SkeletonBone`); empty fields fall back to name heuristics in `BoneNameConventions`

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

- **`openwiki/agents/` is yours.** It exists for the detailed, operational material you need to work here efficiently: exact invariants, hazards and "do NOT fix this" notes, editing and verification workflows, per-subsystem internals. Organise it as one subfolder per subsystem. Write to it freely — you do not need permission. It currently holds `agents/tooling/`, `agents/animation-tools/` and `agents/python/`; add a subfolder the first time you have something operational worth keeping about a subsystem that has none.
- **Everything else is for human readers.** Concept-first, plain language, explaining what a system is for and how it behaves before naming files and symbols. No exhaustive inventories — link to the matching `agents/` page for that depth.

Working rules:

- **Update the wiki as part of finishing a task, and commit the wiki change with the code.** A behaviour change that leaves its page stale is not finished. There is no "only when asked" restriction.
- Use the OpenWiki MCP lifecycle rather than editing blind: `openwiki_begin` at the repo root, `openwiki_inspect_claims` before materially editing an existing factual page, `openwiki_resolve_claims` for new or changed propositions, `openwiki_finish` at the end. Do not hand-edit Claims sidecars, indexes, logs, provenance, run metadata, or the scheduled workflow.
- **`openwiki_finish` rewrites the block between the `OPENWIKI:BEGIN`/`END` markers in this file with its own default text, discarding this policy.** Check `git diff AGENTS.md` after every finish and restore the section if it was replaced.
- Treat source code as authoritative. A page's unknowns and review items are verification gaps, not automatic requirements.
- Prefer the narrowest quiet validation that proves the changed behavior. Preserve complete failure output.
- These rules bind any agent asked to organise, refresh, or restructure the wiki, not just agents changing code.

It is still optional just-in-time context, not required startup reading. Full brief: [`openwiki/INSTRUCTIONS.md`](openwiki/INSTRUCTIONS.md).

<!-- OPENWIKI:END -->
