# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, and others reading this file) when working with code in this repository.

## Project Overview

**MoSynth** is a motion synthesis system for character animation that combines motion matching with neural motion fields. It's a Unity project (v6000.4.4) that synthesizes realistic character locomotion by blending traditional motion matching with learned motion fields. The system uses a pipeline architecture where pose data flows through multiple "stages" that transform it, with interop to Python via PythonNET for neural network inference.

### Code Style
- Do not write comments in generated code that depend on the context of the conversation that produced them (e.g. referencing the task, a fix, a prior approach, or "why we're changing this now"). Comments should only explain non-obvious WHY that a reader with just the code in front of them would need.
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
  - Barracuda (3.0.2) - neural network inference in Unity
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
      └── Other stages (RootMotionCorrection, PoseSetVisualizer, etc.)
```

**Key classes:**
- `MoSynthStage` (abstract base): interface for all stages with `Init()`, `Apply(PoseBuffer, deltaTime)`, `GetSkeleton()`, `OnDestroy()`
- `MotionSynthesisComponent`: main orchestrator that runs stages each frame; manages skeleton transforms and pose updates
- `PoseBuffer`: the mutable pose data structure carrying joint positions, rotations, velocities, contact states
- `Skeleton`: a bone hierarchy defined by a Transform tree — serializes as a single root Transform, and the bone list is the preorder depth-first walk from that root (bone 0 = root); index+1 convention for bone IDs, index 0 = SimulationBone in a pose skeleton
- `SkeletonBone`: one bone of a `Skeleton` — a `Skeleton` + `Transform` pair; also the serializable type used for a standalone bone-reference field (contact bones, root motion bone)
- `SkeletonBoneOverrides`: binds a `Skeleton` (an asset rig at rest) to a different, live rig by name, with per-bone overrides; `MotionSynthesisComponent.characterRig` is one of these

### Data Flow: Pose Representation

Poses are packed into flat arrays for efficiency with Python interop:
- **Format**: `(..., num_bones + 2, 4)` array
  - Index 0: root position (xyz) + padding (1 float)
  - Index 1: hips position (xyz) + padding (1 float)
  - Indices 2+: joint rotations as quaternions (xyzw per joint)
- **Velocities**: parallel arrays using same format but represent per-second rates (scaled by `frame_time` to get one-frame deltas)
- **Python classes**: `Pose` (immutable), `PoseDelta` (velocity), `Skeleton` (FK/IK)

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

**⚠️ Known Issues:**
- Python paths are hard-coded (`MotionFieldStage:40`, `MotionFieldStage:47`) — should be made configurable
- Requires specific Python 3.13 DLL location
- Requires virtual environment setup at a specific path
- Should use `Application.streamingAssetsPath` pattern like the animation data does

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
│   ├── Editor/
│   │   └── SkeletonBoneDrawer.cs    [Inspector dropdown for bone selection]
│   └── (other runtime & editor features)
├── MotionField/
│   ├── MotionFieldStage.cs          [stage implementing neural field]
│   ├── MfConnector.cs               [data connector for field]
│   └── MotionFieldData.cs           [serializable config]
├── MotionMatching/
│   ├── Runtime/Core/
│   │   ├── MotionMatchingStage.cs   [database search stage]
│   │   └── RootMotionCorrectionStage.cs
│   ├── Runtime/Pose/
│   │   └── PoseBufferFK.cs          [motion matching pose FK extension]
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
├── action_predictor.py              [animation loading & conversion]
├── motion_field_io.py               [.mffield.npz value function format]
├── motion_field_trainer.py          [fitted value iteration]
├── motion_field_embedding.py        [UMAP projection for the debug visualizer]
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
# The MotionFieldStage expects a venv at a specific path (currently hard-coded)
# Python 3.13 is required for PythonNET compatibility
python -m venv D:\iitbpg\MoSynth\AnimationTech\.anim_env
D:\iitbpg\MoSynth\AnimationTech\.anim_env\Scripts\activate

# Install dependencies
pip install numpy scipy torch

# Test Python modules
python Python/MotionField.py
```

### Building for Distribution
The project uses standard Unity build pipeline:
1. `File → Build Settings` in Unity Editor
2. Select target platform and scenes
3. Configure player settings (scripts-only for development)

### Testing & Validation
- **Editor play mode**: test motion synthesis visually
- **Python validation**: `Python/test_server.py` (if it exists) or manual import tests
- **Motion database**: verify animation data loads in `StreamingAssets/MMDatabases/MotionMatchingData`

## Important Patterns & Conventions

### Stage Development
When adding a new `MoSynthStage`:
1. Inherit from `MoSynthStage` abstract class
2. Override `Init(MotionSynthesisComponent)` for setup (called once at startup)
3. Override `Apply(PoseBuffer pose, float deltaTime)` to transform pose (called every frame); return `false` to halt the pipeline
4. Override `GetSkeleton(Skeleton inSkeleton)` if the stage modifies skeleton structure (e.g., adding simulation bones)
5. Implement `IDisposable.Dispose()` if holding unmanaged resources (e.g., Python state)
6. `OnDestroy()` is called automatically when the scene unloads

### Python Interop with PythonNET
- Always wrap Python calls in `using (Py.GIL()) { ... }` to acquire the Global Interpreter Lock
- Use `dynamic` types for Python objects in C#
- Python modules should be added to `sys.path` or placed in `Assets/../Python/`
- For module hot-reloading during development, use `importlib.reload()` (see `MotionFieldStage:63-65`)

### Skeleton & Bone Binding
- A `Skeleton` is a Transform tree: bone identity is a Transform reference, not a name string. The bone list is the preorder depth-first walk from the skeleton's root Transform, built lazily and cached per root. A name fallback exists only for cross-rig resolution (`SkeletonBone.ResolveIndex`), e.g. matching a contact bone picked on one rig against another that's structurally the same
- Bone indices follow the index+1 convention (bone ID = index + 1); index 0 is the SimulationBone in a pose skeleton
- `SkeletonBoneOverrides` binds a `Skeleton` (an asset rig at rest) to a different, live rig by name, with per-bone overrides for mismatched names; `SkeletonBone` is drawn by `SkeletonBoneDrawer` with a rig-aware dropdown
- **Two invariants nothing enforces at compile time:**
  1. A `Skeleton`'s rest pose is read live off its Transforms, so it must point at an ASSET rig (imported FBX or prefab), never a live scene rig — a skeleton over an animated scene rig reports the current pose as the rest pose and silently corrupts FK. `MotionSynthesisComponent` deliberately takes its `Skeleton` from the stages (asset-backed) and binds it to the scene rig via `SkeletonBoneOverrides`
  2. Everything under a skeleton's root Transform becomes a bone, so a rig must have nothing but bones beneath its skeleton root — a mesh node, IK helper, or attachment point there would shift every index after it

### `SkeletonAnimation` has one source of truth
- A `SkeletonAnimation` (and its `AnnotatedAnimationClip` subclass) stores a clip and a `Skeleton`, and nothing else about where the bones are. There is no separate rig field: a second reference could disagree with the skeleton, and did — an earlier `GameObject` → `Transform` change orphaned every asset's rig
- `AnimationClipBaker` still needs the asset's main object, because `clip.SampleAnimation` matches curve paths against the hierarchy it is handed and importers write those paths relative to the main object (the FBX root, or the container `BvhImporter` puts its bones under). It derives that as `skeleton.Root.root` — the root bone's topmost ancestor — so the bake target cannot drift from the skeleton
- The root-bone guess (single child, else a `*Hips` descendant) now happens once, in `CreateAnnotatedClipMenu`, and is written into the asset. Nothing re-derives it at access time
- A `Skeleton` field is drawn by `SkeletonDrawer` as a rig object field plus a root-bone dropdown, because a bone *inside* an imported rig is not reachable from the Project window or the object picker — only the asset's main object is. Drop the rig in to seed the field, then pick the actual root from the dropdown. It deliberately does not guess: a clip's skeleton starts at the root bone, a config's at the armature node. `SkeletonDrawer` and `SkeletonBoneDrawer` share their listing code via `BonePopup`
- `TryValidate` rejects a skeleton root that is not `EditorUtility.IsPersistent`, which is the only automatic check on invariant 1 above

### `SkeletonBone` naming collision
`AnimationTools.SkeletonBone` collides with the built-in `UnityEngine.SkeletonBone`. Any file outside the `AnimationTools*` namespaces that names the type needs `using SkeletonBone = AnimationTools.SkeletonBone;`.

### Config assets and their skeletons
- `MotionMatchingData` and `MotionFieldConfig` each carry an explicit T-pose `Skeleton` field whose root is the rig's identity armature node. That node **is** the SimulationBone at index 0, so index 1 is the first real bone (Hips) and nothing is prepended at load
- A clip's own `Skeleton` starts at its root bone, so it is one bone shorter. The compatibility check is therefore `skeleton.MatchesFrom(1, clip.Skeleton)`, not `StructurallyEqual`
- Both assets expose `TryValidate(out string error)`. `GetOrImportPoseSet()`/`GetOrImportFeatureSet()` return null silently when it fails, because `OnValidate` reaches them every Inspector repaint; the custom Inspector shows the reason in a HelpBox and disables Generate. `SkeletonAnimation` follows the same pattern
- The `.mmskeleton` format is gone — the skeleton comes from the config's field. `.mmpose` carries an `MMPS` v2 magic + version header, so databases written before this must be regenerated from the MotionMatchingData editor

### Pose Data Handling
- `PoseBuffer` is mutable and used during synthesis; holds positions, rotations, velocities, and contact states
- `Pose` (Python class) is immutable; unpack with `.from_array()`, pack with `.pack()`
- Always use consistent `frame_time` when scaling velocities (stored in metadata)
- Root velocity is in local space (rotated by root bone before applying); hips use world space

### File Paths
- Use `Application.dataPath` for Assets folder (e.g., `Path.Combine(Application.dataPath, "../Python")`)
- Use `Application.streamingAssetsPath` for packaged read-only data (e.g., animation databases)
- Avoid hard-coded absolute paths like `D:\iitbpg\...` and `C:\Users\...` — these break on other machines

## Subagent Delegation

Project subagents live in `.claude/agents/`. Design, review, and integration stay in the main agent; the routine subtask goes to the cheapest agent that can do it well.

| Agent | Model | Use it for |
|---|---|---|
| `code-locator` | haiku | "Where is X / who calls Y / what files touch Z" — returns file:line, not analysis |
| `compile-checker` | haiku | Verifying C# edits build; reading Unity/Rider errors. Never enters play mode |
| `python-runner` | haiku | Running a module, test, or probe in the venv and reporting real output |
| `docs-updater` | haiku | Mechanical doc/comment upkeep against already-established facts |
| `csharp-implementer` | sonnet | Writing a C#/Unity change from a spec that already names files and design |
| `python-implementer` | sonnet | Writing a Python-side change from a decided approach |

Implementer agents need a spec that names the files, the intended design, and the acceptance check — they will not make architecture decisions, and vague briefs produce guesswork. Keep concurrent subagents to about 5. Run the C# implementer and `compile-checker` as a pair: the implementer cannot verify its own build.

## Known Issues & TODOs

1. **Hard-coded Python paths** (`MotionFieldStage:40`, `:47`) — should be configurable via Inspector or config file
2. **Module reloading** uses `importlib.reload()` for development convenience but should be removed in production
3. **Unused code** in `MotionField.py` (see dead branches in `get_pose()` method lines 169-171)
4. **PoseBuffer vs Pose confusion**: dual representation exists; unify or document the split
5. **MotionSynthesisComponent TODO** at line 11: should decouple from `MotionMatchingData` dependency
6. **Foot-contact detection bones** are configurable on `MotionMatchingData`/`MotionFieldConfig` via `leftContactBone`/`rightContactBone` (`SkeletonBone`); empty fields fall back to name heuristics in `BoneNameConventions`

## Testing & Debugging

- **Unity Console** shows Python errors with proper stack traces (thanks to PythonNET error propagation)
- **Python print() statements** flush to Unity's Debug output when wrapped in `Py.GIL()`
- **Profiler**: Enable in Player settings to monitor stage performance; each stage's `Apply()` runs in sequence
- **Scene playback**: ExampleSimpleMMStages is the main test scene; use it to validate pose quality and blending

## Before Your First Commit

- Verify hard-coded paths are not leaking (search for `D:\`, `C:\Users\`)
- Test on a fresh Unity project if adding dependencies
- Update `Assets/MotionField/MotionFieldData.cs` or config if adding configurable parameters
- Run Python linting on any modified `.py` files (PEP8 style preferred)

