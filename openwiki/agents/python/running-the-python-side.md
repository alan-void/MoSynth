---
type: Agent Reference
title: Running the Python side
description: Which interpreter to use, how the flat module layout constrains imports, the scipy shapes that bite, and how to read a generated database from a shell.
tags: [agents, python, numpy, scipy, workflow]
sources:
  - id: openwiki-source-839acd5f9c92d76d722dddc3
    resource: repo://Assets/MotionField/PythonRuntime.cs
  - id: openwiki-source-030d30d689203655d06f8a6b
    resource: repo://Python/tests/test_gait_phase.py
generated: {by: "claude-code", at: "2026-08-29T23:23:48.822Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-08-29T23:23:48.822Z
---

# Running the Python side

## Which interpreter

`Python/` needs numpy, scipy and (for the motion field trainer) torch, so the system interpreter is
usually not it. Resolution order, and the same one `PythonRuntime.EnsureInitialized` uses:

1. `MOSYNTH_PYTHON_VENV` / `MOSYNTH_PYTHON_DLL`
2. `MotionFieldConfig.pythonVenvPath` / `.pythonDllPath` — the serialized fallback, and the reason a
   checked-in asset can name someone else's drive letter
3. `PYTHONNET_PYDLL`, for the DLL only

Read the config asset when neither variable is set:

```bash
grep -n "pythonVenvPath\|pythonDllPath" Assets/Animation/MotionField/MotionFieldConfig.asset
```

Run through `<venv>/Scripts/python.exe` directly rather than activating; activation does not survive
between tool calls.

**Python 3.13 is required** for PythonNET compatibility. The venv has torch, umap and scikit-learn,
and does **not** have pytest — the test suites are stdlib `unittest` for exactly that reason.

## The modules are flat, not a package

`Python/` has no `__init__.py`. Modules import each other by bare name (`import gait_phase`,
`from Animation import PoseSet`), so anything importing them needs `Python/` on `sys.path`:

- **Unity** appends it in `PythonRuntime.EnsureInitialized`.
- **A shell** needs `cd Python` first, or a `sys.path.insert`.
- **The tests** each do `sys.path.insert(0, dirname(dirname(abspath(__file__))))` at the top, which
  is why they run from anywhere — and why `discover -t Python` fails while
  `discover -s Python/tests -t Python/tests` works.

Adding an `__init__.py` anywhere under `Python/` would break the Unity import path, which imports by
bare module name. Do not.

## Module reloading in the Editor

`PythonRuntime.InvalidateProjectModules` drops **every** already-imported module under `Python/` so
the next import re-reads them all. A per-module `importlib.reload` is not equivalent and was tried:
it leaves dependencies cached, so reloading a module after adding a symbol to one of its imports dies
on `ImportError`. Call it once before a group of related imports, not per import, so they all end up
on one generation of the code.

## scipy shapes that bite

`scipy.spatial.transform.Rotation` takes flat stacks only.

```python
# (n, n_bones, 4) will not go in directly.
rotations = Rotation.from_quat(quats.reshape(-1, 4))

# Applying one frame rotation per bone means repeating it per bone first.
inverse_frame = Rotation.from_quat(np.repeat(frame_rotations, n_bones, axis=0)).inv()
```

`from_euler` with a single-axis sequence wants a trailing axis of size 1:

```python
Rotation.from_euler('y', yaws)                  # ValueError on an (n,) array
Rotation.from_euler('y', yaws[:, np.newaxis])   # correct
```

Quaternions are **xyzw** throughout this package, matching both the packed pose layout and scipy.

## Reading a generated database

```bash
cd Python
"$VENV/Scripts/python.exe" - <<'PY'
from pose_set_importer import deserialize_pose_set
from feature_set_importer import read_feature_set

db, name = '../Assets/StreamingAssets/MMDatabases/MotionMatchingData', 'MotionMatchingData'
poses = deserialize_pose_set(db, name)
features = read_feature_set(db, name)
print(poses.local_positions.shape, features.features.shape, features.pose_offset)
PY
```

`training_data.py` runs as a script and writes the whole training set as one `.npz`:

```bash
python training_data.py ../Assets/StreamingAssets/MMDatabases/MotionMatchingData MotionMatchingData
python training_data.py ../Assets/StreamingAssets/MotionFields/MotionFieldConfig MotionFieldConfig --no-features
```

A `MotionFieldConfig` database has no `.mmfeatures`, hence `--no-features`.

## Sanity numbers for the demo database

Useful as a smell test after changing anything in the conversion chain — these are what the checked-in
demo database produces:

| Quantity | Value |
| --- | --- |
| Frames / bones / clips | 7839 / 85 / 1 |
| Frame time | 1/30 s |
| Bone 0 in the character frame | x and z within 1e-5 of zero, y ≈ 0.87 |
| Root speed | mean 0.68 m/s, max 2.11 m/s |
| Foot contact duty | ≈ 0.50 / 0.49 |
| Reconstructed stride | ≈ 1.49 s |

Bone 0's x and z being zero is not a coincidence to be explained away: the character frame *is* bone
0 ground-projected, so anything else means the frame transform is wrong.
