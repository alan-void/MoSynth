---
type: Agent Reference
title: Running the Python side
description: Which interpreter to use, how the package layout constrains imports, the scipy shapes that bite, and how to read a generated database from a shell.
tags: [agents, python, numpy, scipy, workflow]
sources:
  - id: openwiki-source-ea70eb6c045047448e446296
    resource: repo://.gitignore
  - id: openwiki-source-51bcadd56f350b558690d51e
    resource: repo://Assets/AnimationTools/Runtime/Python/PythonPathSettings.cs
  - id: openwiki-source-fd8b7d29f1e6a937c4ae1988
    resource: repo://Assets/AnimationTools/Runtime/Python/PythonRuntime.cs
  - id: openwiki-source-44aeec7103fbb90f34a34626
    resource: repo://Python/tests/test_training_data.py
generated: {by: "claude-code", at: "2026-09-21T19:17:12.006Z"}
verified:
  - by: openwiki/0.3.3
    at: 2026-09-21T19:17:12.006Z
---

# Running the Python side

## Which interpreter

`Python/` needs numpy, scipy and (for the motion field trainer) torch, so the system interpreter is
usually not it. Resolution order, and the same one `PythonRuntime.EnsureInitialized` uses:

1. `MOSYNTH_PYTHON_VENV` / `MOSYNTH_PYTHON_DLL`
2. `UserSettings/MoSynthPython.json`, which *Project Settings → MoSynth → Python* writes
3. `PYTHONNET_PYDLL`, for the DLL only

So find the venv the same way the Editor does — the variables first, then the file:

```bash
echo "$MOSYNTH_PYTHON_VENV"
cat UserSettings/MoSynthPython.json
```

That file is gitignored and per user, so **on a fresh clone it does not exist** and neither path is
set until someone fills the settings page in. Paths were serialized on `MotionFieldConfig` until
they moved here; do not reach for the asset, it no longer carries them.

Run through `<venv>/Scripts/python.exe` directly rather than activating; activation does not survive
between tool calls.

**Python 3.13 is required** for PythonNET compatibility. The venv has torch, umap and scikit-learn,
and does **not** have pytest — the test suites are stdlib `unittest` for exactly that reason.

## The modules are packages under an import root

`Python/` is a set of packages — `core`, `formats`, `training`, `pfnn`, `lmm`, `motion_field`,
`debugging` — each with an `__init__.py`. **`Python/` itself is the import root and is not a
package**, because it is the one folder `PythonRuntime` puts on `sys.path`; a module is therefore
addressed `pfnn.runtime`, never `Python.pfnn.runtime`. Everything importing them needs `Python/` on
`sys.path`:

- **Unity** appends it in `PythonRuntime.EnsureInitialized`, and names modules by the same dotted
  path: `PythonRuntime.Import("pfnn.runtime")`.
- **A shell** needs `cd Python` first, or a `sys.path.insert`.
- **The tests** each do `sys.path.insert(0, dirname(dirname(abspath(__file__))))` at the top, which
  is why they run from anywhere — and why `discover -t Python` fails while
  `discover -s Python/tests -t Python/tests` works.

Imports between modules are **absolute from that root** (`from core.pose import Pose`), never
relative, so one spelling works whether Unity or a shell did the importing. Within its own package a
module is imported bare (`from lmm import dataset`); from outside it, or when the bare name shadows
a stdlib one, it is aliased (`from pfnn import io as pfnn_io`).

Two things this layout will not tolerate:

- **Do not add an `__init__.py` to `Python/` itself.** It would make `Python` a package and every
  `PythonRuntime.Import` string wrong.
- **Do not run a module by path.** `python pfnn/trainer.py` puts `Python/pfnn/` on `sys.path`
  instead of `Python/`, and every absolute import in it fails. Use `python -m pfnn.trainer` from
  `Python/`.

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
from formats.pose_set_importer import deserialize_pose_set
from formats.feature_set_importer import read_feature_set

db, name = '../Assets/StreamingAssets/MMDatabases/MotionMatchingData', 'MotionMatchingData'
poses = deserialize_pose_set(db, name)
features = read_feature_set(db, name)
print(poses.local_positions.shape, features.features.shape, features.pose_offset)
PY
```

`training.training_data` runs as a script and writes the whole training set as one `.npz`:

```bash
python -m training.training_data ../Assets/StreamingAssets/MMDatabases/MotionMatchingData MotionMatchingData
python -m training.training_data ../Assets/StreamingAssets/MotionFields/MotionFieldConfig MotionFieldConfig --no-features
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
