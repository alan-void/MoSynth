# MoSynth

A Unity testbed for comparing character locomotion synthesis methods on equal terms:

- **motion matching** (MM)
- **motion fields** (MF)
- **phase-functioned neural networks** (PFNN, Holden et al. 2017)
- **learned motion matching** (LMM, Holden et al. 2020)

Every method runs as a stage in the same pose pipeline, steered by the same control inputs, on the
same character. A benchmark drives each method round a set of paths and measures three things:
path following, motion quality (footskate, jerk, discontinuities) and cost.

This README covers reproducing the results. For how the code is organised, see
[`AGENTS.md`](AGENTS.md) and the generated wiki in [`openwiki/`](openwiki/).

## Requirements

| | Version | Notes |
|---|---|---|
| Windows 10/11 | | The tool scripts are PowerShell, and the Unity setup is Windows-only |
| Unity | 6000.4.4f1 | Install through Unity Hub; the scripts find it at the Hub's default path |
| Python | 3.13 exactly | PythonNET embeds this version inside the Editor |
| Blender | 5.1 or 5.2 | Only for the retarget stage |
| git | with git-lfs installed | Submodules; the dataset clones skip LFS objects they don't need |
| GPU | CUDA 13 capable, recommended | Training runs on CPU too, more slowly |
| Disk | about 20 GB | Raw data 5 GB, retargeted FBX 4 GB, databases and checkpoints 6 GB, recordings |

You also need a free Adobe ID to download the Mixamo Y Bot character; see [`DATA.md`](DATA.md).

## Setup

```powershell
git clone --recursive https://github.com/alanaman/MoSynth.git
cd MoSynth

# Python environment, kept anywhere you like
py -3.13 -m venv C:\envs\mosynth
C:\envs\mosynth\Scripts\pip install -r Python\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

# Tell the tools where things are (or set them in Project Settings > MoSynth > Python)
setx MOSYNTH_PYTHON_VENV "C:\envs\mosynth"
setx MOSYNTH_PYTHON_DLL  "C:\Users\<you>\AppData\Local\Programs\Python\Python313\python313.dll"
# Only if Unity or Blender is not in its default install location:
setx MOSYNTH_UNITY_EXE   "D:\Unity\6000.4.4f1\Editor\Unity.exe"
setx MOSYNTH_BLENDER_EXE "D:\Blender\blender.exe"
```

Open a new terminal so the variables take effect. If you cloned without `--recursive`, run
`git submodule update --init` first, or Unity cannot find the `GameplayTags` and MuJoCo packages.

Check the Python side with:

```powershell
cd Python; python -m unittest discover -s tests -t tests; cd ..
```

## Reproducing the results

**Close Unity before every stage.** The retarget writes FBX files that Unity would otherwise import
under fresh GUIDs, which breaks every clip asset that refers to them. The Unity stages also need the
project to themselves.

```powershell
.\Tools\reproduce.ps1
```

That runs every stage in order. All output for the run goes to `Benchmarks\reproduction_<timestamp>\`.
Each stage skips work whose output already exists, so an interrupted run resumes when started again,
and `-Stage` runs a subset:

| Stage | What it does | Typical time |
|---|---|---|
| `fetch` | Downloads LAFAN1, Bandai-Namco and Edinburgh at pinned versions and verifies every file (`Tools/Data/fetch_datasets.py`). Checks the Y Bot you downloaded. | ~10 min |
| `retarget` | Rebuilds `Assets/LFS`: converts Edinburgh to BVH, builds the Y Bot retarget setups from `Tools/Retargeting/setups/`, retargets every shard in Blender, then restores the FBX `.meta` files (`Tools/Retargeting/reproduce_all.py`) | several hours |
| `databases` | Checks every paper config for missing references, then builds the pose and feature databases (`ReproductionPipeline.Validate`, `.BuildDatabases`) | ~30 min |
| `train` | Trains PFNN, LMM (all stages), the motion-field value function and its embedding. Records wall-clock time and hardware per run in `training_log_*.json` | hours, GPU-bound |
| `benchmark` | Runs every sweep listed in `Assets/Benchmarks/PaperReproduction.asset` | ~1 h per sweep |
| `report` | Reduces the sweeps to per-method tables, `summary.md` and `summary.csv` (`python -m benchmark.report`) | seconds |

```powershell
.\Tools\reproduce.ps1 -Stage fetch,retarget          # data only
.\Tools\reproduce.ps1 -Stage train,benchmark,report  # after changing a model
```

The Unity steps also live in the Editor under **MoSynth > Reproduction**, and a single sweep can be
run with `.\Tools\run-benchmark.ps1 -Config Assets/Benchmarks/<name>.asset`.

### What the paper's results consist of

`Assets/Benchmarks/PaperReproduction.asset` lists the benchmark configs that make up the results.
Everything else is derived from those configs: the method prefabs they run and the MM, MF, PFNN
and LMM configs those prefabs use. That is also how the `databases` and `train` stages decide what
to build. To add a sweep to the reproduction, add its config to that asset.

### Expected variation

PFNN, LMM and the motion-field embedding are all seeded from their configs (`seed`, `umapSeed`).
Even so, PyTorch on a GPU is not bit-for-bit deterministic, and the benchmark's cost column is
wall-clock time inside the Editor. So expect:

- motion and path metrics close to the paper's, but not identical;
- cost figures that are only comparable between methods within one run on one machine.

## Repository layout

| Path | Contents |
|---|---|
| `Assets/AnimationTools/` | The shared pipeline: skeletons, poses, stages, control inputs, benchmark, recording |
| `Assets/MotionMatching/`, `MotionField/`, `Pfnn/`, `Lmm/` | One folder per synthesis method |
| `Assets/Animation/` | Annotated clips and per-method configs. Clips refer to `Assets/LFS` FBX files, which you rebuild yourself |
| `Assets/Benchmarks/` | Benchmark configs, method prefabs, paths, and the reproduction manifest |
| `Assets/Editor/Reproduction/` | The batchmode reproduction pipeline |
| `Python/` | Training and inference for each method, file-format readers, benchmark reports. Run modules with `python -m` from inside `Python/` |
| `Tools/Data/` | Dataset download and checksums, plus the FBX `.meta` mirror |
| `Tools/Retargeting/` | Blender retargeting: stripped setups, shard lists, batch drivers ([README](Tools/Retargeting/README.md)) |
| `External/mujoco`, `Packages/com.alanvoid.gameplaytags` | Git submodules |

The folders `Assets/LFS/`, `External/LFS/`, `Assets/StreamingAssets/` and `Benchmarks/` are
gitignored. They hold data you rebuild yourself, or output.

## Licence

The code is MIT licensed; see [`LICENSE`](LICENSE). The motion-capture datasets and the Mixamo
character keep their own licences, which are non-commercial for LAFAN1 and Bandai-Namco. See
[`DATA.md`](DATA.md) for each one and for how to cite it. If you use this repository, please cite
it as described in [`CITATION.cff`](CITATION.cff).
