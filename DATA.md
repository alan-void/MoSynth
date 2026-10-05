# Data

This repository ships **no motion-capture data and no character models**. The licences of the
datasets do not allow it: LAFAN1 is NoDerivatives, and Adobe forbids redistributing Mixamo files
on their own. What the repository does ship is everything needed to rebuild the data on your own
machine, identically:

- `Tools/Data/fetch_datasets.py` downloads each dataset at a pinned version and checks every file
  against `Tools/Data/checksums.json`.
- `Tools/Retargeting/setups/` holds the retarget setups with meshes and motion removed. These are the
  hand-authored calibration: each source skeleton's cleaned rest pose, its constraints and its bone map.
- `Tools/Retargeting/shards/` lists exactly which BVH clips go into which output FBX, and with
  which settings.
- `Tools/Data/lfs-meta/` holds the Unity `.meta` files of the retargeted FBX. The 4,000-odd
  annotated clip assets under `Assets/Animation/` refer to those FBX by GUID. Restoring the
  `.meta` beside a regenerated FBX is what lets those assets find it again. Every FBX is imported
  with `fileIdsGeneration: 2`, so sub-asset IDs are derived from take and bone names, not stored.

Everything rebuilt lands in two gitignored folders: `External/LFS/` holds the raw downloads, and
`Assets/LFS/` holds the converted BVH and the retargeted FBX.

## Datasets

| Dataset | Used for | Version pinned | Licence |
|---|---|---|---|
| LAFAN1 (Ubisoft La Forge) | `LafanHolden` clips (Holden benchmark), LAFAN shards | `lafan1.zip` by sha256 | CC BY-NC-ND 4.0 |
| Bandai-Namco Research Motion Dataset, dataset-2 | `BandaiWalk*` clips (PFNN BandaiWalk config) | git commit `74ead3ba` | CC BY-NC 4.0 |
| Edinburgh Locomotion MOCAP Database | `Edinburg` / `EdinburghLoco` clips | git commit `38fb87f0` | none stated by the authors |
| Mixamo Y Bot (Adobe) | target character for every paper result | manual download, sha256 checked | Mixamo terms; no redistribution |

### LAFAN1

- Source: <https://github.com/ubisoft/ubisoft-laforge-animation-dataset>. The script downloads
  `lafan1/lafan1.zip` from the repository's LFS URL, which needs no login and no git-lfs.
- 77 BVH files from 5 subjects, 496,672 frames. They are extracted to `Assets/LFS/Animation/lafan1/bvh/`.
- Licence: CC BY-NC-ND 4.0, see `license.txt` in that repository.
- **Holden subset.** The `LafanHolden` clips are the ranges Daniel Holden's
  [Motion-Matching](https://github.com/orangeduck/Motion-Matching) demo builds its database from
  (`resources/generate_database.py`):

  | File | Frames |
  |---|---|
  | `pushAndStumble1_subject5` | 194–351 |
  | `run1_subject5` | 90–7086 |
  | `walk1_subject5` | 80–7791 |

- Cite:

```bibtex
@article{harvey2020robust,
  author  = {F{\'e}lix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and Christopher Pal},
  title   = {Robust Motion In-Betweening},
  journal = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
  volume  = {39}, number = {4}, year = {2020}
}
```

### Bandai-Namco Research Motion Dataset

- Source: <https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset>, cloned
  and checked out at commit `74ead3ba1ae4696404e6086233779f60de8bf9ef` into
  `External/LFS/Bandai-Namco-Research-Motiondataset/`.
- Only dataset-2 is used: 2,902 BVH files, 384,931 frames at 30 fps.
- Licence: CC BY-NC 4.0, see `dataset/Bandai-Namco-Research-Motiondataset-2/LICENSE`.
- Cite:

```bibtex
@misc{kobayashi2023motion,
  title  = {Motion Capture Dataset for Practical Use of AI-based Motion Editing and Stylization},
  author = {Makito Kobayashi and Chen-Chieh Liao and Keito Inoue and Sentaro Yojima and Masafumi Takahashi},
  year   = {2023}, eprint = {2306.08861}, archivePrefix = {arXiv}, primaryClass = {cs.CV}
}
```

### Edinburgh Locomotion MOCAP Database

- Source: <https://bitbucket.org/jonathan-schwarz/edinburgh_locomotion_mocap_dataset>, cloned and
  checked out at commit `38fb87f0874bdfcdfd2cd37b27e4aceabef1109d` into
  `External/LFS/edinburgh_locomotion_mocap_dataset/`.
- 1,855 sequences of 240 frames: 1,835 for training and 20 for testing. They are stored as 21 joint
  *positions* per frame, with no rotations.
- `Tools/Retargeting/edinburgh_npz_to_bvh.py` solves the rotations and writes BVH to
  `Assets/LFS/Animation/Edinburgh/bvh/`. The `EdinburghLoco` configs use 1,743 of the 1,835
  training clips; the selection is stored in the config assets themselves.
- Licence: the repository states none. Ask its authors before using it beyond research.
- Cite, as the repository's README asks:

```bibtex
@inproceedings{ikhansul17-vaelstm,
  author    = {Habibie, Ikhansul and Holden, Daniel and Schwarz, Jonathan and Yearsley, Joe and Komura, Taku},
  booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
  title     = {A Recurrent Variational Autoencoder for Human Motion Synthesis},
  year      = {2017}
}
```

### Mixamo Y Bot

Every paper result is synthesised on Mixamo's **Y Bot**. Adobe's terms allow using it in research
but not redistributing the file, so you download it yourself:

1. Sign in at <https://www.mixamo.com> with a free Adobe ID.
2. On the Characters tab, choose **Y Bot**.
3. Download it as **FBX Binary**, in **T-pose**, with no animation.
4. Save it as `Assets/LFS/Characters/Y Bot.fbx`.

`fetch_datasets.py --check` compares the file against the checksum of the copy the paper used. A
mismatch is only a warning, because Adobe may re-export the character. The bone names and rest pose
are what matter, and the retarget step checks those.
