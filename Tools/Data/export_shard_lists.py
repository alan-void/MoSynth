"""Copy the retarget runs' shard lists out of Assets/LFS into the tracked tree.

`run_batch_all.py` leaves each run's clip lists and settings in a `.batch/` folder beside its
FBX output, inside the unpublished `Assets/LFS` repository. This writes them to
`Tools/Retargeting/shards/<character>/<dataset>/` with machine-specific absolute paths made
repo-relative: one `<shard>.txt` per FBX listing its source BVH files in take order, and a
`run.json` with the settings the run used.

    python Tools/Data/export_shard_lists.py
"""

import json
import sys
from pathlib import Path, PureWindowsPath

REPO_ROOT = Path(__file__).resolve().parents[2]
ANIMATION_ROOT = REPO_ROOT / "Assets" / "LFS" / "Animation"
SHARDS_ROOT = REPO_ROOT / "Tools" / "Retargeting" / "shards"
CHARACTERS = ["YBot", "LafanCorrected"]
RUN_SETTINGS = ["script", "simplify", "keep_capture_position"]
# Manifests written before run_batch_all.py had --script ran its default engine.
DEFAULT_SCRIPT = "batch"
# Runs whose FBX were moved after the run: folder holding `.batch` -> folder now holding the FBX.
RELOCATED_RUNS = {"LafanCorrected": "LafanCorrected/Lafan1"}


def say(message=""):
    print(message, flush=True)


def repo_relative(recorded, run_root):
    """Turn a path recorded on the machine that made the run into a repo-relative one.

    `run_root` is that machine's repository root, so lists made on another checkout still convert.
    """
    path = PureWindowsPath(recorded)
    root = PureWindowsPath(run_root)
    if [part.lower() for part in path.parts[:len(root.parts)]] != \
            [part.lower() for part in root.parts]:
        raise ValueError(f"{recorded} is not under {run_root}")
    return "/".join(path.parts[len(root.parts):])


def run_root_of(manifest, batch):
    """The repository root of the machine that wrote the manifest.

    Read from `out_dir`, the folder the run wrote to, which is the `.batch` folder's parent even
    when that folder has been renamed since the run.
    """
    out_dir = PureWindowsPath(manifest["out_dir"])
    tail = len(batch.parent.relative_to(REPO_ROOT).parts)
    return str(PureWindowsPath(*out_dir.parts[:len(out_dir.parts) - tail]))


def export_run(batch, out_dir):
    """Write the shard lists and run.json of the run in `batch`, whose FBX are in out_dir."""
    with open(batch / "manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    run_root = run_root_of(manifest, batch)

    shard_names = [shard["shard"] for shard in manifest["shards"]]
    lists = {path.stem: path for path in batch.glob("*.txt")}
    if sorted(lists) != sorted(shard_names):
        raise ValueError(f"{batch}: shard lists {sorted(lists)} do not match the manifest's "
                         f"{sorted(shard_names)}")
    missing = [name for name in shard_names if not (out_dir / f"{name}.fbx").exists()]
    if missing:
        raise ValueError(f"{out_dir} lacks the run's output for {missing}")

    label = out_dir.relative_to(ANIMATION_ROOT).as_posix()
    target = SHARDS_ROOT / label
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("*.txt"):
        if stale.stem not in lists:
            stale.unlink()

    clips = 0
    for name in shard_names:
        lines = lists[name].read_text(encoding="utf-8").splitlines()
        paths = [repo_relative(line.strip(), run_root) for line in lines if line.strip()]
        clips += len(paths)
        with open(target / f"{name}.txt", "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(paths) + "\n")

    run = {"setup": repo_relative(manifest["setup"], run_root),
           "script": manifest.get("script", DEFAULT_SCRIPT),
           "simplify": manifest["simplify"],
           "keep_capture_position": manifest["keep_capture_position"],
           "out_dir": out_dir.relative_to(REPO_ROOT).as_posix()}
    with open(target / "run.json", "w", encoding="utf-8", newline="\n") as stream:
        json.dump(run, stream, indent=2)
        stream.write("\n")

    say(f"{label}: {len(shard_names)} shard(s), {clips} clip(s)"
        + ("" if clips == manifest.get("clips", clips)
           else f" (manifest records {manifest['clips']})"))


def main():
    for character in CHARACTERS:
        root = ANIMATION_ROOT / character
        runs = {}
        for folder in [root, *root.iterdir()]:
            if (folder / ".batch" / "manifest.json").exists():
                label = folder.relative_to(ANIMATION_ROOT).as_posix()
                runs[folder] = ANIMATION_ROOT / RELOCATED_RUNS.get(label, label)
        for folder in sorted(root.iterdir()):
            if folder.is_dir() and not folder.name.startswith(".") \
                    and folder not in runs.values():
                say(f"{character}/{folder.name}: no retarget run recorded, skipped")
        for folder, out_dir in sorted(runs.items()):
            export_run(folder / ".batch", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
