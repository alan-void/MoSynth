"""Rebuilds every retargeted FBX of one character from the raw BVH, by replaying the recorded runs.

    python Tools/Retargeting/reproduce_all.py --dry-run
    python Tools/Retargeting/reproduce_all.py [--rig YBot] [--dataset Lafan1,edinburgh]

Unity must be closed while this runs. Each run writes FBX files into Assets/LFS, and Unity would
import every one of them under a fresh .meta -- a fresh GUID -- before the mirrored metas are put
back at the end, which breaks every clip asset that refers to them.

Each `Tools/Retargeting/shards/<rig>/<dataset>/` holds the run that made that dataset's FBX: one
`<shard>.txt` per FBX listing its BVH in take order, and a `run.json` with the setup, the retarget
script and its flags. Per dataset this
  1. converts the Edinburgh .npz to BVH if the listed BVH are missing,
  2. builds the setup blend the run names, from a published setup under Tools/Retargeting/setups/,
     if it is missing (see SETUP_RECIPES),
  3. replays the shards through run_batch_all.py --shard-list-dir, which skips any FBX already
     built,
and finally restores the mirrored .meta files with Tools/Data/lfs_meta.py restore.

Stdlib only, and it runs outside Blender. The Edinburgh conversion needs numpy, so run this with
the project's Python environment.
"""

import argparse
import json
import os
import subprocess
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(TOOLS_DIR))
SHARDS_ROOT = os.path.join(TOOLS_DIR, "shards")

sys.path.insert(0, TOOLS_DIR)
import run_batch_all  # noqa: E402

LAFAN_BVH = "Assets/LFS/Animation/lafan1/bvh/"
BANDAI_BVH = ("External/LFS/Bandai-Namco-Research-Motiondataset/dataset/"
              "Bandai-Namco-Research-Motiondataset-2/data/")
EDINBURGH_BVH = "Assets/LFS/Animation/Edinburgh/bvh"
EDINBURGH_NPZ = "External/LFS/edinburgh_locomotion_mocap_dataset/data/"
MIXAMO_CHARACTER = "Assets/LFS/Characters/Y Bot.fbx"

# Every setup blend a recorded run names, and how to get it without the private Assets/LFS tree.
#
# "mixamo" builds it with make_mixamo_setup.py from a published setup. The clips are the ones the
# original blend had parked, in the same order: the character's height above the floor is measured
# over them, so a different set gives a different RT_ground. "published" uses the published blend
# in place; it is the original stripped of meshes and parked clips, so its exports carry the
# skeleton and the takes but not the corrected rig's skinned mesh.
SETUP_RECIPES = {
    "Assets/LFS/Retargeting/ybot/lafan_bvh_to_ybot.blend": {
        "kind": "mixamo",
        "from": "Tools/Retargeting/setups/lafan_bvh_to_lafan_corrected.blend",
        "clips": [LAFAN_BVH + "walk1_subject1.bvh", LAFAN_BVH + "run1_subject2.bvh",
                  LAFAN_BVH + "jumps1_subject1.bvh"],
        "keep_capture_position": False,
    },
    "Assets/LFS/Retargeting/ybot/bandai_namco_to_ybot.blend": {
        "kind": "mixamo",
        "from": "Tools/Retargeting/setups/bandai_namco_to_lafan_corrected.blend",
        "clips": [BANDAI_BVH + "dataset-2_walk_normal_001.bvh",
                  BANDAI_BVH + "dataset-2_run_normal_001.bvh",
                  BANDAI_BVH + "dataset-2_walk-turn-left_normal_001.bvh"],
        "keep_capture_position": True,
    },
    "Assets/LFS/Retargeting/ybot/edinburgh_bvh_to_ybot.blend": {
        "kind": "mixamo",
        "from": "Tools/Retargeting/setups/edinburgh_rest_authoring.blend",
        "clips": [EDINBURGH_BVH + "/edin_locomotion_{}.bvh".format(i)
                  for i in ("0000", "0300", "0700", "1100", "1500")]
                 + [EDINBURGH_BVH + "/edin_test_00.bvh"],
        "keep_capture_position": False,
    },
    "Assets/LFS/Retargeting/lafan_bvh_to_lafan_corrected.blend": {
        "kind": "published",
        "from": "Tools/Retargeting/setups/lafan_bvh_to_lafan_corrected.blend",
    },
    "Assets/LFS/Retargeting/edinburgh_rest_authoring.blend": {
        "kind": "published",
        "from": "Tools/Retargeting/setups/edinburgh_rest_authoring.blend",
    },
}

# Both splits in one run: every BVH has to declare one skeleton, which the converter fits across
# every clip it is given. See edinburgh_npz_to_bvh.py.
EDINBURGH_CONVERSION = [
    "Tools/Retargeting/edinburgh_npz_to_bvh.py",
    "--npz", EDINBURGH_NPZ + "edinburgh_locomotion_train.npz",
    EDINBURGH_NPZ + "edinburgh_locomotion_test.npz",
    "--name", "edin_locomotion", "edin_test",
    "--out-dir", EDINBURGH_BVH,
]

UNITY_WARNING = """\
WARNING: Unity must be closed for this whole run. It writes FBX files into Assets/LFS, and an open
         Editor imports each one under a fresh .meta, breaking every asset that refers to it."""


def say(message=""):
    print(message, flush=True)


def resolve(path):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


def quote(command):
    return " ".join('"{}"'.format(part) if " " in part else part for part in command)


class Step:
    """One command of the plan, with what it is for and the file it must leave behind, if any."""

    def __init__(self, label, command, produces=None):
        self.label = label
        self.command = command
        self.produces = produces


def read_run(folder):
    with open(os.path.join(folder, "run.json"), encoding="utf-8") as handle:
        run = json.load(handle)
    lists = sorted(n for n in os.listdir(folder) if n.endswith(".txt"))
    clips = []
    for name in lists:
        with open(os.path.join(folder, name), encoding="utf-8") as handle:
            clips.extend(line.strip() for line in handle if line.strip())
    run["shards"] = [os.path.splitext(n)[0] for n in lists]
    run["clips"] = clips
    return run


def setup_step(run, blender, options):
    """The setup blend a run should use, and the step that builds it if it has to be built."""
    setup = run["setup"]
    if os.path.isfile(resolve(setup)):
        return resolve(setup), None
    recipe = SETUP_RECIPES.get(setup)
    if recipe is None:
        sys.exit("{} is missing and there is no recipe for rebuilding it; add one to "
                 "SETUP_RECIPES.".format(setup))
    published = resolve(recipe["from"])
    if not os.path.isfile(published) and not options.dry_run:
        sys.exit("The published setup {} is missing.".format(recipe["from"]))
    if recipe["kind"] == "published":
        return published, None

    # Blender exits 0 after an uncaught exception unless told otherwise.
    command = [blender, "--background", "--python-exit-code", "1", "--python",
               os.path.join(TOOLS_DIR, "make_mixamo_setup.py"), "--",
               "--setup", published, "--model-fbx", resolve(MIXAMO_CHARACTER),
               "--out", resolve(setup), "--clips"] + [resolve(c) for c in recipe["clips"]]
    if recipe["keep_capture_position"]:
        command.append("--keep-capture-position")
    return resolve(setup), Step("build " + setup, command, produces=resolve(setup))


def plan(options):
    rig_dir = os.path.join(SHARDS_ROOT, options.rig)
    if not os.path.isdir(rig_dir):
        sys.exit("No shard lists for rig {} under {}".format(options.rig, SHARDS_ROOT))
    datasets = sorted(d for d in os.listdir(rig_dir)
                      if os.path.isfile(os.path.join(rig_dir, d, "run.json")))
    if options.dataset:
        unknown = sorted(set(options.dataset) - set(datasets))
        if unknown:
            sys.exit("No recorded run for {} under {}; there are: {}".format(
                ", ".join(unknown), rig_dir, ", ".join(datasets)))
        datasets = [d for d in datasets if d in options.dataset]

    blender = run_batch_all.find_blender(options.blender) if not options.dry_run \
        else (options.blender or os.environ.get(run_batch_all.BLENDER_ENV) or "blender")
    steps, converted = [], False
    for dataset in datasets:
        folder = os.path.join(rig_dir, dataset)
        run = read_run(folder)
        missing = [c for c in run["clips"] if not os.path.isfile(resolve(c))]
        say("{}/{}: {} shard(s), {} clip(s), {} missing; setup {}".format(
            options.rig, dataset, len(run["shards"]), len(run["clips"]), len(missing),
            run["setup"]))

        if missing and not converted and all(
                c.replace("\\", "/").startswith(EDINBURGH_BVH + "/") for c in missing):
            steps.append(Step("convert the Edinburgh .npz to BVH",
                              [sys.executable] + [resolve(EDINBURGH_CONVERSION[0])]
                              + EDINBURGH_CONVERSION[1:]))
            converted = True
        elif missing and not options.dry_run:
            sys.exit("{} BVH listed for {} are missing, starting with {}. Fetch the dataset first "
                     "(Tools/Data/fetch_datasets.py).".format(len(missing), dataset, missing[0]))

        setup, build = setup_step(run, blender, options)
        if build is not None:
            steps.append(build)

        out_dir = os.path.join(options.out_root, dataset) if options.out_root \
            else resolve(run["out_dir"])
        command = [sys.executable, os.path.join(TOOLS_DIR, "run_batch_all.py"),
                   "--setup", setup, "--shard-list-dir", folder, "--out-dir", out_dir,
                   "--script", run["script"], "--simplify", repr(float(run["simplify"])),
                   "--workers", str(options.workers)]
        if run["keep_capture_position"]:
            command.append("--keep-capture-position")
        if blender and not options.dry_run:
            command += ["--blender", blender]
        if options.only:
            command += ["--only", options.only]
        if options.limit:
            command += ["--limit", str(options.limit)]
        if options.force:
            command.append("--force")
        steps.append(Step("retarget {}/{}".format(options.rig, dataset), command))

    if options.out_root:
        say("--out-root is set, so nothing lands in Assets/LFS and the .meta restore is skipped")
    else:
        steps.append(Step("restore the mirrored .meta files",
                          [sys.executable, resolve("Tools/Data/lfs_meta.py"), "restore"]))
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rig", default="YBot",
                        help="character folder under Tools/Retargeting/shards (default: YBot)")
    parser.add_argument("--dataset", default="",
                        help="comma-separated dataset folders to run; default every one")
    parser.add_argument("--workers", type=int, default=4,
                        help="Blender processes per dataset (default: 4)")
    parser.add_argument("--blender", default="",
                        help="path to blender.exe; else $MOSYNTH_BLENDER_EXE, else the usual "
                             "install folders")
    parser.add_argument("--force", action="store_true", help="rebuild shards already built")
    parser.add_argument("--only", default="", help="comma-separated shard names (smoke runs)")
    parser.add_argument("--limit", type=int, default=0,
                        help="take at most N clips from each shard (smoke runs)")
    parser.add_argument("--out-root", default="",
                        help="write each dataset's FBX to <out-root>/<dataset> instead of where "
                             "the run recorded (smoke runs)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    options = parser.parse_args()
    options.dataset = [d.strip() for d in options.dataset.split(",") if d.strip()]
    if options.out_root:
        options.out_root = os.path.abspath(options.out_root)

    say(UNITY_WARNING)
    say()
    steps = plan(options)
    say()
    for index, step in enumerate(steps, 1):
        say("[{}/{}] {}".format(index, len(steps), step.label))
        say("    " + quote(step.command))
    if options.dry_run:
        return 0

    skipped = False
    for index, step in enumerate(steps, 1):
        say()
        say("=== [{}/{}] {}".format(index, len(steps), step.label))
        code = subprocess.call(step.command, cwd=REPO_ROOT)
        if code == 0 and step.produces and not os.path.isfile(step.produces):
            code = run_batch_all.EXIT_FATAL
            say("{} was not written".format(step.produces))
        # run_batch_all.py exits 2 when clips were skipped but every shard still exported.
        skipped |= code == run_batch_all.EXIT_SKIPPED
        if code not in (run_batch_all.EXIT_OK, run_batch_all.EXIT_SKIPPED):
            say("FAILED: {} (exit {})".format(step.label, code))
            return code
    say()
    if skipped:
        say("done, but some clips were skipped; each run's .batch/manifest.json lists them")
        return run_batch_all.EXIT_SKIPPED
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
