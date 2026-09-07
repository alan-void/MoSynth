"""Retargets a whole BVH dataset by running batch_retarget.py over it in shards.

    python Tools/Retargeting/run_batch_all.py \
        --setup   Assets/LFS/Retargeting/bandai-namco/bandai_namco_retarget_to_corrected_lafan_claude.blend \
        --bvh-dir External/LFS/Bandai-Namco-Research-Motiondataset/dataset/Bandai-Namco-Research-Motiondataset-2/data \
        --out-dir Assets/LFS/Animation/bandai-namco/retargeted \
        --simplify 1 --workers 6

batch_retarget.py exports one FBX holding a take per clip, which does not scale to a few thousand:
every retargeted action stays resident until that single export, and Rokoko's rest-pose drift
accumulates across the run until it trips the guard. Both reset only when the process exits, so a
dataset is retargeted as several independent runs rather than one long one. That also makes the
batch resumable and lets it use more than one core.

Shards follow the dataset's own motion labels, so an output file is something a person can name.
Stdlib only, and it runs outside Blender.
"""

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_retarget.py")

# The same candidates RetargetBatchWindow probes, so the two launchers agree on which Blender runs.
BLENDER_CANDIDATES = [
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
]

# batch_retarget.py's exit codes.
EXIT_OK, EXIT_FATAL, EXIT_SKIPPED = 0, 1, 2

# Everything the driver writes that is not an FBX. The leading dot keeps Unity from importing any
# of it, since the output folder lives under Assets/.
WORK_DIRNAME = ".batch"

_print_lock = threading.Lock()


def say(message):
    with _print_lock:
        print(message, flush=True)


def resolve(path):
    """Paths are given relative to the repository root, as the Unity launcher does."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


def find_blender(explicit):
    if explicit:
        if not os.path.isfile(explicit):
            sys.exit("No Blender at " + explicit)
        return explicit
    for candidate in BLENDER_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    sys.exit("Could not find Blender; pass --blender with the path to blender.exe")


# --- sharding -------------------------------------------------------------------------


def parse_clip_name(filename):
    """`dataset-2_walk-turn-left_normal_005.bvh` -> ("dataset-2", "walk-turn-left").

    The underscore is the field separator and never appears inside a token, so the dataset and
    motion labels can be read straight off the name; `cfg/content_label.txt` lists the same
    vocabulary.
    """
    parts = os.path.splitext(filename)[0].split("_")
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def split_evenly(items, maximum):
    """Contiguous parts, none longer than `maximum`, sizes within one of each other.

    Contiguous rather than interleaved so that a part still groups clips of one style together,
    the files being sorted by name.
    """
    count = max(1, int(math.ceil(len(items) / float(maximum))))
    base, remainder = divmod(len(items), count)
    parts, start = [], 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        parts.append(items[start:start + size])
        start += size
    return parts


def build_shards(bvh_dir, prefix, max_per_file, only, limit):
    """Groups the folder's clips by motion label into the shards a run is made of."""
    names = sorted(n for n in os.listdir(bvh_dir) if n.lower().endswith(".bvh"))
    if not names:
        sys.exit("No .bvh files in " + bvh_dir)

    groups = {}
    for name in names:
        dataset, motion = parse_clip_name(name)
        if motion is None:
            say("  ignoring unparseable name: " + name)
            continue
        if only and motion not in only:
            continue
        groups.setdefault((dataset, motion), []).append(name)

    if not groups:
        sys.exit("No clips left after --only " + ",".join(sorted(only)))

    shards = []
    for (dataset, motion), clips in sorted(groups.items()):
        parts = split_evenly(clips, max_per_file)
        for index, part in enumerate(parts, 1):
            if limit:
                part = part[:limit]
            stem = "{}_{}_{}".format(prefix, dataset, motion)
            if len(parts) > 1:
                stem += "_part{:02d}".format(index)
            shards.append({"name": stem,
                           "clips": [os.path.join(bvh_dir, c) for c in part]})
    return shards


# --- running --------------------------------------------------------------------------


def run_shard(shard, options):
    """One Blender process over one shard, filling the shard in with the outcome."""
    started = time.time()
    listing = os.path.join(options.work_dir, shard["name"] + ".txt")
    with open(listing, "w", encoding="utf-8") as handle:
        handle.write("\n".join(shard["clips"]) + "\n")

    command = [
        options.blender, "--background", options.setup,
        "--python", BATCH_SCRIPT, "--",
        "--files-from", listing,
        "--out", shard["fbx"],
        "--manifest", shard["manifest"],
        "--continue-on-error",
        "--simplify", repr(float(options.simplify)),
    ]
    if options.keep_capture_position:
        command.append("--keep-capture-position")

    log_path = os.path.join(options.work_dir, shard["name"] + ".log")
    say("  start {} ({} clips)".format(shard["name"], len(shard["clips"])))
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            line = line.rstrip("\n")
            log.write(line + "\n")
            # Blender's own chatter is dropped; of the rest, the per-clip counters would be six
            # interleaved streams of noise, so only real events are echoed.
            if not line.startswith("[retarget]"):
                continue
            body = line[len("[retarget] "):].strip()
            if body.startswith(("SKIP", "ERROR", "WARNING", "wrote", "retargeted")) \
                    or "-> take" in body:
                say("  [{}] {}".format(shard["name"], body))
        process.wait()

    shard["exit_code"] = process.returncode
    shard["seconds"] = round(time.time() - started, 1)
    shard["log"] = log_path
    if process.returncode == EXIT_FATAL:
        say("  FAILED {} (exit 1) - see {}".format(shard["name"], log_path))
    return shard


def main():
    parser = argparse.ArgumentParser(prog="run_batch_all", description=__doc__)
    parser.add_argument("--setup", required=True, help="the retarget setup .blend")
    parser.add_argument("--bvh-dir", required=True, help="folder of source BVH files")
    parser.add_argument("--out-dir", required=True, help="where the FBX files are written")
    parser.add_argument("--prefix", default="bandai",
                        help="leading token of each output file name (default: bandai)")
    parser.add_argument("--max-per-file", type=int, default=250,
                        help="split a motion into parts so no shard exceeds this many clips")
    parser.add_argument("--workers", type=int, default=6,
                        help="Blender processes to run at once (default: 6)")
    parser.add_argument("--simplify", type=float, default=0.0,
                        help="FBX keyframe reduction passed to each shard; 0 keeps every key")
    parser.add_argument("--only", default="",
                        help="comma-separated motion labels to include, e.g. walk,run")
    parser.add_argument("--limit", type=int, default=0,
                        help="take at most N clips from each shard (smoke runs)")
    parser.add_argument("--keep-capture-position", action="store_true",
                        help="hold each clip where its channels were recorded; needed when a "
                             "dataset varies the rest offset from clip to clip, as "
                             "Bandai-Namco dataset-2 does")
    parser.add_argument("--blender", default="", help="path to blender.exe")
    parser.add_argument("--force", action="store_true",
                        help="redo shards whose FBX already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the shards and stop")
    options = parser.parse_args()

    options.setup = resolve(options.setup)
    options.bvh_dir = resolve(options.bvh_dir)
    options.out_dir = resolve(options.out_dir)
    for label, path in (("setup blend", options.setup), ("BVH folder", options.bvh_dir)):
        if not os.path.exists(path):
            sys.exit("No such {}: {}".format(label, path))
    if not os.path.isfile(BATCH_SCRIPT):
        sys.exit("No batch_retarget.py beside this script: " + BATCH_SCRIPT)

    only = {token.strip() for token in options.only.split(",") if token.strip()}
    shards = build_shards(options.bvh_dir, options.prefix, options.max_per_file,
                          only, options.limit)

    options.work_dir = os.path.join(options.out_dir, WORK_DIRNAME)
    for shard in shards:
        shard["fbx"] = os.path.join(options.out_dir, shard["name"] + ".fbx")
        shard["manifest"] = os.path.join(options.work_dir, shard["name"] + ".json")

    total_clips = sum(len(s["clips"]) for s in shards)
    say("{} shard(s), {} clip(s) total".format(len(shards), total_clips))
    for shard in shards:
        built = os.path.exists(shard["fbx"])
        say("  {:<46s} {:>4d} clips{}".format(
            shard["name"], len(shard["clips"]), "   [done]" if built else ""))
    if options.dry_run:
        return EXIT_OK

    os.makedirs(options.work_dir, exist_ok=True)
    options.blender = find_blender(options.blender)
    say("blender: " + options.blender)

    pending = [s for s in shards if options.force or not os.path.exists(s["fbx"])]
    for shard in shards:
        if shard not in pending:
            say("  skip {} (already built)".format(shard["name"]))
    if not pending:
        say("Nothing to do; pass --force to rebuild.")
        return EXIT_OK

    say("running {} shard(s) on {} worker(s)".format(len(pending), options.workers))
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = [pool.submit(run_shard, shard, options) for shard in pending]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    return report(shards, pending, options, time.time() - started)


def report(shards, attempted, options, elapsed):
    """Merges the shard manifests and says plainly what came out."""
    merged, clips, frames, skipped, failed = [], 0, 0, [], []
    for shard in shards:
        if not os.path.exists(shard["manifest"]):
            if shard in attempted:
                failed.append(shard["name"])
            continue
        with open(shard["manifest"], encoding="utf-8") as handle:
            record = json.load(handle)
        record["shard"] = shard["name"]
        record["megabytes"] = (round(os.path.getsize(shard["fbx"]) / 1e6, 1)
                               if os.path.exists(shard["fbx"]) else 0.0)
        merged.append(record)
        clips += record["clips"]
        frames += record["frames"]
        skipped.extend(record["skipped"])

    merged_path = os.path.join(options.work_dir, "manifest.json")
    with open(merged_path, "w", encoding="utf-8") as handle:
        json.dump({"out_dir": options.out_dir, "setup": options.setup,
                   "simplify": options.simplify,
                   "keep_capture_position": options.keep_capture_position, "shards": merged,
                   "clips": clips, "frames": frames, "skipped": skipped},
                  handle, indent=2)

    say("")
    say("{:<46s} {:>6s} {:>9s} {:>8s} {:>8s} {:>10s}".format(
        "shard", "clips", "frames", "MB", "skipped", "drift"))
    for record in sorted(merged, key=lambda r: r["shard"]):
        say("{:<46s} {:>6d} {:>9d} {:>8.1f} {:>8d} {:>10.2e}".format(
            record["shard"], record["clips"], record["frames"], record["megabytes"],
            len(record["skipped"]), record["rest_drift_m"]))
    say("{:<46s} {:>6d} {:>9d} {:>8.1f} {:>8d}".format(
        "TOTAL", clips, frames, sum(r["megabytes"] for r in merged), len(skipped)))
    say("")
    say("{} shard(s) in {:.0f}s ({:.1f} min)".format(len(attempted), elapsed, elapsed / 60.0))
    say("manifest " + merged_path)

    if failed:
        say("FAILED shard(s), nothing exported: " + ", ".join(failed))
        return EXIT_FATAL
    if skipped:
        say("{} clip(s) skipped; they are listed in the manifest".format(len(skipped)))
        return EXIT_SKIPPED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
