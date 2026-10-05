"""Keep the GUIDs of the private Assets/LFS files that tracked assets refer to.

Tracked clip and config assets point at retargeted FBX files in `Assets/LFS` by the GUID in each
FBX's `.meta`. That tree is a separate, unpublished repository, so someone rebuilding it from
`fetch_datasets.py` and the retarget pipeline would get fresh metas with fresh GUIDs and every
reference would break. This tool mirrors the metas that matter into `Tools/Data/lfs-meta/`.
The FBX metas use `fileIdsGeneration: 2`, which derives sub-asset IDs from names, so an FBX
regenerated under its original `.meta` binds to the existing assets again.

    python Tools/Data/lfs_meta.py export            # refresh the mirror from Assets/LFS
    python Tools/Data/lfs_meta.py restore [--force] # put mirrored metas beside rebuilt files
    python Tools/Data/lfs_meta.py check             # report what is out of step

Run `restore` before opening Unity on a rebuilt tree: Unity writes a fresh `.meta` for any file
that lacks one the moment it sees it, and restoring over that needs `--force`.
"""

import argparse
import filecmp
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LFS_ROOT = REPO_ROOT / "Assets" / "LFS"
MIRROR_ROOT = Path(__file__).with_name("lfs-meta")

# Kept whole, referenced or not: a fresh clone needs the character before anything else.
ALWAYS_EXPORTED = ["Characters"]
# Sources and Blender backups are never regenerated, so no rebuild needs their GUIDs kept.
SKIPPED_SUFFIXES = (".bvh", ".blend", ".blend1")
TRACKED_ASSET_PATTERNS = [":(glob)Assets/**/*.asset", ":(glob)Assets/**/*.prefab",
                          ":(glob)Assets/**/*.unity"]

GUID_LINE = re.compile(r"^guid: ([0-9a-f]{32})\s*$", re.MULTILINE)
GUID_REFERENCE = re.compile(rb"guid: ([0-9a-f]{32})")


def say(message=""):
    print(message, flush=True)


def read_guid(meta_path):
    match = GUID_LINE.search(meta_path.read_text(encoding="utf-8", errors="replace"))
    return match.group(1) if match else None


def asset_of(meta_path):
    """The file or folder a `.meta` describes."""
    return meta_path.with_name(meta_path.name[:-len(".meta")])


def lfs_metas():
    """`{path relative to Assets/LFS: guid}` for every meta Unity would keep there."""
    metas = {}
    for meta in LFS_ROOT.rglob("*.meta"):
        relative = meta.relative_to(LFS_ROOT)
        # Unity ignores dot-folders, so nothing in them has a GUID anyone can reference.
        if any(part.startswith(".") for part in relative.parts):
            continue
        if asset_of(meta).name.lower().endswith(SKIPPED_SUFFIXES):
            continue
        guid = read_guid(meta)
        if guid:
            metas[relative.as_posix()] = guid
    return metas


def tracked_references():
    """`{guid: [tracked asset paths referring to it]}` over the main repository's assets."""
    listing = subprocess.run(["git", "ls-files", "-z", "--", *TRACKED_ASSET_PATTERNS],
                             cwd=REPO_ROOT, check=True, capture_output=True).stdout
    references = defaultdict(list)
    for name in filter(None, listing.decode("utf-8").split("\0")):
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        for guid in set(GUID_REFERENCE.findall(path.read_bytes())):
            references[guid.decode("ascii")].append(name)
    return references


def folder_metas_above(relative_meta):
    """Metas of each folder between Assets/LFS and the given meta's asset."""
    folder = Path(relative_meta).parent
    return [ancestor.as_posix() + ".meta" for ancestor in [folder, *folder.parents]
            if ancestor != Path(".")]


def export(_args):
    metas = lfs_metas()
    references = tracked_references()

    referenced = {meta: guid for meta, guid in metas.items() if guid in references}
    selected = set(referenced)
    for meta in metas:
        if meta.split("/", 1)[0] in ALWAYS_EXPORTED:
            selected.add(meta)
    for meta in list(selected):
        selected.update(folder for folder in folder_metas_above(meta) if folder in metas)

    say(f"{len(referenced)} Assets/LFS file(s) referenced by tracked assets:")
    say(f"  {'LFS file':<72} {'guid':<32}  refs  e.g.")
    for meta in sorted(referenced):
        users = references[referenced[meta]]
        say(f"  {meta[:-len('.meta')]:<72} {referenced[meta]}  {len(users):>4}  {users[0]}")

    by_top = defaultdict(int)
    for meta in metas:
        if meta not in selected and not asset_of(LFS_ROOT / meta).is_dir():
            by_top["/".join(meta.split("/")[:3])] += 1
    if by_top:
        say("\nunreferenced metas left out (no tracked asset uses their GUID):")
        for folder, count in sorted(by_top.items()):
            say(f"  {folder}: {count}")

    stale = {path.relative_to(MIRROR_ROOT).as_posix() for path in MIRROR_ROOT.rglob("*.meta")} \
        if MIRROR_ROOT.exists() else set()
    for meta in sorted(selected):
        target = MIRROR_ROOT / meta
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LFS_ROOT / meta, target)
        stale.discard(meta)
    for meta in stale:
        (MIRROR_ROOT / meta).unlink()
    for folder in sorted(MIRROR_ROOT.rglob("*"), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()

    say(f"\nmirrored {len(selected)} meta(s) into {MIRROR_ROOT.relative_to(REPO_ROOT).as_posix()}"
        + (f", removed {len(stale)} stale" if stale else ""))
    return 0


def mirrored_metas():
    if not MIRROR_ROOT.exists():
        return []
    return sorted(path.relative_to(MIRROR_ROOT).as_posix() for path in MIRROR_ROOT.rglob("*.meta"))


def restore(args):
    restored = conflicts = 0
    for meta in mirrored_metas():
        source, target = MIRROR_ROOT / meta, LFS_ROOT / meta
        if not asset_of(target).exists():
            continue
        if not target.exists():
            shutil.copyfile(source, target)
            restored += 1
            continue
        if filecmp.cmp(source, target, shallow=False):
            continue
        mirror_guid, local_guid = read_guid(source), read_guid(target)
        if mirror_guid != local_guid:
            say(f"GUID DIFFERS: {meta}  local {local_guid}, tracked assets expect {mirror_guid}")
        else:
            say(f"settings differ: {meta}")
        if args.force:
            shutil.copyfile(source, target)
            restored += 1
        else:
            conflicts += 1
    say(f"restored {restored} meta(s)"
        + (f"; {conflicts} differing left alone (re-run with --force to overwrite)"
           if conflicts else ""))
    return 1 if conflicts else 0


def check(_args):
    problems = 0
    for fbx in sorted(LFS_ROOT.rglob("*.fbx")):
        if any(part.startswith(".") for part in fbx.relative_to(LFS_ROOT).parts):
            continue
        if not fbx.with_name(fbx.name + ".meta").exists():
            say(f"no meta: {fbx.relative_to(LFS_ROOT).as_posix()}")
            problems += 1
    for meta in mirrored_metas():
        target = LFS_ROOT / meta
        if not asset_of(target).exists():
            say(f"missing file: {meta[:-len('.meta')]}")
            problems += 1
        elif not target.exists():
            say(f"meta not restored: {meta}")
            problems += 1
        elif read_guid(target) != read_guid(MIRROR_ROOT / meta):
            say(f"GUID DIFFERS: {meta}  local {read_guid(target)}, "
                f"tracked assets expect {read_guid(MIRROR_ROOT / meta)}")
            problems += 1
    say(f"{problems} problem(s)" if problems else "Assets/LFS metas agree with the mirror")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("export", help="refresh Tools/Data/lfs-meta from Assets/LFS")
    restore_parser = commands.add_parser("restore", help="copy mirrored metas into Assets/LFS")
    restore_parser.add_argument("--force", action="store_true",
                                help="overwrite metas that differ from the mirror")
    commands.add_parser("check", help="report metas out of step with the mirror")
    args = parser.parse_args()
    return {"export": export, "restore": restore, "check": check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
