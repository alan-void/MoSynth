"""Fetch and verify the raw motion datasets the retarget pipeline reads.

Every dataset lands exactly where the shard lists under `Tools/Retargeting/shards/` and the
retarget setups expect it, and is verified file by file against `checksums.json`. A dataset that
already verifies is left alone, so the script is safe to re-run and picks up where an interrupted
run stopped.

    python Tools/Data/fetch_datasets.py                 # fetch whatever is missing, then verify
    python Tools/Data/fetch_datasets.py --check         # report only, touch nothing
    python Tools/Data/fetch_datasets.py --only lafan1,bandai
    python Tools/Data/fetch_datasets.py --record [--lafan-zip PATH]

`--record` re-hashes the local copies into `checksums.json`; it is how the file was made, and is
only for whoever owns the reference copies.

The Mixamo character cannot be fetched: Adobe does not allow redistributing it, so it is
checked and, if missing, explained.

Run with the repository root as working directory; paths are resolved from this file, so any
working directory works.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKSUMS_PATH = Path(__file__).with_name("checksums.json")
DOWNLOAD_DIR = REPO_ROOT / "External" / "LFS" / ".downloads"

LAFAN_URL = ("https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/master/"
             "lafan1/lafan1.zip")
# Lowercase because the shard lists and the Assets/LFS index spell it that way.
LAFAN_DIR = REPO_ROOT / "Assets" / "LFS" / "Animation" / "lafan1" / "bvh"

BANDAI_URL = "https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset"
BANDAI_COMMIT = "74ead3ba1ae4696404e6086233779f60de8bf9ef"
BANDAI_CHECKOUT = REPO_ROOT / "External" / "LFS" / "Bandai-Namco-Research-Motiondataset"
BANDAI_DATA = "dataset/Bandai-Namco-Research-Motiondataset-2/data"

EDINBURGH_URL = "https://bitbucket.org/jonathan-schwarz/edinburgh_locomotion_mocap_dataset.git"
EDINBURGH_COMMIT = "38fb87f0874bdfcdfd2cd37b27e4aceabef1109d"
EDINBURGH_CHECKOUT = REPO_ROOT / "External" / "LFS" / "edinburgh_locomotion_mocap_dataset"
EDINBURGH_DATA = "data"

CHARACTERS_DIR = REPO_ROOT / "Assets" / "LFS" / "Characters"
CHARACTER_FILES = ["Y Bot.fbx"]

MIXAMO_INSTRUCTIONS = """\
    Download it yourself from https://www.mixamo.com (free Adobe account):
      Characters tab -> "Y Bot" -> Download, with
        Format: FBX Binary (.fbx)    Pose: T-pose
      and save it as Assets/LFS/Characters/Y Bot.fbx.
    The reference copy is a binary FBX (version 7700, written by FBX SDK 2020.2), about 1.9 MB,
    a 65-bone mixamorig skeleton standing in T-pose. Then run
      python Tools/Data/lfs_meta.py restore
    BEFORE opening Unity, so the FBX keeps the GUID every tracked config refers to."""

GIT_ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
CHUNK = 1 << 20


class DatasetError(Exception):
    pass


def say(message=""):
    print(message, flush=True)


def relative(path):
    try:
        return Path(path).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root, pattern):
    """`{relative posix path: sha256}` for every file under root matching pattern."""
    return {path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.glob(pattern)) if path.is_file()}


def verify_files(root, expected):
    """Return (missing, mismatched) relative paths of `expected` under root."""
    missing, mismatched = [], []
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif sha256_file(path) != digest:
            mismatched.append(name)
    return missing, mismatched


def describe(missing, mismatched, total):
    good = total - len(missing) - len(mismatched)
    text = f"{good}/{total} files verify"
    if missing:
        text += f", {len(missing)} missing (e.g. {missing[0]})"
    if mismatched:
        text += f", {len(mismatched)} differ (e.g. {mismatched[0]})"
    return text


def git(*args, cwd=None, check=True):
    return subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=check,
                          capture_output=True, text=True)


def has_commit(checkout, commit):
    return git("cat-file", "-e", commit + "^{commit}", cwd=checkout, check=False).returncode == 0


def head_commit(checkout):
    result = git("rev-parse", "--verify", "-q", "HEAD", cwd=checkout, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def ensure_checkout(url, commit, checkout, paths):
    """Bring `paths` of `commit` into `checkout`, cloning only what that commit needs.

    An existing checkout keeps its HEAD and has just those paths restored from the commit, so a
    clone someone is working in is not moved.
    """
    if not (checkout / ".git").exists():
        if checkout.exists() and any(checkout.iterdir()):
            raise DatasetError(f"{relative(checkout)} exists but is not a git checkout; "
                               "move it aside and re-run")
        checkout.mkdir(parents=True, exist_ok=True)
        git("init", "-q", cwd=checkout)
        git("remote", "add", "origin", url, cwd=checkout)
    # Checksums are of the committed bytes, so line-ending conversion would break every one.
    git("config", "core.autocrlf", "false", cwd=checkout)

    if not has_commit(checkout, commit):
        say(f"  fetching {commit[:12]} from {url}")
        shallow = git("fetch", "--depth", "1", "origin", commit, cwd=checkout, check=False)
        if shallow.returncode != 0:
            say("  the remote refused a single-commit fetch; fetching full history")
            git("fetch", "origin", cwd=checkout)
        if not has_commit(checkout, commit):
            raise DatasetError(f"commit {commit} is not on {url}")

    if head_commit(checkout) is None:
        git("checkout", "-q", "--detach", commit, cwd=checkout)
    else:
        git("checkout", commit, "--", *paths, cwd=checkout)


def download(url, target):
    """Download url to target, resuming a previous partial download if one is there."""
    partial = target.with_name(target.name + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    have = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers))
    except urllib.error.HTTPError as error:
        if error.code != 416:
            raise
        # 416: the partial file already holds the whole body.
        partial.replace(target)
        return
    with response:
        resuming = have and response.status == 206
        if have and not resuming:
            have = 0
        total = response.headers.get("Content-Length")
        total = int(total) + have if total else None
        say(f"  {'resuming at ' + format_mb(have) if resuming else 'downloading'} {url}")
        reported = have
        with open(partial, "ab" if resuming else "wb") as stream:
            for block in iter(lambda: response.read(CHUNK), b""):
                stream.write(block)
                have += len(block)
                if have - reported >= 16 * CHUNK:
                    reported = have
                    say(f"    {format_mb(have)}" + (f" / {format_mb(total)}" if total else ""))
    partial.replace(target)


def format_mb(size):
    return f"{size / CHUNK:.1f} MB"


class Dataset:
    name = ""
    title = ""
    licence = ""

    def check(self, sums):
        """Print status; return True if the dataset is present and verifies."""
        raise NotImplementedError

    def fetch(self, sums):
        raise NotImplementedError

    def record(self, sums, args):
        """Return this dataset's entry for checksums.json from the local copy."""
        raise NotImplementedError

    def header(self):
        say(f"[{self.name}] {self.title}")
        say(f"  licence: {self.licence}")


class Lafan1(Dataset):
    name = "lafan1"
    title = "Ubisoft La Forge Animation Dataset (Harvey et al. 2020)"
    licence = "CC BY-NC-ND 4.0"

    def check(self, sums):
        expected = sums[self.name]["files"]
        missing, mismatched = verify_files(LAFAN_DIR, expected)
        say(f"  {relative(LAFAN_DIR)}: {describe(missing, mismatched, len(expected))}")
        return not missing and not mismatched

    def fetch(self, sums):
        if self.check(sums):
            return
        entry = sums[self.name]
        archive = DOWNLOAD_DIR / "lafan1.zip"
        if not archive.exists():
            download(LAFAN_URL, archive)
        digest = sha256_file(archive)
        if digest != entry["zip_sha256"]:
            # The archive may be repacked upstream; the per-file hashes below are what count.
            say(f"  WARNING: lafan1.zip sha256 is {digest}, recorded {entry['zip_sha256']}")

        expected = entry["files"]
        LAFAN_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            members = {Path(info.filename).name: info for info in bundle.infolist()
                       if not info.is_dir()}
            for name, wanted in expected.items():
                path = LAFAN_DIR / name
                if path.is_file() and sha256_file(path) == wanted:
                    continue
                if name not in members:
                    raise DatasetError(f"lafan1.zip has no {name}")
                staging = path.with_name(path.name + ".part")
                with bundle.open(members[name]) as source, open(staging, "wb") as target:
                    shutil.copyfileobj(source, target, CHUNK)
                staging.replace(path)

        if not self.check(sums):
            archive.unlink()
            raise DatasetError("extracted BVH files do not match checksums.json; the archive "
                               "was deleted so the next run downloads it again")
        archive.unlink()

    def record(self, sums, args):
        entry = {"zip_sha256": sums.get(self.name, {}).get("zip_sha256", "")}
        if args.lafan_zip:
            entry["zip_sha256"] = sha256_file(Path(args.lafan_zip))
        if not entry["zip_sha256"]:
            say("  WARNING: no zip hash recorded; pass --lafan-zip")
        entry["files"] = hash_tree(LAFAN_DIR, "*.bvh")
        return entry


class GitDataset(Dataset):
    url = ""
    commit = ""
    checkout = None
    data = ""
    pattern = ""

    def check(self, sums):
        expected = sums[self.name]["files"]
        if not (self.checkout / ".git").exists():
            say(f"  {relative(self.checkout)}: not cloned")
            return False
        head = head_commit(self.checkout)
        if head != self.commit:
            say(f"  note: checkout HEAD is {head}, files are checked against {self.commit}")
        missing, mismatched = verify_files(self.checkout / self.data, expected)
        say(f"  {relative(self.checkout / self.data)}: "
            f"{describe(missing, mismatched, len(expected))}")
        return not missing and not mismatched

    def fetch(self, sums):
        if self.check(sums):
            return
        ensure_checkout(self.url, self.commit, self.checkout, [self.data])
        if not self.check(sums):
            raise DatasetError(f"{self.name} does not match checksums.json after checkout")

    def record(self, sums, args):
        return {"files": hash_tree(self.checkout / self.data, self.pattern)}


class Bandai(GitDataset):
    name = "bandai"
    title = "Bandai-Namco Research Motion Dataset, dataset-2"
    licence = "CC BY-NC 4.0 (dataset-2/LICENSE)"
    url = BANDAI_URL
    commit = BANDAI_COMMIT
    checkout = BANDAI_CHECKOUT
    data = BANDAI_DATA
    pattern = "*.bvh"


class Edinburgh(GitDataset):
    name = "edinburgh"
    title = "Edinburgh Locomotion MOCAP Database (Habibie et al., BMVC 2017)"
    licence = ("TODO: none stated in the repository; confirm terms with the authors "
               "before redistributing anything derived from it")
    url = EDINBURGH_URL
    commit = EDINBURGH_COMMIT
    checkout = EDINBURGH_CHECKOUT
    data = EDINBURGH_DATA
    pattern = "*.npz"


class Mixamo(Dataset):
    name = "mixamo"
    title = "Mixamo Y Bot character (manual download)"
    licence = "Adobe Mixamo terms: free to use, not to redistribute as a raw file"

    def check(self, sums):
        healthy = True
        for name, digest in sums[self.name]["files"].items():
            path = CHARACTERS_DIR / name
            if not path.is_file():
                say(f"  {relative(path)}: MISSING")
                say(MIXAMO_INSTRUCTIONS)
                healthy = False
            elif sha256_file(path) != digest:
                # Mixamo re-exports on every download, so a fresh copy rarely matches byte for byte.
                say(f"  WARNING: {relative(path)} differs from the reference copy. That is "
                    "expected for a fresh download; make sure it was exported as:")
                say(MIXAMO_INSTRUCTIONS)
            else:
                say(f"  {relative(path)}: verifies")
        return healthy

    def fetch(self, sums):
        if not self.check(sums):
            raise DatasetError("the Mixamo character has to be downloaded by hand (see above)")

    def record(self, sums, args):
        return {"files": {name: sha256_file(CHARACTERS_DIR / name) for name in CHARACTER_FILES
                          if (CHARACTERS_DIR / name).is_file()}}


DATASETS = [Lafan1(), Bandai(), Edinburgh(), Mixamo()]


def load_checksums():
    if not CHECKSUMS_PATH.exists():
        return {}
    with open(CHECKSUMS_PATH, encoding="utf-8") as stream:
        return json.load(stream)


def save_checksums(sums):
    with open(CHECKSUMS_PATH, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(sums, stream, indent=1, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report status only")
    mode.add_argument("--record", action="store_true",
                      help="hash the local copies into checksums.json")
    parser.add_argument("--only", default="",
                        help="comma-separated subset of: "
                             + ",".join(dataset.name for dataset in DATASETS))
    parser.add_argument("--lafan-zip", default="",
                        help="with --record, a lafan1.zip whose sha256 to record")
    args = parser.parse_args()

    selected = DATASETS
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        unknown = wanted - {dataset.name for dataset in DATASETS}
        if unknown:
            parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
        selected = [dataset for dataset in DATASETS if dataset.name in wanted]

    sums = load_checksums()
    if args.record:
        for dataset in selected:
            entry = dataset.record(sums, args)
            if not entry.get("files"):
                say(f"[{dataset.name}] nothing found locally; keeping the recorded entry")
                continue
            sums[dataset.name] = entry
            say(f"[{dataset.name}] recorded {len(entry['files'])} file(s)")
        save_checksums(sums)
        say(f"wrote {relative(CHECKSUMS_PATH)}")
        return 0

    failed = []
    for dataset in selected:
        dataset.header()
        if dataset.name not in sums:
            say(f"  no entry in {relative(CHECKSUMS_PATH)}")
            failed.append(dataset.name)
            continue
        try:
            if args.check:
                healthy = dataset.check(sums)
            else:
                dataset.fetch(sums)
                healthy = True
        except (DatasetError, OSError, urllib.error.URLError,
                subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", "") or ""
            say(f"  FAILED: {error} {detail.strip()}")
            healthy = False
        if not healthy:
            failed.append(dataset.name)
        say()

    if failed:
        say(f"not ready: {', '.join(failed)}")
        return 1
    say("all selected datasets verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
