"""
Fréchet Motion Distance: how far each method's generated motion sits from the pose database,
as a distribution, in the latent space of an autoencoder trained on that database alone.

The FID recipe applied to motion windows. An autoencoder is fitted to one-second windows of the
database's motion features (``benchmark.motion_features``); database and generated windows are
encoded; a Gaussian is fitted to each set of latents, and the score is the Fréchet distance between
the two. Lower is closer to the data. Because the encoder sees only the database, one scorer judges
every method in a sweep.

Two scores are reported per method:

* **matched FMD** (the headline) -- the reference is not the whole database but, for every
  generated window, the ``k`` database windows whose root motion (sideways speed, forward speed,
  turning rate, averaged over the window) is closest. It asks whether the body moves like the data
  does *when the data travels like this*. A benchmark path asks for a narrow band of travel, and
  against the whole database even real walking windows score far from zero; whether the requested
  travel was delivered is the path-following metrics' job.
* **FMD** -- the plain score against every database window. It also charges a method for the parts
  of the database the task never asked for, so it measures coverage as much as quality.

Two reference numbers come with them:

* **floor** -- both scores between two halves of the database itself, split into alternating
  ten-second chunks, one half playing the method. A method cannot be expected to beat the data's
  own variation.
* **± (bootstrap)** -- the spread of a method's matched score when its runs are resampled with
  replacement, since paths, not windows, are the independent samples.

Every method here draws on the database it is scored against, so this measures staying on the
data's distribution, not generalising beyond it.

Usage, from the ``Python/`` folder::

    python -m benchmark.fmd ../Benchmarks/<sweep> \\
        --database ../Assets/StreamingAssets/MMDatabases/MM_YBot_Holden

The sweep must have been recorded with ``recordFullPose`` on. Writes ``fmd.csv`` (per method),
``fmd_runs.csv`` (per run) and ``fmd_encoder.pt`` into the sweep folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from scipy import linalg
from scipy.spatial import cKDTree
from torch import nn

from benchmark.motion_features import (database_sequences, feature_groups, read_recording,
                                       recording_features, windows)
from formats.pose_set_importer import deserialize_pose_set

FLOOR_CHUNK_SECONDS = 10.0

# The last three feature columns are the frame's sideways speed, forward speed and yaw rate; see
# benchmark.motion_features.sequence_features.
_FRAME_MOTION = slice(-3, None)


class MotionAutoencoder(nn.Module):
    """A temporal convolutional encoder to a small latent, and an MLP decoding the window back."""

    def __init__(self, n_features: int, window: int, latent: int, hidden: int = 256):
        super().__init__()
        self.window = window
        self.n_features = n_features
        self.latent = latent
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, hidden, 3, padding=1), nn.ELU(),
            nn.Conv1d(hidden, hidden, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv1d(hidden, hidden, 3, stride=2, padding=1), nn.ELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(
            nn.Linear(latent, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, window * n_features))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, window, features) -> (batch, latent)."""
        return self.encoder(x.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x)).view(-1, self.window, self.n_features)


class Normaliser:
    """
    Per-feature mean, and one standard deviation per feature group.

    Sharing the deviation across a group keeps a joint the database never moves (a finger in a
    capture without hands) from being divided by zero and dominating the distance.
    """

    def __init__(self, data: np.ndarray, n_bones: int):
        self.mean = data.mean(axis=0)
        self.std = np.ones_like(self.mean)
        per_feature = data.std(axis=0)
        for group in feature_groups(n_bones):
            self.std[group] = max(float(per_feature[group].mean()), 1e-6)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return ((data - self.mean) / self.std).astype(np.float32)


class TravelMatcher:
    """Finds, for a window's travel descriptor, the database windows that travel most like it."""

    def __init__(self, descriptors: np.ndarray, neighbours: int):
        self.scale = descriptors.std(axis=0) + 1e-6
        self.tree = cKDTree(descriptors / self.scale)
        self.neighbours = neighbours

    def __call__(self, descriptors: np.ndarray) -> np.ndarray:
        """Indices into the database windows, ``neighbours`` per query, repeats kept."""
        _, indices = self.tree.query(descriptors / self.scale, k=self.neighbours)
        return np.asarray(indices).reshape(-1)


def travel_descriptors(raw_windows: np.ndarray) -> np.ndarray:
    """(m, 3) mean sideways speed, forward speed and yaw rate of each un-normalised window."""
    return raw_windows[:, :, _FRAME_MOTION].mean(axis=1)


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between Gaussians fitted to two (n, d) sample sets."""
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
    trace = np.trace(cov_a + cov_b - 2.0 * np.real(root))
    return float(np.sum((mean_a - mean_b) ** 2) + trace)


def train_encoder(sequences: list[np.ndarray], window: int, latent: int, steps: int,
                  batch_size: int, device: str, seed: int) -> tuple[MotionAutoencoder, float]:
    """Fits the autoencoder to windows drawn at random from every clip; returns it and its final loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    data = torch.from_numpy(np.concatenate(sequences)).to(device)
    starts, offset = [], 0
    for sequence in sequences:
        starts.append(offset + np.arange(max(sequence.shape[0] - window + 1, 0)))
        offset += sequence.shape[0]
    starts = np.concatenate(starts)
    steps_in_window = torch.arange(window, device=device)

    model = MotionAutoencoder(data.shape[1], window, latent).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)

    recent = []
    for step in range(steps):
        batch_starts = torch.from_numpy(rng.choice(starts, batch_size)).to(device)
        batch = data[batch_starts[:, None] + steps_in_window[None, :]]
        loss = torch.mean((model(batch) - batch) ** 2)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()
        recent = (recent + [loss.item()])[-100:]
        if (step + 1) % 1000 == 0:
            print(f"  step {step + 1}/{steps}  reconstruction {np.mean(recent):.4f}")

    model.eval()
    return model, float(np.mean(recent))


@torch.no_grad()
def encode(model: MotionAutoencoder, batch: np.ndarray, device: str) -> np.ndarray:
    out = [model.encode(torch.from_numpy(batch[i:i + 4096]).to(device)).cpu().numpy()
           for i in range(0, batch.shape[0], 4096)]
    return np.concatenate(out) if out else np.zeros((0, model.latent), dtype=np.float32)


def floor_halves(sequences: list[np.ndarray], frame_time: float, window: int,
                 stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Database windows split into two halves by alternating fixed-length chunks of each clip."""
    chunk = int(round(FLOOR_CHUNK_SECONDS / frame_time))
    halves = ([], [])
    parity = 0
    for sequence in sequences:
        for start in range(0, sequence.shape[0], chunk):
            halves[parity].append(windows(sequence[start:start + chunk], window, stride))
            parity ^= 1
    return np.concatenate(halves[0]), np.concatenate(halves[1])


def matched_distance(latents: np.ndarray, matches: np.ndarray, reference: np.ndarray) -> float:
    return frechet_distance(latents, reference[matches])


def bootstrap_spread(runs: list[tuple[np.ndarray, np.ndarray]], reference: np.ndarray,
                     samples: int, rng: np.random.Generator) -> float:
    """Standard deviation of the matched score over runs resampled with replacement."""
    scores = []
    for _ in range(samples):
        picked = [runs[i] for i in rng.integers(0, len(runs), len(runs))]
        scores.append(matched_distance(np.concatenate([latents for latents, _ in picked]),
                                       np.concatenate([matches for _, matches in picked]),
                                       reference))
    return float(np.std(scores))


def load_runs(sweep: str) -> tuple[list[dict], float]:
    """The sweep's results rows that have a recording, and the settle time it ran with."""
    with open(os.path.join(sweep, 'results.csv'), newline='', encoding='utf-8') as f:
        rows = [row for row in csv.DictReader(f) if row['recordingFile'] and not row['error']]
    with open(os.path.join(sweep, 'results.json'), encoding='utf-8') as f:
        settle_time = float(json.load(f)['settleTime'])
    return rows, settle_time


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0])
    parser.add_argument('sweep', help='sweep folder recorded with recordFullPose on')
    parser.add_argument('--database', required=True, help='folder holding the reference .mmpose')
    parser.add_argument('--name', help='database name; defaults to the folder name')
    parser.add_argument('--window', type=int, default=30, help='window length in frames')
    parser.add_argument('--stride', type=int, default=5, help='window stride when scoring')
    parser.add_argument('--latent', type=int, default=32)
    parser.add_argument('--steps', type=int, default=8000, help='encoder training steps')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--neighbours', type=int, default=4,
                        help='database windows matched to each generated window by travel')
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    name = args.name or os.path.basename(os.path.normpath(args.database))
    pose_set = deserialize_pose_set(args.database, name)
    n_bones = len(list(pose_set.skeleton))
    frame_time = float(pose_set.frameTime)

    raw_database = database_sequences(pose_set)
    normalise = Normaliser(np.concatenate(raw_database), n_bones)
    database = [normalise(sequence) for sequence in raw_database]

    print(f"Training the encoder on {name} ({sum(len(s) for s in database)} frames, "
          f"{len(database)} clips) on {device}")
    started = time.perf_counter()
    model, reconstruction = train_encoder(database, args.window, args.latent, args.steps,
                                          args.batch_size, device, args.seed)
    print(f"  {time.perf_counter() - started:.0f} s")

    raw_reference = np.concatenate([windows(s, args.window, args.stride) for s in raw_database])
    reference = encode(model, normalise(raw_reference), device)
    matcher = TravelMatcher(travel_descriptors(raw_reference), args.neighbours)

    raw_a, raw_b = floor_halves(raw_database, frame_time, args.window, args.stride)
    latents_a = encode(model, normalise(raw_a), device)
    latents_b = encode(model, normalise(raw_b), device)
    floor = frechet_distance(latents_b, latents_a)
    matched_floor = matched_distance(
        latents_b, TravelMatcher(travel_descriptors(raw_a), args.neighbours)(travel_descriptors(raw_b)),
        latents_a)

    rows, settle_time = load_runs(args.sweep)
    recordings = os.path.join(args.sweep, 'recordings')
    by_method: dict[str, list[tuple[dict, np.ndarray, np.ndarray]]] = {}
    for row in rows:
        recording = read_recording(os.path.join(recordings, row['recordingFile']))
        raw = windows(recording_features(recording, pose_set, settle_time), args.window, args.stride)
        latents = encode(model, normalise(raw), device)
        if latents.shape[0] > 1:
            by_method.setdefault(row['method'], []).append(
                (row, latents, matcher(travel_descriptors(raw))))

    rng = np.random.default_rng(args.seed)
    summary = []
    for method, runs in by_method.items():
        pairs = [(latents, matches) for _, latents, matches in runs]
        all_latents = np.concatenate([latents for latents, _ in pairs])
        summary.append({
            'method': method,
            'runs': len(runs),
            'windows': len(all_latents),
            'matchedFmd': matched_distance(all_latents,
                                           np.concatenate([matches for _, matches in pairs]),
                                           reference),
            'bootstrapStd': bootstrap_spread(pairs, reference, args.bootstrap, rng),
            'fmd': frechet_distance(all_latents, reference),
        })

    with open(os.path.join(args.sweep, 'fmd.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'runs', 'windows', 'matchedFmd',
                                               'bootstrapStd', 'fmd', 'matchedFloor', 'floor',
                                               'referenceWindows', 'database'])
        writer.writeheader()
        for entry in summary:
            writer.writerow({**entry, 'matchedFloor': matched_floor, 'floor': floor,
                             'referenceWindows': len(reference), 'database': name})

    with open(os.path.join(args.sweep, 'fmd_runs.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'path', 'windows', 'matchedFmd', 'fmd'])
        writer.writeheader()
        for method, runs in by_method.items():
            for row, latents, matches in runs:
                writer.writerow({'method': method, 'path': row['path'], 'windows': len(latents),
                                 'matchedFmd': matched_distance(latents, matches, reference),
                                 'fmd': frechet_distance(latents, reference)})

    torch.save({'state_dict': model.state_dict(), 'mean': normalise.mean, 'std': normalise.std,
                'database': name, 'args': vars(args), 'reconstruction': reconstruction},
               os.path.join(args.sweep, 'fmd_encoder.pt'))

    print(f"\n{os.path.basename(os.path.normpath(args.sweep))} against {name}")
    print(f"  {'':<16} {'matched FMD':>16} {'FMD':>9}")
    print(f"  {'floor':<16} {matched_floor:>16.2f} {floor:>9.2f}   (database half vs half)")
    for entry in sorted(summary, key=lambda e: e['matchedFmd']):
        matched = f"{entry['matchedFmd']:.2f} +- {entry['bootstrapStd']:.2f}"
        print(f"  {entry['method']:<16} {matched:>16} {entry['fmd']:>9.2f}"
              f"   ({entry['runs']} runs, {entry['windows']} windows)")
    print(f"  reference: {len(reference)} database windows")


if __name__ == '__main__':
    main()
