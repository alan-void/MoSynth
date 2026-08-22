"""
UMAP embedding of the motion field, for the Unity debug visualizer.

The motion field is a cloud of motion states in a 138-dimensional similarity
space (46 weighted joint position/velocity triples). Nothing about a stalled
policy is visible in that space directly, so this module projects it to 3D once,
offline, and writes the result next to the trained value function. Unity draws
the projection and overlays the live query, its k nearest neighbours and the tug
target, which turns "the character froze" into a picture of *where* in the field
it froze.

The field's own k-NN metric is a sum of per-joint L2 norms -- an L2,1 mixed
norm, not the flat L2 that UMAP would otherwise assume. Embedding under the
wrong metric would draw neighbourhoods the field does not actually see, so the
k-NN graph is built here with the field's metric and handed to UMAP via
`precomputed_knn`.

One consequence of `precomputed_knn`: `reducer.transform()` is unavailable, so a
novel pose cannot be projected exactly. Unity instead places the live state at
the similarity-weighted barycenter of its neighbours' embeddings, which costs
nothing and is the same interpolation the field already uses for values.

By default only the *position* half of the feature is projected. Embedding the
whole thing lays out a phase space, where one pose held at two speeds lands in
two places and nothing on screen distinguishes "a different pose" from "the same
pose, moving differently" -- which is what made the first version of this cloud
hard to read. Positions alone lay out a pose manifold instead.

Velocity does not disappear from the picture, it changes form. Since
`pose_v[i] = (Pose(i+1) - Pose(i)) / frame_time`, a state's one-frame lookahead
`x + v * frame_time` *is* the next state, so in a position-only projection the
velocity of state i points exactly at the point for state i+1 -- the
frame-adjacency edge Unity already draws. The edges become the velocity field.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import action_predictor
from MotionField import MotionField, load_bone_weights_file, resolve_device

# 'field'     -- k-NN graph under the field's own sum-of-per-joint-L2 metric.
# 'euclidean' -- flat L2 on the flattened feature, what UMAP assumes by default.
METRIC_FIELD = 'field'
METRIC_EUCLIDEAN = 'euclidean'

# 'position' -- joint positions only: a pose manifold, with velocity carried by
#               the frame-adjacency edges. See the module docstring.
# 'full'     -- positions and velocities, the original phase-space projection.
FEATURE_POSITION = 'position'
FEATURE_FULL = 'full'

DEFAULT_KNN_CHUNK = 256

def _noop_progress(stage: str, fraction: float) -> None:
    pass


def compute_knn(features: np.ndarray, k: int, device: str = None,
                chunk: int = DEFAULT_KNN_CHUNK, progress=_noop_progress):
    """
    k nearest neighbours of every state under the motion field's own metric.

    `MotionField.get_batched_knn` already does this but returns similarity
    weights; UMAP needs the raw distances, so the search is repeated here rather
    than widening that function's contract for one caller.

    :param features: (states, feature_rows, 3) float32, i.e. `state_features`.
    :param k: neighbours per state, including the state itself.
    :return: (indices int32 (states, k), distances float32 (states, k)),
        sorted nearest first.
    """
    device = resolve_device(device)
    states = torch.from_numpy(features).to(device, torch.float32)
    count = states.shape[0]
    k = min(int(k), count)

    indices = np.empty((count, k), dtype=np.int32)
    distances = np.empty((count, k), dtype=np.float32)

    chunk = max(1, int(chunk))
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        query = states[start:stop].unsqueeze(1)              # (c, 1, F, 3)
        # Sum of per-joint L2 norms -- the metric MotionField.get_knn uses.
        block = torch.norm(states.unsqueeze(0) - query, dim=3).sum(2)
        nearest = torch.topk(block, k=k, largest=False)
        indices[start:stop] = nearest.indices.cpu().numpy()
        distances[start:stop] = nearest.values.cpu().numpy()
        progress('Building the k-NN graph', stop / count)

    return indices, distances


def build_edges(state_frames: np.ndarray):
    """
    Directed state->state+1 links, and the clip each state came from.

    `action_predictor.build_state_indices` drops the last frame of every clip,
    so its output is one contiguous run of source frames per clip. Two adjacent
    states are a real transition exactly when their source frames are adjacent
    too -- deriving the links from that rather than re-deriving clip arithmetic
    keeps this in step with whatever boundary rule the state builder uses, and
    stops a link being drawn across a cut the database never contains.

    :param state_frames: (states,) source frame index per state.
    :return: (edges int32 (m, 2), state_clip int32 (states,))
    """
    state_frames = np.asarray(state_frames).ravel()
    states_count = state_frames.size
    if states_count == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.int32)

    contiguous = np.diff(state_frames) == 1
    # A break in the run starts a new clip.
    state_clip = np.concatenate([[0], np.cumsum(~contiguous)]).astype(np.int32)

    starts = np.nonzero(contiguous)[0].astype(np.int32)
    edges = np.stack([starts, starts + 1], axis=1).astype(np.int32)
    return edges, state_clip


def compute_embedding(data_dir: str, db_name: str, out_path: str,
                      n_components: int = 3,
                      n_neighbors: int = 80,
                      min_dist: float = 0.1,
                      metric_mode: str = METRIC_FIELD,
                      feature_mode: str = FEATURE_POSITION,
                      device: str = 'auto',
                      seed: int = 42,
                      knn_chunk: int = DEFAULT_KNN_CHUNK,
                      pos_weight: float = 1.0,
                      vel_weight: float = 0.06,
                      bone_weights=None,
                      progress=None) -> dict:
    """
    Fit the UMAP projection and write `<name>.mfembed.npz`.

    `progress(stage, fraction)` matches `motion_field_trainer.train`, so the
    Unity progress-bar callback works unchanged for both.

    :param feature_mode: `FEATURE_POSITION` to project the joint-position rows
        alone (the default -- see the module docstring), `FEATURE_FULL` for the
        whole position+velocity feature.

        Two consequences of position mode worth knowing. `vel_weight` stops
        affecting the projection entirely, and `pos_weight` only scales it
        uniformly, which cannot reorder neighbours and which UMAP normalises
        away regardless; per-joint *position* weights do still shape the layout.
        Both weights are still recorded in the file. And `load_embedding` will
        still warn when the bone weights change, including a velocity-only
        change that cannot have moved a single point -- over-warning is the safe
        direction for a staleness check, so it is left alone.
    :return: summary dict for logging on the C# side.
    """
    progress = progress or _noop_progress
    started = time.perf_counter()

    progress('Loading pose database', 0.0)
    skeleton, poses_x, poses_v, poses_y, _, frame_time, pose_set = \
        action_predictor.load_animations(data_dir, db_name)

    # Built only for its similarity features; no value function is needed.
    motion_field = MotionField(poses_x, poses_v, poses_y, skeleton, frame_time,
                               device=device, pos_weight=pos_weight,
                               vel_weight=vel_weight, bone_weights=bone_weights)
    features = motion_field.state_features
    if feature_mode == FEATURE_POSITION:
        # build_motion_states stacks bone_count weighted position rows on top of
        # bone_count weighted velocity rows; the first half is the pose. Slicing
        # here covers both uses below -- the k-NN graph and the matrix UMAP fits
        # are derived from this one array.
        features = np.ascontiguousarray(features[:, :motion_field.bone_count, :])

    states_count = int(features.shape[0])
    flat = features.reshape(states_count, -1)

    n_neighbors = int(max(2, min(n_neighbors, states_count - 1)))

    precomputed = None
    if metric_mode == METRIC_FIELD:
        indices, distances = compute_knn(features, n_neighbors, motion_field.device,
                                         knn_chunk, progress)
        precomputed = (indices, distances)
    else:
        progress('Building the k-NN graph', 1.0)

    progress('Fitting UMAP', 0.0)
    # Imported late: the numba warm-up costs ~8 s and should not be paid by
    # callers that only want load_embedding().
    import umap

    reducer = umap.UMAP(n_components=int(n_components),
                        n_neighbors=n_neighbors,
                        min_dist=float(min_dist),
                        metric='euclidean',
                        precomputed_knn=precomputed,
                        random_state=int(seed))
    embedding = np.ascontiguousarray(reducer.fit_transform(flat), dtype=np.float32)
    progress('Fitting UMAP', 1.0)

    n_poses = int(np.shape(pose_set.local_positions)[0])
    state_frames = action_predictor.build_state_indices(pose_set.clips, n_poses)
    edges, state_clip = build_edges(state_frames)
    speed = np.linalg.norm(poses_v[:, 0, :3], axis=1).astype(np.float32)

    progress('Writing embedding', 1.0)
    save_embedding(out_path, embedding, edges, state_clip, speed,
                   params={
                       'states_count': states_count,
                       'bone_count': int(motion_field.bone_count),
                       'n_neighbors': n_neighbors,
                       'n_components': int(n_components),
                       'min_dist': float(min_dist),
                       'metric_mode': str(metric_mode),
                       'feature_mode': str(feature_mode),
                       'seed': int(seed),
                       'pos_weight': float(pos_weight),
                       'vel_weight': float(vel_weight),
                   })

    summary = {
        'out_path': out_path,
        'states_count': states_count,
        'edges_count': int(edges.shape[0]),
        'n_components': int(n_components),
        'n_neighbors': n_neighbors,
        'metric_mode': str(metric_mode),
        'feature_mode': str(feature_mode),
        'feature_rows': int(features.shape[1]),
        'device': motion_field.device,
        'seconds': round(time.perf_counter() - started, 2),
    }
    print(f"[MotionField] embedded {summary['states_count']} states -> "
          f"{summary['n_components']}D, {summary['edges_count']} edges, "
          f"features={summary['feature_mode']} ({summary['feature_rows']} rows), "
          f"metric={summary['metric_mode']}, {summary['seconds']}s "
          f"on {summary['device']}")
    return summary


def save_embedding(out_path: str, embedding: np.ndarray, edges: np.ndarray,
                   state_clip: np.ndarray, speed: np.ndarray,
                   params: dict) -> None:
    """Write `<name>.mfembed.npz`, pickle-free so it loads under allow_pickle=False."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    payload = {
        'embedding': np.ascontiguousarray(embedding, dtype=np.float32),
        'edges': np.ascontiguousarray(edges, dtype=np.int32),
        'state_clip': np.ascontiguousarray(state_clip, dtype=np.int32),
        'speed': np.ascontiguousarray(speed, dtype=np.float32),
        'states_count': np.int64(params['states_count']),
        'bone_count': np.int64(params['bone_count']),
        'n_neighbors': np.int64(params['n_neighbors']),
        'n_components': np.int64(params['n_components']),
        'seed': np.int64(params['seed']),
        'min_dist': np.float32(params['min_dist']),
        'pos_weight': np.float32(params['pos_weight']),
        'vel_weight': np.float32(params['vel_weight']),
        'metric_mode': np.str_(params['metric_mode']),
        'feature_mode': np.str_(params.get('feature_mode', FEATURE_FULL)),
    }

    np.savez(out_path, **payload)


def load_embedding(path: str, states_count: int = None, log=print):
    """
    Load `<name>.mfembed.npz`.

    Every row of the embedding is a row of the pose database, so an embedding
    loaded against a re-extracted `.mmpose` addresses the wrong states. The only
    check left is the state count: nothing here hashes the database, so an
    embedding fitted on a *different* database of the same length is loaded and
    drawn as if it were current. Recompute the embedding after every Generate
    Pose Database.

    Returns `None` rather than raising on any problem -- a missing or unreadable
    embedding must degrade to "draw nothing", never throw inside `Py.GIL()`.

    :return: dict with embedding/edges/state_clip/speed, or None.
    """
    if not path or not os.path.isfile(path):
        log(f'[MotionField] no embedding at {path}')
        return None

    try:
        with np.load(path, allow_pickle=False) as data:
            loaded = {
                'embedding': np.asarray(data['embedding'], dtype=np.float32),
                'edges': np.asarray(data['edges'], dtype=np.int32),
                'state_clip': np.asarray(data['state_clip'], dtype=np.int32),
                'speed': np.asarray(data['speed'], dtype=np.float32),
                'states_count': int(data['states_count']),
                'n_components': int(data['n_components']),
                'n_neighbors': int(data['n_neighbors']),
                'metric_mode': str(data['metric_mode']),
                'feature_mode': str(data['feature_mode']),
            }
    except (OSError, ValueError, KeyError) as error:
        log(f'[MotionField] could not read embedding {path}: {error}')
        return None

    if states_count is not None and loaded['states_count'] != int(states_count):
        log(f"[MotionField] embedding covers {loaded['states_count']} states but the "
            f'field has {states_count}, ignoring {path}')
        return None

    log(f"[MotionField] loaded embedding {path} "
        f"({loaded['states_count']} states, {loaded['edges'].shape[0]} edges, "
        f"{loaded['n_components']}D, features={loaded['feature_mode']}, "
        f"metric={loaded['metric_mode']})")
    return loaded


def load_embedding_arrays(path: str, states_count: int = None, log=None):
    """
    `load_embedding` flattened for PythonNET, following the same convention as
    `action_predictor.get_pose_arrays`.

    :param log: optional `callable(str)`. Unity passes one in so rejection
        reasons reach the Editor console -- the default `print` goes to stdout,
        which PythonNET does not forward, making a stale embedding look like a
        silent no-op.
    :return: (embedding_xyz, edges_pairs, speed, states_count) as flat Python
        lists, or ([], [], [], 0) when there is no usable embedding.
    """
    loaded = load_embedding(path, states_count, log=log or print)
    if loaded is None:
        return [], [], [], 0

    embedding = loaded['embedding']
    if embedding.shape[1] < 3:                    # pad a 2D embedding to xyz
        embedding = np.pad(embedding, ((0, 0), (0, 3 - embedding.shape[1])))

    return (np.ascontiguousarray(embedding[:, :3], dtype=np.float32).flatten().tolist(),
            np.ascontiguousarray(loaded['edges'], dtype=np.int32).flatten().tolist(),
            np.ascontiguousarray(loaded['speed'], dtype=np.float32).tolist(),
            int(loaded['states_count']))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='motion_field_embedding')
    parser.add_argument('--data-dir', required=True,
                        help='directory containing <db-name>.mmpose')
    parser.add_argument('--db-name', default=None,
                        help='database base name (defaults to the data-dir name)')
    parser.add_argument('--out', default=None,
                        help='destination .mfembed.npz (defaults to <data-dir>/<db-name>.mfembed.npz)')
    parser.add_argument('--n-components', type=int, default=3)
    parser.add_argument('--n-neighbors', type=int, default=80)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--metric-mode', default=METRIC_FIELD,
                        choices=[METRIC_FIELD, METRIC_EUCLIDEAN])
    parser.add_argument('--feature-mode', default=FEATURE_POSITION,
                        choices=[FEATURE_POSITION, FEATURE_FULL],
                        help='what to project: joint positions alone (a pose manifold, '
                             'with the frame-adjacency edges carrying velocity) or the '
                             'whole position+velocity feature')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--knn-chunk', type=int, default=DEFAULT_KNN_CHUNK)
    parser.add_argument('--pos-weight', type=float, default=1.0)
    parser.add_argument('--vel-weight', type=float, default=0.06)
    parser.add_argument('--bone-weights', default=None,
                        help='JSON file of {"JointName": [pos, vel]} per-joint metric weights')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    data_dir = os.path.abspath(args.data_dir)
    db_name = args.db_name or os.path.basename(data_dir.rstrip('\\/'))
    out_path = args.out or os.path.join(data_dir, f'{db_name}.mfembed.npz')

    def report(stage, fraction):
        print(f'  {stage}: {fraction * 100:5.1f}%', end='\r', flush=True)

    compute_embedding(data_dir, db_name, out_path,
                      n_components=args.n_components,
                      n_neighbors=args.n_neighbors,
                      min_dist=args.min_dist,
                      metric_mode=args.metric_mode,
                      feature_mode=args.feature_mode,
                      device=args.device,
                      seed=args.seed,
                      knn_chunk=args.knn_chunk,
                      pos_weight=args.pos_weight,
                      vel_weight=args.vel_weight,
                      bone_weights=load_bone_weights_file(args.bone_weights),
                      progress=report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
