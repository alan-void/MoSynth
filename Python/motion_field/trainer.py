"""
Fitted value iteration for the motion field (Lee et al. 2010, section 6).

The controller needs to know more than which action looks best right now --
turning a walk cycle takes several steps of setup, so a one-step-greedy policy
oscillates instead of committing. Training fits a value function V(state, goal
heading) over the whole database, after which the runtime can act on
`Q = R + gamma * V(s')` and anticipate.

Pipeline:

  1. Precompute, for every database state and each of its K candidate actions,
     the yaw that action produces and the neighbourhood of the state it lands
     in. This is the expensive part, and it is rebuilt on every train.
  2. Iterate the Bellman backup over that fixed transition structure until the
     residual stops moving.
  3. Write the value function to StreamingAssets.

Both the task parameter (the goal heading, quantised onto a grid of
`theta_count` values) and the motion state are interpolated, so although values
are only stored at discrete database states the result behaves as a continuous
function over the field.

Runs from the Unity Editor through PythonNET, or standalone:

    python -m motion_field.trainer --data-dir ../Assets/StreamingAssets/MotionFields/MotionFieldData
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from motion_field import io as mfio
from motion_field.field import (MotionField, load_bone_weights_file, resolve_device,
                         root_yaw, state_locomotion_scores)
from motion_field.action_predictor import load_animations

# States per precompute chunk. The dominant allocation is the neighbour gather,
# states_per_chunk * K * K * (bones+2) * 4 floats -- about 46 MB at 512 states
# with K=15. Gathering all 7838 states at once would need ~700 MB.
DEFAULT_STATE_CHUNK = 512


def _noop_progress(stage: str, fraction: float) -> None:
    pass


def build_tables(motion_field: MotionField, k_neighbors: int, tug_ratio: float,
                 state_chunk: int = DEFAULT_STATE_CHUNK, knn_chunk: int = 64,
                 progress=_noop_progress):
    """
    Precompute the transition structure the Bellman backup iterates over.

    For every database state s and every candidate action a:
      * `delta_yaw[s, a]` -- the yaw the action turns the character through.
      * `value_indices[s, a, :]` / `value_weights[s, a, :]` -- the k nearest
        database states to wherever the action lands, and their similarity
        weights, so V(s') can be read off as a weighted sum.

    Fully batched. Integrating one action at a time the way the reference
    implementation does costs roughly 5 minutes here; this takes about 15 s.

    :return: (delta_yaw (S,K) f32, value_indices (S,K,K) i32, value_weights (S,K,K) f32)
    """
    states_count = motion_field.states_count
    k = k_neighbors
    frame_time = motion_field.frame_time

    progress('Finding neighbourhoods', 0.0)
    indices, weights = motion_field.get_batched_knn(
        motion_field.state_features, k, chunk=knn_chunk)

    # One action per neighbour: pin that neighbour's weight to 1, re-normalise.
    diagonal = np.arange(k)
    actions = np.repeat(weights[:, np.newaxis, :], k, axis=1)  # (S, K, K)
    actions[:, diagonal, diagonal] = 1.0
    actions /= np.sum(actions, axis=2, keepdims=True)

    delta_yaw = np.zeros((states_count, k), dtype=np.float32)
    value_indices = np.zeros((states_count, k, k), dtype=np.int32)
    value_weights = np.zeros((states_count, k, k), dtype=np.float32)

    for start in range(0, states_count, state_chunk):
        stop = min(start + state_chunk, states_count)
        span = stop - start
        count = span * k  # actions in this chunk

        # Flatten (state, action) into one action axis so a chunk is a single
        # batched integrate rather than span*k separate ones.
        chunk_indices = np.repeat(indices[start:stop], k, axis=0)          # (N, K)
        chunk_actions = actions[start:stop].reshape(count, k)              # (N, K)
        chunk_starts = np.repeat(motion_field.poses_x[start:stop], k, axis=0)
        # Each action is tugged toward the neighbour it emphasises, which is
        # what keeps the K outcomes distinguishable from one another.
        chunk_tug = chunk_indices[np.arange(count), np.tile(np.arange(k), span)]

        xs, vs = motion_field.compute_new_states(
            chunk_starts, frame_time, chunk_indices, chunk_actions,
            tug_indices=chunk_tug, tug_ratio=tug_ratio)

        delta_yaw[start:stop] = root_yaw(xs).reshape(span, k)

        successor_indices, successor_weights = motion_field.get_batched_knn(
            motion_field.build_motion_states(xs, vs), k, chunk=knn_chunk)
        value_indices[start:stop] = successor_indices.reshape(span, k, k)
        value_weights[start:stop] = successor_weights.reshape(span, k, k)

        progress('Precomputing transitions', stop / states_count)

    return delta_yaw, value_indices, value_weights

def theta_grid(theta_count: int) -> np.ndarray:
    """The task-parameter grid: `theta_count` headings spanning [-pi, pi)."""
    return np.linspace(-np.pi, np.pi, theta_count + 1, dtype=np.float32)[:theta_count]

def train_value_function(delta_yaw, value_indices, value_weights,
                         state_scores=None, locomotion_factor=1.0,
                         theta_count=17, epochs=300, gamma=0.99,
                         device=None, residual_tolerance=1e-5,
                         progress=_noop_progress):
    """
    Fitted value iteration.

        bonus[s,a]    = factor * sum_j w[s,a,j] * state_scores[s'[s,a,j]]
        reward[s,a,t] = -|wrap(theta[t] + delta_yaw[s,a])| + bonus[s,a]
        Q[s,a,t]      = reward + gamma * sum_j w[s,a,j] * V(s'[s,a,j], theta[t] + delta_yaw[s,a])
        V[s,t]        = max over a of Q[s,a,t]

    The heading term is how badly the character faces away from the goal after
    the action. On its own it is never positive, which makes any idle state the
    reward can reach a perfect score once aligned -- the bonus term (the
    reference implementation's `state_score * factor`, keyed on root speed and
    paid on the arrival state) is what makes continuing to move strictly better
    than freezing. Because the action rotates the character, the goal moves in
    its frame -- hence the successor value is read at `theta + delta_yaw`, not
    at `theta`, and that shifted lookup is interpolated across the two
    bracketing grid headings.

    :param delta_yaw: (states, actions) f32 -- yaw each action turns the
        character through, from `build_tables`.
    :param value_indices: (states, actions, neighbours) i32 -- the k database
        states nearest each action's arrival state.
    :param value_weights: (states, actions, neighbours) f32 -- similarity
        weights over `value_indices`; each (state, action) row sums to 1.
    :param state_scores: (states,) per-state locomotion score in [0, 1], or
        None to train on the heading term alone.
    :param locomotion_factor: scale on the bonus term; 0 disables it even when
        `state_scores` is given.
    :param theta_count: number of goal headings on the task grid, spanning
        [-pi, pi). Must match what the runtime policy will use.
    :param epochs: maximum Bellman backup iterations.
    :param gamma: discount factor; also bounds V at 1 / (1 - gamma) per unit
        of per-step reward.
    :param device: torch device name, or None to pick via `resolve_device`.
    :param residual_tolerance: stop early once the mean Bellman residual for
        an epoch falls below this.
    :param progress: optional callable(stage_name, fraction_0_to_1).
    :return: (V (states, theta_count) f32, scores (epochs_run, 3) f32) where
        scores columns are the min / max / mean Bellman residual per epoch.
    """
    device = resolve_device(device)
    states_count, k = delta_yaw.shape

    values = torch.zeros((states_count, theta_count), device=device)
    delta = torch.from_numpy(delta_yaw).to(device, torch.float32)
    neighbour_ids = torch.from_numpy(value_indices).to(device, torch.long)
    neighbour_weights = torch.from_numpy(value_weights).to(device, torch.float32)
    thetas = torch.from_numpy(theta_grid(theta_count)).to(device)

    # Goal heading after the action, per (state, action, starting heading).
    next_theta = thetas.view(1, 1, theta_count) + delta.view(states_count, k, 1)
    next_theta = (next_theta + torch.pi) % (2.0 * torch.pi) - torch.pi
    rewards = -torch.abs(next_theta)

    if state_scores is not None and locomotion_factor != 0.0:
        # Interpolated through the same successor neighbourhood as V, and
        # heading-independent, so it broadcasts over the theta axis.
        scores = torch.from_numpy(state_scores).to(device, torch.float32)
        bonus = (scores[neighbour_ids] * neighbour_weights).sum(dim=2)
        rewards = rewards + locomotion_factor * bonus.unsqueeze(-1)

    # Fractional position of next_theta on the heading grid, split into the two
    # bracketing grid indices and a blend factor. Fixed for all epochs.
    spacing = (2.0 * torch.pi) / theta_count
    coordinate = (next_theta + torch.pi) / spacing
    lower = torch.floor(coordinate)
    alpha = (coordinate - lower).unsqueeze(2)
    lower_id = (lower.to(torch.long) % theta_count).unsqueeze(2).expand(-1, -1, k, -1)
    upper_id = ((lower.to(torch.long) + 1) % theta_count).unsqueeze(2).expand(-1, -1, k, -1)

    scores = np.zeros((epochs, 3), dtype=np.float32)
    epochs_run = 0

    for epoch in range(epochs):
        neighbour_values = values[neighbour_ids]                     # (S, A, J, T)
        at_lower = torch.gather(neighbour_values, 3, lower_id)
        at_upper = torch.gather(neighbour_values, 3, upper_id)

        # Interpolate over the task grid first, then over motion space.
        by_task = (1.0 - alpha) * at_lower + alpha * at_upper
        expected_next = (by_task * neighbour_weights.unsqueeze(-1)).sum(dim=2)

        q_values = rewards + gamma * expected_next
        updated = q_values.max(dim=1).values

        residual = torch.abs(values - updated)
        scores[epoch] = (residual.min().item(), residual.max().item(), residual.mean().item())
        values = updated
        epochs_run = epoch + 1

        progress('Value iteration', epochs_run / epochs)
        if scores[epoch, 2] < residual_tolerance:
            break

    return values.cpu().numpy(), scores[:epochs_run]


def train(data_dir: str,
          db_name: str,
          out_path: str,
          k_neighbors: int = 15,
          theta_count: int = 17,
          epochs: int = 300,
          gamma: float = 0.99,
          tug_ratio: float = 0.1,
          pos_weight: float = 1.0,
          vel_weight: float = 0.06,
          bone_weights=None,
          locomotion_factor: float = 1.0,
          locomotion_speed_threshold: float = 0.5,
          device: str = 'auto',
          knn_chunk: int = 64,
          state_chunk: int = DEFAULT_STATE_CHUNK,
          residual_tolerance: float = 1e-5,
          progress=None) -> dict:
    """
    Train a motion field value function and write it to `out_path`.

    :param data_dir: directory holding `<db_name>.mmpose`
    :param out_path: destination `.mffield.npz`, normally under StreamingAssets
    :param bone_weights: optional per-joint emphasis on the similarity metric,
        see `MotionField.resolve_bone_weights`. Changing it invalidates any
        previously trained value function.
    :param locomotion_factor: reward bonus for actions landing in moving
        states; see `MotionField.state_locomotion_scores`.
    :param locomotion_speed_threshold: root speed, m/s, at which a state
        counts as fully moving.
    :param progress: optional callable(stage_name, fraction_0_to_1)
    :return: a summary dict (also useful as the value Unity logs)
    """
    progress = progress or _noop_progress
    started = time.perf_counter()

    progress('Loading pose database', 0.0)
    skeleton, poses_x, poses_v, poses_y, _, frame_time, _ = load_animations(data_dir, db_name)

    motion_field = MotionField(poses_x, poses_v, poses_y, skeleton, frame_time,
                               device=device, pos_weight=pos_weight, vel_weight=vel_weight,
                               k_neighbors=k_neighbors, tug_ratio=tug_ratio,
                               knn_chunk=knn_chunk, bone_weights=bone_weights,
                               locomotion_factor=locomotion_factor,
                               locomotion_speed_threshold=locomotion_speed_threshold)

    delta_yaw, value_indices, value_weights = build_tables(
        motion_field, k_neighbors, tug_ratio,
        state_chunk=state_chunk, knn_chunk=knn_chunk, progress=progress)

    values, scores = train_value_function(
        delta_yaw, value_indices, value_weights,
        state_scores=motion_field.state_scores,
        locomotion_factor=locomotion_factor,
        theta_count=theta_count, epochs=epochs, gamma=gamma,
        device=motion_field.device, residual_tolerance=residual_tolerance,
        progress=progress)

    progress('Writing value function', 1.0)
    mfio.save_value_function(out_path, values, scores,
                             theta_count=int(theta_count), gamma=float(gamma),
                             k_neighbors=int(k_neighbors))

    summary = {
        'out_path': out_path,
        'states_count': int(motion_field.states_count),
        'theta_count': int(theta_count),
        'epochs_run': int(scores.shape[0]),
        'final_residual': float(scores[-1, 2]) if scores.size else float('nan'),
        'device': motion_field.device,
        'seconds': round(time.perf_counter() - started, 2),
    }
    print(f'[MotionField] trained {summary["states_count"]} states x '
          f'{summary["theta_count"]} headings in {summary["seconds"]}s on '
          f'{summary["device"]} ({summary["epochs_run"]} epochs, '
          f'final residual {summary["final_residual"]:.6f}) -> {out_path}')
    return summary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python -m motion_field.trainer')
    parser.add_argument('--data-dir', required=True,
                        help='directory containing <db-name>.mmpose')
    parser.add_argument('--db-name', default=None,
                        help='database base name (defaults to the data-dir name)')
    parser.add_argument('--out', default=None,
                        help='destination .mffield.npz (defaults to <data-dir>/<db-name>.mffield.npz)')
    parser.add_argument('--k-neighbors', type=int, default=15)
    parser.add_argument('--theta-count', type=int, default=17)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tug-ratio', type=float, default=0.1)
    parser.add_argument('--pos-weight', type=float, default=1.0)
    parser.add_argument('--vel-weight', type=float, default=0.06)
    parser.add_argument('--locomotion-factor', type=float, default=1.0)
    parser.add_argument('--locomotion-speed-threshold', type=float, default=0.5)
    parser.add_argument('--bone-weights', default=None,
                        help='JSON file of {"JointName": [pos, vel]} per-joint metric weights')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--knn-chunk', type=int, default=64)
    parser.add_argument('--state-chunk', type=int, default=DEFAULT_STATE_CHUNK)
    return parser.parse_args(argv)


# def main(argv=None) -> int:
#     args = _parse_args(argv)
#     data_dir = os.path.abspath(args.data_dir)
#     db_name = args.db_name or os.path.basename(data_dir.rstrip('\\/'))
#     out_path = args.out or os.path.join(data_dir, f'{db_name}.mffield.npz')
#
#     def report(stage, fraction):
#         print(f'  {stage}: {fraction * 100:5.1f}%', end='\r', flush=True)
#
#     train(data_dir, db_name, out_path,
#           k_neighbors=args.k_neighbors, theta_count=args.theta_count,
#           epochs=args.epochs, gamma=args.gamma, tug_ratio=args.tug_ratio,
#           pos_weight=args.pos_weight, vel_weight=args.vel_weight,
#           bone_weights=load_bone_weights_file(args.bone_weights),
#           locomotion_factor=args.locomotion_factor,
#           locomotion_speed_threshold=args.locomotion_speed_threshold,
#           device=args.device, knn_chunk=args.knn_chunk,
#           state_chunk=args.state_chunk, progress=report)
#     return 0
#
#
# if __name__ == '__main__':
#     raise SystemExit(main())
