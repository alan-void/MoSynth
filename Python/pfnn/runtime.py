"""
Runs a trained PFNN one frame at a time, for the Unity stage and for offline inspection.

The policy here is **stateless**. It takes a whole input and returns a whole output; the phase, the
previous pose and the trajectory history live on the C# side. That is the opposite of
``motion_field.field``, and deliberately: a motion field's state is an index into a database, which is
cheap to keep in Python, whereas a PFNN's state is a pose that Unity has to write into a
``PoseBuffer`` anyway. Keeping it in C# means a benchmark run can restart the character without
reaching across the boundary, and the state is visible to the inspector and the profiler.

Forward kinematics is likewise the caller's job. The network predicts rotations; turning those into
joint positions needs the skeleton's rest offsets, and the authority on those is the rig -- so the
stage does it with ``SkeletonData``, and :func:`rollout` below does it with the pose set, rather
than a third copy living in here.

Everything crossing the PythonNET boundary is a flat Python list of floats, which pythonnet
marshals straight into a C# ``float[]``; see the same note in ``motion_field.action_predictor.get_pose_arrays``.

Standalone, to judge a checkpoint before Unity is involved::

    python -m pfnn.runtime <checkpoint>.pfnn.npz \\
        --database ../Assets/StreamingAssets/MMDatabases/MotionMatchingData \\
        --name MotionMatchingData --rollout 300
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from pfnn import dataset
from pfnn import io as pfnn_io
from pfnn.model import PhaseFunctionedNetwork, resolve_device
from training.training_data import rotations_from_6d


class PfnnPolicy:
    """
    A loaded checkpoint, ready to step.

    :param checkpoint_path: the ``.pfnn.npz`` to load.
    :param log: where to report a load failure. Unity passes its own callable.
    :param device: ``auto``, ``cpu`` or ``cuda``. A single-sample forward pass is small enough that
        CPU is usually the faster choice once the per-call transfer is counted, so ``auto`` is not
        obviously right for inference -- but it keeps one story for both directions.
    """

    def __init__(self, checkpoint_path: str, log=print, device: str = 'cpu'):
        checkpoint = pfnn_io.load_checkpoint(checkpoint_path, log=log)
        if checkpoint is None:
            raise FileNotFoundError(
                f'no usable PFNN checkpoint at {checkpoint_path}. Press Train on the config.')

        self.checkpoint = checkpoint
        self.device = resolve_device(device)

        self.network = PhaseFunctionedNetwork(
            checkpoint.input_size, checkpoint.output_size,
            checkpoint.hidden_units, checkpoint.dropout).to(self.device)

        with torch.no_grad():
            for layer, weights, biases in zip(self.network.layers,
                                              checkpoint.weights, checkpoint.biases):
                layer.weights.copy_(torch.from_numpy(weights))
                layer.biases.copy_(torch.from_numpy(biases))
        self.network.eval()

        self._x_mean = torch.from_numpy(checkpoint.x_mean).to(self.device)
        self._x_std = torch.from_numpy(checkpoint.x_std).to(self.device)
        self._y_mean = checkpoint.y_mean
        self._y_std = checkpoint.y_std

        self.spec = dataset.PfnnSpec(
            window_offsets=checkpoint.window_offsets,
            bone_indices=np.arange(checkpoint.n_bones, dtype=np.int64),
            bone_names=tuple(checkpoint.bone_names),
            frame_time=checkpoint.frame_time)
        self._output_layout = self.spec.output_layout()

        # Before anything is fed through it, not at the first odd-looking frame.
        dataset.check_output_blocks(checkpoint.output_blocks, self._output_layout)
        if checkpoint.output_size != self.spec.output_size:
            raise ValueError(f'checkpoint predicts {checkpoint.output_size} floats where this '
                             f'layout reads {self.spec.output_size}; it has to be retrained')

    # Description of the packing, for the stage to validate against its own skeleton -------------

    def bone_names(self) -> list:
        """The bones this model predicts, in the order its output blocks are packed."""
        return list(self.checkpoint.bone_names)

    def window_offsets(self) -> list:
        """Frame offsets of the trajectory window the model expects, negative for the past."""
        return [int(offset) for offset in self.checkpoint.window_offsets]

    def frame_time(self) -> float:
        """Seconds per frame of the database this was trained on."""
        return float(self.checkpoint.frame_time)

    def describe(self) -> str:
        checkpoint = self.checkpoint
        return (f'{checkpoint.n_bones} bones, {checkpoint.input_size} -> {checkpoint.output_size}, '
                f'{checkpoint.hidden_units} hidden, window {self.window_offsets()}, '
                f'final val loss {checkpoint.final_loss:.5f}')

    # The step ------------------------------------------------------------------------------------

    def step(self, trajectory_positions, trajectory_directions,
             joint_positions, joint_velocities, contacts, phase: float):
        """
        Predict the next frame.

        Every array is flat and in the order :meth:`PfnnSpec.input_layout` declares: the trajectory
        pair is ``(x, z)`` per sample over :meth:`window_offsets`, and the joint pair is ``(x, y, z)``
        per bone over :meth:`bone_names`, all in the character frame of the frame being queried.

        :return: ``(rotations_6d, joint_velocities, root_height, dx, dz, dyaw, phase_delta,
            left_contact, right_contact, future_positions, future_directions)``. The rotations are
            6 floats per bone in the same bone order; ``dx``/``dz``/``dyaw`` are the step into the
            next character frame, expressed in the current one; ``phase_delta`` is in radians. The
            two future arrays are ``(x, z)`` per positive window offset, in the *next* character
            frame -- the trajectory the caller should hand back as the future half of its next
            input.
        """
        packed = np.concatenate([
            np.asarray(trajectory_positions, dtype=np.float32).ravel(),
            np.asarray(trajectory_directions, dtype=np.float32).ravel(),
            np.asarray(joint_positions, dtype=np.float32).ravel(),
            np.asarray(joint_velocities, dtype=np.float32).ravel(),
            np.asarray(contacts, dtype=np.float32).ravel(),
        ])
        if packed.size != self.checkpoint.input_size:
            raise ValueError(f'built a {packed.size}-float input but the checkpoint expects '
                             f'{self.checkpoint.input_size}')

        y = self._forward(packed, float(phase))
        return self._unpack(y)

    def _forward(self, packed: np.ndarray, phase: float) -> np.ndarray:
        with torch.no_grad():
            x = (torch.from_numpy(packed).to(self.device) - self._x_mean) / self._x_std
            phase_tensor = torch.tensor([phase], dtype=torch.float32, device=self.device)
            normalised = self.network(x.unsqueeze(0), phase_tensor)[0].cpu().numpy()
        return normalised * self._y_std + self._y_mean

    def _unpack(self, y: np.ndarray):
        layout = self._output_layout
        rotations = dataset.block(y, layout, 'joint_rotations_6d')
        velocities = dataset.block(y, layout, 'joint_velocities')
        root_delta = dataset.block(y, layout, 'root_delta')
        contacts = dataset.block(y, layout, 'contacts')
        future_positions = dataset.block(y, layout, 'future_positions')
        future_directions = dataset.block(y, layout, 'future_directions')

        return (
            rotations.astype(np.float32).tolist(),
            velocities.astype(np.float32).tolist(),
            float(dataset.block(y, layout, 'root_height')[0]),
            float(root_delta[0]), float(root_delta[1]), float(root_delta[2]),
            float(dataset.block(y, layout, 'phase_delta')[0]),
            bool(contacts[0] > 0.5), bool(contacts[1] > 0.5),
            future_positions.astype(np.float32).tolist(),
            future_directions.astype(np.float32).tolist(),
        )


def rollout(policy: PfnnPolicy, training_set, frames: int = 300, start_frame: int = 0,
            log=print) -> dict:
    """
    Run the model autoregressively against its own predictions, driven by the database's own path.

    This is the cheapest honest test of a checkpoint: a model that has learned nothing still scores
    a plausible per-frame loss, because the pose barely changes in a thirtieth of a second, but it
    cannot survive being fed its own output for ten seconds. Feeding the *true* future trajectory
    isolates the question -- if the character still falls apart, the controller is not the problem.

    :return: a summary dict with the distance travelled, the mean speed to compare against the
        database's own, the largest joint excursion seen, and how far the predicted future
        trajectory fell from the one the character really walked.
    """
    spec = dataset.build_spec(training_set, _excluded_for(policy, training_set),
                                   window_radius=int(abs(policy.checkpoint.window_offsets).max()),
                                   window_stride=int(_stride_of(policy.checkpoint.window_offsets)))
    bones = spec.bone_indices
    parents_in_subset = _parents_within(training_set.parents, bones)
    rest_offsets = _rest_offsets(training_set, bones, parents_in_subset)

    window_positions, window_directions = training_set.trajectory_window(spec.window_offsets)

    positions = training_set.positions[start_frame][bones].astype(np.float32)
    velocities = training_set.velocities[start_frame][bones].astype(np.float32)
    contacts = training_set.contacts[start_frame].astype(np.float32)
    phase = float(training_set.phase[start_frame])

    travelled, speeds, extents = 0.0, [], []

    # The predicted future is scored against the window frame i+1 really carries, which is the
    # quantity it was trained on -- so only where that frame exists in the same clip.
    future_mask = spec.future_mask
    scorable = dataset.usable_queries(training_set)
    position_errors, heading_errors = [], []

    for step_index in range(frames):
        frame = min(start_frame + step_index, training_set.n_frames - 1)
        (six, next_velocities, root_height, dx, dz, dyaw, phase_delta, left, right,
         future_positions, future_directions) = policy.step(
            window_positions[frame], window_directions[frame],
            positions, velocities, contacts, phase)

        if scorable[frame]:
            predicted = np.asarray(future_positions, dtype=np.float32).reshape(-1, 2)
            heading = np.asarray(future_directions, dtype=np.float32).reshape(-1, 2)
            truth_positions = window_positions[frame + 1][future_mask]
            truth_headings = window_directions[frame + 1][future_mask]

            position_errors.append(
                float(np.linalg.norm(predicted - truth_positions, axis=1).mean()))
            cosine = (heading * truth_headings).sum(axis=1) / np.maximum(
                np.linalg.norm(heading, axis=1), 1e-6)
            heading_errors.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).mean()))

        rotations = rotations_from_6d(np.asarray(six, dtype=np.float32).reshape(-1, 6))
        positions = _forward_kinematics(rotations, rest_offsets, parents_in_subset, root_height)
        velocities = np.asarray(next_velocities, dtype=np.float32).reshape(-1, 3)
        contacts = np.array([float(left), float(right)], dtype=np.float32)
        phase = (phase + phase_delta) % (2.0 * np.pi)

        step_length = float(np.hypot(dx, dz))
        travelled += step_length
        speeds.append(step_length / training_set.frame_time)
        extents.append(float(np.abs(positions).max()))

    summary = {
        'frames': frames,
        'travelled': travelled,
        'mean_speed': float(np.mean(speeds)) if speeds else float('nan'),
        'database_mean_speed': float(np.linalg.norm(
            training_set.root_velocity[:, [0, 2]], axis=1).mean()),
        'max_joint_extent': max(extents) if extents else float('nan'),
        'final_phase': phase,
        'trajectory_position_error': (float(np.mean(position_errors)) if position_errors
                                      else float('nan')),
        'trajectory_heading_error': (float(np.mean(heading_errors)) if heading_errors
                                     else float('nan')),
    }

    log(f"[PFNN] rollout {frames} frames: travelled {summary['travelled']:.2f} m at "
        f"{summary['mean_speed']:.3f} m/s (database {summary['database_mean_speed']:.3f}), "
        f"largest joint extent {summary['max_joint_extent']:.2f} m")
    log(f"[PFNN] predicted future trajectory: {summary['trajectory_position_error']:.3f} m and "
        f"{summary['trajectory_heading_error']:.1f} deg from the path actually walked")
    return summary


def _excluded_for(policy: PfnnPolicy, training_set) -> list:
    """The bones the checkpoint does not predict, as names, against this database's skeleton."""
    predicted = set(policy.bone_names())
    missing = predicted.difference(training_set.bone_names)
    if missing:
        raise ValueError('this checkpoint predicts bones the database does not have: '
                         + ', '.join(sorted(missing)))
    return [name for name in training_set.bone_names if name not in predicted]


def _stride_of(offsets: np.ndarray) -> int:
    spacing = np.diff(np.asarray(offsets))
    return int(spacing[0]) if spacing.size else 1


def _parents_within(parents: np.ndarray, bones: np.ndarray) -> np.ndarray:
    """Parent index of each selected bone, expressed within the selection. -1 for its root."""
    position = {int(bone): i for i, bone in enumerate(bones)}
    return np.array([position.get(int(parents[bone]), -1) for bone in bones], dtype=np.int64)


def _rest_offsets(training_set, bones: np.ndarray, parents_in_subset: np.ndarray) -> np.ndarray:
    """
    Each selected bone's offset from its parent, in the parent's own space.

    Recovered from the first frame's character-space pose rather than read from the skeleton,
    because a ``TrainingSet`` does not carry rest transforms -- and it is constant, which the
    caller can check by taking a different frame and getting the same answer.
    """
    positions = training_set.positions[0][bones].astype(np.float64)
    rotations = training_set.rotations[0][bones].astype(np.float64)

    offsets = np.zeros((len(bones), 3))
    for i, parent in enumerate(parents_in_subset):
        if parent < 0:
            continue
        delta = positions[i] - positions[parent]
        offsets[i] = _rotate(_conjugate(rotations[parent]), delta)
    return offsets


def _forward_kinematics(rotations: np.ndarray, rest_offsets: np.ndarray,
                        parents_in_subset: np.ndarray, root_height: float) -> np.ndarray:
    """
    Joint positions in the character frame, from character-frame rotations and rest offsets.

    The root sits on the frame's own vertical axis by construction -- its x and z are what the frame
    transform removed -- so only its height is predicted.
    """
    positions = np.zeros((rotations.shape[0], 3), dtype=np.float64)
    positions[0] = (0.0, root_height, 0.0)

    for i in range(1, rotations.shape[0]):
        parent = parents_in_subset[i]
        positions[i] = positions[parent] + _rotate(rotations[parent], rest_offsets[i])
    return positions.astype(np.float32)


def _conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])


def _rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    axis = quaternion[:3]
    w = quaternion[3]
    return (vector + 2.0 * np.cross(axis, np.cross(axis, vector) + w * vector))


def _main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('checkpoint', help='the .pfnn.npz to load')
    parser.add_argument('--database', default=None, help='database folder, for --rollout')
    parser.add_argument('--name', default=None, help='database base file name, for --rollout')
    parser.add_argument('--rollout', type=int, default=0, help='frames to run autoregressively')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--no-features', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)

    policy = PfnnPolicy(args.checkpoint, device=args.device)
    print(f'[PFNN] {args.checkpoint}: {policy.describe()}')

    if args.rollout <= 0:
        return
    if not args.database or not args.name:
        parser.error('--rollout needs --database and --name')

    from training.training_data import load_database
    training_set = load_database(args.database, args.name, with_features=not args.no_features)
    rollout(policy, training_set, args.rollout, args.start_frame)


if __name__ == '__main__':
    _main()
