"""
Runs a trained style network one tick at a time, for the Unity stage and for offline inspection.

The policy is **stateless**: the causal phase, the descriptor history and the crossfade live on the
C# side, and :meth:`StylePolicy.step` returns the deltas without composing them. The stage writes
``R_out = R_in * P^(w_p * g) * D^(w_d * g)`` itself, because the intensities, region weights and
gate smoothing are per-character state.

Everything crossing the PythonNET boundary is a flat Python list of floats, which pythonnet
marshals straight into a C# ``float[]``; see the same note in ``pfnn.runtime``.

Standalone, to judge a checkpoint before Unity is involved::

    python -m style.runtime <checkpoint>.style.npz \\
        --database ../Assets/StreamingAssets/StyleDatabases/Style100 --name Style100 --style Old
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from core.simulation_frame import canonical_quaternions, frame_yaw, yaw_quaternion
from lmm.fk import forward_kinematics_reference, local_rotations, rest_offsets
from pfnn.model import resolve_device
from style import dataset
from style import io as style_io
from style.model import StyleNetwork
from style.phase import OnlineGaitPhase
from training.training_data import rotations_from_6d, rotations_to_6d

# The gate is 1 up to the threshold and falls smoothly to 0 at this multiple of it.
GATE_FALLOFF = 2.0


def gate_from_score(score: np.ndarray, threshold: float) -> np.ndarray:
    """Novelty score to gate in [0, 1]: a smoothstep from ``threshold`` down to zero."""
    span = (GATE_FALLOFF - 1.0) * threshold
    t = np.clip((np.asarray(score, dtype=np.float64) - threshold) / span, 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def scale_rotation(quaternions: np.ndarray, weight) -> np.ndarray:
    """``slerp(I, q, w)``: each rotation's angle scaled by its weight, xyzw."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    weight = np.broadcast_to(np.asarray(weight, dtype=np.float64), quaternions.shape[:-1])
    rotvec = Rotation.from_quat(quaternions.reshape(-1, 4)).as_rotvec()
    return Rotation.from_rotvec(rotvec * weight.reshape(-1, 1)).as_quat().reshape(
        quaternions.shape)


def compose(rotations: np.ndarray, posture: np.ndarray, dynamics: np.ndarray,
            posture_weight=1.0, dynamics_weight=1.0) -> np.ndarray:
    """
    ``R_in * slerp(I, P, w_p) * slerp(I, D, w_d)`` per bone -- the composition the stage writes.

    :param rotations: (..., 4) parent-local input rotations, xyzw.
    :param posture: (..., 4) posture deltas. :param dynamics: (..., 4) dynamics deltas.
    :param posture_weight: broadcastable to ``rotations.shape[:-1]``, gate already folded in.
    :return: (..., 4) xyzw, w >= 0.
    """
    shape = np.asarray(rotations).shape
    result = (Rotation.from_quat(np.asarray(rotations, dtype=np.float64).reshape(-1, 4))
              * Rotation.from_quat(scale_rotation(posture, posture_weight).reshape(-1, 4))
              * Rotation.from_quat(scale_rotation(dynamics, dynamics_weight).reshape(-1, 4)))
    return canonical_quaternions(result.as_quat()).reshape(shape)


class StylePolicy:
    """
    A loaded checkpoint, ready to step.

    :param checkpoint_path: the ``.style.npz`` to load.
    :param log: where to report a load failure. Unity passes its own callable.
    :param device: ``auto``, ``cpu`` or ``cuda``. CPU by default: one tiny forward pass is faster
        there once the transfer is counted.
    :raises FileNotFoundError: the checkpoint is missing or unreadable.
    """

    def __init__(self, checkpoint_path: str, log=print, device: str = 'cpu'):
        checkpoint = style_io.load_checkpoint(checkpoint_path, log=log)
        if checkpoint is None:
            raise FileNotFoundError(
                f'no usable style checkpoint at {checkpoint_path}. Press Train on the config.')
        self.checkpoint = checkpoint
        self.device = resolve_device(device)

        bone_names = list(checkpoint.bone_names)
        missing = [name for name in checkpoint.styled_bone_names + checkpoint.contact_bone_names
                   if name not in bone_names]
        if missing:
            raise ValueError(f'the checkpoint names bones its own skeleton lacks: {missing}')

        self.spec = dataset.StyleSpec(
            bone_names=tuple(bone_names),
            parents=np.asarray(checkpoint.parents, dtype=np.int64),
            styled_bones=np.array([bone_names.index(name)
                                   for name in checkpoint.styled_bone_names], dtype=np.int64),
            contact_bones=np.array([bone_names.index(name)
                                    for name in checkpoint.contact_bone_names], dtype=np.int64),
            contact_bone_names=tuple(checkpoint.contact_bone_names),
            frame_time=checkpoint.frame_time,
            history_offsets=tuple(float(offset) for offset in checkpoint.history_offsets))
        if (self.spec.input_size, self.spec.output_size) != \
                (checkpoint.input_size, checkpoint.output_size):
            raise ValueError(f'the checkpoint was packed as {checkpoint.input_size} -> '
                             f'{checkpoint.output_size} but its description packs '
                             f'{self.spec.input_size} -> {self.spec.output_size}; retrain it')

        self.contact_bone_names = list(checkpoint.contact_bone_names)

        self.network = StyleNetwork(checkpoint.input_size, checkpoint.output_size,
                                    len(checkpoint.style_names), checkpoint.hidden_units,
                                    checkpoint.layers, checkpoint.embed_dim).to(self.device)
        self.network.load_state_dict({name: torch.from_numpy(value)
                                      for name, value in checkpoint.parameters.items()})
        self.network.eval()

        self._x_mean = checkpoint.x_mean
        self._x_std = checkpoint.x_std
        self._gate_tree = cKDTree(checkpoint.gate_reference)
        self._gate_k = min(int(checkpoint.gate_k), int(checkpoint.gate_reference.shape[0]))

    # Description of the packing, for the stage to validate against its own skeleton -------------

    def bone_names(self) -> list:
        """Every skeleton bone the checkpoint was trained on, in skeleton order."""
        return list(self.spec.bone_names)

    def styled_bone_names(self) -> list:
        """The styled subset, in skeleton order; bone 0 is always first."""
        return list(self.checkpoint.styled_bone_names)

    def style_names(self) -> list:
        """Style tag names; the index is the style index :meth:`step` takes."""
        return list(self.checkpoint.style_names)

    def frame_time(self) -> float:
        return float(self.checkpoint.frame_time)

    def history_offsets(self) -> list:
        """Seconds into the past of each history descriptor, ascending."""
        return [float(offset) for offset in self.spec.history_offsets]

    def descriptor_size(self) -> int:
        return int(self.spec.descriptor_size)

    def describe(self) -> str:
        checkpoint = self.checkpoint
        return (f'{len(checkpoint.style_names)} styles, {self.spec.n_styled}/'
                f'{len(self.spec.bone_names)} bones styled, {checkpoint.input_size} -> '
                f'{checkpoint.output_size}, {checkpoint.layers}x{checkpoint.hidden_units} hidden, '
                f'gaits {checkpoint.gait_names}, history {self.history_offsets()} s, '
                f'final val loss {checkpoint.final_loss:.5f}')

    # The step ------------------------------------------------------------------------------------

    def step(self, rotations, positions, contacts, root_velocity, root_yaw_rate: float,
             phase: float, phase_rate: float, history, style_indices):
        """
        The style deltas for one tick, for one style or two while crossfading.

        :param rotations: B x 4 flat, xyzw, every bone in the character frame.
        :param positions: B x 3 flat, character frame.
        :param contacts: C floats, contact slot order. Not read by the network: timing stays the
            method's, and contact heights are in the descriptor.
        :param root_velocity: 3 floats, the character frame's velocity in its own space, m/s.
        :param history: H x D flat, the descriptors earlier steps returned, history-offset order.
        :param style_indices: 1 or 2 ints, previous then current while crossfading.
        :return: ``(posture_rotations[S*Bs*4], dynamics_rotations[S*Bs*4], posture_height[S],
            dynamics_height[S], speed_ratio[S], root_residual[S*4], gate, descriptor[D])``.
        """
        spec = self.spec
        n_bones = len(spec.bone_names)
        rotations = np.asarray(rotations, dtype=np.float32).reshape(1, n_bones, 4)
        positions = np.asarray(positions, dtype=np.float32).reshape(1, n_bones, 3)
        root_velocity = np.asarray(root_velocity, dtype=np.float32).reshape(1, 3)
        history = np.asarray(history, dtype=np.float32).reshape(1, spec.n_history,
                                                                 spec.descriptor_size)
        styles = np.asarray(style_indices, dtype=np.int64).ravel()
        if not 1 <= styles.size <= 2 or styles.min() < 0 \
                or styles.max() >= len(self.checkpoint.style_names):
            raise ValueError(f'style indices {styles.tolist()} are not 1 or 2 of '
                             f'{len(self.checkpoint.style_names)} styles')

        descriptor = dataset.descriptor(positions, root_velocity, [float(root_yaw_rate)],
                                        spec.contact_bones)
        local_6d = rotations_to_6d(dataset.styled_local_rotations(rotations, spec))
        x = dataset.input_vectors(local_6d, positions[:, 0, 1], descriptor, history,
                                  [float(phase)], [float(phase_rate)])
        weights = dataset.gait_weights(dataset.gait_features(descriptor, history),
                                       self.checkpoint.gait_centroids,
                                       self.checkpoint.gait_sigma)

        n_styles = styles.size
        weights = np.repeat(weights, n_styles, axis=0)
        posture = canonical_quaternions(
            dataset.blend_posture(self.checkpoint.posture_rotations[styles], weights))
        posture_height = (self.checkpoint.posture_heights[styles] * weights).sum(axis=1)
        speed_ratio = (self.checkpoint.speed_ratios[styles] * weights).sum(axis=1)

        output = self._forward(np.repeat(x, n_styles, axis=0), styles)
        n_rotation = 6 * spec.n_styled
        dynamics = rotations_from_6d(
            dataset.IDENTITY_6D + output[:, :n_rotation].reshape(n_styles, spec.n_styled, 6))

        gate = float(self._gate(descriptor, phase_rate)[0])

        return (posture.astype(np.float32).ravel().tolist(),
                dynamics.astype(np.float32).ravel().tolist(),
                posture_height.astype(np.float32).tolist(),
                output[:, n_rotation].astype(np.float32).tolist(),
                speed_ratio.astype(np.float32).tolist(),
                output[:, n_rotation + 1:].astype(np.float32).ravel().tolist(),
                gate,
                descriptor[0].astype(np.float32).tolist())

    def _forward(self, x: np.ndarray, styles: np.ndarray) -> np.ndarray:
        normalised = ((x - self._x_mean) / self._x_std).astype(np.float32)
        with torch.no_grad():
            out = self.network(torch.from_numpy(normalised).to(self.device),
                               torch.from_numpy(styles).to(self.device)).cpu().numpy()
        return out * self.checkpoint.y_std + self.checkpoint.y_mean

    def _gate(self, descriptors: np.ndarray, phase_rate) -> np.ndarray:
        vectors = dataset.gate_vectors(descriptors, np.atleast_1d(phase_rate))
        normalised = (vectors - self.checkpoint.gate_mean) / self.checkpoint.gate_std
        distance, _ = self._gate_tree.query(normalised, k=self._gate_k)
        score = np.asarray(distance, dtype=np.float64).reshape(normalised.shape[0], -1).mean(1)
        return gate_from_score(score, self.checkpoint.gate_threshold)

    def style_index(self, name: str) -> int:
        """The index of a style, matched by tag leaf ignoring case."""
        keys = [dataset.style_key(style) for style in self.checkpoint.style_names]
        if dataset.style_key(name) not in keys:
            raise ValueError(f'no style {name!r} in this checkpoint; it has '
                             f'{", ".join(self.checkpoint.style_names)}')
        return keys.index(dataset.style_key(name))


# Offline check ---------------------------------------------------------------------------------

def check(policy: StylePolicy, training_set, style: str, forward_axis, frames: int = 600,
          log=print) -> dict:
    """
    Stylise a database's frames as the stage would, and report what the style did.

    Frames are stepped clip by clip from the start of the database, each clip with a fresh causal
    phase and history, exactly as a character would feed the stage. The deltas are composed at
    full intensity with the gate applied, bone 0's yaw change removed as the stage removes it.

    :param forward_axis: bone 0's local forward axis, ``PoseSet.sim_frame_forward``.
    :return: delta magnitudes in degrees, gate activity, and contact-bone slide in m/s before
        and after -- after once with the method's root (``Preserve``) and once restyled
        (``Stylize``).
    """
    if list(training_set.bone_names) != policy.bone_names():
        raise ValueError('the database skeleton differs from the one the checkpoint was trained on')

    spec = policy.spec
    s = policy.style_index(style)
    tracker = OnlineGaitPhase()
    speed = dataset.planar_speed(training_set.root_velocity)
    history_frames = spec.history_frames

    chosen, posture, dynamics, heights, ratios, residuals, gates = [], [], [], [], [], [], []
    for start, end in training_set.clip_ranges:
        if len(chosen) >= frames:
            break
        tracker.reset()
        descriptors = []
        for frame in range(start, min(end, start + frames - len(chosen))):
            contact = training_set.contacts[frame] > 0.5
            phase, rate = tracker.update(training_set.frame_time, contact[0], contact[1],
                                         speed[frame])
            local = frame - start
            history = [descriptors[max(0, local - back)] if descriptors else None
                       for back in history_frames]
            result = policy.step(
                training_set.rotations[frame].ravel(), training_set.positions[frame].ravel(),
                training_set.contacts[frame], training_set.root_velocity[frame],
                float(training_set.root_yaw_rate[frame]), phase, rate,
                np.concatenate([entry if entry is not None else _first_descriptor(
                    policy, training_set, frame) for entry in history]), [s])
            descriptors.append(np.asarray(result[7], dtype=np.float32))

            chosen.append(frame)
            posture.append(result[0])
            dynamics.append(result[1])
            heights.append(result[2][0] + result[3][0])
            ratios.append(result[4][0])
            residuals.append(result[5])
            gates.append(result[6])

    chosen = np.asarray(chosen, dtype=np.int64)
    n_styled = spec.n_styled
    posture = np.asarray(posture, dtype=np.float64).reshape(-1, n_styled, 4)
    dynamics = np.asarray(dynamics, dtype=np.float64).reshape(-1, n_styled, 4)
    gates = np.asarray(gates, dtype=np.float64)
    ratios = np.asarray(ratios, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1, 4)

    local_in = local_rotations(training_set.rotations[chosen], spec.parents)
    local_out = np.array(local_in, dtype=np.float64)
    local_out[:, spec.styled_bones] = compose(local_in[:, spec.styled_bones], posture, dynamics,
                                              gates[:, np.newaxis], gates[:, np.newaxis])

    # The character frame is derived from bone 0, so its delta may not turn it.
    turn = frame_yaw(Rotation.from_quat(local_out[:, 0]), forward_axis) \
        - frame_yaw(Rotation.from_quat(local_in[:, 0]), forward_axis)
    local_out[:, 0] = (Rotation.from_quat(yaw_quaternion(-turn))
                       * Rotation.from_quat(local_out[:, 0])).as_quat()

    offsets = rest_offsets(training_set.positions[0], training_set.rotations[0], spec.parents)
    height = training_set.positions[chosen, 0, 1] + gates * np.asarray(heights)
    styled_positions, _ = forward_kinematics_reference(rotations_to_6d(local_out), height,
                                                       offsets, spec.parents)

    velocity = np.asarray(training_set.root_velocity[chosen], dtype=np.float64)
    restyled_velocity = (1.0 + (ratios - 1.0) * gates)[:, np.newaxis] * velocity \
        + gates[:, np.newaxis] * residuals[:, :3]
    contacts = training_set.contacts[chosen] > 0.5

    def slide(positions, root_velocity):
        feet = positions[:, spec.contact_bones]
        consecutive = np.flatnonzero(np.diff(chosen) == 1)
        planar = (feet[consecutive + 1] - feet[consecutive]) / training_set.frame_time \
            + root_velocity[consecutive, np.newaxis]
        speed_of = np.linalg.norm(planar[..., [0, 2]], axis=-1)
        weight = contacts[consecutive]
        return float((speed_of * weight).sum() / max(1, weight.sum()))

    def degrees(quaternions):
        return float(np.degrees(Rotation.from_quat(quaternions.reshape(-1, 4)).magnitude()).mean())

    summary = {
        'style': policy.style_names()[s],
        'frames': int(chosen.size),
        'posture_degrees': degrees(posture),
        'dynamics_degrees': degrees(dynamics),
        'gate_mean': float(gates.mean()) if gates.size else float('nan'),
        'gate_closed_fraction': float((gates < 0.5).mean()) if gates.size else float('nan'),
        'speed_ratio_mean': float(ratios.mean()) if ratios.size else float('nan'),
        'slide_before': slide(training_set.positions[chosen].astype(np.float64), velocity),
        'slide_after_preserve': slide(styled_positions, velocity),
        'slide_after_stylize': slide(styled_positions, restyled_velocity),
    }

    log(f"[Style] {summary['style']} over {summary['frames']} frames: posture "
        f"{summary['posture_degrees']:.2f} deg, dynamics {summary['dynamics_degrees']:.2f} deg "
        f"mean per bone; gate {summary['gate_mean']:.2f} mean, closed on "
        f"{100.0 * summary['gate_closed_fraction']:.0f}% of frames; speed x"
        f"{summary['speed_ratio_mean']:.2f}")
    log(f"[Style] contact slide {summary['slide_before']:.3f} m/s before, "
        f"{summary['slide_after_preserve']:.3f} after (root preserved), "
        f"{summary['slide_after_stylize']:.3f} after (root stylised)")
    return summary


def _first_descriptor(policy: StylePolicy, training_set, frame: int) -> np.ndarray:
    """A clip's first descriptor, before any step has returned one: the oldest available."""
    return dataset.descriptor(training_set.positions[frame:frame + 1],
                              training_set.root_velocity[frame:frame + 1],
                              training_set.root_yaw_rate[frame:frame + 1],
                              policy.spec.contact_bones)[0]


def _main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('checkpoint', help='the .style.npz to load')
    parser.add_argument('--database', default=None, help='database folder to stylise')
    parser.add_argument('--name', default=None, help='database base file name')
    parser.add_argument('--style', default=None, help='the style to apply, e.g. Old')
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)

    policy = StylePolicy(args.checkpoint, device=args.device)
    print(f'[Style] {args.checkpoint}: {policy.describe()}')
    if not args.database:
        return
    if not args.name or not args.style:
        parser.error('--database needs --name and --style')

    from formats.pose_set_importer import deserialize_pose_set
    from training.training_data import build_training_set
    pose_set = deserialize_pose_set(args.database, args.name)
    check(policy, build_training_set(pose_set), args.style, pose_set.sim_frame_forward,
          args.frames)


if __name__ == '__main__':
    _main()
