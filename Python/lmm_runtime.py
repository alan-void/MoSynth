"""
Runs a trained Learned Motion Matching model one frame at a time, for the Unity stage and for
offline inspection.

The policy here is **stateless**, as ``pfnn_runtime``'s is: it takes a whole query and returns a
whole pose, while the character's state -- which feature vector it is holding, which latent -- lives
on the C# side, where the profiler and the inspector can see it and a benchmark run can reset it
without reaching across the boundary.

**The latent table stays in Python.** Phase A's runtime mode reads a latent per tick by database
frame, and that lookup is free on this side of the boundary while marshalling a 200 000-row table
across it at startup is not. So the stage passes a frame index and gets a pose back, in the one
call it was already making. Phase B changes that -- a stepper advances a latent the stage has to
carry -- and :meth:`LmmPolicy.decompress` is the entry point it will use.

Forward kinematics is the caller's job, as it is for the PFNN: the network predicts **joint-local**
rotations, and the authority on the rest offsets that turn those into a posed character is the rig.
The stage does it with ``SkeletonData``; :func:`reconstruction_report` below does it with
:mod:`lmm_fk`, which is the same definition the loss was written in.

Everything crossing the PythonNET boundary is a flat Python list of floats, never an ndarray --
pythonnet marshals a list straight into a C# ``float[]``.

Standalone, to judge a checkpoint before Unity is involved::

    python lmm_runtime.py <checkpoint>.lmm.npz \\
        --database ../Assets/StreamingAssets/MMDatabases/MM_LafanCorrected \\
        --name MM_LafanCorrected --report
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import lmm_dataset
import lmm_fk
import lmm_io
import neural_packing
from lmm_model import Decompressor, resolve_device

# Frames scored per forward pass by the offline report. Nothing runtime-critical depends on it.
REPORT_BATCH = 4096


class LmmPolicy:
    """
    A loaded checkpoint, ready to decompress.

    :param checkpoint_path: the ``.lmm.npz`` to load.
    :param log: where to report a load failure. Unity passes its own callable.
    :param device: ``auto``, ``cpu`` or ``cuda``. A single-sample forward pass is small enough that
        CPU usually beats the per-call transfer to a GPU.
    """

    def __init__(self, checkpoint_path: str, log=print, device: str = 'cpu'):
        checkpoint = lmm_io.load_checkpoint(checkpoint_path, log=log)
        if checkpoint is None:
            raise FileNotFoundError(
                f'no usable LMM checkpoint at {checkpoint_path}. Press Train on the config.')

        self.checkpoint = checkpoint
        self.device = resolve_device(device)

        self._pose_layout = lmm_dataset.pose_vector_layout(checkpoint.n_bones)
        # Before anything is fed through it, not at the first odd-looking frame.
        neural_packing.check_blocks(checkpoint.output_blocks, self._pose_layout, what='pose')

        expected = neural_packing.vector_size(self._pose_layout)
        if checkpoint.pose_size != expected:
            raise ValueError(f'checkpoint predicts {checkpoint.pose_size} pose floats where this '
                             f'layout reads {expected}; it has to be retrained')

        self.decompressor = Decompressor(
            checkpoint.feature_size, checkpoint.latent_size, checkpoint.pose_size,
            hidden_units=checkpoint.decompressor_weights[0].shape[0],
            hidden_layers=len(checkpoint.decompressor_weights) - 1).to(self.device)
        self.decompressor.load_parameters(
            checkpoint.decompressor_weights, checkpoint.decompressor_biases)
        self.decompressor.eval()

        self._x_mean = torch.from_numpy(checkpoint.x_mean).to(self.device)
        self._x_std = torch.from_numpy(checkpoint.x_std).to(self.device)
        self._y_mean = checkpoint.y_mean
        self._y_std = checkpoint.y_std
        self._latents = checkpoint.latents

    # Description of the packing, for the stage to validate against its own database --------------

    def bone_names(self) -> list:
        """The bones this model predicts, in the order the pose blocks are packed."""
        return list(self.checkpoint.bone_names)

    def pose_blocks(self) -> list:
        """The names of the pose vector's blocks, in order."""
        return [name for name, _, _ in self._pose_layout]

    def pose_block_offsets(self) -> list:
        """``[offset, count]`` per pose block, flattened -- the slices the stage reads."""
        return [value for _, offset, count in self._pose_layout for value in (offset, count)]

    def feature_names(self) -> list:
        """The matching features the query vector was packed from, in vector order."""
        return list(self.checkpoint.feature_names)

    def feature_widths(self) -> list:
        return [int(width) for width in self.checkpoint.feature_widths]

    def feature_counts(self) -> list:
        return [int(count) for count in self.checkpoint.feature_counts]

    def feature_weights(self) -> list:
        """The authored search weights the model was fitted against, one per float of the query."""
        return [float(weight) for weight in self.checkpoint.feature_weights]

    def feature_size(self) -> int:
        return int(self.checkpoint.feature_size)

    def latent_size(self) -> int:
        return int(self.checkpoint.latent_size)

    def n_frames(self) -> int:
        """Database frames the latent table is indexed by; the staleness check at load."""
        return int(self.checkpoint.n_frames)

    def frame_time(self) -> float:
        return float(self.checkpoint.frame_time)

    def stages_trained(self) -> list:
        """Which of ``autoencoder``, ``stepper``, ``projector`` this checkpoint carries."""
        return list(self.checkpoint.stages_trained)

    def invalid_latent_frames(self) -> list:
        """
        Database frames with no latent -- the last frame of every clip, and any frame whose
        matching feature vector Unity marked invalid.

        Returned as the exceptions rather than as a per-frame mask because there are a few thousand
        of them against a few hundred thousand frames, and marshalling the mask would cost more than
        the information in it.
        """
        return [int(frame) for frame in np.flatnonzero(~self.checkpoint.latent_valid)]

    def describe(self) -> str:
        checkpoint = self.checkpoint
        return (f'{checkpoint.n_bones} bones, X {checkpoint.feature_size} + Z '
                f'{checkpoint.latent_size} -> Y {checkpoint.pose_size}, '
                f'{checkpoint.n_frames} frames, stages {"+".join(checkpoint.stages_trained)}, '
                f'final val loss {checkpoint.final_loss:.5f}')

    # The tick --------------------------------------------------------------------------------

    def decompress_frame(self, features, frame: int) -> list:
        """
        Reconstruct a pose from a query vector and the latent baked for one database frame.

        The phase A path: the stage decides which frame it is holding and this looks the latent up,
        so nothing but the query and the pose crosses the boundary per tick.

        :param features: the matching feature vector, normalised exactly as the database is -- i.e.
            what ``MotionMatchingStage.FillQueryVector`` produces.
        :param frame: a database frame index that has a latent; see :meth:`invalid_latent_frames`.
        """
        return self.decompress(features, self._latents[frame])

    def decompress(self, features, latent) -> list:
        """
        Reconstruct a pose from a query vector and an arbitrary latent.

        The general entry point, used by the ablation diagnostics now and by the phase B stepper
        later, when the latent the stage carries is no longer one of the baked rows.

        :return: ``pose_size`` floats, in the order :meth:`pose_blocks` declares.
        """
        packed = np.asarray(features, dtype=np.float32).ravel()
        if packed.size != self.checkpoint.feature_size:
            raise ValueError(f'got a {packed.size}-float query but the checkpoint expects '
                             f'{self.checkpoint.feature_size}')

        code = np.asarray(latent, dtype=np.float32).ravel()
        if code.size != self.checkpoint.latent_size:
            raise ValueError(f'got a {code.size}-float latent but the checkpoint expects '
                             f'{self.checkpoint.latent_size}')

        with torch.no_grad():
            x = (torch.from_numpy(packed).to(self.device) - self._x_mean) / self._x_std
            y = self.decompressor.decode(x.unsqueeze(0),
                                         torch.from_numpy(code).to(self.device).unsqueeze(0))
            normalised = y[0].cpu().numpy()

        return (normalised * self._y_std + self._y_mean).astype(np.float32).tolist()

    def latent(self, frame: int) -> list:
        """One baked latent, for diagnostics and for seeding the phase B stepper."""
        return self._latents[frame].astype(np.float32).tolist()

    # Batched, for the offline report ------------------------------------------------------------

    def decompress_batch(self, features: np.ndarray, latents: np.ndarray) -> np.ndarray:
        """Many poses at once, in the database's own units. Not for the tick path."""
        with torch.no_grad():
            x = (torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(self.device)
                 - self._x_mean) / self._x_std
            code = torch.from_numpy(np.ascontiguousarray(latents, dtype=np.float32)).to(self.device)
            normalised = self.decompressor.decode(x, code).cpu().numpy()
        return normalised * self._y_std + self._y_mean


def reconstruction_report(policy: LmmPolicy, training_set, frames: int = 0,
                          holdout: float = 0.0, log=print) -> dict:
    """
    Score a checkpoint against the database it was trained on, in metres rather than loss units.

    This is the honest test of phase A. The training loss is computed on weighted blocks in mixed
    units, which makes it comparable between runs and comparable to nothing else -- it cannot say
    whether a foot is a centimetre or a hand's breadth out of place.

    Joint positions come from **forward kinematics of the predicted rotations**, which is the only
    place they can come from: the decompressor predicts joint-local rotations and the root's height,
    and nothing else. That is also what the stage poses the rig with, so the number measured here is
    the number the character exhibits.

    :param frames: how many latent-carrying frames to score, evenly spread; 0 scores all of them.
    :param holdout: score only the last fraction of the training pairs -- the same contiguous tail
        the trainer validated on. 0 scores the whole database, which reports how well the model
        memorised it rather than how well it generalises.
    :return: a summary dict, also logged.
    """
    spec = lmm_dataset.build_spec(training_set, _excluded_for(policy, training_set),
                                  policy.latent_size())
    x, y, q, latent_exists = lmm_dataset.build_vectors(training_set, spec)

    if holdout:
        pairs = lmm_dataset.training_pairs(training_set)
        scored = pairs[max(1, int(round(pairs.size * (1.0 - holdout)))):]
    else:
        scored = np.flatnonzero(latent_exists).astype(np.int64)

    if scored.size == 0:
        raise ValueError('no frame of this database has a latent, so there is nothing to score')
    if frames and frames < scored.size:
        scored = scored[np.linspace(0, scored.size - 1, frames).astype(np.int64)]

    pose_layout = spec.pose_layout()
    character_layout = spec.character_layout()
    offsets = torch.from_numpy(spec.rest_offsets)
    hierarchy = lmm_fk.Hierarchy(spec.parents)

    fk_errors, height_errors, ablation = [], [], []
    speeds, true_speeds, contact_hits = [], [], []

    for start in range(0, scored.size, REPORT_BATCH):
        batch = scored[start:start + REPORT_BATCH]
        predicted = policy.decompress_batch(x[batch], policy.checkpoint.latents[batch])
        without = policy.decompress_batch(x[batch], np.zeros_like(policy.checkpoint.latents[batch]))
        truth = y[batch]

        local_six = torch.from_numpy(
            np.ascontiguousarray(neural_packing.block(predicted, pose_layout, 'rotations_6d')
                                 .reshape(-1, spec.n_bones, 6)))
        height = torch.from_numpy(
            np.ascontiguousarray(neural_packing.block(predicted, pose_layout, 'root_height')
                                 .reshape(-1)))
        with torch.no_grad():
            positions, _ = lmm_fk.forward_kinematics(local_six, height, offsets, hierarchy)

        true_positions = neural_packing.block(q[batch], character_layout, 'positions') \
            .reshape(-1, spec.n_bones, 3)
        fk_errors.append(np.linalg.norm(positions.numpy() - true_positions, axis=2))
        height_errors.append(np.abs(
            neural_packing.block(predicted, pose_layout, 'root_height')
            - neural_packing.block(truth, pose_layout, 'root_height')))
        ablation.append(np.abs(without - truth).mean() / max(1e-9, np.abs(predicted - truth).mean()))

        speeds.append(np.linalg.norm(
            neural_packing.block(predicted, pose_layout, 'root_velocity')[:, [0, 2]], axis=1))
        true_speeds.append(np.linalg.norm(
            neural_packing.block(truth, pose_layout, 'root_velocity')[:, [0, 2]], axis=1))
        contact_hits.append((neural_packing.block(predicted, pose_layout, 'contacts') > 0.5) ==
                            (neural_packing.block(truth, pose_layout, 'contacts') > 0.5))

    fk_error = np.concatenate(fk_errors)
    mean_speed = float(np.concatenate(speeds).mean())
    true_mean_speed = float(np.concatenate(true_speeds).mean())

    summary = {
        'frames_scored': int(scored.size),
        'held_out': bool(holdout),
        'mean_joint_error': float(fk_error.mean()),
        'joint_error_std': float(fk_error.std()),
        'worst_bone_error': float(fk_error.mean(axis=0).max()),
        'worst_bone': spec.bone_names[int(fk_error.mean(axis=0).argmax())],
        'root_height_error': float(np.concatenate(height_errors).mean()),
        'mean_root_speed': mean_speed,
        'database_root_speed': true_mean_speed,
        'root_speed_ratio': float(mean_speed / true_mean_speed) if true_mean_speed else float('nan'),
        'contact_agreement': float(np.concatenate(contact_hits).mean()),
        'latent_ablation_ratio': float(np.mean(ablation)),
    }

    where = 'held-out' if holdout else 'all'
    log(f"[LMM] reconstruction over {summary['frames_scored']} {where} frames: mean joint error "
        f"{summary['mean_joint_error'] * 100:.2f} cm, std {summary['joint_error_std'] * 100:.2f} cm, "
        f"worst bone {summary['worst_bone']} at {summary['worst_bone_error'] * 100:.2f} cm, "
        f"root height {summary['root_height_error'] * 100:.2f} cm")
    log(f"[LMM] root speed {summary['mean_root_speed']:.3f} m/s against the database's "
        f"{summary['database_root_speed']:.3f} ({summary['root_speed_ratio'] * 100:.1f}%), "
        f"contacts agree {summary['contact_agreement'] * 100:.1f}% of the time")
    log(f"[LMM] latent ablation: D(X||0) is {summary['latent_ablation_ratio']:.2f}x the error of "
        f"D(X||Z) -- a diagnostic of how informative X is, not a pass mark")
    return summary


def _excluded_for(policy: LmmPolicy, training_set) -> list:
    """The bones the checkpoint does not predict, as names, against this database's skeleton."""
    predicted = set(policy.bone_names())
    missing = predicted.difference(training_set.bone_names)
    if missing:
        raise ValueError('this checkpoint predicts bones the database does not have: '
                         + ', '.join(sorted(missing)))
    return [name for name in training_set.bone_names if name not in predicted]


def _main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('checkpoint', help='the .lmm.npz to load')
    parser.add_argument('--database', default=None, help='database folder, for --report')
    parser.add_argument('--name', default=None, help='database base file name, for --report')
    parser.add_argument('--report', action='store_true',
                        help='score the checkpoint against its database, in metres')
    parser.add_argument('--holdout', type=float, nargs='?', const=0.1, default=0.0,
                        help='score only the last fraction of the pairs, as the trainer validated')
    parser.add_argument('--frames', type=int, default=0,
                        help='frames to score, evenly spread; 0 scores all of them')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)

    policy = LmmPolicy(args.checkpoint, device=args.device)
    print(f'[LMM] {args.checkpoint}: {policy.describe()}')

    if not args.report:
        return
    if not args.database or not args.name:
        parser.error('--report needs --database and --name')

    from training_data import load_database
    reconstruction_report(policy, load_database(args.database, args.name), args.frames,
                          holdout=args.holdout)


if __name__ == '__main__':
    _main()
