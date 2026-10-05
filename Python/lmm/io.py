"""
On-disk format for a trained Learned Motion Matching model: ``<name>.lmm.npz``.

One artefact per config, written into StreamingAssets so it ships with the player. It holds every
network's parameters, the normalisation the vectors were packed with, the baked latents, and a full
description of the packing -- the bones, the contact bones, the feature schema, the authored feature
weights -- because a model fed a differently shaped input does not throw, it just produces bad
motion.

**Every training stage writes the same contents; what changes is which arrays are populated.**
``stages_trained`` records how far training got, and a runtime refuses a checkpoint that lacks what
its mode needs, naming the missing stage.

:func:`save_checkpoint` rewrites the file whole, so retraining the autoencoder *drops* the stepper
and the projector. That is intended: both are functions of the latent space that was just replaced.

Unversioned: everything is regenerated when a format moves, and the content checks guard better
than a version byte would.

Numpy only -- no torch, and no ``allow_pickle``, so inspecting a checkpoint neither loads a deep
learning framework nor executes what is inside the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# The training stages, in the order they run. A checkpoint carries the prefix of this list that was
# actually fitted.
STAGE_AUTOENCODER = 'autoencoder'
STAGE_STEPPER = 'stepper'
STAGE_PROJECTOR = 'projector'
STAGES = (STAGE_AUTOENCODER, STAGE_STEPPER, STAGE_PROJECTOR)


@dataclass
class LmmCheckpoint:
    """
    A loaded model.

    The ``*_weights``/``*_biases`` lists hold one ``(out, in)`` and one ``(out,)`` array per linear
    layer, in forward order -- the shape :meth:`lmm.model.Mlp.load_parameters` reads. Stepper and
    projector entries are empty until the phase that trains them.
    """

    compressor_weights: list
    compressor_biases: list
    decompressor_weights: list
    decompressor_biases: list

    # (feature_size,) standardisation of X. The .mmfeatures is already normalised, so the trainer
    # writes an identity; the runtime applies whatever it is given.
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray      # (pose_size,) standardisation of Y
    y_std: np.ndarray
    q_mean: np.ndarray      # (character_size,) standardisation of Q
    q_std: np.ndarray
    z_mean: np.ndarray      # (latent_size,) statistics of the baked latents
    z_std: np.ndarray

    latents: np.ndarray       # (n_frames, latent_size) float32, zero where no latent exists
    latent_valid: np.ndarray  # (n_frames,) bool

    feature_weights: np.ndarray  # (feature_size,) the authored search weights, expanded per float
    bone_names: list             # the bones the decompressor predicts, in order
    output_blocks: list          # the Y layout it was packed with, in order
    character_blocks: list       # the Q layout the compressor and the FK loss used
    parents: np.ndarray          # (n_bones,) parent index within the predicted set, -1 for the root
    rest_offsets: np.ndarray     # (n_bones, 3) each bone's offset from its parent, parent-local

    feature_names: list
    feature_widths: np.ndarray
    feature_counts: np.ndarray
    n_trajectory_features: int
    pose_offset: int

    latent_size: int
    frame_time: float
    n_frames: int
    stages_trained: list
    losses: np.ndarray       # (epochs, len(loss_columns)) float32
    loss_columns: list

    # The stepper and the projector. Present and empty until each is trained.
    stepper_weights: list = field(default_factory=list)
    stepper_biases: list = field(default_factory=list)
    # (feature_size + latent_size,) standardisation of the per-second rate the stepper regresses.
    xz_rate_mean: np.ndarray | None = None
    xz_rate_std: np.ndarray | None = None
    # (iterations, len(stepper_loss_columns)) float32; its own curve, since its terms differ.
    stepper_losses: np.ndarray | None = None
    stepper_loss_columns: list = field(default_factory=list)
    projector_weights: list = field(default_factory=list)
    projector_biases: list = field(default_factory=list)
    # (iterations, len(projector_loss_columns)) float32, as `stepper_losses`.
    projector_losses: np.ndarray | None = None
    projector_loss_columns: list = field(default_factory=list)

    # The bone behind each contact slot, in slot order. None for a checkpoint written before the
    # names were stored, which carries the legacy two slots and cannot be checked by name.
    contact_bone_names: list | None = None

    @property
    def feature_size(self) -> int:
        return int(self.x_mean.size)

    @property
    def pose_size(self) -> int:
        return int(self.y_mean.size)

    @property
    def character_size(self) -> int:
        return int(self.q_mean.size)

    @property
    def n_bones(self) -> int:
        return len(self.bone_names)

    @property
    def final_loss(self) -> float:
        """Last epoch's validation loss, which is the last column."""
        return float(self.losses[-1, -1]) if self.losses.size else float('nan')

    def has_stage(self, stage: str) -> bool:
        return stage in self.stages_trained


def checkpoint_arguments(checkpoint: LmmCheckpoint) -> dict:
    """
    Every :func:`save_checkpoint` argument, read back off a loaded checkpoint.

    A later training stage updates this dict with whatever it fitted and splats it into
    :func:`save_checkpoint`, so no field is silently lost when the file is rewritten whole.
    """
    return {
        'compressor_weights': checkpoint.compressor_weights,
        'compressor_biases': checkpoint.compressor_biases,
        'decompressor_weights': checkpoint.decompressor_weights,
        'decompressor_biases': checkpoint.decompressor_biases,
        'x_mean': checkpoint.x_mean, 'x_std': checkpoint.x_std,
        'y_mean': checkpoint.y_mean, 'y_std': checkpoint.y_std,
        'q_mean': checkpoint.q_mean, 'q_std': checkpoint.q_std,
        'z_mean': checkpoint.z_mean, 'z_std': checkpoint.z_std,
        'latents': checkpoint.latents,
        'latent_valid': checkpoint.latent_valid,
        'feature_weights': checkpoint.feature_weights,
        'bone_names': checkpoint.bone_names,
        'output_blocks': checkpoint.output_blocks,
        'character_blocks': checkpoint.character_blocks,
        'parents': checkpoint.parents,
        'rest_offsets': checkpoint.rest_offsets,
        'feature_names': checkpoint.feature_names,
        'feature_widths': checkpoint.feature_widths,
        'feature_counts': checkpoint.feature_counts,
        'n_trajectory_features': checkpoint.n_trajectory_features,
        'pose_offset': checkpoint.pose_offset,
        'latent_size': checkpoint.latent_size,
        'frame_time': checkpoint.frame_time,
        'n_frames': checkpoint.n_frames,
        'losses': checkpoint.losses,
        'loss_columns': checkpoint.loss_columns,
        'stepper_weights': checkpoint.stepper_weights,
        'stepper_biases': checkpoint.stepper_biases,
        'xz_rate_mean': checkpoint.xz_rate_mean,
        'xz_rate_std': checkpoint.xz_rate_std,
        'stepper_losses': checkpoint.stepper_losses,
        'stepper_loss_columns': checkpoint.stepper_loss_columns,
        'projector_weights': checkpoint.projector_weights,
        'projector_biases': checkpoint.projector_biases,
        'projector_losses': checkpoint.projector_losses,
        'projector_loss_columns': checkpoint.projector_loss_columns,
        'contact_bone_names': checkpoint.contact_bone_names,
    }


def _add_loss_table(arrays: dict, prefix: str, rows, columns) -> None:
    """
    One fit's loss curve, under its own columns, or nothing at all when it did not run.

    A checkpoint whose stepper or projector was never fitted simply has no such key.
    """
    if not columns:
        return

    columns = list(columns)
    # Not `rows or []`: a curve passed through from a loaded checkpoint is an ndarray, whose truth
    # value raises.
    rows = [] if rows is None else list(rows)
    arrays[f'{prefix}_losses'] = (
        np.ascontiguousarray(rows, dtype=np.float32).reshape(len(rows), -1) if len(rows)
        else np.zeros((0, len(columns)), dtype=np.float32))
    arrays[f'{prefix}_loss_columns'] = np.array(columns, dtype=np.str_)


def _layer_arrays(prefix: str, weights, biases) -> dict:
    arrays = {f'{prefix}_layer_count': np.int64(len(weights))}
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        arrays[f'{prefix}_layer{i}_weights'] = np.ascontiguousarray(weight, dtype=np.float32)
        arrays[f'{prefix}_layer{i}_biases'] = np.ascontiguousarray(bias, dtype=np.float32)
    return arrays


def _read_layers(f, prefix: str):
    count = int(f[f'{prefix}_layer_count'])
    weights = [np.ascontiguousarray(f[f'{prefix}_layer{i}_weights'], dtype=np.float32)
               for i in range(count)]
    biases = [np.ascontiguousarray(f[f'{prefix}_layer{i}_biases'], dtype=np.float32)
              for i in range(count)]
    return weights, biases


def save_checkpoint(out_path: str, *,
                    compressor_weights, compressor_biases,
                    decompressor_weights, decompressor_biases,
                    x_mean, x_std, y_mean, y_std, q_mean, q_std, z_mean, z_std,
                    latents, latent_valid, feature_weights,
                    bone_names, output_blocks, character_blocks, parents, rest_offsets,
                    feature_names, feature_widths, feature_counts,
                    n_trajectory_features: int, pose_offset: int,
                    latent_size: int, frame_time: float, n_frames: int,
                    losses, loss_columns,
                    stepper_weights=None, stepper_biases=None,
                    xz_rate_mean=None, xz_rate_std=None,
                    stepper_losses=None, stepper_loss_columns=(),
                    projector_weights=None, projector_biases=None,
                    projector_losses=None, projector_loss_columns=(),
                    contact_bone_names=None) -> None:
    """
    Write ``<name>.lmm.npz``.

    Every argument up to ``loss_columns`` describes the autoencoder and is required. The stepper
    and projector arguments are what later stages add; **leaving them out removes them from the
    file** -- see this module's docstring. ``contact_bone_names`` left as None writes no names,
    which reads back as a legacy checkpoint.

    ``stages_trained`` is derived from which of them arrived rather than passed in, so it cannot
    claim a stage the file does not carry.
    """
    stages = [STAGE_AUTOENCODER]
    if stepper_weights:
        stages.append(STAGE_STEPPER)
    if projector_weights:
        if STAGE_STEPPER not in stages:
            raise ValueError('a projector cannot be saved without the stepper it was trained '
                             'alongside: the runtime mode that uses one needs both')
        stages.append(STAGE_PROJECTOR)

    arrays = {
        'x_mean': np.ascontiguousarray(x_mean, dtype=np.float32),
        'x_std': np.ascontiguousarray(x_std, dtype=np.float32),
        'y_mean': np.ascontiguousarray(y_mean, dtype=np.float32),
        'y_std': np.ascontiguousarray(y_std, dtype=np.float32),
        'q_mean': np.ascontiguousarray(q_mean, dtype=np.float32),
        'q_std': np.ascontiguousarray(q_std, dtype=np.float32),
        'z_mean': np.ascontiguousarray(z_mean, dtype=np.float32),
        'z_std': np.ascontiguousarray(z_std, dtype=np.float32),
        'latents': np.ascontiguousarray(latents, dtype=np.float32),
        'latent_valid': np.ascontiguousarray(latent_valid, dtype=bool),
        'feature_weights': np.ascontiguousarray(feature_weights, dtype=np.float32),
        # Fixed-width unicode arrays rather than object arrays, so the file loads without
        # allow_pickle -- reading a checkpoint should never mean executing what is inside it.
        'bone_names': np.array(list(bone_names), dtype=np.str_),
        'output_blocks': np.array(list(output_blocks), dtype=np.str_),
        'character_blocks': np.array(list(character_blocks), dtype=np.str_),
        'parents': np.ascontiguousarray(parents, dtype=np.int32),
        'rest_offsets': np.ascontiguousarray(rest_offsets, dtype=np.float32),
        'feature_names': np.array(list(feature_names), dtype=np.str_),
        'feature_widths': np.ascontiguousarray(feature_widths, dtype=np.int32),
        'feature_counts': np.ascontiguousarray(feature_counts, dtype=np.int32),
        'n_trajectory_features': np.int64(n_trajectory_features),
        'pose_offset': np.int64(pose_offset),
        'latent_size': np.int64(latent_size),
        'frame_time': np.float32(frame_time),
        'n_frames': np.int64(n_frames),
        'stages_trained': np.array(stages, dtype=np.str_),
        'losses': np.ascontiguousarray(losses, dtype=np.float32).reshape(len(losses), -1)
        if len(losses) else np.zeros((0, len(loss_columns)), dtype=np.float32),
        'loss_columns': np.array(list(loss_columns), dtype=np.str_),
    }

    arrays.update(_layer_arrays('compressor', compressor_weights, compressor_biases))
    arrays.update(_layer_arrays('decompressor', decompressor_weights, decompressor_biases))
    arrays.update(_layer_arrays('stepper', stepper_weights or [], stepper_biases or []))
    arrays.update(_layer_arrays('projector', projector_weights or [], projector_biases or []))

    if contact_bone_names is not None:
        arrays['contact_bone_names'] = np.array(list(contact_bone_names), dtype=np.str_)

    if xz_rate_mean is not None and xz_rate_std is not None:
        arrays['xz_rate_mean'] = np.ascontiguousarray(xz_rate_mean, dtype=np.float32)
        arrays['xz_rate_std'] = np.ascontiguousarray(xz_rate_std, dtype=np.float32)

    _add_loss_table(arrays, 'stepper', stepper_losses, stepper_loss_columns)
    _add_loss_table(arrays, 'projector', projector_losses, projector_loss_columns)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    np.savez(out_path, **arrays)


def load_checkpoint(path: str, log=print):
    """
    Load a ``.lmm.npz``, or return ``None``.

    Returns ``None`` rather than raising: callers run inside ``Py.GIL()`` from Unity, where an
    exception arrives as an opaque managed error.
    """
    if not path or not os.path.isfile(path):
        return None

    try:
        with np.load(path, allow_pickle=False) as f:
            compressor_weights, compressor_biases = _read_layers(f, 'compressor')
            decompressor_weights, decompressor_biases = _read_layers(f, 'decompressor')
            stepper_weights, stepper_biases = _read_layers(f, 'stepper')
            projector_weights, projector_biases = _read_layers(f, 'projector')

            return LmmCheckpoint(
                compressor_weights=compressor_weights,
                compressor_biases=compressor_biases,
                decompressor_weights=decompressor_weights,
                decompressor_biases=decompressor_biases,
                x_mean=np.ascontiguousarray(f['x_mean'], dtype=np.float32),
                x_std=np.ascontiguousarray(f['x_std'], dtype=np.float32),
                y_mean=np.ascontiguousarray(f['y_mean'], dtype=np.float32),
                y_std=np.ascontiguousarray(f['y_std'], dtype=np.float32),
                q_mean=np.ascontiguousarray(f['q_mean'], dtype=np.float32),
                q_std=np.ascontiguousarray(f['q_std'], dtype=np.float32),
                z_mean=np.ascontiguousarray(f['z_mean'], dtype=np.float32),
                z_std=np.ascontiguousarray(f['z_std'], dtype=np.float32),
                latents=np.ascontiguousarray(f['latents'], dtype=np.float32),
                latent_valid=np.ascontiguousarray(f['latent_valid'], dtype=bool),
                feature_weights=np.ascontiguousarray(f['feature_weights'], dtype=np.float32),
                bone_names=[str(name) for name in f['bone_names']],
                output_blocks=[str(name) for name in f['output_blocks']],
                character_blocks=[str(name) for name in f['character_blocks']],
                parents=np.ascontiguousarray(f['parents'], dtype=np.int32),
                rest_offsets=np.ascontiguousarray(f['rest_offsets'], dtype=np.float32),
                feature_names=[str(name) for name in f['feature_names']],
                feature_widths=np.ascontiguousarray(f['feature_widths'], dtype=np.int32),
                feature_counts=np.ascontiguousarray(f['feature_counts'], dtype=np.int32),
                n_trajectory_features=int(f['n_trajectory_features']),
                pose_offset=int(f['pose_offset']),
                latent_size=int(f['latent_size']),
                frame_time=float(f['frame_time']),
                n_frames=int(f['n_frames']),
                stages_trained=[str(name) for name in f['stages_trained']],
                losses=np.ascontiguousarray(f['losses'], dtype=np.float32),
                loss_columns=[str(name) for name in f['loss_columns']],
                stepper_weights=stepper_weights,
                stepper_biases=stepper_biases,
                xz_rate_mean=(np.ascontiguousarray(f['xz_rate_mean'], dtype=np.float32)
                              if 'xz_rate_mean' in f else None),
                xz_rate_std=(np.ascontiguousarray(f['xz_rate_std'], dtype=np.float32)
                             if 'xz_rate_std' in f else None),
                # Absent from a checkpoint with no stepper fitted, as `stages_trained` says.
                stepper_losses=(np.ascontiguousarray(f['stepper_losses'], dtype=np.float32)
                                if 'stepper_losses' in f else None),
                stepper_loss_columns=([str(name) for name in f['stepper_loss_columns']]
                                      if 'stepper_loss_columns' in f else []),
                projector_weights=projector_weights,
                projector_biases=projector_biases,
                projector_losses=(np.ascontiguousarray(f['projector_losses'], dtype=np.float32)
                                  if 'projector_losses' in f else None),
                projector_loss_columns=([str(name) for name in f['projector_loss_columns']]
                                        if 'projector_loss_columns' in f else []),
                contact_bone_names=([str(name) for name in f['contact_bone_names']]
                                    if 'contact_bone_names' in f else None))
    except (OSError, ValueError, KeyError) as exc:
        # KeyError: a file written by an older layout, which has no version to say so.
        log(f'[LMM] could not read checkpoint {path}: {exc}. Press Train on the config to rebuild it.')
        return None
