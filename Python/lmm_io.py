"""
On-disk format for a trained Learned Motion Matching model: ``<name>.lmm.npz``.

One artefact per config, written into StreamingAssets so it ships with the player. It holds every
network's parameters, the normalisation the vectors were packed with, the baked latents, and a full
description of the packing -- the bones, the feature schema, the authored feature weights. All of
the description is there for the same reason ``pfnn_io`` stores bone names: a model fed a
differently shaped input does not throw, it just produces bad motion.

**The contents are the same at every training phase; what changes is which arrays are populated.**
``stages_trained`` records how far training got, and a runtime refuses a checkpoint that lacks what
its mode needs, naming the missing stage rather than failing at the first odd-looking frame.

``numpy.savez`` cannot append, so :func:`save_checkpoint` rewrites the file whole -- and retraining
the autoencoder therefore *drops* the stepper and the projector. That is the correct behaviour
rather than a limitation: both are functions of a latent space that has just been replaced, so
keeping them would leave a file whose parts describe different models. It is enforced here, in the
writer, rather than being left to every caller to remember.

Unversioned, per the project's standing decision: everything is regenerated when a format moves, so
a version byte would guard nothing that the content checks do not already guard better.

Numpy only -- no torch, and no ``allow_pickle``. A caller that just wants to know what a checkpoint
contains should neither have to load a deep learning framework to find out nor execute what is
inside the file.
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
    layer, in forward order -- the shape :meth:`lmm_model.Mlp.load_parameters` reads. Stepper and
    projector entries are empty until the phase that trains them.
    """

    compressor_weights: list
    compressor_biases: list
    decompressor_weights: list
    decompressor_biases: list

    # (feature_size,) standardisation of X. Unity's FeatureSet already applies the paper's own
    # recipe when it bakes the .mmfeatures, so the trainer writes an identity here rather than
    # standardising twice; the field stays because the runtime applies whatever it is given.
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

    # Phase B and C. Present and empty until those phases train them.
    stepper_weights: list = field(default_factory=list)
    stepper_biases: list = field(default_factory=list)
    xz_rate_mean: np.ndarray | None = None
    xz_rate_std: np.ndarray | None = None
    projector_weights: list = field(default_factory=list)
    projector_biases: list = field(default_factory=list)

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
                    projector_weights=None, projector_biases=None) -> None:
    """
    Write ``<name>.lmm.npz``.

    Every argument up to ``loss_columns`` describes the autoencoder and is required, because a
    checkpoint without one is not a checkpoint. The stepper and projector arguments are what a
    later phase adds; **leaving them out removes them from the file**, which is the whole point --
    see this module's docstring.

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

    if xz_rate_mean is not None and xz_rate_std is not None:
        arrays['xz_rate_mean'] = np.ascontiguousarray(xz_rate_mean, dtype=np.float32)
        arrays['xz_rate_std'] = np.ascontiguousarray(xz_rate_std, dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    np.savez(out_path, **arrays)


def load_checkpoint(path: str, log=print):
    """
    Load a ``.lmm.npz``, or return ``None``.

    Returns ``None`` rather than raising for the reason ``pfnn_io`` does: callers run inside
    ``Py.GIL()`` from Unity, where an exception arrives as an opaque managed error, and an unusable
    checkpoint should be a legible message plus a stage that declines to run.
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
                projector_weights=projector_weights,
                projector_biases=projector_biases)
    except (OSError, ValueError, KeyError) as exc:
        # KeyError is how a file written by an older layout shows up: nothing records a format
        # version, so the first missing key is the symptom.
        log(f'[LMM] could not read checkpoint {path}: {exc}. Press Train on the config to rebuild it.')
        return None
