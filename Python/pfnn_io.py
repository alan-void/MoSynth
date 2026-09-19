"""
On-disk format for a trained PFNN: ``<name>.pfnn.npz``.

Training produces one artefact, written into StreamingAssets so it ships with the player. It holds
the network's parameters, the normalisation the vectors were packed with, and the description of
that packing -- the bone names and the trajectory horizons -- because a model fed a differently
shaped input does not throw, it just produces bad motion. The output block names go in for the
same reason from the other side: blocks are appended, so an older file's are all still readable and
only the names say the layout has moved on.

Bone **names** are stored, never bone indices. An index is only meaningful against one database, so
a stored one can go stale silently the moment a rig gains a joint; a name is checked against the
live skeleton at load, which is the same check ``.mmpose`` does with its skeleton block. Whether the
checkpoint still matches the config that will run it is otherwise Unity's job, exactly as it is for
the motion field: ``PfnnConfig.hasTrained`` is cleared the moment anything the packing depends on is
edited, so the answer shows up in the inspector rather than in play mode.

Unversioned, per the project's standing decision: everything is regenerated when a format moves, so
a version byte would guard nothing that the content checks do not already guard better.

Numpy only -- no torch. A caller that just wants to know what a checkpoint contains should not have
to load a deep learning framework to find out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Layers are stored under indexed keys because their shapes differ, so they cannot be one stacked
# array. The count is fixed by the architecture, not a parameter.
LAYER_COUNT = 3


@dataclass
class PfnnCheckpoint:
    """A loaded PFNN."""

    weights: list          # LAYER_COUNT arrays, each (control_points, out, in) float32
    biases: list           # LAYER_COUNT arrays, each (control_points, out) float32
    x_mean: np.ndarray     # (input_size,) float32
    x_std: np.ndarray      # (input_size,) float32
    y_mean: np.ndarray     # (output_size,) float32
    y_std: np.ndarray      # (output_size,) float32
    bone_names: list       # the bones this model predicts, in order
    output_blocks: list    # the output layout it was packed with, in order
    window_offsets: np.ndarray  # (n_offsets,) int64 frame offsets of the trajectory window
    frame_time: float
    hidden_units: int
    dropout: float
    losses: np.ndarray     # (epochs, 2) float32 -- train and validation loss per epoch

    @property
    def input_size(self) -> int:
        return int(self.x_mean.size)

    @property
    def output_size(self) -> int:
        return int(self.y_mean.size)

    @property
    def n_bones(self) -> int:
        return len(self.bone_names)

    @property
    def final_loss(self) -> float:
        return float(self.losses[-1, 1]) if self.losses.size else float('nan')


def save_checkpoint(out_path: str, weights, biases, x_mean, x_std, y_mean, y_std,
                    bone_names, output_blocks, window_offsets, frame_time: float,
                    hidden_units: int, dropout: float, losses) -> None:
    """Write ``<name>.pfnn.npz``."""
    if len(weights) != LAYER_COUNT or len(biases) != LAYER_COUNT:
        raise ValueError(f'expected {LAYER_COUNT} layers, got {len(weights)}/{len(biases)}')

    arrays = {
        'x_mean': np.ascontiguousarray(x_mean, dtype=np.float32),
        'x_std': np.ascontiguousarray(x_std, dtype=np.float32),
        'y_mean': np.ascontiguousarray(y_mean, dtype=np.float32),
        'y_std': np.ascontiguousarray(y_std, dtype=np.float32),
        # A fixed-width unicode array rather than an object array, so the file loads without
        # allow_pickle -- reading a checkpoint should never mean executing what is inside it.
        'bone_names': np.array(list(bone_names), dtype=np.str_),
        # Output blocks are appended over time, and an older file's earlier blocks all still slice
        # out correctly -- so a width alone cannot tell a stale checkpoint from a current one.
        'output_blocks': np.array(list(output_blocks), dtype=np.str_),
        'window_offsets': np.ascontiguousarray(window_offsets, dtype=np.int64),
        'frame_time': np.float32(frame_time),
        'hidden_units': np.int64(hidden_units),
        'dropout': np.float32(dropout),
        'losses': np.ascontiguousarray(losses, dtype=np.float32).reshape(-1, 2),
    }

    for i in range(LAYER_COUNT):
        arrays[f'layer{i}_weights'] = np.ascontiguousarray(weights[i], dtype=np.float32)
        arrays[f'layer{i}_biases'] = np.ascontiguousarray(biases[i], dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    np.savez(out_path, **arrays)


def load_checkpoint(path: str, log=print):
    """
    Load a ``.pfnn.npz``, or return ``None``.

    Returns ``None`` rather than raising for the reason ``motion_field_io`` does: callers run inside
    ``Py.GIL()`` from Unity, where an exception arrives as an opaque managed error, and an unusable
    checkpoint should be a legible message plus a stage that declines to run.
    """
    if not path or not os.path.isfile(path):
        return None

    try:
        with np.load(path, allow_pickle=False) as f:
            return PfnnCheckpoint(
                weights=[np.ascontiguousarray(f[f'layer{i}_weights'], dtype=np.float32)
                         for i in range(LAYER_COUNT)],
                biases=[np.ascontiguousarray(f[f'layer{i}_biases'], dtype=np.float32)
                        for i in range(LAYER_COUNT)],
                x_mean=np.ascontiguousarray(f['x_mean'], dtype=np.float32),
                x_std=np.ascontiguousarray(f['x_std'], dtype=np.float32),
                y_mean=np.ascontiguousarray(f['y_mean'], dtype=np.float32),
                y_std=np.ascontiguousarray(f['y_std'], dtype=np.float32),
                bone_names=[str(name) for name in f['bone_names']],
                output_blocks=[str(name) for name in f['output_blocks']],
                window_offsets=np.ascontiguousarray(f['window_offsets'], dtype=np.int64),
                frame_time=float(f['frame_time']),
                hidden_units=int(f['hidden_units']),
                dropout=float(f['dropout']),
                losses=np.ascontiguousarray(f['losses'], dtype=np.float32))
    except (OSError, ValueError, KeyError) as exc:
        # KeyError is how a file written by an older layout shows up: nothing records a format
        # version, so the first missing key is the symptom.
        log(f'[PFNN] could not read checkpoint {path}: {exc}. Press Train on the config to rebuild it.')
        return None
