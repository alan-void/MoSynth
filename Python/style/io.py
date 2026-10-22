"""
On-disk format for a trained style network: ``<name>.style.npz``.

It holds the network's parameters and everything the packing and the closed-form terms depend on:
style and gait names, gait centroids, the posture and speed-ratio tables, normalisation, bone and
contact-bone names, the history offsets, and the gate's reference set. A model fed a differently
packed input does not throw, it just stylises badly, so the packing travels with the weights.

Bone names, never indices, so a checkpoint is checked against the live skeleton at load.
Unversioned, as every format here is: regenerate when it moves.

Numpy only -- no torch -- so a caller that only wants to read a checkpoint need not load one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

_PARAMETER_PREFIX = 'param.'


@dataclass
class StyleCheckpoint:
    """A loaded style network: S styles, G gaits, B bones, Bs styled bones, D descriptor."""

    parameters: dict            # torch state-dict name -> float32 array
    hidden_units: int
    layers: int
    embed_dim: int
    style_names: list           # S, index = style index; [0] is the neutral style
    gait_names: list            # G
    gait_centroids: np.ndarray  # (G, 2) float32
    gait_sigma: float
    posture_rotations: np.ndarray  # (S, G, Bs, 4) float32, xyzw
    posture_heights: np.ndarray    # (S, G) float32
    speed_ratios: np.ndarray       # (S, G) float32
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    bone_names: list            # B, skeleton order
    parents: np.ndarray         # (B,) int64
    styled_bone_names: list     # Bs, skeleton order, bone 0 first
    contact_bone_names: list    # C, slot order
    frame_time: float
    history_offsets: np.ndarray    # (H,) float64 seconds, ascending
    gate_reference: np.ndarray     # (m, D + 1) float32, normalised
    gate_mean: np.ndarray          # (D + 1,)
    gate_std: np.ndarray           # (D + 1,)
    gate_threshold: float
    gate_k: int
    losses: np.ndarray             # (epochs, 2) float32 train and validation loss

    @property
    def input_size(self) -> int:
        return int(self.x_mean.size)

    @property
    def output_size(self) -> int:
        return int(self.y_mean.size)

    @property
    def final_loss(self) -> float:
        return float(self.losses[-1, 1]) if self.losses.size else float('nan')


_ARRAYS = {
    'gait_centroids': np.float32, 'posture_rotations': np.float32,
    'posture_heights': np.float32, 'speed_ratios': np.float32,
    'x_mean': np.float32, 'x_std': np.float32, 'y_mean': np.float32, 'y_std': np.float32,
    'parents': np.int64, 'history_offsets': np.float64, 'gate_reference': np.float32,
    'gate_mean': np.float32, 'gate_std': np.float32, 'losses': np.float32,
}
_NAMES = ('style_names', 'gait_names', 'bone_names', 'styled_bone_names', 'contact_bone_names')
_INTS = ('hidden_units', 'layers', 'embed_dim', 'gate_k')
_FLOATS = ('gait_sigma', 'frame_time', 'gate_threshold')


def save_checkpoint(out_path: str, checkpoint: StyleCheckpoint) -> None:
    """Write ``<name>.style.npz``."""
    arrays = {}
    for name, dtype in _ARRAYS.items():
        arrays[name] = np.ascontiguousarray(getattr(checkpoint, name), dtype=dtype)
    arrays['losses'] = arrays['losses'].reshape(-1, 2)
    for name in _NAMES:
        # Fixed-width unicode rather than objects, so the file loads without allow_pickle.
        arrays[name] = np.array(list(getattr(checkpoint, name)), dtype=np.str_)
    for name in _INTS:
        arrays[name] = np.int64(getattr(checkpoint, name))
    for name in _FLOATS:
        arrays[name] = np.float64(getattr(checkpoint, name))
    for name, value in checkpoint.parameters.items():
        arrays[_PARAMETER_PREFIX + name] = np.ascontiguousarray(value, dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    np.savez(out_path, **arrays)


def load_checkpoint(path: str, log=print):
    """
    Load a ``.style.npz``, or return ``None``.

    Returns ``None`` rather than raising: callers run inside ``Py.GIL()`` from Unity, where an
    exception arrives as an opaque managed error.
    """
    if not path or not os.path.isfile(path):
        return None

    try:
        with np.load(path, allow_pickle=False) as f:
            fields = {name: np.ascontiguousarray(f[name], dtype=dtype)
                      for name, dtype in _ARRAYS.items()}
            fields.update({name: [str(value) for value in f[name]] for name in _NAMES})
            fields.update({name: int(f[name]) for name in _INTS})
            fields.update({name: float(f[name]) for name in _FLOATS})
            fields['parameters'] = {
                key[len(_PARAMETER_PREFIX):]: np.ascontiguousarray(f[key], dtype=np.float32)
                for key in f.files if key.startswith(_PARAMETER_PREFIX)}
            return StyleCheckpoint(**fields)
    except (OSError, ValueError, KeyError) as exc:
        # KeyError: a file written by an older layout, which has no version to say so.
        log(f'[Style] could not read checkpoint {path}: {exc}. Press Train on the config to '
            'rebuild it.')
        return None
