"""
On-disk format for the trained motion field value function.

Training produces one artefact: `<name>.mffield.npz`, the value function itself,
written into StreamingAssets so it ships with the player. The transition tables
it is fitted on are intermediate and rebuilt on every train.

Whether a value function still matches its config is Unity's job
(`MotionFieldConfig.hasTrained`), so the loader takes whatever file it is handed
and the file carries only what the loader reads back. The layout carries no
version: a file from an older layout is recognised by its arrays' shapes and
refused with a request to retrain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class ValueFunctionData:
    """A loaded value function."""
    values: np.ndarray          # (states, thetas, speeds) float32
    scores: np.ndarray          # (epochs, 3) float32 -- Bellman residual min/max/mean
    theta_count: int
    gamma: float
    k_neighbors: int
    speeds: np.ndarray          # (speeds,) float32, ascending -- the requested-speed grid, m/s

    @property
    def speed_count(self) -> int:
        return int(self.speeds.shape[0])

    @property
    def theta_spacing(self) -> float:
        """Radians between adjacent headings of `theta_grid(theta_count)`."""
        return 2.0 * np.pi / self.theta_count

    @property
    def final_residual(self) -> float:
        return float(self.scores[-1, 2]) if self.scores.size else float('nan')


def save_value_function(out_path: str, values: np.ndarray, scores: np.ndarray,
                        theta_count: int, gamma: float, k_neighbors: int,
                        speeds: np.ndarray) -> None:
    """
    Write `<name>.mffield.npz`.

    Alongside the arrays go the heading grid size, the requested-speed grid,
    and the `gamma` and `k_neighbors` the fit assumed, which the policy's
    lookahead must reuse.
    """
    speeds = np.asarray(speeds, dtype=np.float32).reshape(-1)
    expected = (theta_count, speeds.shape[0])
    if values.ndim != 3 or values.shape[1:] != expected:
        raise ValueError(f'values must be (states, {expected[0]}, {expected[1]}), got {values.shape}')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez(
        out_path,
        value_function=np.ascontiguousarray(values, dtype=np.float32),
        scores=np.ascontiguousarray(scores, dtype=np.float32),
        theta_count=np.int64(theta_count),
        gamma=np.float32(gamma),
        k_neighbors=np.int64(k_neighbors),
        speeds=speeds,
    )


def load_value_function(path: str, log=print) -> ValueFunctionData | None:
    """
    Load a `.mffield.npz`.

    Returns None for a missing or unreadable file rather than raising: callers
    run inside `Py.GIL()` from Unity, where an exception surfaces as an opaque
    managed error, and the logged reason is clearer than that.
    """
    if not path or not os.path.isfile(path):
        return None

    try:
        with np.load(path, allow_pickle=False) as f:
            values = np.ascontiguousarray(f['value_function'], dtype=np.float32)
            theta_count = int(f['theta_count'])
            # A heading-only value function, from before speed was a task dimension, is
            # (states, thetas) and has no speed grid to read.
            if values.ndim != 3 or 'speeds' not in f.files:
                log(f'[MotionField] {path} is a heading-only value function {values.shape}, '
                    f'trained before requested speed was part of the task. Retrain it: press '
                    f'Train on the config.')
                return None

            speeds = np.ascontiguousarray(f['speeds'], dtype=np.float32).reshape(-1)
            if values.shape[1:] != (theta_count, speeds.shape[0]):
                log(f'[MotionField] {path} holds values {values.shape} that do not match its '
                    f'{theta_count} headings x {speeds.shape[0]} speeds. Retrain it: press Train '
                    f'on the config.')
                return None

            return ValueFunctionData(
                values=values,
                scores=np.ascontiguousarray(f['scores'], dtype=np.float32),
                theta_count=theta_count,
                gamma=float(f['gamma']),
                k_neighbors=int(f['k_neighbors']),
                speeds=speeds,
            )
    except (OSError, ValueError, KeyError) as exc:
        # KeyError: a file written by an older layout, which has no version to say so.
        log(f'[MotionField] could not read value function {path}: {exc}. '
            f'Retrain it: press Train on the config.')
        return None
