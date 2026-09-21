"""
On-disk format for the trained motion field value function.

Training produces one artefact: `<name>.mffield.npz`, the value function itself,
written into StreamingAssets so it ships with the player. The transition tables
it is fitted on are intermediate and rebuilt on every train.

Deciding whether a value function still matches the config that will run it is
Unity's job, not this module's: `MotionFieldConfig.hasTrained` is set by the
Train button and cleared the moment a field is edited, so the answer is known in
the inspector rather than at load time with the character already in play mode.
The loader here therefore takes whatever file it is handed, and the file carries
only what the loader reads back -- nothing describing the run that produced it,
since nothing would consult it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class ValueFunctionData:
    """A loaded value function."""
    values: np.ndarray          # (states, thetas) float32
    scores: np.ndarray          # (epochs, 3) float32 -- Bellman residual min/max/mean
    theta_count: int
    gamma: float
    k_neighbors: int

    @property
    def theta_spacing(self) -> float:
        """Radians between adjacent headings of `theta_grid(theta_count)`."""
        return 2.0 * np.pi / self.theta_count

    @property
    def final_residual(self) -> float:
        return float(self.scores[-1, 2]) if self.scores.size else float('nan')




def save_value_function(out_path: str, values: np.ndarray, scores: np.ndarray,
                        theta_count: int, gamma: float, k_neighbors: int) -> None:
    """
    Write `<name>.mffield.npz`.

    Alongside the arrays go the three scalars the runtime cannot recover on its
    own: the heading grid the values are sampled on, and the `gamma` and
    `k_neighbors` the fit assumed -- which the policy has to reuse for its
    one-step lookahead to score actions the way training scored them.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez(
        out_path,
        value_function=np.ascontiguousarray(values, dtype=np.float32),
        scores=np.ascontiguousarray(scores, dtype=np.float32),
        theta_count=np.int64(theta_count),
        gamma=np.float32(gamma),
        k_neighbors=np.int64(k_neighbors),
    )


def load_value_function(path: str, log=print) -> ValueFunctionData | None:
    """
    Load a `.mffield.npz`.

    Whether this file belongs with the config about to use it is settled in
    Unity before the call; here a missing or unreadable file is the only reason
    to decline. Returns None rather than raising either way -- callers run inside
    `Py.GIL()` from Unity, where an exception surfaces as an opaque managed
    error, and an unusable value function should degrade to greedy control
    rather than take the player down.
    """
    if not path or not os.path.isfile(path):
        return None

    try:
        with np.load(path, allow_pickle=False) as f:
            return ValueFunctionData(
                values=np.ascontiguousarray(f['value_function'], dtype=np.float32),
                scores=np.ascontiguousarray(f['scores'], dtype=np.float32),
                theta_count=int(f['theta_count']),
                gamma=float(f['gamma']),
                k_neighbors=int(f['k_neighbors']),
            )
    except (OSError, ValueError, KeyError) as exc:
        # KeyError is how a file written by an older layout shows up: nothing
        # records a format version, so the first missing key is the symptom.
        log(f'[MotionField] could not read value function {path}: {exc}. '
            f'Press Train on the config to rebuild it.')
        return None
