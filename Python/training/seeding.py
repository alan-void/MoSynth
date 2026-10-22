"""
One call that seeds every random number generator a training run can draw from.

A run is repeated under several seeds to measure how much of a result is the method and how much
the draw, so a generator left unseeded would leak run-to-run noise into what reads as the seed's
effect. Python's ``random``, numpy's legacy global generator and torch (CPU and every CUDA device)
are all seeded; code that wants its own stream should still build a ``torch.Generator`` or a
``numpy.random.default_rng`` from the same seed.
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy.random`` and, when it is installed, ``torch``."""
    seed = int(seed)
    random.seed(seed)
    # The legacy global generator only takes 32 bits.
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
    except ImportError:
        return
    # Also seeds every CUDA device.
    torch.manual_seed(seed)
