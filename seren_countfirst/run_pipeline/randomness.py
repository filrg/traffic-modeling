from __future__ import annotations

from enum import IntEnum

import numpy as np

from .config import SEED


class SeedStream(IntEnum):
    BASELINE = 0
    EVALUATION = 1
    RECONSTRUCTION_POOL = 3
    LOAD_PATH = 4
    PLACEMENT_CHECK = 6
    CERTIFICATION = 1000
    RECONSTRUCTION_MARKS = 5000
    WORKLOAD_MARKS = 7000


def seed(stream: SeedStream, replicate: int = 0, *, base: int = SEED) -> int:
    """Return a legacy-compatible seed for one named random stream."""
    return base + int(stream) + int(replicate)


def generator(stream: SeedStream, replicate: int = 0, *, base: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed(stream, replicate, base=base))
