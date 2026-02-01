"""search/space.py — Compositional search-space enumeration."""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray


def build_composition_grid(
    n_elements: int,
    resolution: float = 0.05,
    min_fraction: float = 0.0,
) -> NDArray[np.float64]:
    """Enumerate all compositions at *resolution* steps that sum to 1.

    Parameters
    ----------
    n_elements:
        Dimensionality of the composition space.
    resolution:
        Compositional step size (0.05 → 5 % increments).
    min_fraction:
        Minimum allowed fraction for each element (0 = element may be absent).

    Returns
    -------
    grid : ndarray of shape (N, n_elements)
    """
    steps = round(1.0 / resolution)
    min_step = round(min_fraction / resolution)
    grid: list[NDArray[np.float64]] = []
    for combo in product(range(min_step, steps + 1), repeat=n_elements):
        if sum(combo) == steps:
            grid.append(np.array(combo, dtype=np.float64) / steps)
    return np.array(grid)
