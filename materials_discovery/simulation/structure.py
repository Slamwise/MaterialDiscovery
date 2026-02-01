"""simulation/structure.py — Composition vector utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def composition_to_vector(
    composition: dict[str, float], elements: list[str]
) -> NDArray[np.float64]:
    """Convert a {element: fraction} dict to a fixed-order numpy vector."""
    vec = np.array([composition.get(el, 0.0) for el in elements], dtype=np.float64)
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def vector_to_composition(
    vec: NDArray[np.float64], elements: list[str]
) -> dict[str, float]:
    """Inverse of ``composition_to_vector``."""
    return {el: float(vec[i]) for i, el in enumerate(elements)}
