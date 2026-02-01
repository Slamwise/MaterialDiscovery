"""fitness/scorer.py — Multi-objective fitness scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FitnessWeights


@dataclass
class FitnessResult:
    """Container for one evaluated composition."""

    composition: dict[str, float]
    temperature_K: float
    elastic_modulus_GPa: float
    thermal_conductivity_WmK: float
    cohesive_energy_eV: float
    fitness_score: float


def _clip_norm(value: float, lo: float, hi: float) -> float:
    if np.isnan(value):
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def aerospace_fitness(
    E: float,
    kappa: float,
    E_coh: float,
    w: FitnessWeights | None = None,
) -> float:
    """Aerospace-grade score: high stiffness, decent κ, strong bonds."""
    if w is None:
        w = FitnessWeights()
    E_n = _clip_norm(E, 200, 600)
    k_n = _clip_norm(kappa, 5, 40)
    c_n = _clip_norm(abs(E_coh), 5, 10)
    return w.elastic_modulus * E_n + w.thermal_conductivity * k_n + w.cohesive_energy * c_n


def fusion_fitness(
    E: float,
    kappa: float,
    E_coh: float,
    w: FitnessWeights | None = None,
) -> float:
    """Fusion-grade score: prioritise κ and radiation tolerance."""
    if w is None:
        w = FitnessWeights(elastic_modulus=0.3, thermal_conductivity=0.4, cohesive_energy=0.3)
    E_n = _clip_norm(E, 100, 400)
    k_n = _clip_norm(kappa, 10, 80)
    c_n = _clip_norm(abs(E_coh), 4, 9)
    return w.elastic_modulus * E_n + w.thermal_conductivity * k_n + w.cohesive_energy * c_n
