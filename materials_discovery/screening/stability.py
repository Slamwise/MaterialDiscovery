"""screening/stability.py — Predict thermodynamic stability with CHGNet."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Composition, Lattice, Structure

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """Predicted stability for a candidate composition."""

    composition: dict[str, float]
    formation_energy_eV_per_atom: float
    e_above_hull_eV_per_atom: float | None
    is_stable: bool
    relaxed_energy_eV_per_atom: float
    relaxed_lattice_constant_A: float
    relaxed_structure: object | None = None  # pymatgen Structure


def _build_rocksalt_structure(
    composition: dict[str, float],
    n_atoms: int = 64,
    lattice_constant: float = 4.5,
) -> Structure:
    """Build a disordered rock-salt supercell from fractional composition.

    Metals go on the (0,0,0) FCC sublattice, nonmetals on (0.5,0.5,0.5).
    """
    import random

    metals = {"Hf", "Zr", "Ta", "Ti", "Nb", "W", "Mo", "V", "Cr"}
    nonmetals = {"C", "N", "B", "O"}

    metal_fracs = {k: v for k, v in composition.items() if k in metals and v > 1e-6}
    nonmetal_fracs = {k: v for k, v in composition.items() if k in nonmetals and v > 1e-6}

    if not metal_fracs or not nonmetal_fracs:
        metal_fracs = {k: v for k, v in composition.items() if v > 1e-6}
        nonmetal_fracs = {}

    n_sites_per_sublattice = n_atoms // 2 if nonmetal_fracs else n_atoms

    def _distribute(fracs: dict[str, float], n: int) -> list[str]:
        total = sum(fracs.values())
        normed = {k: v / total for k, v in fracs.items()}
        symbols: list[str] = []
        for el, frac in normed.items():
            count = round(frac * n)
            symbols.extend([el] * count)
        while len(symbols) < n:
            symbols.append(max(normed, key=normed.get))
        while len(symbols) > n:
            symbols.pop()
        random.Random(42).shuffle(symbols)
        return symbols

    metal_symbols = _distribute(metal_fracs, n_sites_per_sublattice)
    nonmetal_symbols = _distribute(nonmetal_fracs, n_sites_per_sublattice) if nonmetal_fracs else []

    n_sc = max(1, round(n_sites_per_sublattice ** (1 / 3)))
    a = lattice_constant * n_sc
    lattice = Lattice.cubic(a)

    species: list[str] = []
    coords: list[list[float]] = []
    mi, ni = 0, 0

    for ix in range(n_sc):
        for iy in range(n_sc):
            for iz in range(n_sc):
                if mi < len(metal_symbols):
                    species.append(metal_symbols[mi])
                    coords.append([
                        (ix + 0.0) / n_sc,
                        (iy + 0.0) / n_sc,
                        (iz + 0.0) / n_sc,
                    ])
                    mi += 1
                if ni < len(nonmetal_symbols):
                    species.append(nonmetal_symbols[ni])
                    coords.append([
                        (ix + 0.5) / n_sc,
                        (iy + 0.5) / n_sc,
                        (iz + 0.5) / n_sc,
                    ])
                    ni += 1

    return Structure(lattice, species, coords)


# Cache the CHGNet model so it's only loaded once across all calls.
_chgnet_model = None
_chgnet_optimizer = None


def _get_chgnet():
    global _chgnet_model, _chgnet_optimizer
    if _chgnet_model is None:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import StructOptimizer
        _chgnet_model = CHGNet.load()
        _chgnet_optimizer = StructOptimizer(model=_chgnet_model)
    return _chgnet_optimizer


def predict_stability(
    composition: dict[str, float],
    api_key: str | None = None,
    n_atoms: int = 64,
    lattice_constant: float = 4.5,
    stability_threshold_eV: float = 0.050,
) -> StabilityResult:
    """Predict whether a composition is thermodynamically stable.

    Uses CHGNet to relax the structure and predict energy per atom.
    Negative energy per atom is used as a heuristic for stability.
    """
    structure = _build_rocksalt_structure(composition, n_atoms, lattice_constant)
    logger.info(
        "Built %d-atom structure for %s",
        len(structure),
        Composition(composition).reduced_formula,
    )

    optimizer = _get_chgnet()
    result = optimizer.relax(structure, fmax=0.05, steps=200, verbose=False)
    relaxed = result["final_structure"]
    energy_per_atom = result["trajectory"].energies[-1] / len(relaxed)
    relaxed_a = relaxed.lattice.a

    # Heuristic: negative energy per atom indicates a bound state.
    # More negative = more stable. We use a simple threshold.
    is_stable = energy_per_atom < 0.0

    logger.info(
        "CHGNet: %s  E=%.4f eV/atom  stable=%s  a=%.3f A",
        Composition(composition).reduced_formula,
        energy_per_atom,
        is_stable,
        relaxed_a,
    )

    return StabilityResult(
        composition=composition,
        formation_energy_eV_per_atom=energy_per_atom,
        e_above_hull_eV_per_atom=None,
        is_stable=is_stable,
        relaxed_energy_eV_per_atom=energy_per_atom,
        relaxed_lattice_constant_A=relaxed_a,
        relaxed_structure=relaxed,
    )
