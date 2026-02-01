"""fitness/mechanical.py — Elastic modulus via Voigt–Reuss–Hill averaging."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ..simulation.engine import LammpsEngine

logger = logging.getLogger(__name__)


def elastic_modulus_vrh(
    engine: LammpsEngine,
    strain_mag: float = 0.003,
) -> float:
    """Compute isotropic Young's modulus *E* (GPa) via the VRH average.

    Applies six independent Voigt strains of magnitude ``strain_mag``,
    reads the stress response, assembles the 6×6 elastic-constant matrix
    C_ij, and returns E from the Voigt–Reuss–Hill scheme.
    """
    engine.minimize()
    ref_stress = engine.get_stress_voigt()

    C = np.zeros((6, 6), dtype=np.float64)
    strain_basis = np.eye(6, dtype=np.float64)

    for i in range(6):
        eps = strain_basis[i] * strain_mag

        engine.apply_voigt_strain(eps)
        engine.minimize()
        perturbed = engine.get_stress_voigt()
        C[:, i] = (perturbed - ref_stress) / strain_mag

        # Reverse the strain to restore the reference state.
        engine.apply_voigt_strain(-eps)
        engine.minimize()

    # Symmetrise (numerical noise breaks exact symmetry).
    C = 0.5 * (C + C.T)

    # Voigt averages
    B_V = (C[0, 0] + C[1, 1] + C[2, 2]
           + 2.0 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    G_V = (C[0, 0] + C[1, 1] + C[2, 2]
           - C[0, 1] - C[0, 2] - C[1, 2]
           + 3.0 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0

    # Reuss averages (need compliance S = C⁻¹)
    try:
        S = np.linalg.inv(C)
        B_R = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2]
                      + 2.0 * (S[0, 1] + S[0, 2] + S[1, 2]))
        G_R = 15.0 / (4.0 * (S[0, 0] + S[1, 1] + S[2, 2])
                       - 4.0 * (S[0, 1] + S[0, 2] + S[1, 2])
                       + 3.0 * (S[3, 3] + S[4, 4] + S[5, 5]))
    except np.linalg.LinAlgError:
        logger.warning("Singular C matrix; falling back to Voigt only.")
        B_R, G_R = B_V, G_V

    # Hill averages
    B = 0.5 * (B_V + B_R)
    G = 0.5 * (G_V + G_R)

    # Young's modulus
    if (3.0 * B + G) < 1e-12:
        return 0.0
    E = 9.0 * B * G / (3.0 * B + G)

    logger.info("C_ij diag = %s  E = %.2f GPa", np.diag(C).tolist(), E)
    return float(E)
