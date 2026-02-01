"""fitness/thermal.py — Thermal conductivity via the Green–Kubo relation."""

from __future__ import annotations

import logging

from ..simulation.engine import LammpsEngine

logger = logging.getLogger(__name__)

# ── Unit-conversion constants (LAMMPS *metal* units) ──────────────
# Energy : eV          (1 eV = 1.6022e-19 J)
# Time   : ps          (1 ps = 1e-12 s)
# Length : Å           (1 Å  = 1e-10 m)
# Volume : ų          (1 ų = 1e-30 m³)
# k_B in eV/K
_KB_EV = 8.617333262e-5
# Conversion factor:  eV² / (Å · ps · K²)  →  W / (m · K)
# W = J/s = eV·1.6022e-19 / (1e-12 s)  →  factor = 1.6022e-7
# Full:  V[ų]*1e-30 / (k_B[eV/K] * T² * dt[ps]*1e-12)  ×  <JJ> [eV²/(Å² ps²)]
#        × 1.6022e-19 [J/eV]  / 1e-10 [m/Å]  / 1e-12 [s/ps]
# Simplifies to  CONV = 1.6022e-19 / (1e-10 * 1e-12) = 1.6022e3  ← per (eV·ps/ų)
_CONV = 1.6022e3


def thermal_conductivity_green_kubo(
    engine: LammpsEngine,
    T: float,
    correlation_length: int = 20_000,
    sample_interval: int = 10,
    dt: float = 0.001,
) -> float:
    r"""Compute isotropic κ [W/(m·K)] via the Green–Kubo relation.

    .. math::
        \kappa = \frac{V}{3\,k_B\,T^2}
                 \int_0^\infty \langle \mathbf{J}(0)\cdot\mathbf{J}(t)\rangle\,dt

    Parameters
    ----------
    engine:
        A ``LammpsEngine`` with a built & equilibrated supercell.
    T:
        Temperature in Kelvin.
    correlation_length:
        Total MD steps for the NVE collection run.
    sample_interval:
        Sampling interval for the autocorrelation fix.
    dt:
        Timestep in ps.
    """
    V_A3 = engine.get_volume()  # ų

    engine.setup_heat_flux_computes()
    Jxx, Jyy, Jzz = engine.run_green_kubo_collection(
        correlation_length=correlation_length,
        sample_interval=sample_interval,
        dt=dt,
    )
    engine.teardown_heat_flux_computes()

    J_total = Jxx + Jyy + Jzz  # sum of three diagonal autocorrelation integrals

    # κ = V / (3 k_B T²) × ∫<J·J>dt  ×  unit conversion
    prefactor = (V_A3 * 1e-30) / (3.0 * _KB_EV * 1.6022e-19 * T * T)
    kappa = prefactor * J_total * _CONV * dt * 1e-12 * sample_interval

    kappa = abs(kappa)
    logger.info("κ(Green–Kubo) = %.4f W/(m·K)  at T = %.0f K", kappa, T)
    return kappa
