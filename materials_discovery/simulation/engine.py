"""
simulation/engine.py — Production LAMMPS driver for Prometheus.

Zero-file-IO design: all atomic data and thermodynamic observables pass through
the C-library interface exposed by the ``lammps`` Python package.  No dump files,
no data files, no log files.  This eliminates the I/O bottleneck that cripples
high-throughput screening loops.

GPU notes
---------
When ``gpu_enabled=True`` the engine issues ``package kokkos`` directives so that
MACE/NequIP pair-style evaluation runs on the allocated H100 device.  LAMMPS must
be compiled with KOKKOS + CUDA support (``-D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON``).

Memory-leak mitigation
----------------------
Every public method that mutates LAMMPS state is wrapped so that intermediate
fixes / computes / variables are cleaned up before the method returns.  The class
also implements the context-manager protocol for use in ``with`` blocks.
"""

from __future__ import annotations

import contextlib
import logging
import random
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

# lammps C-library bindings
from lammps import lammps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Atomic masses for elements likely to appear in the UHTC / fusion search
# spaces.  Sourced from IUPAC 2021.  Using a local lookup avoids importing
# all of ASE just for mass data.
_ATOMIC_MASS: dict[str, float] = {
    "Hf": 178.49,
    "Zr": 91.224,
    "Ta": 180.948,
    "C": 12.011,
    "N": 14.007,
    "W": 183.84,
    "Ti": 47.867,
    "Si": 28.085,
    "B": 10.81,
    "Nb": 92.906,
    "Mo": 95.95,
    "V": 50.942,
    "Cr": 51.996,
    "Re": 186.207,
}


class LammpsEngine:
    """In-memory LAMMPS simulation driver.

    Parameters
    ----------
    potential_path:
        Filesystem path to the serialised ML-IAP model file
        (e.g. ``models/mace_uhtc_v1.model``).
    model_type:
        ``"mace"`` or ``"nequip"``.
    gpu_enabled:
        If *True*, configure LAMMPS/KOKKOS for GPU execution.
    gpu_device_id:
        CUDA device ordinal.
    seed:
        Master RNG seed.  Sub-seeds for velocity creation are derived
        deterministically so runs are reproducible.
    """

    # --------------------------------------------------------------------- #
    #  Lifecycle                                                             #
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        potential_path: str,
        model_type: str = "mace",
        gpu_enabled: bool = True,
        gpu_device_id: int = 0,
        seed: int | None = None,
    ) -> None:
        self.potential_path = potential_path
        self.model_type = model_type.lower()
        self.gpu_enabled = gpu_enabled
        self.gpu_device_id = gpu_device_id
        self._rng = random.Random(seed)

        # Build LAMMPS command-line args — suppress all disk output.
        cmdargs: list[str] = ["-log", "none", "-screen", "none", "-nocite"]
        if gpu_enabled:
            cmdargs += [
                "-k", "on", "g", "1",
                "-sf", "kk",
                "-pk", "kokkos", f"gpu/aware on neigh full",
            ]

        self._lmp = lammps(cmdargs=cmdargs)
        self._n_atoms: int = 0
        self._type_map: dict[str, int] = {}
        self._elements: list[str] = []
        # Track computes/fixes/variables we create so we can tear them down.
        self._active_computes: list[str] = []
        self._active_fixes: list[str] = []
        self._active_variables: list[str] = []

    # Context-manager support
    def __enter__(self) -> "LammpsEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Destroy the LAMMPS instance and release all memory."""
        if self._lmp is not None:
            self._cleanup_all()
            self._lmp.close()
            self._lmp = None  # type: ignore[assignment]
            logger.debug("LAMMPS instance destroyed.")

    # --------------------------------------------------------------------- #
    #  Build supercell                                                       #
    # --------------------------------------------------------------------- #

    def build(
        self,
        composition: dict[str, float],
        n_atoms: int = 256,
        lattice_constant: float = 4.5,
    ) -> None:
        """Create a random solid-solution supercell **entirely in memory**.

        Parameters
        ----------
        composition:
            Element → mole-fraction mapping, e.g. ``{"Hf": 0.4, "C": 0.6}``.
            Fractions are re-normalised internally.
        n_atoms:
            Target atom count; the actual count will be the nearest multiple
            of 4 (FCC conventional cell has 4 atoms).
        lattice_constant:
            Lattice parameter in Ångström.
        """
        self._cmd("clear")
        self._cleanup_all()

        # Alphabetical ordering gives deterministic type IDs.
        elements = sorted(composition.keys())
        self._elements = elements
        self._type_map = {el: idx + 1 for idx, el in enumerate(elements)}
        n_types = len(elements)

        # Determine supercell replication factor.
        atoms_per_side = max(2, round((n_atoms / 4) ** (1 / 3)))
        self._n_atoms = 4 * atoms_per_side ** 3

        # --- Core LAMMPS initialisation ---
        self._cmd("units metal")
        self._cmd("dimension 3")
        self._cmd("boundary p p p")
        self._cmd("atom_style atomic")
        self._cmd("atom_modify map array sort 0 0.0")
        self._cmd(f"lattice fcc {lattice_constant}")
        self._cmd(
            f"region simbox block 0 {atoms_per_side} "
            f"0 {atoms_per_side} 0 {atoms_per_side}"
        )
        self._cmd(f"create_box {n_types} simbox")
        self._cmd("create_atoms 1 box")

        # --- Assign species stochastically ---
        fracs = np.array([composition[el] for el in elements], dtype=np.float64)
        fracs /= fracs.sum()
        cumulative = np.cumsum(fracs)

        for atom_id in range(1, self._n_atoms + 1):
            r = self._rng.random()
            atom_type = int(np.searchsorted(cumulative, r)) + 1
            # Clamp to valid range (searchsorted can return n_types).
            atom_type = min(atom_type, n_types)
            self._cmd(f"set atom {atom_id} type {atom_type}")

        # --- Masses ---
        for el, tid in self._type_map.items():
            mass = _ATOMIC_MASS.get(el)
            if mass is None:
                raise ValueError(
                    f"No mass data for element '{el}'.  "
                    f"Add it to _ATOMIC_MASS in engine.py."
                )
            self._cmd(f"mass {tid} {mass}")

        # --- Potential ---
        self._load_potential()

        # --- Neighbor list ---
        self._cmd("neighbor 2.0 bin")
        self._cmd("neigh_modify every 1 delay 0 check yes")

        logger.info(
            "Built %d-atom supercell  elements=%s  fracs=%s",
            self._n_atoms,
            elements,
            fracs.tolist(),
        )

    # --------------------------------------------------------------------- #
    #  Potential loading                                                     #
    # --------------------------------------------------------------------- #

    def _load_potential(self) -> None:
        el_str = " ".join(self._elements)
        if self.model_type == "mace":
            self._cmd("pair_style mace no_domain_decomposition")
            self._cmd(f"pair_coeff * * {self.potential_path} {el_str}")
        elif self.model_type == "nequip":
            self._cmd("pair_style nequip")
            self._cmd(f"pair_coeff * * {self.potential_path} {el_str}")
        else:
            raise ValueError(f"Unsupported potential type: {self.model_type!r}")

    def swap_potential(self, new_path: str, model_type: str | None = None) -> None:
        """Hot-swap the ML-IAP model without rebuilding the cell.

        Use this after an online fine-tuning cycle to inject the updated
        potential into a running loop.
        """
        if model_type is not None:
            self.model_type = model_type.lower()
        self.potential_path = new_path
        self._load_potential()
        logger.info("Potential swapped → %s (%s)", new_path, self.model_type)

    # --------------------------------------------------------------------- #
    #  Energy minimisation                                                   #
    # --------------------------------------------------------------------- #

    def minimize(
        self,
        etol: float = 1e-8,
        ftol: float = 1e-10,
        maxiter: int = 20_000,
        maxeval: int = 200_000,
    ) -> float:
        """Conjugate-gradient minimisation.  Returns final PE (eV)."""
        self._cmd("min_style cg")
        self._cmd(f"minimize {etol} {ftol} {maxiter} {maxeval}")
        return self.get_thermo("pe")

    # --------------------------------------------------------------------- #
    #  MD runs                                                               #
    # --------------------------------------------------------------------- #

    def run_nvt(
        self, T: float, steps: int, dt: float = 0.001, damping: float = 0.1
    ) -> None:
        """Nosé–Hoover NVT with velocity initialisation."""
        fix_id = "_prom_nvt"
        self._cmd(f"timestep {dt}")
        self._cmd(
            f"velocity all create {T} {self._rng.randint(1, 99999)} "
            f"dist gaussian"
        )
        self._cmd(f"fix {fix_id} all nvt temp {T} {T} {damping}")
        self._active_fixes.append(fix_id)
        self._cmd(f"run {steps}")
        self._unfix(fix_id)

    def run_npt(
        self,
        T: float,
        P: float = 0.0,
        steps: int = 10_000,
        dt: float = 0.001,
        T_damp: float = 0.1,
        P_damp: float = 1.0,
    ) -> None:
        """Isotropic NPT (Nosé–Hoover barostat)."""
        fix_id = "_prom_npt"
        self._cmd(f"timestep {dt}")
        self._cmd(
            f"velocity all create {T} {self._rng.randint(1, 99999)} "
            f"dist gaussian"
        )
        self._cmd(
            f"fix {fix_id} all npt temp {T} {T} {T_damp} "
            f"iso {P} {P} {P_damp}"
        )
        self._active_fixes.append(fix_id)
        self._cmd(f"run {steps}")
        self._unfix(fix_id)

    def run_nve(self, steps: int, dt: float = 0.001) -> None:
        """Micro-canonical (NVE) production run."""
        fix_id = "_prom_nve"
        self._cmd(f"timestep {dt}")
        self._cmd(f"fix {fix_id} all nve")
        self._active_fixes.append(fix_id)
        self._cmd(f"run {steps}")
        self._unfix(fix_id)

    # --------------------------------------------------------------------- #
    #  Thermo / observable extraction  (all in-memory)                       #
    # --------------------------------------------------------------------- #

    def get_thermo(self, keyword: str) -> float:
        """Read a scalar thermo keyword (pe, ke, temp, press, vol, …)."""
        return float(self._lmp.get_thermo(keyword))

    def get_stress_voigt(self) -> NDArray[np.float64]:
        """Return Voigt stress [σxx, σyy, σzz, σxy, σxz, σyz] in **GPa**.

        LAMMPS *metal* units report pressure in bar → multiply by 1 e-4.
        """
        labels = ("pxx", "pyy", "pzz", "pxy", "pxz", "pyz")
        # LAMMPS sign convention: negative = compressive.
        # We negate so that tensile stress is positive (Voigt convention).
        return np.array(
            [-self.get_thermo(lb) * 1e-4 for lb in labels], dtype=np.float64
        )

    def get_pe_per_atom(self) -> float:
        return self.get_thermo("pe") / self._n_atoms

    def get_volume(self) -> float:
        """Simulation box volume in ų."""
        return self.get_thermo("vol")

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    @property
    def elements(self) -> list[str]:
        return list(self._elements)

    # --------------------------------------------------------------------- #
    #  Green–Kubo heat-flux computes                                        #
    # --------------------------------------------------------------------- #

    def setup_heat_flux_computes(self) -> None:
        """Install the per-atom and global computes needed for κ(Green–Kubo).

        Call this **once** after ``build()`` and before the NVE production run
        that collects the heat-flux autocorrelation.
        """
        for cid in ("_prom_ke_atom", "_prom_pe_atom",
                     "_prom_stress_atom", "_prom_flux"):
            self._safe_uncompute(cid)
        for vid in ("_prom_Jx", "_prom_Jy", "_prom_Jz"):
            self._safe_unvariable(vid)

        self._cmd("compute _prom_ke_atom all ke/atom")
        self._cmd("compute _prom_pe_atom all pe/atom")
        self._cmd("compute _prom_stress_atom all stress/atom NULL")
        self._cmd(
            "compute _prom_flux all heat/flux "
            "_prom_ke_atom _prom_pe_atom _prom_stress_atom"
        )
        self._active_computes += [
            "_prom_ke_atom", "_prom_pe_atom",
            "_prom_stress_atom", "_prom_flux",
        ]
        self._cmd("variable _prom_Jx equal c__prom_flux[1]")
        self._cmd("variable _prom_Jy equal c__prom_flux[2]")
        self._cmd("variable _prom_Jz equal c__prom_flux[3]")
        self._active_variables += ["_prom_Jx", "_prom_Jy", "_prom_Jz"]

    def run_green_kubo_collection(
        self,
        correlation_length: int = 20_000,
        sample_interval: int = 10,
        dt: float = 0.001,
    ) -> tuple[float, float, float]:
        """Run NVE and accumulate heat-flux autocorrelation.

        Returns the three diagonal components of the integrated
        autocorrelation (Jx·Jx, Jy·Jy, Jz·Jz) in LAMMPS metal units.
        The caller is responsible for the κ unit conversion.
        """
        n_corr_samples = correlation_length // sample_interval

        fix_id = "_prom_gk"
        nve_id = "_prom_gk_nve"
        self._safe_unfix(fix_id)
        self._safe_unfix(nve_id)

        self._cmd(f"timestep {dt}")
        self._cmd(f"fix {nve_id} all nve")
        self._cmd(
            f"fix {fix_id} all ave/correlate "
            f"{sample_interval} {n_corr_samples} {correlation_length} "
            f"v__prom_Jx v__prom_Jy v__prom_Jz type auto ave running"
        )
        self._active_fixes += [nve_id, fix_id]
        self._cmd(f"run {correlation_length}")

        # Extract the running-average trap-integrated autocorrelation.
        trap = np.zeros(3, dtype=np.float64)
        for col_idx in range(3):
            trap[col_idx] = self._lmp.extract_fix(
                fix_id, 0, 1, 0, col_idx + 1  # style global, type vector
            )

        self._unfix(fix_id)
        self._unfix(nve_id)
        return float(trap[0]), float(trap[1]), float(trap[2])

    def teardown_heat_flux_computes(self) -> None:
        """Remove heat-flux computes/variables to free memory."""
        for vid in ("_prom_Jx", "_prom_Jy", "_prom_Jz"):
            self._safe_unvariable(vid)
        for cid in ("_prom_flux", "_prom_stress_atom",
                     "_prom_pe_atom", "_prom_ke_atom"):
            self._safe_uncompute(cid)

    # --------------------------------------------------------------------- #
    #  Elastic-constant helpers                                              #
    # --------------------------------------------------------------------- #

    def apply_voigt_strain(self, eps: NDArray[np.float64]) -> None:
        """Apply a 6-component Voigt strain to the simulation box.

        Parameters
        ----------
        eps : array of shape (6,)
            [εxx, εyy, εzz, εxy, εxz, εyz]
        """
        # Ensure box is triclinic so tilt factors can be applied.
        try:
            self._cmd("change_box all triclinic")
        except Exception:
            pass  # Already triclinic
        self._cmd(
            f"change_box all "
            f"x scale {1.0 + eps[0]:.12g} "
            f"y scale {1.0 + eps[1]:.12g} "
            f"z scale {1.0 + eps[2]:.12g} "
            f"xy delta {eps[3]:.12g} "
            f"xz delta {eps[4]:.12g} "
            f"yz delta {eps[5]:.12g} "
            f"remap"
        )

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _cmd(self, cmd: str) -> None:
        self._lmp.command(cmd)

    def _unfix(self, fix_id: str) -> None:
        with contextlib.suppress(Exception):
            self._cmd(f"unfix {fix_id}")
        with contextlib.suppress(ValueError):
            self._active_fixes.remove(fix_id)

    def _safe_unfix(self, fix_id: str) -> None:
        self._unfix(fix_id)

    def _safe_uncompute(self, compute_id: str) -> None:
        with contextlib.suppress(Exception):
            self._cmd(f"uncompute {compute_id}")
        with contextlib.suppress(ValueError):
            self._active_computes.remove(compute_id)

    def _safe_unvariable(self, var_id: str) -> None:
        with contextlib.suppress(Exception):
            self._cmd(f"variable {var_id} delete")
        with contextlib.suppress(ValueError):
            self._active_variables.remove(var_id)

    def _cleanup_all(self) -> None:
        """Remove every compute / fix / variable that *we* created."""
        for vid in list(self._active_variables):
            self._safe_unvariable(vid)
        for fid in list(self._active_fixes):
            self._safe_unfix(fid)
        for cid in list(self._active_computes):
            self._safe_uncompute(cid)
