"""ml_iap/dft_oracle.py — DFT single-point calculation interface.

This is a thin wrapper that submits single-point calculations to
VASP / CP2K.  In production, this would submit jobs to the cluster
scheduler.  Here we provide the interface and a local-VASP reference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DFTOracle:
    """Submit and collect DFT single-point calculations."""

    def __init__(
        self,
        code: str = "vasp",
        work_dir: str | Path = "dft_runs",
        pseudopotential_dir: str | Path = "potentials/paw_pbe",
    ) -> None:
        self.code = code
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.pp_dir = Path(pseudopotential_dir)

    def single_point(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run a single-point DFT and return energy + forces.

        Parameters
        ----------
        config:
            Must contain ``positions``, ``cell``, ``atomic_numbers``.

        Returns
        -------
        dict with ``energy`` (eV) and ``forces`` (N×3, eV/Å).
        """
        logger.info("DFT single-point requested for %d atoms", len(config["atomic_numbers"]))
        # Placeholder — in production this calls ASE's VASP calculator
        # or submits a Slurm job.
        raise NotImplementedError(
            "Wire this up to your cluster's VASP/CP2K installation."
        )
