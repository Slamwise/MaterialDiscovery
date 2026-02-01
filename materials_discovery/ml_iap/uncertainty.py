"""ml_iap/uncertainty.py — Committee-disagreement monitor."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class UncertaintyMonitor:
    """Flag configurations where the ML-IAP ensemble disagrees."""

    def __init__(self, threshold_eV_per_atom: float = 0.05) -> None:
        self.threshold = threshold_eV_per_atom
        self.flagged_count = 0

    def check(self, energies_per_atom: Sequence[float]) -> bool:
        """Return *True* if the committee spread exceeds threshold."""
        std = float(np.std(energies_per_atom))
        if std > self.threshold:
            self.flagged_count += 1
            logger.warning(
                "High IAP uncertainty: σ = %.4f eV/atom (threshold %.4f). "
                "Flagging for DFT.  Total flagged: %d",
                std, self.threshold, self.flagged_count,
            )
            return True
        return False
