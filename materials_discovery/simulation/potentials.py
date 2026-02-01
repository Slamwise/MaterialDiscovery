"""simulation/potentials.py — Potential file management."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PotentialManager:
    """Registry of available ML-IAP model files."""

    def __init__(self, model_dir: str | Path = "models") -> None:
        self.model_dir = Path(model_dir)

    def latest(self, prefix: str = "mace_uhtc") -> Path:
        """Return the most-recently-modified model matching *prefix*."""
        candidates = sorted(
            self.model_dir.glob(f"{prefix}*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No model files matching '{prefix}*' in {self.model_dir}"
            )
        logger.info("Selected potential: %s", candidates[0])
        return candidates[0]
