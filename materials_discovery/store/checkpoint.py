"""store/checkpoint.py — Save / resume loop state."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    epoch: int,
    iteration: int,
    optimizer_state: dict[str, Any],
) -> None:
    path = Path(path)
    data = {
        "epoch": epoch,
        "iteration": iteration,
        "optimizer_X": [x.tolist() for x in optimizer_state.get("X", [])],
        "optimizer_y": list(optimizer_state.get("y", [])),
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Checkpoint saved → %s  (epoch=%d, iter=%d)", path, epoch, iteration)


def load_checkpoint(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    logger.info("Checkpoint loaded ← %s", path)
    return data
