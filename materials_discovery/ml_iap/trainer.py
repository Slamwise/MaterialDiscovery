"""ml_iap/trainer.py — Online fine-tuning stub for MACE / NequIP.

The actual training loop depends on the installed MACE or NequIP version.
This module provides the interface contract and a reference implementation
for MACE fine-tuning using ``mace.tools.finetuning``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OnlineTrainer:
    """Fine-tune a MACE model on a small DFT batch.

    Parameters
    ----------
    base_model_path:
        Path to the currently deployed ``.model`` file.
    output_dir:
        Where to write the updated model checkpoint.
    lr:
        Learning rate for fine-tuning.
    max_epochs:
        Maximum epochs per fine-tune cycle.
    """

    def __init__(
        self,
        base_model_path: str | Path,
        output_dir: str | Path = "models",
        lr: float = 1e-4,
        max_epochs: int = 100,
    ) -> None:
        self.base_model_path = Path(base_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lr = lr
        self.max_epochs = max_epochs
        self._version = 0

    def fine_tune(self, dft_data: list[dict[str, Any]]) -> Path:
        """Run a fine-tuning cycle and return the path to the new model.

        Parameters
        ----------
        dft_data:
            List of dicts, each containing:
            - ``positions``: (N, 3) array
            - ``cell``: (3, 3) array
            - ``atomic_numbers``: (N,) array
            - ``energy``: float (eV)
            - ``forces``: (N, 3) array (eV/Å)
        """
        self._version += 1
        new_path = self.output_dir / f"mace_uhtc_ft_v{self._version}.model"

        try:
            from mace.tools.finetuning import run_finetuning  # type: ignore[import]

            run_finetuning(
                model_path=str(self.base_model_path),
                train_data=dft_data,
                output_path=str(new_path),
                lr=self.lr,
                max_num_epochs=self.max_epochs,
            )
        except ImportError:
            logger.error(
                "mace package not installed — cannot fine-tune.  "
                "Returning base model unchanged."
            )
            return self.base_model_path

        logger.info("Fine-tuned model saved → %s  (%d DFT configs)", new_path, len(dft_data))
        self.base_model_path = new_path
        return new_path
