"""
search/bayesian_opt.py — Production Bayesian-optimisation driver.

This module owns the surrogate model, the acquisition function, and the
suggest → observe loop that steers the Prometheus active-learning campaign.

Design notes
------------
* **Surrogate choice** — Gaussian Process (scikit-learn) is the default and
  the correct choice for < 10 k observations.  When the ledger grows beyond
  ~5 k points the GP cubic scaling becomes painful; at that point we auto-
  switch to a Random-Forest surrogate whose predict + variance call is O(n·d).
* **Acquisition** — UCB with κ = 2.576 (≈ 99 % CI) is the launch default.
  EI and PI are available as drop-in replacements via ``AcquisitionFn``.
* **Batch selection** — We use greedy batch UCB with a "kriging believer"
  heuristic: after picking the top-UCB candidate we hallucinate its predicted
  mean as the observation, re-compute UCB, and pick the next one.  This
  produces diverse batches without the cost of full q-UCB.
* **Determinism** — all randomness is seeded through numpy ``Generator``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .acquisition import (
    expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
)
from .space import build_composition_grid

logger = logging.getLogger(__name__)

# How many observations before we auto-switch GP → RF.
_GP_OBSERVATION_LIMIT = 5_000


# ------------------------------------------------------------------ #
#  Public data structures                                             #
# ------------------------------------------------------------------ #

class AcquisitionFn(Enum):
    UCB = auto()
    EI = auto()
    PI = auto()


@dataclass
class Candidate:
    """A proposed composition together with the surrogate's prediction."""

    composition: dict[str, float]
    composition_vec: NDArray[np.float64]
    predicted_mean: float
    predicted_std: float
    acquisition_value: float


@dataclass
class _ObservationLedger:
    """Append-only store for (X, y) pairs."""

    X: list[NDArray[np.float64]] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    def as_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.array(self.X), np.array(self.y)


# ------------------------------------------------------------------ #
#  Optimiser                                                          #
# ------------------------------------------------------------------ #

class BayesianOptimizer:
    """Active-learning search driver for the Prometheus loop.

    Parameters
    ----------
    elements:
        Ordered list of element symbols that define the composition axes.
    resolution:
        Compositional grid step (0.05 → 5 %).
    kappa:
        UCB exploration parameter.
    acquisition:
        Which acquisition function to use.
    n_initial_random:
        Number of purely random evaluations before the surrogate is fit.
    seed:
        Master RNG seed for reproducibility.
    """

    def __init__(
        self,
        elements: Sequence[str],
        resolution: float = 0.05,
        kappa: float = 2.576,
        acquisition: AcquisitionFn = AcquisitionFn.UCB,
        n_initial_random: int = 20,
        seed: int = 42,
    ) -> None:
        self.elements = list(elements)
        self.resolution = resolution
        self.kappa = kappa
        self.acquisition = acquisition
        self.n_initial_random = n_initial_random
        self._rng = np.random.default_rng(seed)

        self._ledger = _ObservationLedger()
        self._surrogate: GaussianProcessRegressor | RandomForestRegressor | None = None
        self._surrogate_kind: str = "gp"

        # Pre-compute the discrete search grid once.
        self._grid = build_composition_grid(
            n_elements=len(self.elements),
            resolution=self.resolution,
        )
        logger.info(
            "Search grid: %d compositions  elements=%s  resolution=%.2f",
            len(self._grid),
            self.elements,
            self.resolution,
        )

    # ---------------------------------------------------------------- #
    #  Surrogate construction                                          #
    # ---------------------------------------------------------------- #

    def _build_gp(self) -> GaussianProcessRegressor:
        kernel = Matern(nu=2.5, length_scale=np.ones(len(self.elements))) + WhiteKernel(
            noise_level=1e-4, noise_level_bounds=(1e-10, 1e-1)
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=int(self._rng.integers(0, 2**31)),
        )

    def _build_rf(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=3,
            random_state=int(self._rng.integers(0, 2**31)),
            n_jobs=-1,
        )

    def _fit_surrogate(self) -> None:
        X, y = self._ledger.as_arrays()
        n = len(y)
        if n > _GP_OBSERVATION_LIMIT and self._surrogate_kind == "gp":
            logger.info(
                "Observation count (%d) exceeds GP limit; switching to RF.", n
            )
            self._surrogate_kind = "rf"

        if self._surrogate_kind == "gp":
            self._surrogate = self._build_gp()
            self._surrogate.fit(X, y)
        else:
            self._surrogate = self._build_rf()
            self._surrogate.fit(X, y)

    def _predict(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (mean, std) predictions for *X*."""
        if isinstance(self._surrogate, GaussianProcessRegressor):
            mu, sigma = self._surrogate.predict(X, return_std=True)
            return mu, sigma

        # RandomForest: use per-tree predictions to estimate uncertainty.
        assert isinstance(self._surrogate, RandomForestRegressor)
        preds = np.array(
            [tree.predict(X) for tree in self._surrogate.estimators_]
        )
        return preds.mean(axis=0), preds.std(axis=0)

    # ---------------------------------------------------------------- #
    #  Acquisition                                                     #
    # ---------------------------------------------------------------- #

    def _acquisition_values(
        self,
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.acquisition == AcquisitionFn.UCB:
            return upper_confidence_bound(mu, sigma, kappa=self.kappa)
        y_best = max(self._ledger.y)
        if self.acquisition == AcquisitionFn.EI:
            return expected_improvement(mu, sigma, y_best)
        return probability_of_improvement(mu, sigma, y_best)

    # ---------------------------------------------------------------- #
    #  Suggest                                                         #
    # ---------------------------------------------------------------- #

    def suggest(self, batch_size: int = 5) -> list[Candidate]:
        """Propose the next batch of compositions to simulate.

        During the initial random phase (< ``n_initial_random`` observations),
        compositions are drawn uniformly from the grid.

        After enough data has been collected the surrogate is fit and the
        batch is selected via greedy kriging-believer UCB.
        """
        if len(self._ledger) < self.n_initial_random:
            return self._random_suggestions(batch_size)

        self._fit_surrogate()
        return self._kriging_believer_batch(batch_size)

    def _random_suggestions(self, batch_size: int) -> list[Candidate]:
        idx = self._rng.choice(len(self._grid), size=batch_size, replace=False)
        return [
            Candidate(
                composition=self._vec_to_dict(self._grid[i]),
                composition_vec=self._grid[i].copy(),
                predicted_mean=0.0,
                predicted_std=1.0,
                acquisition_value=float("inf"),
            )
            for i in idx
        ]

    def _kriging_believer_batch(self, batch_size: int) -> list[Candidate]:
        """Greedy batch construction with hallucinated observations."""
        # Work on a copy of the ledger so hallucinations don't persist.
        X_extra: list[NDArray[np.float64]] = []
        y_extra: list[float] = []
        selected: list[Candidate] = []

        for _ in range(batch_size):
            mu, sigma = self._predict(self._grid)
            acq = self._acquisition_values(mu, sigma)

            # Mask already-selected grid points.
            for prev in selected:
                dists = np.linalg.norm(self._grid - prev.composition_vec, axis=1)
                acq[dists < 1e-9] = -np.inf

            best_idx = int(np.argmax(acq))
            best_vec = self._grid[best_idx]

            cand = Candidate(
                composition=self._vec_to_dict(best_vec),
                composition_vec=best_vec.copy(),
                predicted_mean=float(mu[best_idx]),
                predicted_std=float(sigma[best_idx]),
                acquisition_value=float(acq[best_idx]),
            )
            selected.append(cand)

            # Hallucinate: pretend we observed the predicted mean.
            X_extra.append(best_vec)
            y_extra.append(float(mu[best_idx]))
            # Temporarily augment the surrogate's training data.
            X_all, y_all = self._ledger.as_arrays()
            X_aug = np.vstack([X_all, np.array(X_extra)])
            y_aug = np.concatenate([y_all, np.array(y_extra)])
            if isinstance(self._surrogate, GaussianProcessRegressor):
                self._surrogate.fit(X_aug, y_aug)
            else:
                self._surrogate.fit(X_aug, y_aug)

        # Restore original surrogate state (discard hallucinations).
        self._fit_surrogate()
        return selected

    # ---------------------------------------------------------------- #
    #  Observe                                                         #
    # ---------------------------------------------------------------- #

    def observe(self, composition_vec: NDArray[np.float64], fitness: float) -> None:
        """Record a simulation result."""
        self._ledger.X.append(composition_vec.copy())
        self._ledger.y.append(fitness)

    def observe_batch(
        self,
        X: Sequence[NDArray[np.float64]],
        y: Sequence[float],
    ) -> None:
        for xi, yi in zip(X, y):
            self.observe(xi, yi)

    # ---------------------------------------------------------------- #
    #  Utilities                                                       #
    # ---------------------------------------------------------------- #

    def _vec_to_dict(self, vec: NDArray[np.float64]) -> dict[str, float]:
        return {el: round(float(vec[i]), 4) for i, el in enumerate(self.elements)}

    @property
    def n_observed(self) -> int:
        return len(self._ledger)

    def best_so_far(self) -> tuple[dict[str, float], float]:
        """Return the composition and fitness of the best observation."""
        idx = int(np.argmax(self._ledger.y))
        return (
            self._vec_to_dict(self._ledger.X[idx]),
            self._ledger.y[idx],
        )
