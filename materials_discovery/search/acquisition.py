"""search/acquisition.py — Acquisition functions for Bayesian optimisation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def upper_confidence_bound(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    kappa: float = 2.576,
) -> NDArray[np.float64]:
    """UCB: α(x) = μ(x) + κ · σ(x)."""
    return mu + kappa * sigma


def expected_improvement(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    y_best: float,
    xi: float = 0.01,
) -> NDArray[np.float64]:
    """EI with exploration jitter *xi*."""
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = mu - y_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-12] = 0.0
    return ei


def probability_of_improvement(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    y_best: float,
    xi: float = 0.01,
) -> NDArray[np.float64]:
    """PI — useful as a cheap fallback."""
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (mu - y_best - xi) / sigma
        pi = norm.cdf(Z)
        pi[sigma < 1e-12] = 0.0
    return pi
