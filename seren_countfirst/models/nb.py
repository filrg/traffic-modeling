from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def nb2_loglik(n: np.ndarray, lam: np.ndarray, kappa: float,
               weights: np.ndarray | None = None) -> float:
    n = np.asarray(n, dtype=float)
    lam = np.asarray(lam, dtype=float)
    ll = (
        gammaln(n + kappa)
        - gammaln(kappa)
        - gammaln(n + 1.0)
        + kappa * (np.log(kappa) - np.log(kappa + lam))
        + n * (np.log(lam) - np.log(kappa + lam))
    )
    return float(np.sum(ll if weights is None else weights * ll))


def nb2_score_eta(n: np.ndarray, lam: np.ndarray, kappa: float) -> np.ndarray:
    """Return d log p / d eta with lam = exp(eta)."""
    return n - lam * (n + kappa) / (kappa + lam)


def nb2_info_eta(n: np.ndarray, lam: np.ndarray, kappa: float) -> np.ndarray:
    """Return -d^2 log p / d eta^2; strictly positive, so the joint mode is unique."""
    return (n + kappa) * kappa * lam / (kappa + lam) ** 2


def nb2_sample(lam: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Draw NB2 counts as a gamma-Poisson mixture: Var = lam + lam^2 / kappa."""
    g = rng.gamma(shape=kappa, scale=1.0 / kappa, size=lam.shape)
    return rng.poisson(lam * g)
