from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def ar1_precision(n: int, sigma: float, phi: float) -> sp.csc_matrix:
    """Return the precision of a stationary AR(1) with marginal variance sigma^2."""
    c = 1.0 / (sigma ** 2 * (1.0 - phi ** 2))
    main = np.full(n, c * (1.0 + phi ** 2))
    main[0] = c
    main[-1] = c
    off = np.full(n - 1, -c * phi)
    return sp.diags([off, main, off], offsets=[-1, 0, 1], format="csc")


def ar1_path(n_bins: int, sigma: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate one stationary AR(1) path of length `n_bins`.

    The first bin is drawn from the stationary law: no burn-in, and no
    post-hoc rescaling of the path.
    """
    z = np.empty(n_bins)
    z[0] = rng.normal(0.0, sigma)
    innov = rng.normal(0.0, sigma * np.sqrt(1.0 - phi ** 2), size=n_bins - 1)
    for t in range(1, n_bins):
        z[t] = phi * z[t - 1] + innov[t - 1]
    return z


def ar1_logdet_precision(n: int, sigma: float, phi: float) -> float:
    """Return log det Q = -n log sigma^2 - (n-1) log(1 - phi^2)."""
    return -n * np.log(sigma ** 2) - (n - 1) * np.log1p(-phi ** 2)


def half_life_to_phi(tau_bins: float) -> float:
    return float(2.0 ** (-1.0 / max(tau_bins, 1e-6)))


def phi_to_half_life(phi: float) -> float:
    return float(-1.0 / np.log2(max(min(phi, 1 - 1e-12), 1e-12)))


def sparse_logdet(mat: sp.spmatrix) -> float:
    lu = spla.splu(sp.csc_matrix(mat), permc_spec="COLAMD", diag_pivot_thresh=0.0)
    return float(np.sum(np.log(np.abs(lu.U.diagonal()))))
