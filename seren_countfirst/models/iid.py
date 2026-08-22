from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .base import CountModel, FitReport
from .nb import nb2_loglik, nb2_sample


class IIDNegBin(CountModel):
    name = "iid_nb"
    rung = 2

    def fit(self, counts: np.ndarray, **kwargs) -> "IIDNegBin":
        self._check_mutable()
        n = np.asarray(counts, dtype=float)

        def nll(theta):
            mu, log_kappa = theta
            lam = np.exp(np.clip(mu, -30, 30))
            return -nb2_loglik(n, np.full_like(n, lam), np.exp(np.clip(log_kappa, -20, 20)))

        best = None
        for start_kappa in (0.05, 0.5, 5.0):
            res = minimize(nll, x0=[np.log(max(n.mean(), 1e-6)), np.log(start_kappa)],
                           method="Nelder-Mead",
                           options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
            if best is None or res.fun < best.fun:
                best = res
        mu, log_kappa = best.x
        self.params_ = {"mu": float(mu), "kappa": float(np.exp(log_kappa)), "mean": float(np.exp(mu))}
        self.report = FitReport(bool(best.success), int(best.nit),
                                float(np.linalg.norm(best.final_simplex[1] - best.fun)),
                                float(-best.fun), 2, len(n), restarts=3)
        return self

    def simulate(self, n_bins: int, rng: np.random.Generator, alpha: float = 1.0,
                 kappa_rule: str = "proportional", **kwargs) -> np.ndarray:
        lam = np.full(n_bins, alpha * self.params_["mean"])
        return nb2_sample(lam, self.scaled_kappa(alpha, kappa_rule), rng).astype(float)
