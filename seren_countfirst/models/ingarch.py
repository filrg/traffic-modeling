"""Negative-binomial INGARCH(1,1): the observation-driven benchmark, not a rung."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from ..metrics import HOURLY_ACF_LAG_WEIGHTS, hourly_acf1_from_autocov
from .base import CountModel, FitReport
from .nb import nb2_loglik

_EPS = 1e-12


def b_max(s: float, kappa: float) -> float:
    """Return the largest feedback weight with a finite stationary variance, given a + b = s.

    Since `1 - a^2 - 2ab = 1 - s^2 + b^2`, the usual condition
    `(1 + 1/kappa) b^2 < 1 - a^2 - 2ab` reduces to `b^2 < kappa (1 - s^2)`.
    """
    return float(min(s, np.sqrt(max(kappa * (1.0 - s * s), 0.0))))


def _unpack(theta: np.ndarray) -> tuple:
    """Return (d, a, b, kappa) from the unconstrained vector.

    The parameterisation is the admissible region itself: `a + b = expit(.)` is
    below 1 and `b` is a fraction of `b_max`, so every reachable point is
    second-order stationary with `a, b >= 0`. Unconstrained, the MLE on this
    trace leaves the region (a + b = 0.995, no finite variance) and the graded
    moments become undefined.
    """
    d = float(np.exp(np.clip(theta[0], -30.0, 30.0)))
    s = float(expit(theta[1]))                       # a + b, the ACF decay rate
    frac = float(expit(theta[2]))                    # b as a fraction of b_max
    kappa = float(np.exp(np.clip(theta[3], -20.0, 20.0)))
    b = (1.0 - 1e-6) * b_max(s, kappa) * frac
    return d, s - b, b, kappa


def _recursion(n: np.ndarray, d: float, a: float, b: float, lam0: float) -> np.ndarray:
    """Return the conditional means lambda_1..lambda_T given the observed counts."""
    T = len(n)
    lam = np.empty(T)
    lam[0] = lam0
    for t in range(1, T):
        lam[t] = d + a * lam[t - 1] + b * n[t - 1]
    return lam


class NBINGARCH(CountModel):
    """NB-INGARCH(1,1): observation-driven conditional mean, NB2 emission.

        lambda_t = d + a lambda_{t-1} + b N_{t-1},  N_t | F_{t-1} ~ NB2(lambda_t, kappa)

    With no latent state the likelihood is exact, unlike the Laplace
    approximation the latent rungs need. Two structural limits follow: the ACF
    is a single geometric `(a+b)^h`, and the feedback term vanishes whenever
    `N_{t-1} = 0`, so quiet stretches pull the conditional mean to `d/(1-a)`.
    Zero inflation is not fitted here.
    """

    name = "nb_ingarch"
    rung = None  # not a rung: the observation-driven external baseline
    role = "benchmark"

    @staticmethod
    def _moments(d: float, a: float, b: float, kappa: float) -> tuple:
        """Return the stationary (mean, variance); variance is inf where it does not exist.

        Var(lambda) = b^2 Var(N) / (1 - a^2 - 2ab) and
        Var(N) = mean + (Var(lambda) + mean^2)/kappa + Var(lambda), so the
        second moment exists only while `(1 + 1/kappa) b^2 < 1 - a^2 - 2ab`.
        """
        mean = d / max(1.0 - a - b, _EPS)
        denom = 1.0 - a * a - 2.0 * a * b
        if denom <= 0.0:
            return mean, np.inf
        shrink = 1.0 - (1.0 + 1.0 / kappa) * b * b / denom
        if shrink <= 0.0:
            return mean, np.inf
        var = (mean + mean * mean / kappa) / shrink
        return mean, var

    @staticmethod
    def _var_lambda(b: float, s: float, var: float) -> float:
        """Return Var(lambda_t), the share of the count variance carried by the recursion."""
        denom = 1.0 - s * s + b * b
        return b * b * var / denom if (np.isfinite(var) and denom > 0.0) else np.inf

    def _analytic(self, theta: np.ndarray) -> tuple:
        """Return the (mean, IDC, rho_1, rho_4) implied by one unconstrained vector."""
        d, a, b, kappa = _unpack(theta)
        s = a + b
        mean, var = self._moments(d, a, b, kappa)
        if not np.isfinite(var) or var <= 0.0 or mean <= 0.0:
            return mean, np.inf, np.nan, np.nan
        var_lam = self._var_lambda(b, s, var)
        rho1 = (a * var_lam + b * var) / var
        return mean, var / mean, rho1, rho1 * s ** 3

    @classmethod
    def calibrate(cls, mean: float, variance: float, acf_lag1: float, acf_lag4: float,
                  n_restarts: int = 4, seed: int = 0) -> "NBINGARCH":
        """Match the analytic (mean, IDC, ACF lag 1, ACF lag 4); no likelihood is evaluated.

        Matches the frozen candidate's moment calibration, so the comparison is
        like for like. `fit` maximises the conditional likelihood, which is
        indifferent to whether the graded stationary variance exists, and on this
        trace it reaches that boundary. Four targets against four free parameters,
        solved inside the admissible region; residuals are kept in `params_`
        because an exact solution is not guaranteed.
        """
        model = cls()
        targets = np.array([float(mean), float(variance) / float(mean),
                            float(acf_lag1), float(acf_lag4)])
        rng = np.random.default_rng(seed)

        def loss(theta):
            m, dispersion, rho1, rho4 = model._analytic(np.asarray(theta, dtype=float))
            if not np.isfinite(dispersion) or not np.isfinite(rho1):
                return 1e12
            return float(
                np.log(m / targets[0]) ** 2
                + np.log(dispersion / targets[1]) ** 2
                + (rho1 - targets[2]) ** 2
                + (rho4 - targets[3]) ** 2
            )

        grid = [np.array([np.log(mean * 0.3), logit(0.7), logit(0.9), np.log(1.0)]),
                np.array([np.log(mean * 0.1), logit(0.9), logit(0.99), np.log(0.5)]),
                np.array([np.log(mean * 0.5), logit(0.5), logit(0.5), np.log(2.0)])]
        starts = grid + [grid[0] + rng.normal(0.0, 0.4, 4)
                         for _ in range(max(n_restarts - len(grid), 0))]

        best = None
        for theta0 in starts[:max(n_restarts, 1)]:
            res = minimize(loss, theta0, method="Nelder-Mead",
                           options={"maxiter": 6000, "xatol": 1e-9, "fatol": 1e-12})
            if best is None or res.fun < best.fun:
                best = res

        d, a, b, kappa = _unpack(best.x)
        model._store(d, a, b, kappa, len(targets))
        got = np.array(model._analytic(best.x))
        model.params_.update({
            "calibration_loss": float(best.fun),
            "resid_mean": float(got[0] / targets[0] - 1.0),
            "resid_IDC": float(got[1] / targets[1] - 1.0),
            "resid_acf_lag1": float(got[2] - targets[2]),
            "resid_acf_lag4": float(got[3] - targets[3]),
        })
        # moment-calibrated: `report` stays None, so there is no AIC to quote
        return model

    def _store(self, d: float, a: float, b: float, kappa: float, n_params: int) -> None:
        s = a + b
        mean, var = self._moments(d, a, b, kappa)
        self.params_ = {
            "d": d, "a": a, "b": b, "kappa": kappa,
            "persistence": float(s),
            "b_over_b_max": float(b / max(b_max(s, kappa), _EPS)),
            "half_life_bins": float(np.log(0.5) / np.log(min(s, 1.0 - 1e-12))),
            "mean": float(mean),
            "variance": float(var),
            "n_params": int(n_params),
        }

    def _loglik(self, n: np.ndarray, theta: np.ndarray) -> float:
        d, a, b, kappa = _unpack(theta)
        mean, _ = self._moments(d, a, b, kappa)
        lam = _recursion(n, d, a, b, lam0=mean)
        if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
            return -np.inf
        return float(nb2_loglik(n, lam, kappa))

    def fit(self, counts: np.ndarray, n_restarts: int = 3, maxiter: int = 400,
            seed: int = 0, **kwargs) -> "NBINGARCH":
        """Fit by exact conditional MLE, the first bin conditioned on the stationary mean.

        The recursion is observed, so this is the true likelihood and its AIC is
        directly comparable with rungs 2-4.
        """
        self._check_mutable()
        n = np.asarray(counts, dtype=float)
        m = max(float(n.mean()), 1e-6)
        rng = np.random.default_rng(seed)

        def start(s: float, frac: float, kappa: float) -> np.ndarray:
            return np.array([np.log(m * (1.0 - s)), logit(s), logit(frac), np.log(kappa)])

        # starts spread over persistence and over the feedback share of it
        grid = [start(0.95, 0.5, 0.3), start(0.99, 0.9, 0.2), start(0.8, 0.3, 1.0)]
        starts = grid[:max(n_restarts, 1)]
        starts += [grid[0] + rng.normal(0.0, 0.3, 4) for _ in range(max(n_restarts, 1) - len(starts))]

        def nll(theta):
            ll = self._loglik(n, np.asarray(theta, dtype=float))
            return -ll if np.isfinite(ll) else 1e12

        best, best_res = np.inf, None
        for theta0 in starts:
            res = minimize(nll, theta0, method="Nelder-Mead",
                           options={"maxiter": maxiter * 4, "xatol": 1e-7, "fatol": 1e-7})
            if res.fun < best:
                best, best_res = res.fun, res

        d, a, b, kappa = _unpack(best_res.x)
        self._store(d, a, b, kappa, 4)
        p = self.params_

        warnings: List[str] = []
        if p["persistence"] > 0.999:
            warnings.append("a + b at the unit root: the persistence is not stationary in practice")
        if p["b_over_b_max"] > 0.99:
            warnings.append("b pinned at the second-moment boundary b^2 = kappa (1 - (a+b)^2): "
                            "the unconstrained likelihood implies no stationary variance; "
                            "report the moment calibration instead")
        if b < 1e-3:
            warnings.append("b near zero: feedback inert, lambda constant, "
                            "the model collapses to iid NB")

        self.report = FitReport(bool(best_res.success), int(best_res.nit), 0.0, float(-best),
                                4, len(n), warnings, restarts=len(starts))
        return self

    def implied_marginals(self) -> pd.Series:
        p = self.params_
        d, a, b, kappa = p["d"], p["a"], p["b"], p["kappa"]
        s = a + b
        mean, var = self._moments(d, a, b, kappa)
        var_lam = self._var_lambda(b, s, var)
        # rho(h) = (a Var(lambda) + b Var(N)) (a+b)^{h-1} / Var(N)
        rho1 = (a * var_lam + b * var) / var if np.isfinite(var_lam) else np.nan
        gamma = {h: rho1 * s ** (h - 1) * var for h in HOURLY_ACF_LAG_WEIGHTS}
        return pd.Series(
            {
                "mean": float(mean),
                "variance": float(var),
                "IDC_base": float(var / max(mean, _EPS)),
                # same footing as the gated rungs, so the table rows compare
                "acf_1h_lag1": float(hourly_acf1_from_autocov(var, gamma)),
                "acf_lag1": float(rho1),
                "acf_lag4": float(rho1 * s ** 3),
                "acf_lag96": float(rho1 * s ** 95),
                "persistence": float(s),
            }
        )

    def simulate(self, n_bins: int, rng: np.random.Generator, alpha: float = 1.0,
                 kappa_rule: str = "proportional", return_intensity: bool = False, **kwargs):
        """Draw a path; load scaling enters through the intercept, so the mean scales by alpha.

        The recursion reads back its own counts and cannot be vectorised over
        time; only the gamma mixing variates are drawn in one go.
        """
        p = self.params_
        d, a, b = alpha * p["d"], p["a"], p["b"]
        kappa = self.scaled_kappa(alpha, kappa_rule)
        g = rng.gamma(shape=kappa, scale=1.0 / kappa, size=n_bins)
        poisson = rng.poisson
        lam = np.empty(n_bins)
        counts = np.empty(n_bins)
        lam[0] = alpha * p["mean"]  # stationary initialisation, never burned in
        for t in range(n_bins):
            if t:
                lam[t] = d + a * lam[t - 1] + b * counts[t - 1]
            counts[t] = poisson(max(lam[t], _EPS) * g[t])
        return (counts, lam) if return_intensity else counts
