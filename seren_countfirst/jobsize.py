"""Job sizes in work units: the A100 inversion, its fitted law, and its pool."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

from .models.base import FitReport

# A100 latency coefficients at the trace's hardware
A100_ALPHA_T = 0.0050
A100_BETA_T = 0.038
A100_GAMMA_T = 0.0014
# AcmeTrace ran at the default (max) clock, so the inversion is at f = 1 GHz
F_MAX = 1.0

# one component must not collapse onto a single point
_VAR_FLOOR = 1e-8

# location pinned at zero throughout, so every AIC compares laws on (0, inf)
SCIPY_CANDIDATES: Dict[str, Tuple[stats.rv_continuous, int]] = {
    "exponential": (stats.expon, 1),
    "gamma": (stats.gamma, 2),
    "weibull": (stats.weibull_min, 2),
    "lognormal": (stats.lognorm, 2),
    "gen_gamma": (stats.gengamma, 3),
    "loglogistic": (stats.fisk, 2),
    "burr12": (stats.burr12, 3),
    "invgauss": (stats.invgauss, 2),
    "lomax": (stats.lomax, 2),
}


def step_time_s(n_gpus: np.ndarray | float, f_ghz: float = F_MAX,
                alpha_t: float = A100_ALPHA_T, beta_t: float = A100_BETA_T,
                gamma_t: float = A100_GAMMA_T) -> np.ndarray:
    """Return seconds per work unit on `n_gpus` at `f_ghz`.

    The multi-GPU branch adds `gamma_t * n` before dividing, so speedup is
    sublinear and bounded.
    """
    n = np.maximum(1.0, np.asarray(n_gpus, dtype=float))
    f = max(float(f_ghz), 1e-9)
    single = alpha_t + beta_t / f
    multi = (alpha_t + beta_t / f + gamma_t * n) / n
    return np.where(n == 1.0, single, multi)


def work_units(duration_s: np.ndarray, n_gpus: np.ndarray,
               f_ghz: float = F_MAX) -> np.ndarray:
    """Invert a wall-clock duration into work units at the trace's operating point.

    Exact inverse of `step_time_s`; `jobsize_table` reports the round trip.
    """
    return np.asarray(duration_s, dtype=float) / step_time_s(n_gpus, f_ghz)


def jobsize_table(view: pd.DataFrame, f_ghz: float = F_MAX) -> pd.DataFrame:
    """Return work units for every job in the view with a usable duration and GPU count.

    Expects `jobsize_view` output and preserves its submission order, so replay
    stays meaningful.
    """
    usable = view.loc[(view["duration"] > 0) & (view["gpu_num"] > 0)].copy()
    usable["job_size"] = work_units(usable["duration"].to_numpy(),
                                    usable["gpu_num"].to_numpy(), f_ghz)
    return usable.reset_index(drop=True)


def roundtrip_error_s(table: pd.DataFrame, f_ghz: float = F_MAX) -> float:
    """Return the largest absolute second the inversion fails to give back."""
    back = table["job_size"].to_numpy() * step_time_s(table["gpu_num"].to_numpy(), f_ghz)
    return float(np.max(np.abs(back - table["duration"].to_numpy())))


@dataclass
class LogNormalMixture:
    """Sizes whose logarithm is a k-component Gaussian mixture.

    Components are stored sorted by `mu`, so equivalent fits serialise identically.
    """

    weights: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    report: Optional[FitReport] = None
    name: str = "lognormal_mixture"

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        self.mu = np.asarray(self.mu, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        self._frozen = False

    @property
    def n_components(self) -> int:
        return len(self.weights)

    @property
    def n_params(self) -> int:
        # k-1 free weights + k means + k variances
        return 3 * self.n_components - 1

    @staticmethod
    def _component_logpdf(z: np.ndarray, weights: np.ndarray, mu: np.ndarray,
                          sigma: np.ndarray) -> np.ndarray:
        """Return log w + log N(z; mu, sigma) per component, shaped (n_obs, k)."""
        return (np.log(weights)[None, :]
                - np.log(sigma)[None, :]
                - 0.5 * np.log(2.0 * np.pi)
                - 0.5 * ((z[:, None] - mu[None, :]) / sigma[None, :]) ** 2)

    @classmethod
    def _em_once(cls, z: np.ndarray, k: int, rng: np.random.Generator,
                 max_iter: int, tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
        """Run EM once from a random start; return params, loglik, iterations, last gain."""
        n = len(z)
        # means seeded from distinct observations, keeping starts inside the support
        mu = np.sort(rng.choice(z, size=k, replace=False))
        sigma = np.full(k, max(float(z.std(ddof=0)), 1e-3))
        weights = np.full(k, 1.0 / k)
        ll_prev, ll, gain, it = -np.inf, -np.inf, np.inf, 0
        for it in range(1, max_iter + 1):
            comp = cls._component_logpdf(z, weights, mu, sigma)
            total = logsumexp(comp, axis=1)
            ll = float(total.sum())
            gain = ll - ll_prev
            if gain < tol * max(abs(ll), 1.0):
                break
            ll_prev = ll
            resp = np.exp(comp - total[:, None])
            nk = resp.sum(axis=0) + 1e-12
            weights = nk / n
            mu = (resp * z[:, None]).sum(axis=0) / nk
            var = (resp * (z[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk
            sigma = np.sqrt(np.maximum(var, _VAR_FLOOR))
        return weights, mu, sigma, ll, it, gain

    @classmethod
    def fit(cls, sizes: np.ndarray, n_components: int = 3, n_init: int = 10,
            max_iter: int = 500, tol: float = 1e-10,
            seed: int = 42) -> "LogNormalMixture":
        """Fit by EM on log sizes, keeping the best of `n_init` random starts.

        EM converges only to a local optimum; `report.restarts` records the
        number of starts used.
        """
        x = _positive(sizes)
        z = np.log(x)
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(n_init):
            trial = cls._em_once(z, n_components, rng, max_iter, tol)
            if best is None or trial[3] > best[3]:
                best = trial
        weights, mu, sigma, _, n_iter, gain = best
        order = np.argsort(mu)
        model = cls(weights=weights[order], mu=mu[order], sigma=sigma[order])
        # loglik of x, not of log x: AIC must compare with the single-family rows
        model.report = FitReport(
            converged=n_iter < max_iter,
            n_iter=n_iter,
            # EM has no gradient; the convergence measure is the last loglik gain
            grad_norm=float(gain),
            loglik=float(model.logpdf(x).sum()),
            n_params=model.n_params,
            n_obs=len(x),
            restarts=n_init,
        )
        return model

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Return the density of the size itself: log-space mixture, Jacobian included."""
        x = np.asarray(x, dtype=float)
        z = np.log(np.maximum(x, np.finfo(float).tiny))
        return logsumexp(self._component_logpdf(z, self.weights, self.mu, self.sigma),
                         axis=1) - np.log(np.maximum(x, np.finfo(float).tiny))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(x))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = np.log(np.maximum(x, np.finfo(float).tiny))
        total = np.zeros_like(z, dtype=float)
        for w, m, s in zip(self.weights, self.mu, self.sigma):
            total += w * stats.norm.cdf(z, loc=m, scale=s)
        return np.clip(total, 0.0, 1.0)

    def rvs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` work units."""
        which = rng.choice(self.n_components, size=n, p=self.weights)
        return np.exp(rng.normal(loc=self.mu[which], scale=self.sigma[which]))

    def freeze(self) -> "LogNormalMixture":
        self._frozen = True
        return self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_components": self.n_components,
            "weights": self.weights.tolist(),
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "fit": self.report.to_dict() if self.report else None,
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "LogNormalMixture":
        return cls(weights=np.asarray(blob["weights"], dtype=float),
                   mu=np.asarray(blob["mu"], dtype=float),
                   sigma=np.asarray(blob["sigma"], dtype=float))


@dataclass
class ScipyJobSize:
    """A fitted single-family law, so the scoreboard's winner is always usable."""

    name: str
    params: Tuple[float, ...]

    @property
    def dist(self) -> stats.rv_continuous:
        return SCIPY_CANDIDATES[self.name][0]

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self.dist.pdf(x, *self.params)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self.dist.cdf(x, *self.params)

    def rvs(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.dist.rvs(*self.params, size=n, random_state=rng)

    def to_dict(self) -> dict:
        return {"name": self.name, "params": list(map(float, self.params))}


@dataclass
class EmpiricalJobSize:
    """The inverted work units, each paired with the GPU count it ran on.

    `replay` returns them in submission order; `rvs` bootstraps. The paired GPU
    count is advisory: work units are invariant to the scheduler's choice of n.
    """

    sizes: np.ndarray
    gpus: np.ndarray
    name: str = "empirical"

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> "EmpiricalJobSize":
        return cls(sizes=table["job_size"].to_numpy(dtype=float),
                   gpus=table["gpu_num"].to_numpy(dtype=float))

    def replay(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return work units in submission order, paired with their trace GPU count."""
        return self.sizes, self.gpus

    def rvs(self, n: int, rng: np.random.Generator,
            with_gpu: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(self.sizes), size=n)
        if with_gpu:
            return self.sizes[idx], self.gpus[idx]
        return self.sizes[idx]

    def to_npz(self, path: str | Path) -> None:
        np.savez_compressed(Path(path), sizes=self.sizes, gpus=self.gpus)

    @classmethod
    def from_npz(cls, path: str | Path) -> "EmpiricalJobSize":
        blob = np.load(Path(path))
        return cls(sizes=blob["sizes"], gpus=blob["gpus"])


def _positive(sizes: np.ndarray) -> np.ndarray:
    """Keep the finite, strictly positive sizes; every candidate lives on (0, inf)."""
    x = np.asarray(sizes, dtype=float)
    return x[np.isfinite(x) & (x > 0.0)]


def fit_jobsize(sizes: np.ndarray, mixture_components: Sequence[int] = (2, 3),
                n_init: int = 10, seed: int = 42) -> pd.DataFrame:
    """Score every candidate law on the work units, best AIC first.

    KS p-values are reported but do not rank: at this sample size every
    candidate is rejected at any conventional level. AIC orders the table.
    """
    x = _positive(sizes)
    n_obs = len(x)
    rows = []
    for name, (dist, k) in SCIPY_CANDIDATES.items():
        try:
            params = dist.fit(x, floc=0)
            logpdf = dist.logpdf(x, *params)
            if not np.all(np.isfinite(logpdf)):
                continue
            ll = float(np.sum(logpdf))
            ks = stats.kstest(x, dist.cdf, args=params)
            rows.append({"distribution": name, "kind": "scipy", "n_params": k,
                         "loglik": ll, "AIC": 2 * k - 2 * ll,
                         "BIC": k * np.log(n_obs) - 2 * ll,
                         "KS_stat": float(ks.statistic), "KS_pvalue": float(ks.pvalue),
                         "params": tuple(float(p) for p in params)})
        except Exception as exc:  # an unfittable family is dropped, not fatal
            print(f"[skip] {name}: {exc}")
    for k_comp in mixture_components:
        model = LogNormalMixture.fit(x, n_components=k_comp, n_init=n_init, seed=seed)
        report = model.report
        assert report is not None  # attached by fit
        ks = stats.kstest(x, model.cdf)
        rows.append({"distribution": f"mix_lognormal_{k_comp}", "kind": "mixture_lognormal",
                     "n_params": model.n_params, "loglik": report.loglik, "AIC": report.aic,
                     "BIC": report.bic, "KS_stat": float(ks.statistic),
                     "KS_pvalue": float(ks.pvalue), "params": model.to_dict()})
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def write_jobsize_model(model: LogNormalMixture | ScipyJobSize, path: str | Path) -> None:
    """Write the frozen law to `path` as json."""
    Path(path).write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def read_jobsize_model(path: str | Path) -> LogNormalMixture:
    return LogNormalMixture.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
