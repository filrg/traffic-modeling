from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp, ndtr, ndtri

from ..artifacts import write_json
from ..models.base import FitReport

if TYPE_CHECKING:
    from .conditional import ConditionalJobSize
    from .two_part import TwoPartJobSize


# one component must not collapse onto a single point
_VAR_FLOOR = 1e-8

# location pinned at zero throughout, so every AIC compares laws on (0, inf)
SCIPY_CANDIDATES: dict[str, tuple[stats.rv_continuous, int]] = {
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


@dataclass
class LogNormalMixture:
    """Sizes whose logarithm is a k-component Gaussian mixture.

    Components are stored sorted by `mu`, so equivalent fits serialise identically.
    """

    weights: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    report: FitReport | None = None
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
    def _component_logpdf(
        z: np.ndarray, weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        """Return log w + log N(z; mu, sigma) per component, shaped (n_obs, k)."""
        return (
            np.log(weights)[None, :]
            - np.log(sigma)[None, :]
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * ((z[:, None] - mu[None, :]) / sigma[None, :]) ** 2
        )

    @classmethod
    def _em_once(
        cls, z: np.ndarray, k: int, rng: np.random.Generator, max_iter: int, tol: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
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
    def _best_of(
        cls, z: np.ndarray, k: int, n_init: int, max_iter: int, tol: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        """Return the best of `n_init` EM runs on log values, sorted by component mean."""
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(n_init):
            trial = cls._em_once(z, k, rng, max_iter, tol)
            if best is None or trial[3] > best[3]:
                best = trial
        if best is None:
            raise ValueError("n_init must be positive")
        weights, mu, sigma, _, n_iter, gain = best
        order = np.argsort(mu)
        return weights[order], mu[order], sigma[order], n_iter, gain

    @classmethod
    def fit(
        cls,
        sizes: np.ndarray,
        n_components: int = 3,
        n_init: int = 10,
        max_iter: int = 500,
        tol: float = 1e-10,
        seed: int = 42,
    ) -> "LogNormalMixture":
        """Fit by EM on log sizes, keeping the best of `n_init` random starts.

        EM converges only to a local optimum; `report.restarts` records the
        number of starts used.
        """
        x = _positive(sizes)
        weights, mu, sigma, n_iter, gain = cls._best_of(
            np.log(x), n_components, n_init, max_iter, tol, seed
        )
        model = cls(weights=weights, mu=mu, sigma=sigma)
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

    @classmethod
    def fit_log(
        cls,
        z: np.ndarray,
        n_components: int = 3,
        n_init: int = 10,
        max_iter: int = 500,
        tol: float = 1e-10,
        seed: int = 42,
    ) -> "LogNormalMixture":
        """Fit values that are already logarithms; the report's loglik is on that scale.

        For residuals, which carry no Jacobian and may be negative. AIC from this
        report ranks component counts on one residual set, nothing wider.
        """
        z = np.asarray(z, dtype=float)
        z = z[np.isfinite(z)]
        weights, mu, sigma, n_iter, gain = cls._best_of(
            z, n_components, n_init, max_iter, tol, seed
        )
        model = cls(weights=weights, mu=mu, sigma=sigma)
        model.report = FitReport(
            converged=n_iter < max_iter,
            n_iter=n_iter,
            grad_norm=float(gain),
            loglik=float(logsumexp(cls._component_logpdf(z, weights, mu, sigma), axis=1).sum()),
            n_params=model.n_params,
            n_obs=len(z),
            restarts=n_init,
        )
        return model

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Return the density of the size itself: log-space mixture, Jacobian included."""
        x = np.asarray(x, dtype=float)
        z = np.log(np.maximum(x, np.finfo(float).tiny))
        return logsumexp(
            self._component_logpdf(z, self.weights, self.mu, self.sigma), axis=1
        ) - np.log(np.maximum(x, np.finfo(float).tiny))

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
        report = FitReport.from_dict(blob["fit"]) if blob.get("fit") else None
        return cls(
            weights=np.asarray(blob["weights"], dtype=float),
            mu=np.asarray(blob["mu"], dtype=float),
            sigma=np.asarray(blob["sigma"], dtype=float),
            report=report,
        )


@dataclass
class ScipyJobSize:
    """A fitted single-family law, so the scoreboard's winner is always usable."""

    name: str
    params: tuple[float, ...]

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
        return cls(
            sizes=table["job_size"].to_numpy(dtype=float),
            gpus=table["gpu_count"].to_numpy(dtype=float),
        )

    def replay(self) -> tuple[np.ndarray, np.ndarray]:
        """Return work units in submission order, paired with their trace GPU count."""
        return self.sizes, self.gpus

    def rvs(
        self, n: int, rng: np.random.Generator, with_gpu: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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


def _normal_below(bound: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Draw N(0, sigma^2) conditioned to fall below `bound`, elementwise."""
    mass = np.maximum(ndtr(bound / sigma), 1e-12)
    return sigma * ndtri(rng.random(len(bound)) * mass)


def _lognormal_mean_below(
    level: np.ndarray, log_cut: float, sigma: float | np.ndarray
) -> np.ndarray:
    """Return E[exp(level + eta)] for eta ~ N(0, sigma^2) conditioned below the cut.

    Capped at the cut, which is the exact limit as the level runs away above it.
    """
    h = (log_cut - level) / sigma
    mean = np.exp(level + 0.5 * sigma**2) * ndtr(h - sigma) / np.maximum(ndtr(h), 1e-300)
    return np.minimum(mean, np.exp(log_cut))


def _positive(sizes: np.ndarray) -> np.ndarray:
    """Keep the finite, strictly positive sizes; every candidate lives on (0, inf)."""
    x = np.asarray(sizes, dtype=float)
    return x[np.isfinite(x) & (x > 0.0)]


def fit_jobsize(
    sizes: np.ndarray, mixture_components: Sequence[int] = (2, 3), n_init: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Score every candidate law on the work units, best AIC first.

    KS p-values are reported but do not rank; AIC orders the table.
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
            rows.append(
                {
                    "distribution": name,
                    "kind": "scipy",
                    "n_params": k,
                    "loglik": ll,
                    "AIC": 2 * k - 2 * ll,
                    "BIC": k * np.log(n_obs) - 2 * ll,
                    "KS_stat": float(ks.statistic),
                    "KS_pvalue": float(ks.pvalue),
                    "params": tuple(float(p) for p in params),
                }
            )
        except Exception as exc:  # an unfittable family is dropped, not fatal
            print(f"[skip] {name}: {exc}")
    for k_comp in mixture_components:
        model = LogNormalMixture.fit(x, n_components=k_comp, n_init=n_init, seed=seed)
        report = model.report
        assert report is not None  # attached by fit
        ks = stats.kstest(x, model.cdf)
        rows.append(
            {
                "distribution": f"mix_lognormal_{k_comp}",
                "kind": "mixture_lognormal",
                "n_params": model.n_params,
                "loglik": report.loglik,
                "AIC": report.aic,
                "BIC": report.bic,
                "KS_stat": float(ks.statistic),
                "KS_pvalue": float(ks.pvalue),
                "params": model.to_dict(),
            }
        )
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def write_jobsize_model(
    model: LogNormalMixture | ScipyJobSize | ConditionalJobSize | TwoPartJobSize,
    path: str | Path,
) -> None:
    """Write the frozen law to `path` as json."""
    write_json(model.to_dict(), path)


def read_jobsize_model(path: str | Path) -> LogNormalMixture:
    return LogNormalMixture.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
