"""Job sizes in work units: the A100 inversion, its fitted law, and its pool."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import logsumexp, ndtr, ndtri

from .data import BASE_FREQ
from .models.base import FitReport

# A100 latency coefficients at the trace's hardware
A100_ALPHA_T = 0.0050
A100_BETA_T = 0.038
A100_GAMMA_T = 0.0014
# AcmeTrace ran at the default (max) clock, so the inversion is at f = 1 GHz
F_MAX = 1.0

# one component must not collapse onto a single point
_VAR_FLOOR = 1e-8

# component counts scanned when the caller does not fix one
COMPONENT_ORDERS = (1, 2, 3, 4, 5)

# work quantile above which a job is placed per bin rather than per job
TAIL_QUANTILE = 0.99

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


def attach_activity(table: pd.DataFrame, arrivals: pd.DataFrame,
                    freq: str = BASE_FREQ) -> pd.DataFrame:
    """Add each job's bin and that bin's batch count, dropping non-positive sizes.

    Activity counts batches over every submission, not only the completed ones:
    it is the quantity the count model generates, and sizes are the only view
    restricted to completions.
    """
    activity = (arrivals.assign(_bin=arrivals["submit_time_dt"].dt.floor(freq))
                .groupby("_bin")["submit_time_dt"].nunique())
    jobs = table.loc[table["job_size"] > 0.0].copy()
    jobs["bin"] = jobs["submit_time_dt"].dt.floor(freq)
    jobs["activity"] = jobs["bin"].map(activity).to_numpy(dtype=float)
    if not np.all(np.isfinite(jobs["activity"].to_numpy())):
        raise ValueError("a completed job sits in a bin the arrival view does not cover")
    return jobs.reset_index(drop=True)


def batch_frame(jobs: pd.DataFrame) -> pd.DataFrame:
    """Collapse jobs sharing a submission timestamp into one batch row."""
    grouped = jobs.assign(z=np.log(jobs["job_size"].to_numpy(dtype=float))).groupby(
        ["bin", "submit_time_dt"], sort=True)
    return grouped.agg(z_bar=("z", "mean"), cardinality=("z", "size"),
                       activity=("activity", "first")).reset_index()


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
    def _best_of(cls, z: np.ndarray, k: int, n_init: int, max_iter: int, tol: float,
                 seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        """Return the best of `n_init` EM runs on log values, sorted by component mean."""
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(n_init):
            trial = cls._em_once(z, k, rng, max_iter, tol)
            if best is None or trial[3] > best[3]:
                best = trial
        weights, mu, sigma, _, n_iter, gain = best
        order = np.argsort(mu)
        return weights[order], mu[order], sigma[order], n_iter, gain

    @classmethod
    def fit(cls, sizes: np.ndarray, n_components: int = 3, n_init: int = 10,
            max_iter: int = 500, tol: float = 1e-10,
            seed: int = 42) -> "LogNormalMixture":
        """Fit by EM on log sizes, keeping the best of `n_init` random starts.

        EM converges only to a local optimum; `report.restarts` records the
        number of starts used.
        """
        x = _positive(sizes)
        weights, mu, sigma, n_iter, gain = cls._best_of(
            np.log(x), n_components, n_init, max_iter, tol, seed)
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
    def fit_log(cls, z: np.ndarray, n_components: int = 3, n_init: int = 10,
                max_iter: int = 500, tol: float = 1e-10,
                seed: int = 42) -> "LogNormalMixture":
        """Fit values that are already logarithms; the report's loglik is on that scale.

        For residuals, which carry no Jacobian and may be negative. AIC from this
        report ranks component counts on one residual set, nothing wider.
        """
        z = np.asarray(z, dtype=float)
        z = z[np.isfinite(z)]
        weights, mu, sigma, n_iter, gain = cls._best_of(
            z, n_components, n_init, max_iter, tol, seed)
        model = cls(weights=weights, mu=mu, sigma=sigma)
        model.report = FitReport(
            converged=n_iter < max_iter,
            n_iter=n_iter,
            grad_norm=float(gain),
            loglik=float(logsumexp(
                cls._component_logpdf(z, weights, mu, sigma), axis=1).sum()),
            n_params=model.n_params,
            n_obs=len(z),
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


def within_batch_variance(jobs: pd.DataFrame, beta: float,
                          mix: LogNormalMixture) -> np.ndarray:
    """Return the within-batch variance carried by each mixture component.

    Batches are assigned to components by the responsibility of their mean
    residual, which is where the model puts the component: it is a property of
    the submitted workload, not of the individual job. Components holding no
    multi-job batch fall back to the pooled variance.
    """
    z = np.log(jobs["job_size"].to_numpy(dtype=float))
    key = ["bin", "submit_time_dt"]
    grouped = jobs.assign(z=z).groupby(key, sort=True)
    dev2 = (z - grouped["z"].transform("mean").to_numpy()) ** 2
    frame = jobs.assign(z=z, _d=dev2).groupby(key, sort=True).agg(
        ss=("_d", "sum"), card=("z", "size"), zbar=("z", "mean"),
        activity=("activity", "first"))
    resid = frame["zbar"].to_numpy() - beta * np.log(frame["activity"].to_numpy(dtype=float))
    comp = mix._component_logpdf(resid, mix.weights, mix.mu, mix.sigma)
    resp = np.exp(comp - logsumexp(comp, axis=1)[:, None])

    ss, card = frame["ss"].to_numpy(), frame["card"].to_numpy(dtype=float)
    dof = resp.T @ (card - 1.0)
    pooled = float(ss.sum() / max(card.sum() - len(frame), 1.0))
    return np.where(dof > 1.0, (resp.T @ ss) / np.maximum(dof, 1e-12), pooled)


def bin_grid_index(bins, freq: str = BASE_FREQ) -> np.ndarray:
    """Return each row's slot on the regular bin grid, counted from the earliest.

    The latent lives on real time, so gaps between occupied bins have to be real
    gaps; a compacted index would make bins hours apart look adjacent.
    """
    at = pd.DatetimeIndex(bins)
    return ((at - at.min()) // pd.Timedelta(freq)).to_numpy().astype(int)


def _ar1_path(n: int, tau_bins: float, rng: np.random.Generator) -> np.ndarray:
    """Return a unit-variance AR(1) path started from its stationary distribution."""
    phi = 2.0 ** (-1.0 / max(float(tau_bins), 1e-6))
    noise = rng.standard_normal(n) * np.sqrt(max(1.0 - phi ** 2, 0.0))
    out = np.empty(n)
    out[0] = rng.standard_normal()
    for t in range(1, n):
        out[t] = phi * out[t - 1] + noise[t]
    return out


@dataclass
class SizeLatent:
    """Two-scale stationary AR(1) level for log work, on the bin grid.

    Sizes carry memory that bin activity does not explain, and one timescale
    decays too fast to hold it. Same two-component form as the count model's gate
    latent.
    """

    variance: float
    w_long: float
    tau_short_bins: float
    tau_long_bins: float
    name: str = "size_latent_two_ar"

    def acf(self, lags: np.ndarray) -> np.ndarray:
        h = np.asarray(lags, dtype=float)
        short = 2.0 ** (-h / max(self.tau_short_bins, 1e-6))
        long = 2.0 ** (-h / max(self.tau_long_bins, 1e-6))
        return (1.0 - self.w_long ** 2) * short + self.w_long ** 2 * long

    def simulate(self, n_bins: int, rng: np.random.Generator) -> np.ndarray:
        """Draw the level for every bin of the grid, occupied or not."""
        w = float(self.w_long)
        path = (np.sqrt(max(1.0 - w ** 2, 0.0)) * _ar1_path(n_bins, self.tau_short_bins, rng)
                + w * _ar1_path(n_bins, self.tau_long_bins, rng))
        return np.sqrt(max(self.variance, 0.0)) * path

    @classmethod
    def calibrate(cls, jobs: pd.DataFrame, beta: float,
                  lags: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
                  min_batches: int = 4) -> "SizeLatent":
        """Fit the level's variance and two timescales where the level is observable.

        A bin's level is the mean of its batches, so a thin bin measures it badly:
        that noise inflates the variance and attenuates every lag, and debiasing a
        pooled series under-corrects both. The estimate is taken on bins holding at
        least `min_batches`, with the residual noise removed.
        """
        batches = batch_frame(jobs)
        residual = (batches["z_bar"].to_numpy(dtype=float)
                    - beta * np.log(batches["activity"].to_numpy(dtype=float)))
        slot = bin_grid_index(batches["bin"])
        frame = pd.DataFrame({"slot": slot, "u": residual})
        level = frame.groupby("slot")["u"].mean()
        n_batches = frame.groupby("slot")["u"].size()

        # a bin's level is the mean of its batches, so its scatter shrinks with the
        # batch count, not the job count
        within = float(((frame["u"] - frame.groupby("slot")["u"].transform("mean")) ** 2).sum()
                       / max(len(frame) - len(level), 1))
        usable = n_batches.index[n_batches >= max(int(min_batches), 1)]
        if len(usable) < 50:
            usable = n_batches.index
        seen = level.loc[usable].to_numpy(dtype=float)
        noise = float(np.mean(within / n_batches.loc[usable].to_numpy(dtype=float)))
        observed_var = float(np.var(seen, ddof=1))
        variance = max(observed_var - noise, 1e-6)

        series = level.loc[usable].reindex(
            np.arange(int(slot.max()) + 1)).to_numpy(dtype=float)
        series = series - np.nanmean(series)
        target, keep = [], []
        for h in lags:
            a, b = series[:-h], series[h:]
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() < 30:
                continue
            rho = float(np.corrcoef(a[ok], b[ok])[0, 1]) * observed_var / variance
            target.append(min(max(rho, 0.0), 0.999))
            keep.append(float(h))
        if len(keep) < 2:
            raise ValueError("too few usable lags to calibrate the size latent")
        h = np.asarray(keep)
        want = np.asarray(target)

        def loss(p: np.ndarray) -> float:
            trial = cls(variance, p[0], np.exp(p[1]), np.exp(p[2]))
            return float(np.sum((trial.acf(h) - want) ** 2))

        best = optimize.minimize(loss, np.array([0.5, np.log(2.0), np.log(200.0)]),
                                 method="L-BFGS-B",
                                 bounds=[(0.0, 0.95), (np.log(0.2), np.log(8.0)),
                                         (np.log(8.0), np.log(4000.0))])
        return cls(variance=variance, w_long=float(best.x[0]),
                   tau_short_bins=float(np.exp(best.x[1])),
                   tau_long_bins=float(np.exp(best.x[2])))

    def to_dict(self) -> dict:
        return {"name": self.name, "variance": self.variance, "w_long": self.w_long,
                "tau_short_bins": self.tau_short_bins,
                "tau_long_bins": self.tau_long_bins}

    @classmethod
    def from_dict(cls, blob: dict) -> "SizeLatent":
        return cls(variance=float(blob["variance"]), w_long=float(blob["w_long"]),
                   tau_short_bins=float(blob["tau_short_bins"]),
                   tau_long_bins=float(blob["tau_long_bins"]))


def component_scoreboard(jobs: pd.DataFrame, beta: float,
                         orders: Sequence[int] = COMPONENT_ORDERS, n_init: int = 4,
                         seed: int = 42, latent_variance: float = 0.0) -> pd.DataFrame:
    """Score component counts, flagging those whose every component can be deflated.

    A component whose own within-batch spread is as wide as the component itself
    has no batch-level variance, so it is the deflation and not the AIC that
    bounds the order.
    """
    z = np.log(jobs["job_size"].to_numpy(dtype=float))
    residual = z - beta * np.log(jobs["activity"].to_numpy(dtype=float))
    rows = []
    for k in orders:
        mix = LogNormalMixture.fit_log(residual, n_components=k, n_init=n_init, seed=seed)
        slack = float(np.min(mix.sigma ** 2 - within_batch_variance(jobs, beta, mix)
                             - latent_variance))
        rows.append({"k": k, "AIC": mix.report.aic, "tightest_slack": slack,
                     "deflatable": slack > 0.0})
    return pd.DataFrame(rows)


def best_component_order(jobs: pd.DataFrame, beta: float,
                         orders: Sequence[int] = COMPONENT_ORDERS, n_init: int = 4,
                         seed: int = 42, latent_variance: float = 0.0) -> int:
    """Return the best-AIC component count that can be deflated in every component."""
    board = component_scoreboard(jobs, beta, orders, n_init, seed, latent_variance)
    usable = board.loc[board["deflatable"]]
    if usable.empty:
        raise RuntimeError(
            f"no component count in {tuple(orders)} leaves every component wider "
            f"than its own within-batch spread")
    return int(usable.loc[usable["AIC"].idxmin(), "k"])


@dataclass
class ConditionalJobSize:
    """Work units as a batch-level level plus a job-level deviation.

    `log S = mu + beta*log(activity) + eps + eta`. A batch draws one component,
    which fixes both its level `eps ~ N(m_c, s_batch_c^2)` and the spread
    `eta ~ N(0, sigma_w_c^2)` of its jobs around it. The component variances are
    stored at both levels, `s_job` being what the mixture was fitted at, so adding
    `eta` back restores it and the job marginal is not broadened.

    Because the component is a batch attribute, a batch cannot straddle two of
    them: the average within-batch spread is reproduced, its spread is not.
    """

    mu: float
    beta: float
    sigma_w: np.ndarray
    weights: np.ndarray
    m: np.ndarray
    s_batch: np.ndarray
    s_job: np.ndarray
    latent: Optional[SizeLatent] = None
    report: Optional[FitReport] = None
    name: str = "conditional_lognormal_mixture"

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        self.m = np.asarray(self.m, dtype=float)
        self.s_batch = np.asarray(self.s_batch, dtype=float)
        self.s_job = np.asarray(self.s_job, dtype=float)
        self.sigma_w = np.broadcast_to(np.asarray(self.sigma_w, dtype=float),
                                       self.weights.shape).astype(float)
        self._frozen = False

    @property
    def n_components(self) -> int:
        return len(self.weights)

    @property
    def n_params(self) -> int:
        # mu, beta, then k-1 free weights, k-1 free means, k variances and k
        # within-batch variances; the latent adds its variance and two timescales
        return 4 * self.n_components + (3 if self.latent is not None else 0)

    @classmethod
    def fit(cls, jobs: pd.DataFrame, n_components: Optional[int] = None, n_init: int = 10,
            seed: int = 42, beta: Optional[float] = None,
            level: str = "mean_work", with_latent: bool = True) -> "ConditionalJobSize":
        """Calibrate the levels in order, then place `mu`.

        No stage revisits an earlier one: the within-batch variance fixes the job
        level, least squares fixes the activity slope, and the mixture is fitted
        to what remains. Pass `beta` to pin the slope instead of estimating it.

        `level` picks what `mu` is solved against, and the two disagree whenever
        the mixture's analytic mean misses the sample mean: `mean_work` matches
        expected work per job, `log_mean` puts `mu` at the residuals' own mean so
        the fitted marginal is reproduced unshifted. Leave `n_components` unset to
        take the best-AIC order that can be deflated.
        """
        if level not in ("mean_work", "log_mean"):
            raise ValueError("level must be 'mean_work' or 'log_mean'")
        z = np.log(jobs["job_size"].to_numpy(dtype=float))
        activity = jobs["activity"].to_numpy(dtype=float)
        batches = batch_frame(jobs)
        if len(jobs) - len(batches) <= 0:
            raise ValueError("no batch holds a second job: sigma_w is not identified")

        if beta is None:
            x = np.log(batches["activity"].to_numpy(dtype=float))
            coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]),
                                       batches["z_bar"].to_numpy(), rcond=None)
            beta = float(coef[1])

        adjusted = z - beta * np.log(activity)
        latent = SizeLatent.calibrate(jobs, beta) if with_latent else None
        latent_var = latent.variance if latent is not None else 0.0
        if n_components is None:
            n_components = best_component_order(jobs, beta, n_init=n_init, seed=seed,
                                                latent_variance=latent_var)
        mix = LogNormalMixture.fit_log(adjusted, n_components=n_components,
                                       n_init=n_init, seed=seed)
        sigma_w2 = within_batch_variance(jobs, beta, mix)
        # three levels share one component variance: bin, batch, job
        slack = mix.sigma ** 2 - sigma_w2 - latent_var
        if np.any(slack <= 0.0):
            c = int(np.argmin(slack))
            raise RuntimeError(
                f"component {c} of variance {mix.sigma[c] ** 2:.4f} cannot hold a bin "
                f"level of {latent_var:.4f} and a within-batch spread of "
                f"{sigma_w2[c]:.4f}: no batch-level variance exists")

        # the centring constraint moves the mixture's level into mu, placed next
        centre = float(np.sum(mix.weights * mix.mu))
        model = cls(mu=0.0, beta=beta, sigma_w=np.sqrt(sigma_w2),
                    weights=mix.weights, m=mix.mu - centre,
                    s_batch=np.sqrt(slack), s_job=mix.sigma, latent=latent,
                    report=mix.report)
        if level == "mean_work":
            target = float(np.mean(jobs["job_size"].to_numpy(dtype=float)))
            model.mu = float(np.log(target) - np.log(model.mean_work(activity)))
        else:
            model.mu = centre
        return model

    def _eps_scale(self) -> float:
        """Return E[exp(v + eps + eta)]: the levels recombine to the fitted variance."""
        return float(np.sum(self.weights * np.exp(self.m + 0.5 * self.s_job ** 2)))

    def mean_work(self, activity: np.ndarray) -> float:
        """Return expected work per job over a population of bins.

        Exact under the model's independence of activity from both residuals.
        """
        act = np.asarray(activity, dtype=float)
        return float(np.exp(self.mu) * np.mean(act ** self.beta) * self._eps_scale())

    def _draw_batches(self, activity: np.ndarray, rng: np.random.Generator,
                      alpha: float = 1.0, bin_level: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw each batch's log work level and the component that set it."""
        act = _stream_activity(activity, alpha)
        which = rng.choice(self.n_components, size=len(act), p=self.weights)
        level = (self.mu + self.beta * np.log(act)
                 + rng.normal(self.m[which], self.s_batch[which]))
        if bin_level is not None:
            level = level + np.asarray(bin_level, dtype=float)
        return level, which

    def draw_bin_level(self, bin_index: np.ndarray,
                       rng: np.random.Generator) -> Optional[np.ndarray]:
        """Return each batch's share of the slow size level, or None without one."""
        if self.latent is None:
            return None
        slots = np.asarray(bin_index, dtype=int)
        return self.latent.simulate(int(slots.max()) + 1, rng)[slots]

    def batch_levels(self, activity: np.ndarray, rng: np.random.Generator,
                     alpha: float = 1.0) -> np.ndarray:
        """Draw the log work level shared by the jobs of each batch."""
        return self._draw_batches(activity, rng, alpha)[0]

    def rvs(self, activity: np.ndarray, cardinality: np.ndarray,
            rng: np.random.Generator, alpha: float = 1.0,
            bin_index: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw one work unit per job, batch by batch, in the order given.

        `alpha` superposes that many unit-load streams, so a batch reads its own
        stream's share of the bin rather than the aggregate: sizes then do not
        move with load, which is what the superposition reading of load requires.
        """
        if self.latent is not None and bin_index is None:
            raise ValueError("this law carries a bin level: pass bin_index to place it")
        bin_level = None if bin_index is None else self.draw_bin_level(bin_index, rng)
        level, which = self._draw_batches(activity, rng, alpha, bin_level)
        card = np.asarray(cardinality, dtype=int)
        base = np.repeat(level, card)
        return np.exp(base + rng.normal(0.0, 1.0, size=len(base))
                      * np.repeat(self.sigma_w[which], card))

    def freeze(self) -> "ConditionalJobSize":
        self._frozen = True
        return self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_components": self.n_components,
            "mu": self.mu,
            "beta": self.beta,
            "sigma_w": self.sigma_w.tolist(),
            "weights": self.weights.tolist(),
            "m": self.m.tolist(),
            "s_batch": self.s_batch.tolist(),
            "s_job": self.s_job.tolist(),
            "latent": self.latent.to_dict() if self.latent else None,
            "fit": self.report.to_dict() if self.report else None,
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "ConditionalJobSize":
        return cls(mu=float(blob["mu"]), beta=float(blob["beta"]),
                   sigma_w=np.asarray(blob["sigma_w"], dtype=float),
                   weights=np.asarray(blob["weights"], dtype=float),
                   m=np.asarray(blob["m"], dtype=float),
                   s_batch=np.asarray(blob["s_batch"], dtype=float),
                   s_job=np.asarray(blob["s_job"], dtype=float),
                   latent=(SizeLatent.from_dict(blob["latent"])
                           if blob.get("latent") else None))


def _stream_activity(activity: np.ndarray, alpha: float) -> np.ndarray:
    """Return the occupancy one unit-load stream sees inside a bin of `activity`.

    The mean share is `activity / alpha`, but a stream that contributed a batch
    holds at least one, so the floor is what a sparse bin gives. The two limits
    are exact and `alpha = 1` is the identity.
    """
    return np.maximum(np.asarray(activity, dtype=float) / float(alpha), 1.0)


@dataclass
class CompositionJobSize:
    """Sizes as a mixture whose weights move with activity, with the component per job.

    A batch draws a component and a shared offset; each job takes that component
    unless it escapes, which is what lets one submission hold jobs of different
    kinds.

    Activity enters through the weights rather than the level: a busy bin carries
    a different mix of job kinds, not uniformly smaller jobs, so the effect is
    bounded and saturating.
    """

    m: np.ndarray
    s_job: np.ndarray
    s_batch: np.ndarray
    logit_a: np.ndarray
    logit_b: np.ndarray
    escape: float
    mu: float = 0.0
    name: str = "composition_jobsize"

    def __post_init__(self) -> None:
        for field in ("m", "s_job", "s_batch", "logit_a", "logit_b"):
            setattr(self, field, np.asarray(getattr(self, field), dtype=float))
        self._frozen = False

    @property
    def n_components(self) -> int:
        return len(self.m)

    @property
    def n_params(self) -> int:
        # k locations, k job sds, k batch sds, k-1 intercepts, k-1 slopes, escape
        return 5 * self.n_components

    def weights(self, activity: np.ndarray) -> np.ndarray:
        """Return the component weights a bin of that activity draws from."""
        x = np.log(np.maximum(np.asarray(activity, dtype=float), 1.0))
        eta = self.logit_a[None, :] + self.logit_b[None, :] * x[:, None]
        eta -= eta.max(axis=1, keepdims=True)
        w = np.exp(eta)
        return w / w.sum(axis=1, keepdims=True)

    @classmethod
    def fit(cls, jobs: pd.DataFrame, n_components: int = 3, n_init: int = 10,
            seed: int = 42) -> "CompositionJobSize":
        """Fit the mixture, then how its weights and its batches follow activity."""
        z = np.log(jobs["job_size"].to_numpy(dtype=float))
        activity = np.log(np.maximum(jobs["activity"].to_numpy(dtype=float), 1.0))
        mix = LogNormalMixture.fit_log(z, n_components=n_components, n_init=n_init,
                                       seed=seed)
        comp = mix._component_logpdf(z, mix.weights, mix.mu, mix.sigma)
        resp = np.exp(comp - logsumexp(comp, axis=1)[:, None])

        # weights as a multinomial logit on activity, fitted to the responsibilities
        design = np.column_stack([np.ones(len(z)), activity])
        k = mix.n_components

        def nll(theta: np.ndarray) -> float:
            coef = np.vstack([np.zeros(2), theta.reshape(k - 1, 2)])
            eta = design @ coef.T
            return float(-(resp * (eta - logsumexp(eta, axis=1)[:, None])).sum())

        start = np.zeros(2 * (k - 1))
        start[0::2] = np.log(np.maximum(mix.weights[1:], 1e-6) / max(mix.weights[0], 1e-6))
        best = optimize.minimize(nll, start, method="L-BFGS-B")
        coef = np.vstack([np.zeros(2), best.x.reshape(k - 1, 2)])

        # a job's component is its batch's unless it escapes; both are observable
        # once every job is assigned the component it most likely came from
        label = np.argmax(resp, axis=1)
        frame = jobs.assign(_c=label, _z=z)
        modal = frame.groupby(["bin", "submit_time_dt"])["_c"].transform(
            lambda s: s.mode().iat[0])
        escape = float(np.mean(label != modal.to_numpy()))

        # the spread left inside a batch once its component is accounted for
        agree = label == modal.to_numpy()
        s_job = np.empty(k)
        for c in range(k):
            take = agree & (label == c)
            sub = frame.loc[take]
            groups = sub.groupby(["bin", "submit_time_dt"])["_z"]
            dof = len(sub) - groups.ngroups
            s_job[c] = (np.sqrt(((sub["_z"] - groups.transform("mean")) ** 2).sum() / dof)
                        if dof > 30 else mix.sigma[c] * 0.5)
        s_job = np.minimum(s_job, mix.sigma * 0.999)
        model = cls(m=mix.mu, s_job=s_job,
                    s_batch=np.sqrt(np.maximum(mix.sigma ** 2 - s_job ** 2, 1e-9)),
                    logit_a=coef[:, 0], logit_b=coef[:, 1], escape=escape)
        model.solve_level(jobs["activity"].to_numpy(dtype=float),
                          float(jobs["job_size"].to_numpy(dtype=float).mean()))
        return model

    def rvs(self, activity: np.ndarray, cardinality: np.ndarray,
            rng: np.random.Generator, alpha: float = 1.0,
            log_cut: Optional[float] = None) -> np.ndarray:
        """Draw one work unit per job; `log_cut` caps the bulk below a threshold."""
        act = _stream_activity(activity, alpha)
        card = np.asarray(cardinality, dtype=int)
        w = self.weights(act)
        pick = (w.cumsum(axis=1) > rng.random(len(act))[:, None]).argmax(axis=1)
        offset = np.repeat(rng.normal(0.0, 1.0, len(act)) * self.s_batch[pick]
                           + self.m[pick], card)

        n = int(card.sum())
        which = np.repeat(pick, card)
        loose = rng.random(n) < self.escape
        if loose.any():
            spare = np.repeat(w, card, axis=0)[loose]
            which = which.copy()
            which[loose] = (spare.cumsum(axis=1)
                            > rng.random(int(loose.sum()))[:, None]).argmax(axis=1)
            # an escaped job leaves its batch's level behind as well as its kind
            offset = offset.copy()
            offset[loose] = (self.m[which[loose]]
                             + rng.normal(0.0, 1.0, int(loose.sum()))
                             * self.s_batch[which[loose]])
        offset = offset + self.mu
        if log_cut is None:
            return np.exp(offset + rng.normal(0.0, 1.0, n) * self.s_job[which])
        return np.exp(offset + _normal_below(log_cut - offset, self.s_job[which], rng))

    def mean_work(self, activity: np.ndarray, log_cut: Optional[float] = None,
                  n_nodes: int = 24) -> float:
        """Return expected work per job, truncated below `log_cut` when given."""
        act = np.asarray(activity, dtype=float)
        share = self.weights(act)
        nodes, gh = np.polynomial.hermite_e.hermegauss(n_nodes)
        gh = gh / gh.sum()
        level = (self.mu + self.m[None, :, None]
                 + self.s_batch[None, :, None] * nodes[None, None, :])
        mass = share[:, :, None] * gh[None, None, :] / len(act)
        sigma = self.s_job[None, :, None]
        if log_cut is None:
            return float((mass * np.exp(level + 0.5 * sigma ** 2)).sum())
        return float((mass * _lognormal_mean_below(level, log_cut, sigma)).sum())

    def solve_level(self, activity: np.ndarray, target: float,
                    log_cut: Optional[float] = None) -> None:
        """Move the level until expected work per job matches `target`."""
        def gap(mu: float) -> float:
            self.mu = mu
            return self.mean_work(activity, log_cut) - target

        lo = hi = self.mu
        while gap(lo) > 0.0:
            lo -= 1.0
        while gap(hi) < 0.0:
            hi += 1.0
        self.mu = float(optimize.brentq(gap, lo, hi, xtol=1e-10))

    def freeze(self) -> "CompositionJobSize":
        self._frozen = True
        return self

    def to_dict(self) -> dict:
        return {"name": self.name, "n_components": self.n_components,
                "m": self.m.tolist(), "s_job": self.s_job.tolist(),
                "s_batch": self.s_batch.tolist(), "logit_a": self.logit_a.tolist(),
                "logit_b": self.logit_b.tolist(), "escape": self.escape,
                "mu": self.mu}

    @classmethod
    def from_dict(cls, blob: dict) -> "CompositionJobSize":
        return cls(m=np.asarray(blob["m"], dtype=float),
                   s_job=np.asarray(blob["s_job"], dtype=float),
                   s_batch=np.asarray(blob["s_batch"], dtype=float),
                   logit_a=np.asarray(blob["logit_a"], dtype=float),
                   logit_b=np.asarray(blob["logit_b"], dtype=float),
                   escape=float(blob["escape"]), mu=float(blob.get("mu", 0.0)))


def _poisson_loglinear(counts: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """Fit `log lambda = a + b x` by Poisson maximum likelihood."""
    def nll(p: np.ndarray) -> float:
        eta = p[0] + p[1] * x
        return float(np.sum(np.exp(eta) - counts * eta))

    def grad(p: np.ndarray) -> np.ndarray:
        resid = np.exp(p[0] + p[1] * x) - counts
        return np.array([resid.sum(), float(resid @ x)])

    start = np.array([np.log(max(float(counts.mean()), 1e-6)), 0.0])
    return tuple(map(float, optimize.minimize(nll, start, jac=grad, method="BFGS").x))


def _normal_below(bound: np.ndarray, sigma: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Draw N(0, sigma^2) conditioned to fall below `bound`, elementwise."""
    mass = np.maximum(ndtr(bound / sigma), 1e-12)
    return sigma * ndtri(rng.random(len(bound)) * mass)


def _lognormal_mean_below(level: np.ndarray, log_cut: float,
                          sigma: float) -> np.ndarray:
    """Return E[exp(level + eta)] for eta ~ N(0, sigma^2) conditioned below the cut.

    Capped at the cut, which is the exact limit as the level runs away above it.
    """
    h = (log_cut - level) / sigma
    mean = (np.exp(level + 0.5 * sigma ** 2)
            * ndtr(h - sigma) / np.maximum(ndtr(h), 1e-300))
    return np.minimum(mean, np.exp(log_cut))


@dataclass
class TwoPartJobSize:
    """Bulk work drawn per job, large jobs placed per bin.

    Sizes below `cut` come from the bulk law truncated at the cut; a bin of `n`
    jobs promotes Binomial(n, lambda/n) of them to exceedances, with
    `log lambda = rate_a + rate_b * log(activity)`. Per-bin placement stops a
    bin's largest job growing with its job count; the binomial draw stops a bin
    being asked for more exceedances than it holds. The bulk level is set last,
    against the mean the composition delivers after truncation, so it is not the
    level `bulk` was fitted with.
    """

    cut: float
    bulk: ConditionalJobSize
    tail_shape: float
    tail_scale: float
    rate_a: float
    rate_b: float
    tail_kind: str = "genpareto"
    tail_pool: Optional[np.ndarray] = None
    name: str = "two_part_jobsize"

    def __post_init__(self) -> None:
        if self.tail_kind == "empirical" and self.tail_pool is None:
            raise ValueError("an empirical tail needs a pool to draw from")
        self._frozen = False

    @property
    def n_params(self) -> int:
        # the cut is fixed by the quantile, so it costs nothing
        return self.bulk.n_params + 4

    @classmethod
    def fit(cls, jobs: pd.DataFrame, quantile: float = TAIL_QUANTILE,
            n_components: Optional[int] = None, n_init: int = 10, seed: int = 42,
            tail_kind: str = "empirical", beta: Optional[float] = None,
            with_latent: bool = True,
            bulk_kind: str = "composition") -> "TwoPartJobSize":
        """Split the sizes at `quantile`, then fit each part against its own data.

        The generalized Pareto is fitted either way, but the default draws from the
        observed exceedances: the fitted shape has no usable variance, so the
        parametric tail is a diagnostic rather than the frozen law.
        """
        sizes = jobs["job_size"].to_numpy(dtype=float)
        cut = float(np.quantile(sizes, quantile))
        above = sizes >= cut
        if not above.any():
            raise ValueError("the cut leaves no tail")
        below = jobs.loc[~above].reset_index(drop=True)
        if bulk_kind == "composition":
            # no deflation constraint on a per-job component, so k is not narrowed
            bulk = CompositionJobSize.fit(below, n_components=n_components or 3,
                                          n_init=n_init, seed=seed)
        elif bulk_kind == "conditional":
            bulk = ConditionalJobSize.fit(below, n_components=n_components,
                                          n_init=n_init, seed=seed, beta=beta,
                                          with_latent=with_latent)
        else:
            raise ValueError("bulk_kind must be 'conditional' or 'composition'")
        shape, _, scale = stats.genpareto.fit(sizes[above] - cut, floc=0.0)
        per_bin = jobs.assign(_above=above).groupby("bin").agg(
            k=("_above", "sum"), activity=("activity", "first"))
        rate_a, rate_b = _poisson_loglinear(
            per_bin["k"].to_numpy(dtype=float),
            np.log(per_bin["activity"].to_numpy(dtype=float)))
        model = cls(cut=cut, bulk=bulk, tail_shape=float(shape), tail_scale=float(scale),
                    rate_a=rate_a, rate_b=rate_b, tail_kind=tail_kind,
                    tail_pool=sizes[above].copy())
        model._solve_level(jobs, float(sizes.mean()))
        return model

    def _solve_level(self, jobs: pd.DataFrame, target: float, n_nodes: int = 24) -> None:
        """Move the bulk level until the composed mean work matches `target`.

        The bulk mean is integrated exactly: Gauss-Hermite over the mixture, the
        closed form for the truncated lognormal within a batch. The tail is always
        weighted by the observed exceedances, so a fitted tail law leaves the level
        alone.
        """
        placed_all = float(self.tail_rate(
            jobs.groupby("bin")["activity"].first().to_numpy(dtype=float)).sum())
        tail_share = placed_all / len(jobs)
        pool_mean = float(self.tail_pool.mean())
        if isinstance(self.bulk, CompositionJobSize):
            # the composition bulk carries its own level and its own quadrature
            self.bulk.solve_level(jobs["activity"].to_numpy(dtype=float),
                                  (target - tail_share * pool_mean) / (1.0 - tail_share),
                                  float(np.log(self.cut)))
            return
        batches = batch_frame(jobs)
        log_act = np.log(batches["activity"].to_numpy(dtype=float))
        # only beta*log(activity) varies across batches, so equal activities collapse
        uniq, inv = np.unique(log_act, return_inverse=True)
        share = np.bincount(inv, weights=batches["cardinality"].to_numpy(dtype=float))
        share /= share.sum()

        nodes, gh = np.polynomial.hermite_e.hermegauss(n_nodes)
        gh = gh / gh.sum()
        # the bin level is integrated on its own axis: it shifts the batch level,
        # so it moves how much of each component the cut removes
        v_sd = np.sqrt(self.bulk.latent.variance) if self.bulk.latent else 0.0
        v_nodes, v_gh = (nodes, gh) if v_sd > 0 else (np.zeros(1), np.ones(1))
        offset = (self.bulk.beta * uniq[:, None, None, None]
                  + self.bulk.m[None, :, None, None]
                  + self.bulk.s_batch[None, :, None, None] * nodes[None, None, :, None]
                  + v_sd * v_nodes[None, None, None, :])
        mass = (share[:, None, None, None] * self.bulk.weights[None, :, None, None]
                * gh[None, None, :, None] * v_gh[None, None, None, :])
        # the truncation is per component, because so is the within-batch spread
        sigma = self.bulk.sigma_w[None, :, None, None]

        placed = float(self.tail_rate(
            jobs.groupby("bin")["activity"].first().to_numpy(dtype=float)).sum())
        share_tail = placed / len(jobs)
        tail_mean = float(self.tail_pool.mean())
        log_cut = float(np.log(self.cut))

        def gap(mu: float) -> float:
            bulk = float((mass * _lognormal_mean_below(mu + offset, log_cut, sigma)).sum())
            return (1.0 - share_tail) * bulk + share_tail * tail_mean - target

        lo = hi = self.bulk.mu
        while gap(lo) > 0.0:
            lo -= 1.0
        while gap(hi) < 0.0:
            hi += 1.0
        self.bulk.mu = float(optimize.brentq(gap, lo, hi, xtol=1e-10))

    def tail_rate(self, activity: np.ndarray) -> np.ndarray:
        """Return the expected number of large jobs in a bin of that activity."""
        act = np.maximum(np.asarray(activity, dtype=float), 1.0)
        return np.exp(self.rate_a + self.rate_b * np.log(act))

    def _draw_tail(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.tail_kind == "empirical":
            return rng.choice(self.tail_pool, size=n)
        return self.cut + stats.genpareto.rvs(self.tail_shape, scale=self.tail_scale,
                                              size=n, random_state=rng)

    def rvs(self, activity: np.ndarray, cardinality: np.ndarray, bin_index: np.ndarray,
            rng: np.random.Generator, alpha: float = 1.0) -> np.ndarray:
        """Draw one work unit per job; `bin_index` groups the batches into bins.

        `alpha` superposes that many unit-load streams: each reads its own share
        of the bin's occupancy and contributes its own large jobs, so both the
        size law and the rate of large jobs stay put as load rises.
        """
        card = np.asarray(cardinality, dtype=int)
        bins = np.asarray(bin_index, dtype=int)
        n_bins = int(bins.max()) + 1
        if isinstance(self.bulk, CompositionJobSize):
            out = self.bulk.rvs(activity, card, rng, alpha, float(np.log(self.cut)))
        else:
            raw, which = self.bulk._draw_batches(activity, rng, alpha,
                                                 self.bulk.draw_bin_level(bins, rng))
            level = np.repeat(raw, card)
            sigma = np.repeat(self.bulk.sigma_w[which], card)
            out = np.exp(level + _normal_below(np.log(self.cut) - level, sigma, rng))

        bin_of_job = np.repeat(bins, card)
        counts = np.bincount(bin_of_job, minlength=n_bins)
        bin_activity = np.zeros(n_bins)
        bin_activity[bins] = _stream_activity(activity, alpha)
        # binomial, not Poisson: a bin cannot promote more jobs than it holds
        wanted = rng.binomial(counts, np.minimum(
            float(alpha) * self.tail_rate(bin_activity) / np.maximum(counts, 1), 1.0))
        order = np.argsort(bin_of_job, kind="stable")
        starts = np.searchsorted(bin_of_job[order], np.arange(n_bins))
        for b in np.nonzero(wanted)[0]:
            take = int(wanted[b])
            picked = order[starts[b] + rng.choice(counts[b], size=take, replace=False)]
            out[picked] = self._draw_tail(take, rng)
        return out

    @property
    def bulk_kind(self) -> str:
        return "composition" if isinstance(self.bulk, CompositionJobSize) else "conditional"

    def freeze(self) -> "TwoPartJobSize":
        self._frozen = True
        self.bulk.freeze()
        return self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cut": self.cut,
            "tail_kind": self.tail_kind,
            "tail_shape": self.tail_shape,
            "tail_scale": self.tail_scale,
            "rate_a": self.rate_a,
            "rate_b": self.rate_b,
            "bulk_kind": self.bulk_kind,
            "bulk": self.bulk.to_dict(),
        }

    @classmethod
    def from_dict(cls, blob: dict, tail_pool: Optional[np.ndarray] = None) -> "TwoPartJobSize":
        loader = (CompositionJobSize if blob.get("bulk_kind") == "composition"
                  else ConditionalJobSize)
        return cls(cut=float(blob["cut"]), bulk=loader.from_dict(blob["bulk"]),
                   tail_shape=float(blob["tail_shape"]),
                   tail_scale=float(blob["tail_scale"]),
                   rate_a=float(blob["rate_a"]), rate_b=float(blob["rate_b"]),
                   tail_kind=str(blob["tail_kind"]), tail_pool=tail_pool)


def rolling_origin_windows(bins: pd.DatetimeIndex, validation_days: float = 21.0,
                           n_folds: int = 4, min_train_days: float = 30.0
                           ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (train_end, validation_end) pairs; training always precedes validation.

    Same geometry as the count model's folds, so the two sides of the generator
    are validated over comparable stretches of the trace.
    """
    start, stop = bins.min(), bins.max()
    span = (stop - start) / pd.Timedelta(days=1)
    if span < float(min_train_days) + float(validation_days):
        return []
    origins = np.unique(np.linspace(float(min_train_days), span - float(validation_days),
                                    n_folds))
    return [(start + pd.Timedelta(days=float(o)),
             start + pd.Timedelta(days=float(o) + float(validation_days)))
            for o in origins]


def _window_arrays(window: pd.DataFrame):
    batches = batch_frame(window)
    card = batches["cardinality"].to_numpy(dtype=int)
    return (batches["activity"].to_numpy(dtype=float), card,
            bin_grid_index(batches["bin"]),
            np.repeat(batches["bin"].to_numpy(), card))


def score_window(model: "TwoPartJobSize", window: pd.DataFrame, n_rep: int = 20,
                 seed: int = 42) -> Dict[str, float]:
    """Return what `model` produces on a window's own batch structure."""
    activity, card, index, bin_of_job = _window_arrays(window)
    sizes = window["job_size"].to_numpy(dtype=float)
    n_jobs = window.groupby("bin").size().to_numpy()
    blocks, ks, mean_ratio, med_ratio = [], [], [], []
    for r in range(n_rep):
        drawn = model.rvs(activity, card, index, np.random.default_rng(seed + r))
        work = pd.Series(drawn).groupby(bin_of_job).sum().to_numpy()
        blocks.append(offered_work_stats(work, n_jobs))
        ks.append(float(stats.ks_2samp(drawn, sizes).statistic))
        mean_ratio.append(float(drawn.mean() / sizes.mean()))
        med_ratio.append(float(np.median(drawn) / np.median(sizes)))
    out = {k: float(np.mean([b[k] for b in blocks])) for k in blocks[0]}
    out["ks_job_marginal"] = float(np.mean(ks))
    out["mean_work_ratio"] = float(np.mean(mean_ratio))
    out["median_ratio"] = float(np.mean(med_ratio))
    return out


def fold_report(jobs: pd.DataFrame, validation_days: float = 21.0, n_folds: int = 4,
                min_train_days: float = 30.0, quantile: float = TAIL_QUANTILE,
                n_init: int = 4, seed: int = 42, n_rep: int = 20,
                bulk_kind: str = "composition") -> pd.DataFrame:
    """Fit on each expanding training window and score on the window that follows.

    Three arms per fold: `out_of_sample` is the training-fitted law scored on
    ground it never saw, `in_window` is the same law refitted on the validation
    window and is the ceiling that window admits, and `observed` is the window
    itself. The gap between the first two is the transfer cost; the gap between
    the last two is what the model cannot represent anywhere.
    """
    rows = []
    windows = rolling_origin_windows(pd.DatetimeIndex(jobs["bin"]), validation_days,
                                     n_folds, min_train_days)
    for fold, (train_end, val_end) in enumerate(windows, start=1):
        at = jobs["bin"]
        train = jobs.loc[at < train_end].reset_index(drop=True)
        window = jobs.loc[(at >= train_end) & (at < val_end)].reset_index(drop=True)
        base = {"fold": fold, "train_end": train_end, "n_train": len(train),
                "n_val": len(window)}
        if window.empty or train.empty:
            continue
        for arm, source in (("out_of_sample", train), ("in_window", window)):
            try:
                model = TwoPartJobSize.fit(source, quantile=quantile, n_init=n_init,
                                           seed=seed, bulk_kind=bulk_kind)
            except (ValueError, RuntimeError) as exc:
                rows.append({**base, "arm": arm, "note": str(exc)[:70]})
                continue
            composition = isinstance(model.bulk, CompositionJobSize)
            rows.append({**base, "arm": arm, "k": model.bulk.n_components,
                         "beta": np.nan if composition else model.bulk.beta,
                         "escape": model.bulk.escape if composition else np.nan,
                         "mu": model.bulk.mu, "cut": model.cut,
                         "rate_a": model.rate_a, "rate_b": model.rate_b,
                         **score_window(model, window, n_rep, seed)})
        by_bin = window.groupby("bin")["job_size"]
        rows.append({**base, "arm": "observed",
                     **offered_work_stats(by_bin.sum().to_numpy(),
                                          by_bin.size().to_numpy()),
                     "ks_job_marginal": 0.0, "mean_work_ratio": 1.0,
                     "median_ratio": 1.0})
    return pd.DataFrame(rows)


def assign_work(batch_times_s: np.ndarray, batch_sizes: np.ndarray, counts: np.ndarray,
                bin_seconds: float, model: ConditionalJobSize | TwoPartJobSize,
                rng: np.random.Generator, alpha: float = 1.0) -> np.ndarray:
    """Return one work unit per job of a reconstruction, in job-arrival order.

    Activity is read from the generated counts, so the size model sees the bin
    occupancy the arrival model produced rather than the trace's. Takes arrays
    rather than a reconstruction so the arrival side stays size-agnostic.

    Pass the load factor as `alpha` to read those counts as a superposition of
    that many unit-load streams; leaving it at 1 reads them as one stream that
    became busier. The two readings are different claims about load.
    """
    counts = np.asarray(counts, dtype=float)
    bin_of_batch = (np.asarray(batch_times_s, dtype=float) // float(bin_seconds)).astype(int)
    bin_of_batch = np.clip(bin_of_batch, 0, len(counts) - 1)
    activity = counts[bin_of_batch]
    if np.any(activity < 1.0):
        raise ValueError("a batch sits in a bin the counts record as empty")
    card = np.asarray(batch_sizes, dtype=int)
    # the grid position, not a compacted one: the size latent lives on real time
    if isinstance(model, TwoPartJobSize):
        return model.rvs(activity, card, bin_of_batch, rng, alpha=alpha)
    return model.rvs(activity, card, rng, alpha=alpha, bin_index=bin_of_batch)


def offered_work_stats(work: np.ndarray, n_jobs: np.ndarray) -> Dict[str, float]:
    """Return the spread of work per occupied bin and how it tracks the job count."""
    w = np.asarray(work, dtype=float)
    ranked = np.sort(w)[::-1]
    top = max(int(round(0.05 * len(ranked))), 1)
    return {
        "cv": float(w.std(ddof=1) / w.mean()),
        "q99_over_median": float(np.quantile(w, 0.99) / np.median(w)),
        "spearman_jobs_work": float(stats.spearmanr(np.asarray(n_jobs, dtype=float),
                                                    w).statistic),
        "top5pct_share": float(ranked[:top].sum() / ranked.sum()),
    }


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


def write_jobsize_model(model: LogNormalMixture | ScipyJobSize | ConditionalJobSize,
                        path: str | Path) -> None:
    """Write the frozen law to `path` as json."""
    Path(path).write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def read_jobsize_model(path: str | Path) -> LogNormalMixture:
    return LogNormalMixture.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def read_conditional_jobsize(path: str | Path) -> ConditionalJobSize:
    return ConditionalJobSize.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8")))


def read_two_part_jobsize(path: str | Path, pool_path: str | Path) -> TwoPartJobSize:
    """Read the two-part law, recovering its exceedances from the job pool."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    sizes = EmpiricalJobSize.from_npz(pool_path).sizes
    return TwoPartJobSize.from_dict(blob, tail_pool=sizes[sizes >= float(blob["cut"])])
