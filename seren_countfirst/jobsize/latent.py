from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import logsumexp

from .work import batch_frame, bin_grid_index, submission_column

if TYPE_CHECKING:
    from .laws import LogNormalMixture


def within_batch_variance(jobs: pd.DataFrame, beta: float, mix: LogNormalMixture) -> np.ndarray:
    """Return the within-batch variance carried by each mixture component.

    Batches are assigned to components by the responsibility of their mean
    residual, which is where the model puts the component: it is a property of
    the submitted workload, not of the individual job. Components holding no
    multi-job batch fall back to the pooled variance.
    """
    z = np.log(jobs["job_size"].to_numpy(dtype=float))
    key = ["bin", submission_column(jobs)]
    grouped = jobs.assign(z=z).groupby(key, sort=True)
    dev2 = (z - grouped["z"].transform("mean").to_numpy()) ** 2
    frame = (
        jobs.assign(z=z, _d=dev2)
        .groupby(key, sort=True)
        .agg(
            ss=("_d", "sum"), card=("z", "size"), zbar=("z", "mean"), activity=("activity", "first")
        )
    )
    resid = frame["zbar"].to_numpy() - beta * np.log(frame["activity"].to_numpy(dtype=float))
    comp = mix._component_logpdf(resid, mix.weights, mix.mu, mix.sigma)
    resp = np.exp(comp - logsumexp(comp, axis=1)[:, None])

    ss, card = frame["ss"].to_numpy(), frame["card"].to_numpy(dtype=float)
    dof = resp.T @ (card - 1.0)
    pooled = float(ss.sum() / max(card.sum() - len(frame), 1.0))
    return np.where(dof > 1.0, (resp.T @ ss) / np.maximum(dof, 1e-12), pooled)


def _ar1_path(n: int, tau_bins: float, rng: np.random.Generator) -> np.ndarray:
    """Return a unit-variance AR(1) path started from its stationary distribution."""
    phi = 2.0 ** (-1.0 / max(float(tau_bins), 1e-6))
    noise = rng.standard_normal(n) * np.sqrt(max(1.0 - phi**2, 0.0))
    out = np.empty(n)
    out[0] = rng.standard_normal()
    for t in range(1, n):
        out[t] = phi * out[t - 1] + noise[t]
    return out


@dataclass
class SizeLatent:
    """Two-scale stationary AR(1) level for log work, on the bin grid.

    Same two-component form as the count model's gate latent.
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
        return (1.0 - self.w_long**2) * short + self.w_long**2 * long

    def simulate(self, n_bins: int, rng: np.random.Generator) -> np.ndarray:
        """Draw the level for every bin of the grid, occupied or not."""
        w = float(self.w_long)
        path = np.sqrt(max(1.0 - w**2, 0.0)) * _ar1_path(
            n_bins, self.tau_short_bins, rng
        ) + w * _ar1_path(n_bins, self.tau_long_bins, rng)
        return np.sqrt(max(self.variance, 0.0)) * path

    @classmethod
    def calibrate(
        cls,
        jobs: pd.DataFrame,
        beta: float,
        lags: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        min_batches: int = 4,
    ) -> "SizeLatent":
        """Fit the level's variance and two timescales where the level is observable.

        A bin's level is the mean of its batches, so a thin bin measures it badly.
        The estimate is taken on bins holding at least `min_batches`, with the
        residual noise removed.
        """
        batches = batch_frame(jobs)
        residual = batches["z_bar"].to_numpy(dtype=float) - beta * np.log(
            batches["activity"].to_numpy(dtype=float)
        )
        slot = bin_grid_index(batches["bin"])
        frame = pd.DataFrame({"slot": slot, "u": residual})
        level = frame.groupby("slot")["u"].mean()
        n_batches = frame.groupby("slot")["u"].size()

        # a bin's level is the mean of its batches, so its scatter shrinks with the
        # batch count, not the job count
        within = float(
            ((frame["u"] - frame.groupby("slot")["u"].transform("mean")) ** 2).sum()
            / max(len(frame) - len(level), 1)
        )
        usable = n_batches.index[n_batches >= max(int(min_batches), 1)]
        if len(usable) < 50:
            usable = n_batches.index
        seen = level.loc[usable].to_numpy(dtype=float)
        noise = float(np.mean(within / n_batches.loc[usable].to_numpy(dtype=float)))
        observed_var = float(np.var(seen, ddof=1))
        variance = max(observed_var - noise, 1e-6)

        series = level.loc[usable].reindex(np.arange(int(slot.max()) + 1)).to_numpy(dtype=float)
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
            trial = cls(
                variance,
                float(p[0]),
                float(np.exp(p[1])),
                float(np.exp(p[2])),
            )
            return float(np.sum((trial.acf(h) - want) ** 2))

        best = optimize.minimize(
            loss,
            np.array([0.5, np.log(2.0), np.log(200.0)]),
            method="L-BFGS-B",
            bounds=[(0.0, 0.95), (np.log(0.2), np.log(8.0)), (np.log(8.0), np.log(4000.0))],
        )
        return cls(
            variance=variance,
            w_long=float(best.x[0]),
            tau_short_bins=float(np.exp(best.x[1])),
            tau_long_bins=float(np.exp(best.x[2])),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "variance": self.variance,
            "w_long": self.w_long,
            "tau_short_bins": self.tau_short_bins,
            "tau_long_bins": self.tau_long_bins,
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "SizeLatent":
        return cls(
            variance=float(blob["variance"]),
            w_long=float(blob["w_long"]),
            tau_short_bins=float(blob["tau_short_bins"]),
            tau_long_bins=float(blob["tau_long_bins"]),
        )
