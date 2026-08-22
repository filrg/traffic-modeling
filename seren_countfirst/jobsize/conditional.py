from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..models.base import FitReport
from .work import _stream_activity, batch_frame
from .laws import LogNormalMixture
from .latent import SizeLatent, within_batch_variance


# component counts scanned when the caller does not fix one
COMPONENT_ORDERS = (1, 2, 3, 4, 5)


def component_scoreboard(
    jobs: pd.DataFrame,
    beta: float,
    orders: Sequence[int] = COMPONENT_ORDERS,
    n_init: int = 4,
    seed: int = 42,
    latent_variance: float = 0.0,
) -> pd.DataFrame:
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
        slack = float(
            np.min(mix.sigma**2 - within_batch_variance(jobs, beta, mix) - latent_variance)
        )
        rows.append(
            {"k": k, "AIC": mix.report.aic, "tightest_slack": slack, "deflatable": slack > 0.0}
        )
    return pd.DataFrame(rows)


def best_component_order(
    jobs: pd.DataFrame,
    beta: float,
    orders: Sequence[int] = COMPONENT_ORDERS,
    n_init: int = 4,
    seed: int = 42,
    latent_variance: float = 0.0,
) -> int:
    """Return the best-AIC component count that can be deflated in every component."""
    board = component_scoreboard(jobs, beta, orders, n_init, seed, latent_variance)
    usable = board.loc[board["deflatable"]]
    if usable.empty:
        raise RuntimeError(
            f"no component count in {tuple(orders)} leaves every component wider "
            f"than its own within-batch spread"
        )
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
    latent: SizeLatent | None = None
    report: FitReport | None = None
    name: str = "conditional_lognormal_mixture"

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        self.m = np.asarray(self.m, dtype=float)
        self.s_batch = np.asarray(self.s_batch, dtype=float)
        self.s_job = np.asarray(self.s_job, dtype=float)
        self.sigma_w = np.broadcast_to(
            np.asarray(self.sigma_w, dtype=float), self.weights.shape
        ).astype(float)
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
    def fit(
        cls,
        jobs: pd.DataFrame,
        n_components: int | None = None,
        n_init: int = 10,
        seed: int = 42,
        beta: float | None = None,
        level: str = "mean_work",
        with_latent: bool = True,
    ) -> "ConditionalJobSize":
        """Calibrate the levels in order, then place `mu`.

        No stage revisits an earlier one. Pass `beta` to pin the activity slope
        instead of estimating it.

        `level` picks what `mu` is solved against, and the two disagree whenever the
        mixture's analytic mean misses the sample mean: `mean_work` matches expected
        work per job, `log_mean` puts `mu` at the residuals' own mean so the fitted
        marginal is reproduced unshifted. Leave `n_components` unset to take the
        best-AIC order that can be deflated.
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
            coef, *_ = np.linalg.lstsq(
                np.column_stack([np.ones(len(x)), x]), batches["z_bar"].to_numpy(), rcond=None
            )
            beta = float(coef[1])

        adjusted = z - beta * np.log(activity)
        latent = SizeLatent.calibrate(jobs, beta) if with_latent else None
        latent_var = latent.variance if latent is not None else 0.0
        if n_components is None:
            n_components = best_component_order(
                jobs, beta, n_init=n_init, seed=seed, latent_variance=latent_var
            )
        mix = LogNormalMixture.fit_log(
            adjusted, n_components=n_components, n_init=n_init, seed=seed
        )
        sigma_w2 = within_batch_variance(jobs, beta, mix)
        # three levels share one component variance: bin, batch, job
        slack = mix.sigma**2 - sigma_w2 - latent_var
        if np.any(slack <= 0.0):
            c = int(np.argmin(slack))
            raise RuntimeError(
                f"component {c} of variance {mix.sigma[c] ** 2:.4f} cannot hold a bin "
                f"level of {latent_var:.4f} and a within-batch spread of "
                f"{sigma_w2[c]:.4f}: no batch-level variance exists"
            )

        # the centring constraint moves the mixture's level into mu, placed next
        centre = float(np.sum(mix.weights * mix.mu))
        model = cls(
            mu=0.0,
            beta=beta,
            sigma_w=np.sqrt(sigma_w2),
            weights=mix.weights,
            m=mix.mu - centre,
            s_batch=np.sqrt(slack),
            s_job=mix.sigma,
            latent=latent,
            report=mix.report,
        )
        if level == "mean_work":
            target = float(np.mean(jobs["job_size"].to_numpy(dtype=float)))
            model.mu = float(np.log(target) - np.log(model.mean_work(activity)))
        else:
            model.mu = centre
        return model

    def _eps_scale(self) -> float:
        """Return E[exp(v + eps + eta)]: the levels recombine to the fitted variance."""
        return float(np.sum(self.weights * np.exp(self.m + 0.5 * self.s_job**2)))

    def mean_work(self, activity: np.ndarray) -> float:
        """Return expected work per job over a population of bins.

        Exact under the model's independence of activity from both residuals.
        """
        act = np.asarray(activity, dtype=float)
        return float(np.exp(self.mu) * np.mean(act**self.beta) * self._eps_scale())

    def _draw_batches(
        self,
        activity: np.ndarray,
        rng: np.random.Generator,
        alpha: float = 1.0,
        bin_level: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw each batch's log work level and the component that set it."""
        act = _stream_activity(activity, alpha)
        which = rng.choice(self.n_components, size=len(act), p=self.weights)
        level = self.mu + self.beta * np.log(act) + rng.normal(self.m[which], self.s_batch[which])
        if bin_level is not None:
            level = level + np.asarray(bin_level, dtype=float)
        return level, which

    def draw_bin_level(self, bin_index: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
        """Return each batch's share of the slow size level, or None without one."""
        if self.latent is None:
            return None
        slots = np.asarray(bin_index, dtype=int)
        return self.latent.simulate(int(slots.max()) + 1, rng)[slots]

    def batch_levels(
        self, activity: np.ndarray, rng: np.random.Generator, alpha: float = 1.0
    ) -> np.ndarray:
        """Draw the log work level shared by the jobs of each batch."""
        return self._draw_batches(activity, rng, alpha)[0]

    def rvs(
        self,
        activity: np.ndarray,
        cardinality: np.ndarray,
        rng: np.random.Generator,
        alpha: float = 1.0,
        bin_index: np.ndarray | None = None,
    ) -> np.ndarray:
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
        return np.exp(
            base + rng.normal(0.0, 1.0, size=len(base)) * np.repeat(self.sigma_w[which], card)
        )

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
        return cls(
            mu=float(blob["mu"]),
            beta=float(blob["beta"]),
            sigma_w=np.asarray(blob["sigma_w"], dtype=float),
            weights=np.asarray(blob["weights"], dtype=float),
            m=np.asarray(blob["m"], dtype=float),
            s_batch=np.asarray(blob["s_batch"], dtype=float),
            s_job=np.asarray(blob["s_job"], dtype=float),
            latent=(SizeLatent.from_dict(blob["latent"]) if blob.get("latent") else None),
        )


def read_conditional_jobsize(path: str | Path) -> ConditionalJobSize:
    return ConditionalJobSize.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
