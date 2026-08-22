from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .work import _stream_activity, batch_frame
from .laws import EmpiricalJobSize, _lognormal_mean_below, _normal_below
from .composition import CompositionJobSize
from .conditional import ConditionalJobSize


# work quantile above which a job is placed per bin rather than per job
TAIL_QUANTILE = 0.99


def _poisson_loglinear(counts: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Fit `log lambda = a + b x` by Poisson maximum likelihood."""

    def nll(p: np.ndarray) -> float:
        eta = p[0] + p[1] * x
        return float(np.sum(np.exp(eta) - counts * eta))

    def grad(p: np.ndarray) -> np.ndarray:
        resid = np.exp(p[0] + p[1] * x) - counts
        return np.array([resid.sum(), float(resid @ x)])

    start = np.array([np.log(max(float(counts.mean()), 1e-6)), 0.0])
    return tuple(map(float, optimize.minimize(nll, start, jac=grad, method="BFGS").x))


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
    bulk: CompositionJobSize | ConditionalJobSize
    tail_shape: float
    tail_scale: float
    rate_a: float
    rate_b: float
    tail_kind: str = "genpareto"
    tail_pool: np.ndarray | None = None
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
    def fit(
        cls,
        jobs: pd.DataFrame,
        quantile: float = TAIL_QUANTILE,
        n_components: int | None = None,
        n_init: int = 10,
        seed: int = 42,
        tail_kind: str = "empirical",
        beta: float | None = None,
        with_latent: bool = True,
        bulk_kind: str = "composition",
    ) -> "TwoPartJobSize":
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
            bulk = CompositionJobSize.fit(
                below, n_components=n_components or 3, n_init=n_init, seed=seed
            )
        elif bulk_kind == "conditional":
            bulk = ConditionalJobSize.fit(
                below,
                n_components=n_components,
                n_init=n_init,
                seed=seed,
                beta=beta,
                with_latent=with_latent,
            )
        else:
            raise ValueError("bulk_kind must be 'conditional' or 'composition'")
        shape, _, scale = stats.genpareto.fit(sizes[above] - cut, floc=0.0)
        per_bin = (
            jobs.assign(_above=above)
            .groupby("bin")
            .agg(k=("_above", "sum"), activity=("activity", "first"))
        )
        rate_a, rate_b = _poisson_loglinear(
            per_bin["k"].to_numpy(dtype=float), np.log(per_bin["activity"].to_numpy(dtype=float))
        )
        model = cls(
            cut=cut,
            bulk=bulk,
            tail_shape=float(shape),
            tail_scale=float(scale),
            rate_a=rate_a,
            rate_b=rate_b,
            tail_kind=tail_kind,
            tail_pool=sizes[above].copy(),
        )
        model._solve_level(jobs, float(sizes.mean()))
        return model

    def _solve_level(self, jobs: pd.DataFrame, target: float, n_nodes: int = 24) -> None:
        """Move the bulk level until the composed mean work matches `target`.

        The bulk mean is integrated exactly: Gauss-Hermite over the mixture, the
        closed form for the truncated lognormal within a batch. The tail is always
        weighted by the observed exceedances, so a fitted tail law leaves the level
        alone.
        """
        placed_all = float(
            self.tail_rate(jobs.groupby("bin")["activity"].first().to_numpy(dtype=float)).sum()
        )
        tail_share = placed_all / len(jobs)
        pool_mean = float(self.tail_pool.mean())
        if isinstance(self.bulk, CompositionJobSize):
            # the composition bulk carries its own level and its own quadrature
            self.bulk.solve_level(
                jobs["activity"].to_numpy(dtype=float),
                (target - tail_share * pool_mean) / (1.0 - tail_share),
                float(np.log(self.cut)),
            )
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
        offset = (
            self.bulk.beta * uniq[:, None, None, None]
            + self.bulk.m[None, :, None, None]
            + self.bulk.s_batch[None, :, None, None] * nodes[None, None, :, None]
            + v_sd * v_nodes[None, None, None, :]
        )
        mass = (
            share[:, None, None, None]
            * self.bulk.weights[None, :, None, None]
            * gh[None, None, :, None]
            * v_gh[None, None, None, :]
        )
        # the truncation is per component, because so is the within-batch spread
        sigma = self.bulk.sigma_w[None, :, None, None]

        placed = float(
            self.tail_rate(jobs.groupby("bin")["activity"].first().to_numpy(dtype=float)).sum()
        )
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

    def tail_rate(self, activity: np.ndarray | float) -> np.ndarray:
        """Return the expected number of large jobs in a bin of that activity."""
        act = np.maximum(np.asarray(activity, dtype=float), 1.0)
        return np.exp(self.rate_a + self.rate_b * np.log(act))

    def _draw_tail(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.tail_kind == "empirical":
            return rng.choice(self.tail_pool, size=n)
        return self.cut + stats.genpareto.rvs(
            self.tail_shape, scale=self.tail_scale, size=n, random_state=rng
        )

    def rvs(
        self,
        activity: np.ndarray,
        cardinality: np.ndarray,
        bin_index: np.ndarray,
        rng: np.random.Generator,
        alpha: float = 1.0,
    ) -> np.ndarray:
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
            raw, which = self.bulk._draw_batches(
                activity, rng, alpha, self.bulk.draw_bin_level(bins, rng)
            )
            level = np.repeat(raw, card)
            sigma = np.repeat(self.bulk.sigma_w[which], card)
            out = np.exp(level + _normal_below(np.log(self.cut) - level, sigma, rng))

        bin_of_job = np.repeat(bins, card)
        counts = np.bincount(bin_of_job, minlength=n_bins)
        bin_activity = np.zeros(n_bins)
        bin_activity[bins] = _stream_activity(activity, alpha)
        # binomial, not Poisson: a bin cannot promote more jobs than it holds
        wanted = rng.binomial(
            counts,
            np.minimum(float(alpha) * self.tail_rate(bin_activity) / np.maximum(counts, 1), 1.0),
        )
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
    def from_dict(cls, blob: dict, tail_pool: np.ndarray | None = None) -> "TwoPartJobSize":
        loader = (
            CompositionJobSize if blob.get("bulk_kind") == "composition" else ConditionalJobSize
        )
        return cls(
            cut=float(blob["cut"]),
            bulk=loader.from_dict(blob["bulk"]),
            tail_shape=float(blob["tail_shape"]),
            tail_scale=float(blob["tail_scale"]),
            rate_a=float(blob["rate_a"]),
            rate_b=float(blob["rate_b"]),
            tail_kind=str(blob["tail_kind"]),
            tail_pool=tail_pool,
        )


def read_two_part_jobsize(path: str | Path, pool_path: str | Path) -> TwoPartJobSize:
    """Read the two-part law, recovering its exceedances from the job pool."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    sizes = EmpiricalJobSize.from_npz(pool_path).sizes
    return TwoPartJobSize.from_dict(blob, tail_pool=sizes[sizes >= float(blob["cut"])])
