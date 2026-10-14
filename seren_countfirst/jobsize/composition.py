from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import logsumexp

from .laws import LogNormalMixture, _lognormal_mean_below, _normal_below
from .work import _stream_activity


@dataclass
class CompositionJobSize:
    """Sizes as a mixture whose weights move with activity, with the component per job.

    A batch draws a component and a shared offset; each job takes that component
    unless it escapes, which is what lets one submission hold jobs of different
    kinds. Activity enters through the weights rather than the level.
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
    def fit(
        cls, jobs: pd.DataFrame, n_components: int = 3, n_init: int = 10, seed: int = 42
    ) -> "CompositionJobSize":
        """Fit the mixture, then how its weights and its batches follow activity."""
        z = np.log(jobs["job_size"].to_numpy(dtype=float))
        activity = np.log(np.maximum(jobs["activity"].to_numpy(dtype=float), 1.0))
        mix = LogNormalMixture.fit_log(z, n_components=n_components, n_init=n_init, seed=seed)
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
        key = ["bin", "submitted_at"]
        modal = frame.groupby(key)["_c"].transform(lambda s: s.mode().iat[0])
        escape = float(np.mean(label != modal.to_numpy()))

        # the spread left inside a batch once its component is accounted for
        agree = label == modal.to_numpy()
        s_job = np.empty(k)
        for c in range(k):
            take = agree & (label == c)
            sub = frame.loc[take]
            groups = sub.groupby(key)["_z"]
            dof = len(sub) - groups.ngroups
            s_job[c] = (
                np.sqrt(((sub["_z"] - groups.transform("mean")) ** 2).sum() / dof)
                if dof > 30
                else mix.sigma[c] * 0.5
            )
        s_job = np.minimum(s_job, mix.sigma * 0.999)
        model = cls(
            m=mix.mu,
            s_job=s_job,
            s_batch=np.sqrt(np.maximum(mix.sigma**2 - s_job**2, 1e-9)),
            logit_a=coef[:, 0],
            logit_b=coef[:, 1],
            escape=escape,
        )
        model.solve_level(
            jobs["activity"].to_numpy(dtype=float),
            float(jobs["job_size"].to_numpy(dtype=float).mean()),
        )
        return model

    def rvs(
        self,
        activity: np.ndarray,
        cardinality: np.ndarray,
        rng: np.random.Generator,
        alpha: float = 1.0,
        log_cut: float | None = None,
    ) -> np.ndarray:
        """Draw one work unit per job; `log_cut` caps the bulk below a threshold."""
        act = _stream_activity(activity, alpha)
        card = np.asarray(cardinality, dtype=int)
        w = self.weights(act)
        pick = (w.cumsum(axis=1) > rng.random(len(act))[:, None]).argmax(axis=1)
        offset: np.ndarray = np.repeat(
            rng.normal(0.0, 1.0, len(act)) * self.s_batch[pick] + self.m[pick],
            card,
        )

        n = int(card.sum())
        which: np.ndarray = np.repeat(pick, card)
        loose = rng.random(n) < self.escape
        if loose.any():
            spare = np.repeat(w, card, axis=0)[loose]
            which = which.copy()
            which[loose] = (spare.cumsum(axis=1) > rng.random(int(loose.sum()))[:, None]).argmax(
                axis=1
            )
            # an escaped job leaves its batch's level behind as well as its kind
            offset = offset.copy()
            offset[loose] = (
                self.m[which[loose]]
                + rng.normal(0.0, 1.0, int(loose.sum())) * self.s_batch[which[loose]]
            )
        offset = offset + self.mu
        if log_cut is None:
            return np.exp(offset + rng.normal(0.0, 1.0, n) * self.s_job[which])
        return np.exp(offset + _normal_below(log_cut - offset, self.s_job[which], rng))

    def mean_work(
        self, activity: np.ndarray, log_cut: float | None = None, n_nodes: int = 24
    ) -> float:
        """Return expected work per job, truncated below `log_cut` when given."""
        act = np.asarray(activity, dtype=float)
        share = self.weights(act)
        nodes, gh = np.polynomial.hermite_e.hermegauss(n_nodes)
        gh = gh / gh.sum()
        level = self.mu + self.m[None, :, None] + self.s_batch[None, :, None] * nodes[None, None, :]
        mass = share[:, :, None] * gh[None, None, :] / len(act)
        sigma = self.s_job[None, :, None]
        if log_cut is None:
            return float((mass * np.exp(level + 0.5 * sigma**2)).sum())
        return float((mass * _lognormal_mean_below(level, log_cut, sigma)).sum())

    def solve_level(
        self, activity: np.ndarray, target: float, log_cut: float | None = None
    ) -> None:
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
        return {
            "name": self.name,
            "n_components": self.n_components,
            "m": self.m.tolist(),
            "s_job": self.s_job.tolist(),
            "s_batch": self.s_batch.tolist(),
            "logit_a": self.logit_a.tolist(),
            "logit_b": self.logit_b.tolist(),
            "escape": self.escape,
            "mu": self.mu,
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "CompositionJobSize":
        return cls(
            m=np.asarray(blob["m"], dtype=float),
            s_job=np.asarray(blob["s_job"], dtype=float),
            s_batch=np.asarray(blob["s_batch"], dtype=float),
            logit_a=np.asarray(blob["logit_a"], dtype=float),
            logit_b=np.asarray(blob["logit_b"], dtype=float),
            escape=float(blob["escape"]),
            mu=float(blob.get("mu", 0.0)),
        )
