from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import run_lengths
from .base import CountModel
from .nb import nb2_sample

_EPS = 1e-9


def _chain_from_indicator(active: np.ndarray) -> tuple:
    """Return (p_off_to_on, p_on_to_off) from the observed activity indicator."""
    a = np.asarray(active, dtype=bool)
    prev, nxt = a[:-1], a[1:]
    n_off, n_on = int((~prev).sum()), int(prev.sum())
    p01 = float((~prev & nxt).sum()) / max(n_off, 1)
    p10 = float((prev & ~nxt).sum()) / max(n_on, 1)
    # floored away from zero: an absorbing state has no stationary mixture
    return float(np.clip(p01, _EPS, 1.0)), float(np.clip(p10, _EPS, 1.0))


class ONOFFCounts(CountModel):
    """Two-state gate: OFF emits nothing, ON emits NB2 (or Poisson) counts.

    `N_t = Z_t * E_t` with `Z_t` a two-state Markov chain on {OFF, ON} and
    `E_t ~ NB2(lambda_on, kappa_on)` drawn independently in the ON state.
    Parameters: the two transition probabilities, the ON level, and for the NB
    emission the ON dispersion.
    """

    name = "onoff_nb"
    rung = None  # not a rung: discrete regime chains are out of scope here
    role = "benchmark"

    def __init__(self, emission: str = "nb") -> None:
        super().__init__()
        if emission not in ("nb", "poisson"):
            raise ValueError("emission must be 'nb' or 'poisson'")
        self.emission = emission
        self.name = "onoff_nb" if emission == "nb" else "onoff_pois"

    @classmethod
    def calibrate(cls, counts: np.ndarray, emission: str = "nb") -> "ONOFFCounts":
        """Moment-calibrate against one count series.

        The chain is estimated from the zero/non-zero indicator, so mean ON and
        OFF run lengths are reproduced by construction; `lambda_on` then matches
        the mean and `kappa_on` the base-bin variance. Under the NB emission the
        ON state can emit a zero, so the implied zero fraction exceeds the OFF
        probability; that gap is stored in `params_` rather than corrected.
        """
        n = np.asarray(counts, dtype=float)
        p01, p10 = _chain_from_indicator(n > 0)
        pi_on = p01 / (p01 + p10)
        m, var = float(n.mean()), float(n.var(ddof=1))
        lam_on = m / max(pi_on, _EPS)

        if emission == "poisson":
            kappa = np.inf
        else:
            # E[N^2] = pi_on (lam + lam^2 (1 + 1/kappa)) pins the ON dispersion
            ratio = ((var + m * m) / max(pi_on, _EPS) - lam_on) / max(lam_on * lam_on, _EPS)
            # ratio <= 1: the gate alone already carries the target variance
            kappa = 1.0 / (ratio - 1.0) if ratio > 1.0 + 1e-6 else np.inf

        model = cls(emission=emission)
        model.params_ = {
            "p01": p01, "p10": p10, "pi_on": float(pi_on),
            "lambda_on": float(lam_on), "kappa": float(kappa),
            "mean": float(pi_on * lam_on),
            "mean_off_run_bins": float(1.0 / p01),
            "mean_on_run_bins": float(1.0 / p10),
            "observed_off_run_mean": float(np.mean(run_lengths(n == 0))) if (n == 0).any() else 0.0,
            "observed_on_run_mean": float(np.mean(run_lengths(n > 0))) if (n > 0).any() else 0.0,
        }
        implied = model.implied_marginals()
        model.params_["implied_zero_frac"] = float(implied["P_zero"])
        model.params_["zero_frac_gap"] = float(implied["P_zero"] - (n == 0).mean())
        model.params_["n_params"] = 3 + (emission == "nb")
        # moment-calibrated: `report` stays None, so there is no AIC to quote
        return model

    def fit(self, counts: np.ndarray, emission: str | None = None, **kwargs) -> "ONOFFCounts":
        """Calibrate in place; this benchmark has no likelihood route."""
        self._check_mutable()
        other = ONOFFCounts.calibrate(counts, emission or self.emission)
        self.params_, self.report = other.params_, other.report
        self.emission, self.name = other.emission, other.name
        return self

    def implied_marginals(self) -> pd.Series:
        """Return the exact stationary marginals of the chain-times-emission product."""
        p = self.params_
        pi_on, lam, kap = p["pi_on"], p["lambda_on"], p["kappa"]
        e2_on = lam + lam * lam * (1.0 + (1.0 / kap if np.isfinite(kap) else 0.0))
        m = pi_on * lam
        var = pi_on * e2_on - m * m
        p0_on = np.exp(-lam) if not np.isfinite(kap) else (1.0 + lam / kap) ** (-kap)
        phi = 1.0 - p["p01"] - p["p10"]  # the chain's one geometric timescale
        return pd.Series(
            {
                "mean": float(m),
                "variance": float(var),
                "IDC_base": float(var / max(m, _EPS)),
                "P_zero": float((1.0 - pi_on) + pi_on * p0_on),
                "acf_lag1": float(pi_on * (1.0 - pi_on) * lam * lam * phi / max(var, _EPS)),
                "gate_phi": float(phi),
            }
        )

    def simulate(self, n_bins: int, rng: np.random.Generator, alpha: float = 1.0,
                 kappa_rule: str = "proportional", return_intensity: bool = False, **kwargs):
        p = self.params_
        p01, p10 = p["p01"], p["p10"]
        pi_on = p["pi_on"]
        state = np.empty(n_bins, dtype=bool)
        state[0] = rng.random() < pi_on  # stationary initialisation; no burn-in
        u = rng.random(n_bins)
        for t in range(1, n_bins):
            state[t] = (u[t] >= p10) if state[t - 1] else (u[t] < p01)
        lam = np.where(state, alpha * p["lambda_on"], 0.0)
        if np.isfinite(p["kappa"]):
            counts = np.where(state, nb2_sample(np.maximum(lam, _EPS),
                                                self.scaled_kappa(alpha, kappa_rule), rng), 0.0)
        else:
            counts = np.where(state, rng.poisson(lam), 0.0)
        counts = counts.astype(float)
        return (counts, lam) if return_intensity else counts

    def scaled_kappa(self, alpha: float, rule: str) -> float:
        """Apply the base rule, except that an infinite kappa (Poisson ON) stays infinite."""
        kappa = float(self.params_["kappa"])
        if not np.isfinite(kappa):
            return kappa
        return super().scaled_kappa(alpha, rule)
