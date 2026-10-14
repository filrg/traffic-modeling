from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from traffic_modeling.counts.metrics import HOURLY_ACF_LAG_WEIGHTS, acf, hourly_acf1_from_autocov

from ..models.base import CountModel

_INFEASIBLE = 1e12
# zero-fraction error against log-variance error; the gated calibration's weighting
_ZERO_WEIGHT = 25.0


@dataclass(frozen=True, slots=True)
class GeometricMoments:
    """Stationary moments of a count model whose ACF is `amplitude * decay**h`.

    `zero_prob` is None where the model has no closed form for it.
    """

    variance: float
    amplitude: float
    decay: float
    zero_prob: float | None = None

    def acf(self, lags: Sequence[int]) -> np.ndarray:
        return self.amplitude * self.decay ** np.asarray(lags, dtype=float)

    def hourly_acf1(self) -> float:
        gamma = {h: self.amplitude * self.decay**h * self.variance for h in HOURLY_ACF_LAG_WEIGHTS}
        return float(hourly_acf1_from_autocov(self.variance, gamma))


def solve_debiased(
    moments: Callable[[np.ndarray], GeometricMoments | None],
    build: Callable[[np.ndarray], CountModel],
    starts: Sequence[np.ndarray],
    *,
    variance: float,
    zero_fraction: float,
    acf_1h: float,
    acf_long: Mapping[int, float],
    n_bins: int,
    seed: int = 0,
    n_paths: int = 8,
    n_rounds: int = 2,
    max_inflation: float = 3.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the parameters minimising one loss over the gated calibration's targets.

    The loss sums the squared log-variance error, the weighted zero-fraction
    error where the model has one, the hourly lag-1 ACF error and the long-lag
    ACF errors. Each round measures the long-lag attenuation on `n_paths` paths
    of `n_bins` and re-solves against targets divided by it; the correction is
    one-sided and capped at `max_inflation`. `moments` returns None outside the
    admissible region.
    """
    lags = sorted(int(h) for h in acf_long)
    targets = np.array([float(acf_long[h]) for h in lags])

    def loss(theta: np.ndarray, long_targets: np.ndarray) -> float:
        found = moments(np.asarray(theta, dtype=float))
        if found is None:
            return _INFEASIBLE
        value = np.log(found.variance / variance) ** 2 + (found.hourly_acf1() - acf_1h) ** 2
        value += float(np.sum((found.acf(lags) - long_targets) ** 2))
        if found.zero_prob is not None:
            value += _ZERO_WEIGHT * (found.zero_prob - zero_fraction) ** 2
        return float(value)

    def solve(long_targets: np.ndarray, warm: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        best = None
        for theta0 in (*(() if warm is None else (warm,)), *starts):
            res = minimize(
                loss,
                theta0,
                args=(long_targets,),
                method="Nelder-Mead",
                options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-13},
            )
            if best is None or res.fun < best.fun:
                best = res
        if best is None or best.fun >= _INFEASIBLE:
            raise RuntimeError("no admissible calibration found")
        return np.asarray(best.x, dtype=float), float(best.fun)

    theta, value = solve(targets)
    attenuation = np.ones_like(targets)
    for _ in range(max(int(n_rounds), 1)):
        rng = np.random.default_rng(seed)
        model = build(theta)
        paths = [model.simulate(int(n_bins), rng) for _ in range(n_paths)]
        realised = pd.DataFrame([acf(path, lags) for path in paths]).mean().to_numpy()
        analytic = _moments_at(moments, theta).acf(lags)
        ratio = np.divide(realised, analytic, out=np.ones_like(targets), where=analytic > 1e-9)
        # one-sided: the correction may only undo attenuation
        attenuation = np.clip(np.where(np.isfinite(ratio), ratio, 1.0), 1.0 / max_inflation, 1.0)
        theta, value = solve(targets / attenuation, warm=theta)

    found = _moments_at(moments, theta)
    report = {
        "calibration_loss": value,
        "acf_attenuation": float(np.mean(attenuation)),
        "resid_IDC": float(found.variance / variance - 1.0),
        "resid_acf_1h": float(found.hourly_acf1() - acf_1h),
    }
    if found.zero_prob is not None:
        report["resid_zero_frac"] = float(found.zero_prob - zero_fraction)
    return theta, report


def _moments_at(
    moments: Callable[[np.ndarray], GeometricMoments | None], theta: np.ndarray
) -> GeometricMoments:
    found = moments(theta)
    if found is None:
        raise RuntimeError("calibration left the admissible region")
    return found
