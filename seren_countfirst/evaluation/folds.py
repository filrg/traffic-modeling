from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from ..bootstrap import bootstrap_metric_envelope
from ..data import Split
from ..metrics import effective_windows, metric_vector, z_score


def scale_support(n_bins: int) -> pd.DataFrame:
    """Return effective window counts per scale, and whether each is reportable."""
    nq = effective_windows(n_bins)
    return pd.DataFrame(
        {
            "scale": nq.index,
            "n_effective_windows": nq.to_numpy(),
            "reportable": nq.to_numpy() >= 8,
            "relative_uncertainty_order": 1.0 / np.sqrt(np.maximum(nq.to_numpy(), 1)),
        }
    ).set_index("scale")

# metrics graded per fold; unsupported IDC scales are dropped via `scale_support`
FOLD_CORE_METRICS: tuple[str, ...] = (
    "mean",
    "zero_bin_fraction",
    "IDC_15m", "IDC_1h", "IDC_6h", "IDC_1d", "IDC_2d", "IDC_4d", "IDC_7d",
    "ACF_lag1", "ACF_lag4", "ACF_lag96",
    "busy_run_mean", "quiet_run_q95",
)


def evaluate_on_folds(
    model,
    folds: Sequence[Split],
    busy_threshold: float,
    n_paths: int = 10,
    seed: int = 0,
    boot_rep: int = 100,
    boot_method: str = "stationary_24h",
    metric_names: Sequence[str] = FOLD_CORE_METRICS,
    refit: Callable[[np.ndarray], object] | None = None,
    **sim_kwargs,
) -> pd.DataFrame:
    """Score one model against every rolling-origin validation window.

    Each window is graded against the joint uncertainty of the window (its
    trace-native bootstrap SE) and of the generator (metric spread across
    simulated paths of the same duration).

    The caller's global rate calibration travels in `sim_kwargs` unchanged; no
    per-fold rescaling is applied. Pass `refit` to fit on each fold's training
    part, which grades the estimator rather than the frozen candidate.
    """
    rows: list[dict] = []
    for k, fold in enumerate(folds, start=1):
        val = np.asarray(fold.validation.values, dtype=float)
        n = len(val)
        supported = scale_support(n)
        reportable = set(supported.index[supported["reportable"].to_numpy()])
        keep = [m for m in metric_names if not m.startswith("IDC_") or m[4:] in reportable]

        est = refit(np.asarray(fold.train.values, dtype=float)) if refit is not None else model
        rng = np.random.default_rng(seed + 1009 * k)
        paths = [est.simulate(n, rng, **sim_kwargs) for _ in range(n_paths)]
        syn_frame = pd.DataFrame([metric_vector(p, busy_threshold) for p in paths])
        syn, syn_sd = syn_frame.mean(), syn_frame.std(ddof=1)
        obs = metric_vector(val, busy_threshold)
        env = bootstrap_metric_envelope(
            val, lambda x: metric_vector(x, busy_threshold), method=boot_method,
            n_rep=boot_rep, seed=seed + k,
        )
        for name in keep:
            if name not in obs.index or name not in syn.index:
                continue
            boot_se = float(env["se"].get(name, np.nan))
            path_sd = float(syn_sd.get(name, np.nan))
            joint = np.sqrt(np.nansum([boot_se ** 2, path_sd ** 2]))
            eps = 1e-3 * max(abs(float(obs[name])), 1.0)
            rows.append(
                {
                    "fold": fold.name,
                    "n_bins": n,
                    "metric": name,
                    "validation": float(obs[name]),
                    "synthetic": float(syn[name]),
                    "boot_se": boot_se,
                    "path_sd": path_sd,
                    "Z": z_score(float(syn[name]), float(obs[name]), max(joint, eps)),
                }
            )
    return pd.DataFrame(rows)
