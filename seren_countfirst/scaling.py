from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from traffic_modeling.counts.metrics import acf, idc_curve

from .reconstruct import collision_risk

DEFAULT_ALPHAS: tuple = (0.5, 1.0, 2.0, 4.0, 8.0)


def simulate_at_load(
    model,
    n_bins: int,
    alpha: float,
    rng: np.random.Generator,
    kappa_rule: str = "proportional",
    **kwargs,
) -> np.ndarray:
    """Draw one path at load alpha; the model's frozen structure is not re-tuned."""
    return model.simulate(n_bins, rng, alpha=alpha, kappa_rule=kappa_rule, **kwargs)


def superposition(model, n_bins: int, k: int, rng: np.random.Generator, **kwargs) -> np.ndarray:
    """Sum k independent populations; semantically distinct from alpha = k."""
    return np.sum([model.simulate(n_bins, rng, alpha=1.0, **kwargs) for _ in range(k)], axis=0)


def rate_error(counts_alpha: np.ndarray, counts_ref: np.ndarray, alpha: float) -> float:
    """Return |N_alpha / (alpha N_1) - 1|."""
    denom = alpha * float(np.sum(counts_ref))
    if denom <= 0:
        return np.nan
    return float(abs(float(np.sum(counts_alpha)) / denom - 1.0))


def scaling_report(
    model,
    n_bins: int,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    n_paths: int = 20,
    kappa_rules: Sequence[str] = ("constant", "proportional"),
    bin_seconds: float = 900.0,
    seed: int = 0,
    **sim_kwargs,
) -> pd.DataFrame:
    """Report rate error, IDC/ACF/tail drift and export collision rate per load.

    Neither kappa rule preserves raw IDC and ACF; the drift is measured here.
    """
    rows: list[dict] = []
    for rule in kappa_rules:
        rng = np.random.default_rng(seed)
        ref_paths = [
            simulate_at_load(model, n_bins, 1.0, rng, rule, **sim_kwargs) for _ in range(n_paths)
        ]
        ref_total = float(np.mean([p.sum() for p in ref_paths]))
        ref_idc = idc_curve(np.concatenate(ref_paths[:1]))
        ref_acf = acf(ref_paths[0], [1, 4, 96])
        # the same statistics averaged over every reference path
        ref_idc_mean = pd.DataFrame([idc_curve(p) for p in ref_paths]).mean()
        ref_acf_mean = pd.DataFrame([acf(p, [1, 4, 96]) for p in ref_paths]).mean()

        for alpha in alphas:
            rng = np.random.default_rng(seed + 7919)
            paths = [
                simulate_at_load(model, n_bins, alpha, rng, rule, **sim_kwargs)
                for _ in range(n_paths)
            ]
            totals = np.array([p.sum() for p in paths], dtype=float)
            mean_total = float(totals.mean())
            expected = alpha * float(model.params_["mean"]) * n_bins
            path0 = paths[0]
            coll = collision_risk(path0, bin_seconds)
            idc_a = idc_curve(path0)
            acf_a = acf(path0, [1, 4, 96])
            idc_mean = pd.DataFrame([idc_curve(p) for p in paths]).mean()
            acf_mean = pd.DataFrame([acf(p, [1, 4, 96]) for p in paths]).mean()
            rows.append(
                {
                    "kappa_rule": rule,
                    "alpha": alpha,
                    "n_paths": n_paths,
                    "expected_total": expected,
                    "mean_total": mean_total,
                    "sd_total": float(totals.std(ddof=1)),
                    # gate 8 grades the Monte Carlo error, not the spread
                    "se_total": float(totals.std(ddof=1) / np.sqrt(len(totals))),
                    "rate_error_vs_theory": float(abs(mean_total / expected - 1.0)),
                    "rate_error_vs_alpha1": float(
                        abs(mean_total / max(alpha * ref_total, 1e-12) - 1.0)
                    ),
                    "IDC_15m": float(idc_a["15m"]),
                    "IDC_1d": float(idc_a["1d"]),
                    "IDC_15m_drift_log": float(np.log(idc_a["15m"] / ref_idc["15m"])),
                    "IDC_1d_drift_log": float(np.log(idc_a["1d"] / ref_idc["1d"])),
                    "ACF_lag1": float(acf_a[1]),
                    "ACF_lag1_drift": float(acf_a[1] - ref_acf[1]),
                    "ACF_lag96_drift": float(acf_a[96] - ref_acf[96]),
                    # path-averaged drift, the columns gate 8 grades
                    "IDC_15m_drift_log_mean": float(np.log(idc_mean["15m"] / ref_idc_mean["15m"])),
                    "IDC_1d_drift_log_mean": float(np.log(idc_mean["1d"] / ref_idc_mean["1d"])),
                    "ACF_lag1_drift_mean": float(acf_mean[1] - ref_acf_mean[1]),
                    "ACF_lag96_drift_mean": float(acf_mean[96] - ref_acf_mean[96]),
                    "p_exceed_slots": float(coll["p_exceed_slots"]),
                    "max_fill_ratio": float(coll["max_fill_ratio"]),
                }
            )
    return pd.DataFrame(rows)


def superposition_comparison(
    model,
    n_bins: int,
    k_values: Sequence[int] = (2, 4, 8),
    n_paths: int = 10,
    seed: int = 0,
    **sim_kwargs,
) -> pd.DataFrame:
    """Contrast superposition with intensity multiplication at matched load."""
    rows = []
    for k in k_values:
        rng = np.random.default_rng(seed)
        sup = [superposition(model, n_bins, k, rng, **sim_kwargs) for _ in range(n_paths)]
        rng = np.random.default_rng(seed)
        mult = [
            simulate_at_load(model, n_bins, float(k), rng, "constant", **sim_kwargs)
            for _ in range(n_paths)
        ]
        rows.append(
            {
                "k": k,
                "superposition_mean_total": float(np.mean([p.sum() for p in sup])),
                "multiplication_mean_total": float(np.mean([p.sum() for p in mult])),
                "superposition_IDC_15m": float(np.mean([idc_curve(p)["15m"] for p in sup])),
                "multiplication_IDC_15m": float(np.mean([idc_curve(p)["15m"] for p in mult])),
                "superposition_ACF_lag1": float(np.mean([acf(p, [1])[1] for p in sup])),
                "multiplication_ACF_lag1": float(np.mean([acf(p, [1])[1] for p in mult])),
            }
        )
    return pd.DataFrame(rows)


def calibrate_intercept(
    model,
    n_bins: int,
    target_rate_per_bin: float,
    rng: np.random.Generator,
    n_paths: int = 20,
    **sim_kwargs,
) -> float:
    """Return the global multiplicative correction on alpha.

    Applied to the intercept, never per synthetic path, and only after the
    structural gates have passed.
    """
    paths = [model.simulate(n_bins, rng, alpha=1.0, **sim_kwargs) for _ in range(n_paths)]
    realised = float(np.mean([p.mean() for p in paths]))
    if realised <= 0:
        return 1.0
    return float(target_rate_per_bin / realised)
