"""Standard metric set (guide 1.11) and discrepancy definitions.

Every metric takes a base-resolution count array, so trace, bootstrap and
synthetic paths are scored by exactly the same code.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import SCALES


# aggregation and dispersion
def aggregate(x: np.ndarray, q: int) -> np.ndarray:
    """Non-overlapping sums of q consecutive base bins."""
    x = np.asarray(x, dtype=float)
    if q <= 1:
        return x
    n = (len(x) // q) * q
    if n == 0:
        return np.empty(0)
    return x[:n].reshape(-1, q).sum(axis=1)


def idc(x: np.ndarray) -> float:
    """Index of dispersion for counts, Var/Mean."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.mean() <= 0:
        return np.nan
    return float(x.var(ddof=1) / x.mean())


def idc_curve(x: np.ndarray, scales: Mapping[str, int] = SCALES) -> pd.Series:
    """IDC(scale) with the number of effective windows recorded elsewhere."""
    return pd.Series({name: idc(aggregate(x, q)) for name, q in scales.items()})


def effective_windows(n_bins: int, scales: Mapping[str, int] = SCALES) -> pd.Series:
    """n_q of guide eq. (44): long scales are supported by very few windows."""
    return pd.Series({name: int(n_bins // q) for name, q in scales.items()})


#: contribution of each 15-min lag to the lag-1 ACF of hourly (4-bin) sums
HOURLY_ACF_LAG_WEIGHTS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 2, 7: 1}


def hourly_acf1_from_autocov(gamma0: float, gamma: Mapping[int, float]) -> float:
    """Lag-1 ACF of hourly sums, given base-bin autocovariances gamma_1..gamma_7.

    Summing four consecutive bins makes the hourly covariance a triangular
    blend of the base-bin lags; the model-side calibrators use this to hit a
    target hourly ACF without simulating.
    """
    num = sum(w * gamma[h] for h, w in HOURLY_ACF_LAG_WEIGHTS.items())
    den = 4.0 * gamma0 + 2.0 * sum((4 - h) * gamma[h] for h in (1, 2, 3))
    return num / den


def acf(x: np.ndarray, lags: Sequence[int]) -> pd.Series:
    x = np.asarray(x, dtype=float)
    out = {}
    if len(x) < 3 or np.std(x) == 0:
        return pd.Series({int(l): np.nan for l in lags})
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    for lag in lags:
        lag = int(lag)
        out[lag] = float(np.sum(xc[:-lag] * xc[lag:]) / denom) if 0 < lag < len(x) else np.nan
    return pd.Series(out)


# episodes and tails
def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of the maximal True runs in `mask`."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return np.empty(0, dtype=int)
    d = np.diff(mask.astype(int))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return ends - starts


def episode_stats(x: np.ndarray, busy_threshold: float, quiet_threshold: float = 0.0) -> pd.Series:
    """Busy / quiet run texture. Thresholds must come from the training split."""
    x = np.asarray(x, dtype=float)
    busy = run_lengths(x > busy_threshold)
    quiet = run_lengths(x <= quiet_threshold)
    return pd.Series(
        {
            "n_busy_runs": int(busy.size),
            "busy_run_mean": float(busy.mean()) if busy.size else 0.0,
            "busy_run_q95": float(np.quantile(busy, 0.95)) if busy.size else 0.0,
            "busy_run_max": int(busy.max()) if busy.size else 0,
            "n_quiet_runs": int(quiet.size),
            "quiet_run_q95": float(np.quantile(quiet, 0.95)) if quiet.size else 0.0,
            "quiet_run_max": int(quiet.max()) if quiet.size else 0,
        }
    )


def quiet_run_vs_geometric(counts: np.ndarray) -> pd.Series:
    """Quiet-run tail against a geometric law with the same mean dwell.

    A two-state chain's OFF dwell is geometric by construction; excursions of
    an AR(1) below a threshold are not.  This is the texture-level fingerprint
    separating the gated model from MMPP.
    """
    x = np.asarray(counts, dtype=float)
    runs = run_lengths(x == 0)
    if runs.size < 5:
        return pd.Series(dtype=float)
    L = float(runs.mean())
    p = 1.0 / L

    def geo_q(q: float) -> float:
        return float(np.ceil(np.log(1 - q) / np.log(1 - p)))

    tail_emp = float(np.mean(runs > geo_q(0.99)))
    return pd.Series(
        {
            "n_quiet_runs": float(runs.size),
            "mean_run": L,
            "q95_empirical": float(np.quantile(runs, 0.95)),
            "q95_geometric_same_mean": geo_q(0.95),
            "q99_empirical": float(np.quantile(runs, 0.99)),
            "q99_geometric_same_mean": geo_q(0.99),
            "max_empirical": float(runs.max()),
            "P_exceed_geoQ99_empirical": tail_emp,
            "P_exceed_geoQ99_geometric": 0.01,
        }
    )


def rolling_tails(x: np.ndarray, window: int, quantiles: Sequence[float] = (0.95, 0.99)) -> pd.Series:
    """Scheduling-relevant extremes of a rolling window sum."""
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return pd.Series({f"roll{window}_q{int(q*100)}": np.nan for q in quantiles} | {f"roll{window}_max": np.nan})
    csum = np.r_[0.0, np.cumsum(x)]
    roll = csum[window:] - csum[:-window]
    out = {f"roll{window}_q{int(q*100)}": float(np.quantile(roll, q)) for q in quantiles}
    out[f"roll{window}_max"] = float(roll.max())
    return pd.Series(out)


def wavelet_variance(x: np.ndarray, n_levels: int = 8) -> pd.Series:
    """Haar wavelet variance: persistence from hours to weeks (guide section 1.11)."""
    x = np.asarray(x, dtype=float)
    out = {}
    cur = x.copy()
    for level in range(1, n_levels + 1):
        n = (len(cur) // 2) * 2
        if n < 4:
            break
        pairs = cur[:n].reshape(-1, 2)
        detail = (pairs[:, 0] - pairs[:, 1]) / np.sqrt(2.0)
        out[f"wv_level{level}"] = float(detail.var(ddof=1))
        cur = pairs.sum(axis=1)
    return pd.Series(out)


# microstructure of reconstructed events
def microstructure(job_times_s: np.ndarray, batch_times_s: np.ndarray, counts: np.ndarray) -> pd.Series:
    """Event-reconstruction fidelity: zero-IAT, positive gaps, empty bins."""
    job_times_s = np.sort(np.asarray(job_times_s, dtype=float))
    batch_times_s = np.sort(np.asarray(batch_times_s, dtype=float))
    job_iat = np.diff(job_times_s)
    gaps = np.diff(batch_times_s)
    pos = gaps[gaps > 0]
    return pd.Series(
        {
            "zero_iat_fraction": float(np.mean(job_iat == 0)) if job_iat.size else np.nan,
            "positive_gap_median_s": float(np.median(pos)) if pos.size else np.nan,
            "positive_gap_q99_s": float(np.quantile(pos, 0.99)) if pos.size else np.nan,
            "zero_bin_fraction": float(np.mean(np.asarray(counts) == 0)),
        }
    )


def base_marginals(x: np.ndarray, bins_per_hour: int = 4) -> dict:
    """The four base-bin facts the model is calibrated against, plus the max.

    One definition shared by the pipeline and the experiments, so a trace row
    and a synthetic row are always built by the same code.
    """
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(x.mean()),
        "IDC_15m": idc(x),
        "zero_frac": float((x == 0).mean()),
        "acf_1h_lag1": float(acf(aggregate(x, bins_per_hour), [1])[1]),
        "max_bin": float(x.max()),
    }


def metric_vector(
    x: np.ndarray,
    busy_threshold: float,
    quiet_threshold: float = 0.0,
    acf_lags: Sequence[int] = (1, 2, 4, 8, 24, 48, 96, 192, 384),
    rolling_windows: Sequence[int] = (4, 96),
) -> pd.Series:
    """One flat metric vector per path, used for gates and for Z scores."""
    x = np.asarray(x, dtype=float)
    parts = [
        pd.Series({"mean": float(x.mean()), "total": float(x.sum()), "n_bins": len(x)}),
        idc_curve(x).add_prefix("IDC_"),
        acf(x, acf_lags).rename(lambda l: f"ACF_lag{l}"),
        episode_stats(x, busy_threshold, quiet_threshold),
        wavelet_variance(x),
        pd.Series({"zero_bin_fraction": float((x == 0).mean()), "max_bin": float(x.max())}),
    ]
    for w in rolling_windows:
        parts.append(rolling_tails(x, w))
    return pd.concat(parts)


# discrepancies (guide eq. 22, 23, 24)
def d_idc(syn: np.ndarray, trace: np.ndarray, scales: Mapping[str, int] = SCALES) -> float:
    a, b = idc_curve(syn, scales), idc_curve(trace, scales)
    ratio = np.log(a.to_numpy() / b.to_numpy())
    ratio = ratio[np.isfinite(ratio)]
    return float(np.mean(ratio ** 2)) if ratio.size else np.nan


def d_idc_shape(syn: np.ndarray, trace: np.ndarray, scales: Mapping[str, int] = SCALES) -> float:
    """Same as d_idc after removing the level at the base scale."""
    a, b = np.log(idc_curve(syn, scales)), np.log(idc_curve(trace, scales))
    diff = (a - a.iloc[0]) - (b - b.iloc[0])
    diff = diff.to_numpy()[np.isfinite(diff.to_numpy())]
    return float(np.mean(diff ** 2)) if diff.size else np.nan


def d_acf(syn: np.ndarray, trace: np.ndarray, max_lag: int = 96) -> float:
    lags = np.arange(1, max_lag + 1)
    a, b = acf(syn, lags).to_numpy(), acf(trace, lags).to_numpy()
    w = 1.0 / (1.0 + lags)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return np.nan
    return float(np.sum(w[ok] * (a[ok] - b[ok]) ** 2) / np.sum(w[ok]))


def z_score(m_syn: float, m_trace: float, se_bootstrap: float, eps: float = 1e-9) -> float:
    """Guide eq. (24): penalise relative to the trace's own variability."""
    return float(abs(m_syn - m_trace) / max(se_bootstrap, eps))


def z_table(
    syn_metrics: pd.Series, trace_metrics: pd.Series, boot_se: pd.Series, eps_frac: float = 1e-3
) -> pd.DataFrame:
    rows = []
    for name in trace_metrics.index:
        if name not in syn_metrics.index:
            continue
        se = float(boot_se.get(name, np.nan))
        eps = eps_frac * max(abs(float(trace_metrics[name])), 1.0)
        rows.append(
            {
                "metric": name,
                "trace": float(trace_metrics[name]),
                "synthetic": float(syn_metrics[name]),
                "boot_se": se,
                "Z": z_score(float(syn_metrics[name]), float(trace_metrics[name]), se if np.isfinite(se) else eps, eps),
            }
        )
    return pd.DataFrame(rows).set_index("metric")


# analytic references, used by the metric unit tests (guide section 1.14)
def theoretical_idc_two_ar(
    mean: float, sigma: Sequence[float], phi: Sequence[float], kappa: float, q: int
) -> float:
    """IDC at aggregation scale q for the NB / sum-of-AR(1) model, eq. (41)."""
    V = float(np.sum(np.square(sigma)))
    m = float(mean)
    gamma0 = m + m * m * (np.exp(V) - 1.0 + np.exp(V) / kappa)
    total = q * gamma0
    for h in range(1, q):
        Ch = float(np.sum(np.square(sigma) * np.power(phi, h)))
        total += 2.0 * (q - h) * m * m * (np.exp(Ch) - 1.0)
    return total / (q * m)


def theoretical_acf_two_ar(
    mean: float, sigma: Sequence[float], phi: Sequence[float], kappa: float, lag: int
) -> float:
    """Guide eq. (37)."""
    V = float(np.sum(np.square(sigma)))
    m = float(mean)
    gamma0 = m + m * m * (np.exp(V) - 1.0 + np.exp(V) / kappa)
    Ch = float(np.sum(np.square(sigma) * np.power(phi, lag)))
    return m * m * (np.exp(Ch) - 1.0) / gamma0
