from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from traffic_modeling.counts.metrics import episode_stats, rolling_tails

HORIZON_DAYS: tuple = (1, 7, 30, 90, 180)


def horizon_distribution(
    model,
    horizon_days: Sequence[int] = HORIZON_DAYS,
    n_paths: int = 30,
    bins_per_day: int = 96,
    alpha: float = 1.0,
    busy_threshold: float = 1.0,
    seed: int = 0,
    **sim_kwargs,
) -> pd.DataFrame:
    """Return the distribution of each extensive statistic per horizon.

    Totals, maxima, episode counts and longest runs are comparable only between
    windows of equal duration, so the caller must match trace windows to the
    same horizon.
    """
    rows: list[dict] = []
    for days in horizon_days:
        n_bins = int(days * bins_per_day)
        rng = np.random.default_rng(seed + days)
        stats = []
        for _ in range(n_paths):
            path = model.simulate(n_bins, rng, alpha=alpha, **sim_kwargs)
            hourly = (
                path[: (len(path) // 4) * 4].reshape(-1, 4).sum(axis=1) if n_bins >= 4 else path
            )
            daily = (
                path[: (len(path) // 96) * 96].reshape(-1, 96).sum(axis=1) if n_bins >= 96 else path
            )
            ep = episode_stats(path, busy_threshold)
            stats.append(
                {
                    "total_batches": float(path.sum()),
                    "realised_rate_per_bin": float(path.mean()),
                    "max_hourly": float(hourly.max()) if hourly.size else np.nan,
                    "max_daily": float(daily.max()) if daily.size else np.nan,
                    "n_busy_episodes": float(ep["n_busy_runs"]),
                    "longest_busy_run": float(ep["busy_run_max"]),
                    "longest_quiet_run": float(ep["quiet_run_max"]),
                    "roll4_q99": float(rolling_tails(path, 4, (0.99,)).iloc[0])
                    if n_bins > 4
                    else np.nan,
                }
            )
        frame = pd.DataFrame(stats)
        for col in frame.columns:
            rows.append(
                {
                    "horizon_days": days,
                    "n_bins": n_bins,
                    "statistic": col,
                    "mean": float(frame[col].mean()),
                    "sd": float(frame[col].std(ddof=1)),
                    "cv": float(frame[col].std(ddof=1) / frame[col].mean())
                    if frame[col].mean()
                    else np.nan,
                    "q025": float(frame[col].quantile(0.025)),
                    "q975": float(frame[col].quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)
