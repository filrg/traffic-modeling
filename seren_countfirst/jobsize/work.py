from __future__ import annotations

import numpy as np
import pandas as pd

from traffic_modeling.counts.series import BASE_FREQ

# A100 latency coefficients at the trace's hardware
A100_ALPHA_T = 0.0050
A100_BETA_T = 0.038
A100_GAMMA_T = 0.0014
# AcmeTrace ran at the default (max) clock, so the inversion is at f = 1 GHz
F_MAX = 1.0

# A size is the work the cluster absorbed; the arrival view keeps every state.
JOBSIZE_STATES = ("COMPLETED", "FAILED", "CANCELLED")


def step_time_s(
    n_gpus: np.ndarray | float,
    f_ghz: float = F_MAX,
    alpha_t: float = A100_ALPHA_T,
    beta_t: float = A100_BETA_T,
    gamma_t: float = A100_GAMMA_T,
) -> np.ndarray:
    """Return seconds per work unit on `n_gpus` at `f_ghz`."""
    n = np.maximum(1.0, np.asarray(n_gpus, dtype=float))
    f = max(float(f_ghz), 1e-9)
    single = alpha_t + beta_t / f
    multi = (alpha_t + beta_t / f + gamma_t * n) / n
    return np.where(n == 1.0, single, multi)


def work_units(duration_s: np.ndarray, n_gpus: np.ndarray, f_ghz: float = F_MAX) -> np.ndarray:
    """Invert a wall-clock duration into work units at the trace's operating point.

    Exact inverse of `step_time_s`; `jobsize_table` reports the round trip.
    """
    return np.asarray(duration_s, dtype=float) / step_time_s(n_gpus, f_ghz)


def jobsize_table(view: pd.DataFrame, f_ghz: float = F_MAX) -> pd.DataFrame:
    """Return work units for every job in the view with a usable duration and GPU count.

    Input follows the canonical jobs schema and preserves submission order.
    """
    usable = view.loc[(view["duration_s"] > 0) & (view["gpu_count"] > 0)].copy()
    usable["job_size"] = work_units(
        usable["duration_s"].to_numpy(), usable["gpu_count"].to_numpy(), f_ghz
    )
    return usable.reset_index(drop=True)


def roundtrip_error_s(table: pd.DataFrame, f_ghz: float = F_MAX) -> float:
    """Return the largest absolute second the inversion fails to give back."""
    back = table["job_size"].to_numpy() * step_time_s(table["gpu_count"].to_numpy(), f_ghz)
    return float(np.max(np.abs(back - table["duration_s"].to_numpy())))


def attach_activity(
    table: pd.DataFrame, arrivals: pd.DataFrame, freq: str = BASE_FREQ
) -> pd.DataFrame:
    """Add each job's bin and that bin's batch count, dropping non-positive sizes.

    Activity counts batches over every submission, not only the completed ones:
    it is the quantity the count model generates.
    """
    activity = (
        arrivals.assign(_bin=arrivals["submitted_at"].dt.floor(freq))
        .groupby("_bin")["submitted_at"]
        .nunique()
    )
    jobs = table.loc[table["job_size"] > 0.0].copy()
    jobs["bin"] = jobs["submitted_at"].dt.floor(freq)
    jobs["activity"] = jobs["bin"].map(activity).to_numpy(dtype=float)
    if not np.all(np.isfinite(jobs["activity"].to_numpy())):
        raise ValueError("a completed job sits in a bin the arrival view does not cover")
    return jobs.reset_index(drop=True)


def batch_frame(jobs: pd.DataFrame) -> pd.DataFrame:
    """Collapse jobs sharing a submission timestamp into one batch row."""
    grouped = jobs.assign(z=np.log(jobs["job_size"].to_numpy(dtype=float))).groupby(
        ["bin", "submitted_at"], sort=True
    )
    return grouped.agg(
        z_bar=("z", "mean"), cardinality=("z", "size"), activity=("activity", "first")
    ).reset_index()


def bin_grid_index(bins, freq: str = BASE_FREQ) -> np.ndarray:
    """Return each row's slot on the regular bin grid, counted from the earliest.

    The latent lives on real time, so gaps between occupied bins have to be real
    gaps; a compacted index would make bins hours apart look adjacent.
    """
    at = pd.DatetimeIndex(bins)
    return ((at - at.min()) // pd.Timedelta(freq)).to_numpy().astype(int)


def _stream_activity(activity: np.ndarray, alpha: float) -> np.ndarray:
    """Return the occupancy one unit-load stream sees inside a bin of `activity`.

    The mean share is `activity / alpha`, floored at one because a stream that
    contributed a batch holds at least it. `alpha = 1` is the identity.
    """
    return np.maximum(np.asarray(activity, dtype=float) / float(alpha), 1.0)
