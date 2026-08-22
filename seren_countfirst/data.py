from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
from traffic_modeling.datasets import sha256_file as _sha256_file

BASE_FREQ = "15min"
BASE_BIN_S = 900.0
SECONDS_PER_DAY = 86400.0

# aggregation scales, in base bins
SCALES: dict[str, int] = {
    "15m": 1,
    "1h": 4,
    "6h": 24,
    "1d": 96,
    "2d": 192,
    "4d": 384,
    "7d": 672,
}

# A size is the work the cluster absorbed, arrivals keep every state.
JOBSIZE_STATES = ("COMPLETED", "FAILED", "CANCELLED")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Compatibility wrapper around the shared dataset-integrity helper."""
    return _sha256_file(path, chunk_size=chunk)


def load_trace(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw trace and parse timestamps in their native timezone."""
    raw = pd.read_csv(csv_path, low_memory=False)
    for col in ("submit_time", "start_time", "end_time"):
        if col in raw.columns:
            raw[col + "_dt"] = pd.to_datetime(raw[col], errors="coerce")
    return raw


def arrival_view(raw: pd.DataFrame, job_type: str = "SFT") -> pd.DataFrame:
    """Return every valid submission of the target type, stable-sorted.

    Failed, cancelled and preempted submissions are kept.
    """
    view = raw.loc[raw["type"].eq(job_type)].copy()
    view = view.sort_values(["submit_time_dt", "job_id"], kind="stable")
    return view.reset_index(drop=True)


def jobsize_view(raw: pd.DataFrame, job_type: str = "SFT") -> pd.DataFrame:
    """Return the jobs whose runtime is work the cluster had to absorb."""
    view = arrival_view(raw, job_type)
    return view.loc[view["state"].isin(JOBSIZE_STATES)].reset_index(drop=True)


@dataclass
class BatchTable:
    """Exact-timestamp batches: the atomic arrival unit."""

    times: pd.DatetimeIndex
    sizes: np.ndarray

    @property
    def n_batches(self) -> int:
        return len(self.times)

    @property
    def n_jobs(self) -> int:
        return int(self.sizes.sum())

    def summary(self) -> pd.Series:
        gaps = self.times.to_series().diff().dt.total_seconds().to_numpy()[1:]
        pos = gaps[gaps > 0]
        return pd.Series(
            {
                "n_batches": self.n_batches,
                "n_jobs": self.n_jobs,
                "mean_batch_size": float(self.sizes.mean()),
                "q99_batch_size": float(np.quantile(self.sizes, 0.99)),
                "max_batch_size": int(self.sizes.max()),
                "median_positive_gap_s": float(np.median(pos)) if len(pos) else np.nan,
                "max_positive_gap_days": float(pos.max() / SECONDS_PER_DAY) if len(pos) else np.nan,
            }
        )


def build_batches(view: pd.DataFrame, time_col: str = "submit_time_dt") -> BatchTable:
    """Group jobs that share an exact timestamp into one batch."""
    grouped = view.groupby(time_col, sort=True).size()
    return BatchTable(times=pd.DatetimeIndex(grouped.index), sizes=grouped.to_numpy(dtype=int))


def zero_iat_fraction(view: pd.DataFrame, time_col: str = "submit_time_dt") -> float:
    iat = view[time_col].diff().dt.total_seconds().to_numpy()[1:]
    return float(np.mean(iat == 0)) if len(iat) else np.nan


@dataclass
class CountSeries:
    """Batch counts on a regular grid. Zero bins are data, not gaps."""

    values: np.ndarray
    index: pd.DatetimeIndex
    freq: str = BASE_FREQ
    bin_seconds: float = BASE_BIN_S

    def __len__(self) -> int:
        return len(self.values)

    @property
    def duration_days(self) -> float:
        return len(self.values) * self.bin_seconds / SECONDS_PER_DAY

    def to_series(self) -> pd.Series:
        return pd.Series(self.values, index=self.index, name="batch_count")

    def slice_bins(self, lo: int, hi: int) -> "CountSeries":
        return CountSeries(self.values[lo:hi], self.index[lo:hi], self.freq, self.bin_seconds)

    def summary(self) -> pd.Series:
        v = self.values
        return pd.Series(
            {
                "n_bins": len(v),
                "duration_days": self.duration_days,
                "mean": float(v.mean()),
                "variance": float(v.var(ddof=1)),
                "IDC": float(v.var(ddof=1) / v.mean()) if v.mean() > 0 else np.nan,
                "zero_fraction": float((v == 0).mean()),
                "q99": float(np.quantile(v, 0.99)),
                "max": int(v.max()),
            }
        )


def build_counts(
    batch_times: pd.DatetimeIndex,
    freq: str = BASE_FREQ,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> CountSeries:
    """Bin batch timestamps onto a regular grid, preserving empty bins."""
    start = (start or batch_times.min()).floor(freq)
    end = (end or batch_times.max()).ceil(freq)
    grid = pd.date_range(start, end, freq=freq, tz=batch_times.tz, inclusive="left")
    counts = pd.Series(1, index=batch_times).resample(freq).sum().reindex(grid, fill_value=0)
    bin_s = pd.Timedelta(freq).total_seconds()
    return CountSeries(counts.to_numpy(dtype=float), grid, freq, bin_s)


# splits are contiguous or rolling-origin only, never random K-fold
@dataclass
class Split:
    name: str
    train: CountSeries
    validation: CountSeries
    test: CountSeries | None = None

    def describe(self) -> pd.DataFrame:
        rows = []
        for part in ("train", "validation", "test"):
            cs = getattr(self, part)
            if cs is None:
                continue
            rows.append(
                {"part": part, "start": cs.index[0], "end": cs.index[-1], **cs.summary().to_dict()}
            )
        return pd.DataFrame(rows)


def contiguous_split(
    counts: CountSeries, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> Split:
    n = len(counts)
    a = int(round(n * fractions[0]))
    b = a + int(round(n * fractions[1]))
    return Split(
        "contiguous", counts.slice_bins(0, a), counts.slice_bins(a, b), counts.slice_bins(b, n)
    )


def expanding_window_folds(
    counts: CountSeries,
    validation_days: float = 21.0,
    n_folds: int = 4,
    min_train_days: float = 30.0,
) -> list[Split]:
    """Rolling-origin folds: training always precedes validation in time."""
    per_day = int(round(SECONDS_PER_DAY / counts.bin_seconds))
    val_bins = int(validation_days * per_day)
    min_train = int(min_train_days * per_day)
    n = len(counts)
    folds: list[Split] = []
    if n < min_train + val_bins:
        return folds
    last_origin = n - val_bins
    origins = np.linspace(min_train, last_origin, n_folds).astype(int)
    for k, origin in enumerate(np.unique(origins), start=1):
        folds.append(
            Split(
                f"expanding_{k}",
                counts.slice_bins(0, origin),
                counts.slice_bins(origin, origin + val_bins),
            )
        )
    return folds


@dataclass
class ActivityScheme:
    """Quantile thresholds learned on training data, then frozen everywhere."""

    edges: tuple[float, float, float]
    labels: tuple[str, str, str, str] = ("low", "medium", "high", "extreme")
    busy_threshold: float = 0.0
    quiet_threshold: float = 0.0
    source: str = "train"

    @classmethod
    def fit(
        cls, counts: CountSeries, quantiles=(0.50, 0.80, 0.95), busy_q: float = 0.80
    ) -> "ActivityScheme":
        occupied = counts.values[counts.values > 0]
        if occupied.size == 0:
            raise ValueError("no occupied bins: cannot fit activity thresholds")
        q = cast(
            tuple[float, float, float],
            tuple(float(x) for x in np.asarray(np.quantile(occupied, quantiles))),
        )
        return cls(
            edges=q, busy_threshold=float(np.quantile(occupied, busy_q)), quiet_threshold=0.0
        )

    def classify(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        out = np.full(v.shape, self.labels[3], dtype=object)
        out[v <= self.edges[2]] = self.labels[2]
        out[v <= self.edges[1]] = self.labels[1]
        out[v <= self.edges[0]] = self.labels[0]
        return out

    @classmethod
    def from_dict(cls, blob: Mapping[str, Any]) -> "ActivityScheme":
        raw_edges = tuple(float(value) for value in blob["edges"])
        if len(raw_edges) != 3:
            raise ValueError("activity scheme requires exactly three edges")
        return cls(
            edges=cast(tuple[float, float, float], raw_edges),
            labels=tuple(blob["labels"]),
            busy_threshold=float(blob["busy_threshold"]),
            quiet_threshold=float(blob["quiet_threshold"]),
            source=str(blob["source"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": list(self.edges),
            "labels": list(self.labels),
            "busy_threshold": self.busy_threshold,
            "quiet_threshold": self.quiet_threshold,
            "source": self.source,
        }


def hour_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    """Return the hour-of-week phase from timezone-aware stamps; no fallback phase."""
    return (index.dayofweek * 24 + index.hour).to_numpy()


def segment_profiles(counts: CountSeries, n_segments: int = 3) -> pd.DataFrame:
    """Return normalised hour-of-week profiles, one per contiguous segment."""
    hourly = counts.to_series().resample("1h").sum()
    how = hourly.index.dayofweek * 24 + hourly.index.hour
    frame = pd.DataFrame({"count": hourly.to_numpy(), "how": how})
    chunks = np.array_split(np.arange(len(frame)), n_segments)
    out = {}
    for k, idx in enumerate(chunks, start=1):
        part = frame.iloc[idx]
        profile = part.groupby("how")["count"].mean().reindex(range(168), fill_value=0.0)
        out[f"segment_{k}"] = profile / max(profile.mean(), 1e-12)
    return pd.DataFrame(out)
