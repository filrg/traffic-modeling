from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from .work import batch_frame, bin_grid_index
from .composition import CompositionJobSize
from .two_part import TAIL_QUANTILE, TwoPartJobSize

if TYPE_CHECKING:
    from .conditional import ConditionalJobSize


def rolling_origin_windows(bins: pd.DatetimeIndex, validation_days: float = 21.0,
                           n_folds: int = 4, min_train_days: float = 30.0
                           ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (train_end, validation_end) pairs; training always precedes validation.

    Same geometry as the count model's folds, so the two sides of the generator
    are validated over comparable stretches of the trace.
    """
    start, stop = bins.min(), bins.max()
    span = (stop - start) / pd.Timedelta(days=1)
    if span < float(min_train_days) + float(validation_days):
        return []
    origins = np.unique(np.linspace(float(min_train_days), span - float(validation_days),
                                    n_folds))
    return [(start + pd.Timedelta(days=float(o)),
             start + pd.Timedelta(days=float(o) + float(validation_days)))
            for o in origins]


def _window_arrays(window: pd.DataFrame):
    batches = batch_frame(window)
    card = batches["cardinality"].to_numpy(dtype=int)
    return (batches["activity"].to_numpy(dtype=float), card,
            bin_grid_index(batches["bin"]),
            np.repeat(batches["bin"].to_numpy(), card))


def score_window(model: "TwoPartJobSize", window: pd.DataFrame, n_rep: int = 20,
                 seed: int = 42) -> dict[str, float]:
    """Return what `model` produces on a window's own batch structure."""
    activity, card, index, bin_of_job = _window_arrays(window)
    sizes = window["job_size"].to_numpy(dtype=float)
    n_jobs = window.groupby("bin").size().to_numpy()
    blocks, ks, mean_ratio, med_ratio = [], [], [], []
    for r in range(n_rep):
        drawn = model.rvs(activity, card, index, np.random.default_rng(seed + r))
        work = pd.Series(drawn).groupby(bin_of_job).sum().to_numpy()
        blocks.append(offered_work_stats(work, n_jobs))
        ks.append(float(stats.ks_2samp(drawn, sizes).statistic))
        mean_ratio.append(float(drawn.mean() / sizes.mean()))
        med_ratio.append(float(np.median(drawn) / np.median(sizes)))
    out = {k: float(np.mean([b[k] for b in blocks])) for k in blocks[0]}
    out["ks_job_marginal"] = float(np.mean(ks))
    out["mean_work_ratio"] = float(np.mean(mean_ratio))
    out["median_ratio"] = float(np.mean(med_ratio))
    return out


def fold_report(jobs: pd.DataFrame, validation_days: float = 21.0, n_folds: int = 4,
                min_train_days: float = 30.0, quantile: float = TAIL_QUANTILE,
                n_init: int = 4, seed: int = 42, n_rep: int = 20,
                bulk_kind: str = "composition") -> pd.DataFrame:
    """Fit on each expanding training window and score on the window that follows.

    Three arms per fold: `out_of_sample` is the training-fitted law scored on
    ground it never saw, `in_window` is the same law refitted on the validation
    window and is the ceiling that window admits, and `observed` is the window
    itself.
    """
    rows = []
    windows = rolling_origin_windows(pd.DatetimeIndex(jobs["bin"]), validation_days,
                                     n_folds, min_train_days)
    for fold, (train_end, val_end) in enumerate(windows, start=1):
        at = jobs["bin"]
        train = jobs.loc[at < train_end].reset_index(drop=True)
        window = jobs.loc[(at >= train_end) & (at < val_end)].reset_index(drop=True)
        base = {"fold": fold, "train_end": train_end, "n_train": len(train),
                "n_val": len(window)}
        if window.empty or train.empty:
            continue
        for arm, source in (("out_of_sample", train), ("in_window", window)):
            try:
                model = TwoPartJobSize.fit(source, quantile=quantile, n_init=n_init,
                                           seed=seed, bulk_kind=bulk_kind)
            except (ValueError, RuntimeError) as exc:
                rows.append({**base, "arm": arm, "note": str(exc)[:70]})
                continue
            composition = isinstance(model.bulk, CompositionJobSize)
            rows.append({**base, "arm": arm, "k": model.bulk.n_components,
                         "beta": np.nan if composition else model.bulk.beta,
                         "escape": model.bulk.escape if composition else np.nan,
                         "mu": model.bulk.mu, "cut": model.cut,
                         "rate_a": model.rate_a, "rate_b": model.rate_b,
                         **score_window(model, window, n_rep, seed)})
        by_bin = window.groupby("bin")["job_size"]
        rows.append({**base, "arm": "observed",
                     **offered_work_stats(by_bin.sum().to_numpy(),
                                          by_bin.size().to_numpy()),
                     "ks_job_marginal": 0.0, "mean_work_ratio": 1.0,
                     "median_ratio": 1.0})
    return pd.DataFrame(rows)


def assign_work(batch_times_s: np.ndarray, batch_sizes: np.ndarray, counts: np.ndarray,
                bin_seconds: float, model: ConditionalJobSize | TwoPartJobSize,
                rng: np.random.Generator, alpha: float = 1.0) -> np.ndarray:
    """Return one work unit per job of a reconstruction, in job-arrival order.

    Activity is read from the generated counts, so the size model sees the bin
    occupancy the arrival model produced rather than the trace's. Takes arrays
    rather than a reconstruction so the arrival side stays size-agnostic.

    Pass the load factor as `alpha` to read those counts as a superposition of
    that many unit-load streams; leaving it at 1 reads them as one stream that
    became busier. The two readings are different claims about load.
    """
    counts = np.asarray(counts, dtype=float)
    bin_of_batch = (np.asarray(batch_times_s, dtype=float) // float(bin_seconds)).astype(int)
    bin_of_batch = np.clip(bin_of_batch, 0, len(counts) - 1)
    activity = counts[bin_of_batch]
    if np.any(activity < 1.0):
        raise ValueError("a batch sits in a bin the counts record as empty")
    card = np.asarray(batch_sizes, dtype=int)
    # the grid position, not a compacted one: the size latent lives on real time
    if isinstance(model, TwoPartJobSize):
        return model.rvs(activity, card, bin_of_batch, rng, alpha=alpha)
    return model.rvs(activity, card, rng, alpha=alpha, bin_index=bin_of_batch)


def offered_work_stats(work: np.ndarray, n_jobs: np.ndarray) -> dict[str, float]:
    """Return the spread of work per occupied bin and how it tracks the job count."""
    w = np.asarray(work, dtype=float)
    ranked = np.sort(w)[::-1]
    top = max(int(round(0.05 * len(ranked))), 1)
    return {
        "cv": float(w.std(ddof=1) / w.mean()),
        "q99_over_median": float(np.quantile(w, 0.99) / np.median(w)),
        "spearman_jobs_work": float(stats.spearmanr(np.asarray(n_jobs, dtype=float),
                                                    w).statistic),
        "top5pct_share": float(ranked[:top].sum() / ranked.sum()),
    }
