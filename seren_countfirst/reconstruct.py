"""Event reconstruction: counts -> batch timestamps -> sizes -> jobs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class ReconstructionResult:
    batch_times_s: np.ndarray
    batch_sizes: np.ndarray
    job_times_s: np.ndarray
    counts: np.ndarray
    collision_report: pd.Series

    @property
    def n_batches(self) -> int:
        return len(self.batch_times_s)

    @property
    def n_jobs(self) -> int:
        return int(self.batch_sizes.sum())


@dataclass
class OffsetTemplates:
    """Empirical within-bin offsets, optionally conditional on the count class."""

    pooled: np.ndarray
    by_class: Dict[str, np.ndarray]

    @classmethod
    def from_trace(cls, batch_times_s: np.ndarray, bin_seconds: float, counts: np.ndarray,
                   classes: np.ndarray | None = None) -> "OffsetTemplates":
        offsets = np.mod(np.asarray(batch_times_s, dtype=float), bin_seconds)
        by_class: Dict[str, np.ndarray] = {}
        if classes is not None:
            bin_index = (np.asarray(batch_times_s, dtype=float) // bin_seconds).astype(int)
            bin_index = np.clip(bin_index, 0, len(classes) - 1)
            labels = np.asarray(classes)[bin_index]
            for label in np.unique(labels):
                by_class[str(label)] = offsets[labels == label]
        return cls(pooled=offsets, by_class=by_class)

    def draw(self, n: int, rng: np.random.Generator, bin_seconds: float, label: str | None = None,
             resolution_s: float = 1.0) -> np.ndarray:
        """Return n distinct offsets inside a bin, on a grid of step `resolution_s`.

        The empirical pool is at the trace's one-second resolution. A finer grid
        samples a second with the empirical shape, then spreads uniformly across
        that second's sub-cells, so only the tie-breaking gets finer.
        `resolution_s = 1.0` reproduces one-second placement, RNG stream included.
        """
        cells_per_s = int(round(1.0 / resolution_s))
        slots = int(round(bin_seconds)) * cells_per_s
        if n > slots:
            raise ValueError(f"cannot place {n} distinct batches in {slots} slots of {resolution_s}s")
        pool = self.by_class.get(label, self.pooled) if label is not None else self.pooled
        if pool.size == 0:
            return np.sort(rng.choice(slots, size=n, replace=False)).astype(float) * resolution_s
        # sample with the empirical shape, then drop duplicates without merging
        cand = np.floor(rng.choice(pool, size=min(4 * n + 8, 4 * slots), replace=True)).astype(int)
        if cells_per_s > 1:
            cand = cand * cells_per_s + rng.integers(0, cells_per_s, size=cand.size)
        cand = np.clip(cand, 0, slots - 1)
        chosen: list[int] = []
        seen: set[int] = set()
        for c in cand:
            if c not in seen:
                seen.add(c)
                chosen.append(c)
            if len(chosen) == n:
                break
        if len(chosen) < n:
            remaining = np.setdiff1d(np.arange(slots), np.array(chosen, dtype=int), assume_unique=False)
            extra = rng.choice(remaining, size=n - len(chosen), replace=False)
            chosen.extend(int(e) for e in extra)
        return np.sort(np.array(chosen, dtype=float)) * resolution_s


def collision_risk(counts: np.ndarray, bin_seconds: float) -> pd.Series:
    """Return the slot-overflow probability and the worst bin."""
    slots = int(bin_seconds)
    counts = np.asarray(counts, dtype=float)
    over = counts > slots
    return pd.Series(
        {
            "slots_per_bin": slots,
            "max_count": float(counts.max()),
            "p_exceed_slots": float(over.mean()),
            "n_bins_exceeding": int(over.sum()),
            "max_fill_ratio": float(counts.max() / slots),
        }
    )


def reconstruct_events(
    counts: np.ndarray,
    bin_seconds: float,
    rng: np.random.Generator,
    batch_size_sampler,
    offsets: OffsetTemplates | None = None,
    classes: np.ndarray | None = None,
    on_overflow: str = "auto",
    resolution_s: float = 1.0,
    alpha: float = 1.0,
) -> ReconstructionResult:
    """Place exactly N_t distinct batch timestamps in every bin.

    The one-second grid comes from how the trace was recorded, not from the
    workload, so a finer grid recovers quantised-away resolution. Rounding the
    result back to whole seconds reproduces the saturated counts with inflated
    sizes, so that reading remains available as a rendering step.

    on_overflow:
      'auto'      'raise' at `alpha <= 1`, 'subsecond' above it
      'raise'     stop when a bin cannot hold its count; nothing is merged
      'subsecond' refine that bin only to the coarsest grid that fits, leaving
                  every other bin on whole seconds

    `alpha` is declared by the caller, not inferred: the counts alone cannot
    distinguish a scaled load from an unusual draw at nominal load.
    """
    counts = np.asarray(counts, dtype=float)
    if on_overflow == "auto":
        on_overflow = "subsecond" if float(alpha) > 1.0 else "raise"
    slots = int(round(bin_seconds / resolution_s))
    offsets = offsets or OffsetTemplates(pooled=np.empty(0), by_class={})

    times: list[np.ndarray] = []
    n_overflow_bins = 0
    finest_s = float(resolution_s)
    for t, n_t in enumerate(counts.astype(int)):
        if n_t <= 0:
            continue
        label = str(classes[t]) if classes is not None else None
        res_t = resolution_s
        if n_t > slots:
            n_overflow_bins += 1
            if on_overflow == "raise":
                raise ValueError(
                    f"bin {t} needs {n_t} distinct slots but only {slots} exist; "
                    "widen the bin, use a sub-second grid, or model collisions explicitly"
                )
            # coarsest refinement that fits: resolution degrades only where forced
            res_t = resolution_s / int(np.ceil(n_t / slots))
            finest_s = min(finest_s, res_t)
        local = offsets.draw(n_t, rng, bin_seconds, label, res_t)
        times.append(t * bin_seconds + local)

    batch_times = np.concatenate(times) if times else np.empty(0)
    batch_times.sort()
    sizes = batch_size_sampler(len(batch_times), rng, classes, counts, bin_seconds, batch_times)
    job_times = np.repeat(batch_times, sizes)

    # reconstruction conserves counts exactly
    assert len(batch_times) == int(counts.sum()), "batch count not conserved by reconstruction"
    assert len(job_times) == int(np.sum(sizes)), "job count not conserved by reconstruction"

    report = collision_risk(counts, bin_seconds)
    report["n_overflow_bins"] = n_overflow_bins
    report["grid_resolution_s"] = float(resolution_s)
    report["finest_resolution_s"] = float(finest_s)
    # batches that did not land on a whole second, i.e. the cost of refinement
    report["n_subsecond_placements"] = int(np.sum(np.mod(batch_times, 1.0) != 0.0))
    report["export_retention"] = 1.0  # every batch survives: nothing is ever merged
    return ReconstructionResult(batch_times, np.asarray(sizes, dtype=int), job_times, counts, report)


# batch-size marks: iid versus conditional on activity class
def make_iid_mark_sampler(empirical_sizes: np.ndarray):
    sizes = np.asarray(empirical_sizes, dtype=int)

    def sampler(n, rng, classes, counts, bin_seconds, batch_times):
        return rng.choice(sizes, size=n, replace=True)

    return sampler


def make_conditional_mark_sampler(empirical_sizes_by_class: Dict[str, np.ndarray], fallback: np.ndarray):
    pools = {k: np.asarray(v, dtype=int) for k, v in empirical_sizes_by_class.items()}
    fallback = np.asarray(fallback, dtype=int)

    def sampler(n, rng, classes, counts, bin_seconds, batch_times):
        if classes is None:
            return rng.choice(fallback, size=n, replace=True)
        bin_index = np.clip((np.asarray(batch_times) // bin_seconds).astype(int), 0, len(classes) - 1)
        labels = np.asarray(classes)[bin_index]
        out = np.empty(n, dtype=int)
        for label in np.unique(labels):
            mask = labels == label
            pool = pools.get(str(label), fallback)
            out[mask] = rng.choice(pool if pool.size else fallback, size=int(mask.sum()), replace=True)
        return out

    return sampler


def export_check(result: ReconstructionResult, counts: np.ndarray) -> pd.Series:
    """Return the batch and job conservation checks as a reportable row."""
    return pd.Series(
        {
            "N_export_batch": result.n_batches,
            "sum_counts": int(np.sum(counts)),
            "batch_conserved": result.n_batches == int(np.sum(counts)),
            "N_export_job": result.n_jobs,
            "sum_batch_sizes": int(result.batch_sizes.sum()),
            "job_conserved": result.n_jobs == int(result.batch_sizes.sum()),
            "distinct_timestamps": len(np.unique(result.batch_times_s)) == result.n_batches,
        }
    )
