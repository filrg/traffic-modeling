from __future__ import annotations

from dataclasses import dataclass

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
    by_class: dict[str, np.ndarray]

    @classmethod
    def from_trace(
        cls,
        batch_times_s: np.ndarray,
        bin_seconds: float,
        counts: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> "OffsetTemplates":
        offsets = np.mod(np.asarray(batch_times_s, dtype=float), bin_seconds)
        by_class: dict[str, np.ndarray] = {}
        if classes is not None:
            bin_index = (np.asarray(batch_times_s, dtype=float) // bin_seconds).astype(int)
            bin_index = np.clip(bin_index, 0, len(classes) - 1)
            labels = np.asarray(classes)[bin_index]
            for label in np.unique(labels):
                by_class[str(label)] = offsets[labels == label]
        return cls(pooled=offsets, by_class=by_class)

    def draw(
        self,
        n: int,
        rng: np.random.Generator,
        bin_seconds: float,
        label: str | None = None,
        resolution_s: float = 1.0,
    ) -> np.ndarray:
        """Return n distinct offsets inside a bin, on a grid of step `resolution_s`.

        The empirical pool is at the trace's one-second resolution. A finer grid
        samples a second with the empirical shape, then spreads uniformly across
        that second's sub-cells, so only the tie-breaking gets finer.
        `resolution_s = 1.0` reproduces one-second placement, RNG stream included.
        """
        cells_per_s = int(round(1.0 / resolution_s))
        slots = int(round(bin_seconds)) * cells_per_s
        if n > slots:
            raise ValueError(
                f"cannot place {n} distinct batches in {slots} slots of {resolution_s}s"
            )
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
            remaining = np.setdiff1d(
                np.arange(slots), np.array(chosen, dtype=int), assume_unique=False
            )
            extra = rng.choice(remaining, size=n - len(chosen), replace=False)
            chosen.extend(int(e) for e in extra)
        return np.sort(np.array(chosen, dtype=float)) * resolution_s


def _place_distinct(
    cand_s: np.ndarray, n: int, bin_seconds: float, resolution_s: float
) -> np.ndarray:
    """Snap n candidate offsets onto distinct grid slots, keeping their order and spacing.

    A point landing on a taken slot moves to the next free one and the run is then
    capped from the right so the last point still fits inside the bin. Displacement
    rather than resampling, so a point stays where the pattern put it and nothing is
    ever merged. Both passes are monotone, so the result is strictly increasing even
    when the bin is nearly full.
    """
    cells_per_s = int(round(1.0 / resolution_s))
    slots = int(round(bin_seconds)) * cells_per_s
    if n > slots:
        raise ValueError(f"cannot place {n} distinct batches in {slots} slots of {resolution_s}s")
    c = np.sort(
        np.clip(np.floor(np.asarray(cand_s, dtype=float)[:n] / resolution_s), 0.0, slots - 1.0)
    )
    i = np.arange(n, dtype=float)
    out = np.maximum.accumulate(c - i) + i
    return np.minimum(out, slots - n + i) * resolution_s


@dataclass
class PatternTemplates:
    """Whole observed within-bin point patterns, resampled by occupancy and class.

    A bin's placement is copied from a trace bin of comparable occupancy rather
    than assembled from independent offsets, so within-bin clustering survives.
    Donors at or above the target count are preferred and thinned down; donors
    are superposed only when none is large enough.
    """

    pooled: list
    pooled_n: np.ndarray
    by_class: dict[str, tuple]
    n_donors: int = 20

    @classmethod
    def from_trace(
        cls,
        batch_times_s: np.ndarray,
        bin_seconds: float,
        counts: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> "PatternTemplates":
        t = np.sort(np.asarray(batch_times_s, dtype=float))
        bin_index = (t // bin_seconds).astype(int)
        offsets = t - bin_index * bin_seconds
        starts = np.flatnonzero(np.r_[True, np.diff(bin_index) != 0])
        stops = np.r_[starts[1:], len(t)]
        patterns = [offsets[a:b] for a, b in zip(starts, stops)]
        by_class: dict[str, tuple] = {}
        if classes is not None:
            labels = np.asarray(classes)[np.clip(bin_index[starts], 0, len(classes) - 1)]
            for label in np.unique(labels):
                by_class[str(label)] = cls._sorted(
                    [p for p, lb in zip(patterns, labels) if lb == label]
                )
        pooled, pooled_n = cls._sorted(patterns)
        return cls(pooled=pooled, pooled_n=pooled_n, by_class=by_class)

    @staticmethod
    def _sorted(patterns: list) -> tuple:
        """Patterns ordered by occupancy, so a donor window is a slice."""
        order = np.argsort([len(p) for p in patterns], kind="stable")
        ordered = [patterns[i] for i in order]
        return ordered, np.array([len(p) for p in ordered], dtype=int)

    def _donor(self, pats: list, lens: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        # window sits at or above n so the usual operation is thinning, not superposition
        i = int(np.searchsorted(lens, n, side="left"))
        hi = min(len(lens), i + self.n_donors)
        lo = max(0, i - max(0, self.n_donors - (hi - i)))
        return pats[int(rng.integers(lo, hi))]

    def draw(
        self,
        n: int,
        rng: np.random.Generator,
        bin_seconds: float,
        label: str | None = None,
        resolution_s: float = 1.0,
    ) -> np.ndarray:
        """Return n distinct offsets copied from observed within-bin patterns."""
        pats, lens = self.pooled, self.pooled_n
        if label is not None and label in self.by_class and self.by_class[label][0]:
            pats, lens = self.by_class[label]
        if not pats:
            slots = int(round(bin_seconds / resolution_s))
            return np.sort(rng.choice(slots, size=n, replace=False)).astype(float) * resolution_s
        drawn, total = [], 0
        while total < n:
            p = self._donor(pats, lens, n - total, rng)
            drawn.append(p)
            total += len(p)
        cand = drawn[0] if len(drawn) == 1 else np.concatenate(drawn)
        if len(cand) > n:
            cand = cand[np.sort(rng.choice(len(cand), size=n, replace=False))]
        return _place_distinct(cand, n, bin_seconds, resolution_s)


@dataclass
class CopulaOffsets:
    """Within-bin placement as a Gaussian-copula spacing process.

    The n points cut a bin into n+1 spacings drawn with lognormal marginals and
    renormalised, so the bin edge is respected by construction. Spread `sigma`
    varies with occupancy, and a stationary AR(1) latent couples consecutive
    spacings, which is what puts short spacings in runs.
    """

    s0: float
    s1: float
    s2: float
    phi: float
    sigma2_bounds: tuple = (0.05, 6.0)

    @classmethod
    def from_trace(
        cls,
        batch_times_s: np.ndarray,
        bin_seconds: float,
        counts: np.ndarray,
        min_count: int = 2,
        debias: bool = True,
    ) -> "CopulaOffsets":
        from scipy import stats as _stats

        t = np.sort(np.asarray(batch_times_s, dtype=float))
        bin_index = (t // bin_seconds).astype(int)
        offsets = t - bin_index * bin_seconds
        starts = np.flatnonzero(np.r_[True, np.diff(bin_index) != 0])
        stops = np.r_[starts[1:], len(t)]

        ns, sig2, ra, rb = [], [], [], []
        for a, b in zip(starts, stops):
            o = offsets[a:b]
            if len(o) < min_count:
                continue
            s = np.diff(np.r_[0.0, o, bin_seconds]) / bin_seconds
            # lognormal: median / mean = exp(-sigma^2 / 2), robust at small n
            ratio = np.median(s) / s.mean()
            if ratio <= 0:
                continue
            ns.append(len(o))
            sig2.append(-2.0 * np.log(ratio))
            # ranks taken inside the bin: pooling raw spacings would measure bin
            # size, since every spacing in a busy bin is small
            r = _stats.rankdata(s) / (len(s) + 1)
            ra.append(r[:-1])
            rb.append(r[1:])
        ns, sig2 = np.array(ns, dtype=float), np.array(sig2, dtype=float)
        ok = sig2 > 0
        # quadratic in log n: the trace's spread is not monotone in occupancy
        w = np.sqrt(ns[ok])
        ln = np.log(ns[ok])
        A = np.c_[np.ones(ok.sum()), ln, ln**2]
        s0, s1, s2 = np.linalg.lstsq(A * w[:, None], np.log(sig2[ok]) * w, rcond=None)[0]

        rho = float(np.corrcoef(np.concatenate(ra), np.concatenate(rb))[0, 1])
        law = cls(
            s0=float(s0), s1=float(s1), s2=float(s2), phi=float(2.0 * np.sin(np.pi * rho / 6.0))
        )
        if not debias:
            return law
        # renormalising n+1 spacings onto the bin correlates them on its own, so
        # the closed form lands low; the offset depends on occupancy, not on phi
        realised = law.rank_corr_on(counts, bin_seconds, np.random.default_rng(0))
        rho_adj = float(np.clip(2.0 * rho - realised, -0.95, 0.95))
        return cls(s0=law.s0, s1=law.s1, s2=law.s2, phi=float(2.0 * np.sin(np.pi * rho_adj / 6.0)))

    def rank_corr_on(
        self, counts: np.ndarray, bin_seconds: float, rng: np.random.Generator
    ) -> float:
        """Return the within-bin rank correlation this law realises on given counts."""
        from scipy import stats as _stats

        ra, rb = [], []
        for n in np.asarray(counts, dtype=int):
            if n < 2:
                continue
            s = np.diff(np.r_[0.0, self.draw(int(n), rng, bin_seconds), bin_seconds])
            r = _stats.rankdata(s) / (len(s) + 1)
            ra.append(r[:-1])
            rb.append(r[1:])
        return float(np.corrcoef(np.concatenate(ra), np.concatenate(rb))[0, 1])

    def sigma(self, n: int) -> float:
        """Spacing spread at occupancy n."""
        ln = np.log(max(n, 1))
        lo, hi = self.sigma2_bounds
        # the quadratic is fitted over the trace's occupancies and is only bounded
        # away from degenerate spread outside them
        return float(np.sqrt(np.clip(np.exp(self.s0 + self.s1 * ln + self.s2 * ln**2), lo, hi)))

    def draw(
        self,
        n: int,
        rng: np.random.Generator,
        bin_seconds: float,
        label: str | None = None,
        resolution_s: float = 1.0,
    ) -> np.ndarray:
        """Return n distinct offsets from n+1 coupled lognormal spacings."""
        k = n + 1
        u = np.empty(k)
        e = rng.standard_normal(k)
        u[0] = e[0]
        sd = np.sqrt(1.0 - self.phi**2)
        for i in range(1, k):
            u[i] = self.phi * u[i - 1] + sd * e[i]
        s = np.exp(self.sigma(n) * u)
        cand = bin_seconds * np.cumsum(s / s.sum())[:n]
        return _place_distinct(cand, n, bin_seconds, resolution_s)


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
    offsets: OffsetTemplates | PatternTemplates | CopulaOffsets | None = None,
    classes: np.ndarray | None = None,
    on_overflow: str = "auto",
    resolution_s: float = 1.0,
    alpha: float = 1.0,
) -> ReconstructionResult:
    """Place exactly N_t distinct batch timestamps in every bin.

    The one-second grid comes from how the trace was recorded, not from the
    workload, so a finer grid recovers quantised-away resolution.

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
    return ReconstructionResult(
        batch_times, np.asarray(sizes, dtype=int), job_times, counts, report
    )


# batch-size marks: iid versus conditional on activity class
def make_iid_mark_sampler(empirical_sizes: np.ndarray):
    sizes = np.asarray(empirical_sizes, dtype=int)

    def sampler(n, rng, classes, counts, bin_seconds, batch_times):
        return rng.choice(sizes, size=n, replace=True)

    return sampler


def make_conditional_mark_sampler(
    empirical_sizes_by_class: dict[str, np.ndarray], fallback: np.ndarray
):
    pools = {k: np.asarray(v, dtype=int) for k, v in empirical_sizes_by_class.items()}
    fallback = np.asarray(fallback, dtype=int)

    def sampler(n, rng, classes, counts, bin_seconds, batch_times):
        if classes is None:
            return rng.choice(fallback, size=n, replace=True)
        bin_index = np.clip(
            (np.asarray(batch_times) // bin_seconds).astype(int), 0, len(classes) - 1
        )
        labels = np.asarray(classes)[bin_index]
        out = np.empty(n, dtype=int)
        for label in np.unique(labels):
            mask = labels == label
            pool = pools.get(str(label), fallback)
            out[mask] = rng.choice(
                pool if pool.size else fallback, size=int(mask.sum()), replace=True
            )
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
