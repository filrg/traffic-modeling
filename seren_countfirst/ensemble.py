from __future__ import annotations

import numpy as np

from .metrics import base_marginals
from .paths import STATE_DIR
from .state import load_portable_state, rebuild_model

# must equal run_pipeline.config.SEED; a test pins the equality
SEED = 20260729
BINS_PER_DAY = 96


def state() -> dict:
    """Load the portable pipeline state."""
    return load_portable_state(STATE_DIR)


def calibrated(st: dict, key: str) -> tuple:
    """Return (model, alpha) for one state key, with the global rate calibration."""
    m = rebuild_model(st["models_json"][key])
    return m, st["trace_marginals"]["mean"] / float(m.params_["mean"])


def winner(st: dict) -> tuple:
    """Return (model, alpha) for the frozen winner."""
    return calibrated(st, st["winner"])


def daily_peaks(counts: np.ndarray) -> np.ndarray:
    """Return the largest bin in each whole day; a trailing partial day is dropped."""
    x = np.asarray(counts, dtype=float)
    n_days = len(x) // BINS_PER_DAY
    return x[: n_days * BINS_PER_DAY].reshape(n_days, BINS_PER_DAY).max(axis=1)


# the quantiles a displayed run is judged typical on
PEAK_Q = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
# the shared ensemble every figure that displays a single run draws from
DISPLAY_OFFSET = 4
# its size, shared rather than repeated per module: `representative` ranks
# against the pool's own median, so a different count selects a different run
DISPLAY_CANDIDATES = 200


def _profiles(pool) -> tuple:
    """Return each run's daily-peak quantiles and hourly lag-one autocorrelation."""
    peaks = np.array([np.quantile(daily_peaks(r), PEAK_Q) for r in pool])
    acf = np.clip([base_marginals(r)["acf_1h_lag1"] for r in pool], 1e-3, None)
    return peaks, acf


def _rank(peaks, acf, ref_peaks, ref_acf) -> np.ndarray:
    """Distance to a reference profile; +1 on the quantiles only, an acf is never zero."""
    return (np.abs(np.log((peaks + 1.0) / (ref_peaks + 1.0))).sum(axis=1)
            + np.abs(np.log(acf / ref_acf)))


def _at_volume(runs: list, target_mean: float, tol: float) -> list:
    """Indices of the runs carrying `target_mean` to within `tol`."""
    return [i for i, r in enumerate(runs) if abs(r.mean() / target_mean - 1.0) <= tol]


def closest_to_trace(runs: list, trace: np.ndarray, target_mean: float | None = None,
                     tol: float = 0.20) -> int:
    """Return the index of the run nearest the trace on `representative`'s profile.

    Selects *on the trace*, so it may be applied to a baseline only while the
    winner keeps its typical run, never the reverse. `target_mean` holds the
    comparison at one volume.
    """
    idx = _at_volume(runs, target_mean, tol) if target_mean is not None else []
    idx = idx or list(range(len(runs)))
    peaks, acf = _profiles([runs[i] for i in idx])
    ref_p, ref_a = _profiles([trace])
    return idx[int(np.argmin(_rank(peaks, acf, ref_p[0], ref_a[0])))]


def representative(runs: list, target_mean: float | None = None,
                   tol: float = 0.20) -> int:
    """Return the index of the run closest to the ensemble's own median profile.

    The profile is the daily-peak quantiles plus the hourly lag-one
    autocorrelation as a single term, so amplitude and memory both rank; the +1
    keeps a quiet-day quantile of zero finite. The reference is the ensemble's
    median and never the trace, so one rule serves every series.

    `target_mean` restricts the pick to runs carrying that volume. `tol` is
    loose: it guards a gross mismatch, not a near one.
    """
    def closest(idx):
        peaks, acf = _profiles([runs[i] for i in idx])
        return idx[int(np.argmin(_rank(peaks, acf, np.median(peaks, axis=0),
                                       np.median(acf))))]

    chosen = closest(list(range(len(runs))))
    if target_mean is None or abs(runs[chosen].mean() / target_mean - 1.0) <= tol:
        return chosen
    keep = _at_volume(runs, target_mean, tol)
    return closest(keep) if keep else chosen


def display_pool(model, cal: float, n_bins: int,
                 n_paths: int = DISPLAY_CANDIDATES) -> tuple[list, list]:
    """Draw the shared display ensemble, with the stream state preceding each run.

    The draws are `series_paths`'s for one model. Snapshotting consumes nothing,
    so the runs are unchanged by it; `redraw` needs the states.
    """
    rng = np.random.default_rng(SEED + DISPLAY_OFFSET)
    runs, states = [], []
    for _ in range(n_paths):
        states.append(rng.bit_generator.state)
        runs.append(model.simulate(n_bins, rng, alpha=cal))
    return runs, states


def redraw(model, state: dict, n_bins: int, alpha: float) -> np.ndarray:
    """Redraw one pooled run at another load factor, on the same latent path.

    `simulate` draws the latent first and alpha reaches only the intensity and
    kappa, so restoring the stream reproduces the latent exactly. Re-seeding per
    load factor pairs only a stream's first path: the count draws consume a
    variable number of values and desynchronise the rest.
    """
    # unseeded only for one line: the next statement replaces the whole state,
    # so nothing is ever drawn from the default seeding
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return model.simulate(n_bins, rng, alpha=alpha)
