from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import SCALES
from .metrics import aggregate, idc


def moving_block_bootstrap(
    x: np.ndarray, block: int, rng: np.random.Generator, n_out: int | None = None
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n_out = n_out or len(x)
    n_blocks = int(np.ceil(n_out / block))
    starts = rng.integers(0, max(len(x) - block, 1), size=n_blocks)
    return np.concatenate([x[s : s + block] for s in starts])[:n_out]


def stationary_bootstrap(
    x: np.ndarray, mean_block: int, rng: np.random.Generator, n_out: int | None = None
) -> np.ndarray:
    """Resample with geometric block lengths, so the resample is stationary."""
    x = np.asarray(x, dtype=float)
    n_out = n_out or len(x)
    p = 1.0 / max(mean_block, 1)
    out = np.empty(n_out)
    idx = int(rng.integers(0, len(x)))
    for i in range(n_out):
        out[i] = x[idx]
        idx = int(rng.integers(0, len(x))) if rng.random() < p else (idx + 1) % len(x)
    return out


def phase_matched_bootstrap(
    x: np.ndarray, period: int, rng: np.random.Generator, n_out: int | None = None
) -> np.ndarray:
    """Resample whole periods, preserving calendar phase but not multi-day episodes."""
    x = np.asarray(x, dtype=float)
    n_out = n_out or len(x)
    n_periods = len(x) // period
    if n_periods < 2:
        return moving_block_bootstrap(x, period, rng, n_out)
    picks = rng.integers(0, n_periods, size=int(np.ceil(n_out / period)))
    return np.concatenate([x[p * period : (p + 1) * period] for p in picks])[:n_out]


def segment_bootstrap(
    x: np.ndarray, n_segments: int, block: int, rng: np.random.Generator, n_out: int | None = None
) -> np.ndarray:
    """Bootstrap within one contiguous activity segment, chosen at random."""
    x = np.asarray(x, dtype=float)
    segs = np.array_split(x, n_segments)
    seg = segs[int(rng.integers(0, len(segs)))]
    return moving_block_bootstrap(seg, min(block, max(len(seg) // 4, 1)), rng, n_out or len(x))


BOOTSTRAPS: dict[str, Callable] = {
    "moving_block_24h": lambda x, rng, n: moving_block_bootstrap(x, 96, rng, n),
    "stationary_6h": lambda x, rng, n: stationary_bootstrap(x, 24, rng, n),
    "stationary_12h": lambda x, rng, n: stationary_bootstrap(x, 48, rng, n),
    "stationary_24h": lambda x, rng, n: stationary_bootstrap(x, 96, rng, n),
    "stationary_48h": lambda x, rng, n: stationary_bootstrap(x, 192, rng, n),
    "stationary_72h": lambda x, rng, n: stationary_bootstrap(x, 288, rng, n),
    "phase_matched_day": lambda x, rng, n: phase_matched_bootstrap(x, 96, rng, n),
    "segment_local": lambda x, rng, n: segment_bootstrap(x, 3, 96, rng, n),
}


def bootstrap_metric_envelope(
    x: np.ndarray,
    metric_fn: Callable[[np.ndarray], pd.Series],
    method: str = "stationary_24h",
    n_rep: int = 200,
    seed: int = 0,
    n_out: int | None = None,
) -> pd.DataFrame:
    """Return the median, 2.5%, 97.5% and SE of every metric under a resample."""
    rng = np.random.default_rng(seed)
    fn = BOOTSTRAPS[method]
    rows = [metric_fn(fn(x, rng, n_out or len(x))) for _ in range(n_rep)]
    frame = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "median": frame.median(),
            "q025": frame.quantile(0.025),
            "q975": frame.quantile(0.975),
            "se": frame.std(ddof=1),
        }
    )


def scale_matched_idc_envelope(
    x: np.ndarray,
    scales: Mapping[str, int] = SCALES,
    block_factor: int = 4,
    min_block_bins: int = 96,
    min_blocks: int = 8,
    n_rep: int = 150,
    seed: int = 0,
) -> pd.DataFrame:
    """Return an IDC envelope per scale, each from blocks long enough to carry it.

    Each scale gets `block_factor * q` bins, floored at `min_block_bins`: one
    shared block length cannot serve every scale. A scale whose blocks do not fit
    at least `min_blocks` times is marked unsupported rather than graded. Scales
    sharing a block length share one set of resamples.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    plan = {name: max(int(block_factor * q), int(min_block_bins)) for name, q in scales.items()}

    draws: dict[int, list[np.ndarray]] = {}
    for block in sorted(set(plan.values())):
        rng = np.random.default_rng(seed + block)
        draws[block] = [stationary_bootstrap(x, block, rng, n) for _ in range(n_rep)]

    rows = []
    for name, q in scales.items():
        block = plan[name]
        n_blocks = n / block
        vals = np.array([idc(aggregate(rep, q)) for rep in draws[block]], dtype=float)
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "scale": name,
                "q_bins": int(q),
                "block_bins": int(block),
                "n_blocks": float(n_blocks),
                "supported": bool(n_blocks >= min_blocks),
                "median": float(np.median(vals)) if vals.size else np.nan,
                "q025": float(np.quantile(vals, 0.025)) if vals.size else np.nan,
                "q975": float(np.quantile(vals, 0.975)) if vals.size else np.nan,
                "se": float(vals.std(ddof=1)) if vals.size > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("scale")


def bootstrap_baseline_table(
    x: np.ndarray,
    metric_fn: Callable[[np.ndarray], pd.Series],
    methods: Sequence[str] | None = None,
    n_rep: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """Score every bootstrap method against the trace with the shared metric set."""
    trace = metric_fn(np.asarray(x, dtype=float))
    rows: list[dict] = []
    for method in methods or list(BOOTSTRAPS):
        env = bootstrap_metric_envelope(x, metric_fn, method, n_rep=n_rep, seed=seed)
        for metric in trace.index:
            if metric not in env.index:
                continue
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "trace": float(trace[metric]),
                    "boot_median": float(env.loc[metric, "median"]),
                    "boot_q025": float(env.loc[metric, "q025"]),
                    "boot_q975": float(env.loc[metric, "q975"]),
                    "inside_envelope": bool(
                        env.loc[metric, "q025"] <= trace[metric] <= env.loc[metric, "q975"]
                    ),
                }
            )
    return pd.DataFrame(rows)
