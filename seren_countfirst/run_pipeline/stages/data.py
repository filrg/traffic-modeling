from __future__ import annotations

import time

import numpy as np

from ...data import (
    ActivityScheme,
    arrival_view,
    build_batches,
    build_counts,
    contiguous_split,
    load_trace,
    segment_profiles,
    sha256_file,
    zero_iat_fraction,
)
from ...metrics import base_marginals, idc_curve, metric_vector
from ...bootstrap import bootstrap_metric_envelope, scale_matched_idc_envelope
from ...paths import TRACE_CSV
from ..config import AUDIT, SEED
from ..state import save_state


def stage_data() -> None:
    t0 = time.time()
    state = {"trace_sha256": sha256_file(TRACE_CSV)}
    raw = load_trace(TRACE_CSV)
    view = arrival_view(raw, "SFT")
    batches = build_batches(view)
    counts = build_counts(batches.times)
    z_iat = zero_iat_fraction(view)

    checks = {
        "n_sft_jobs": (len(view), AUDIT["n_sft_jobs"]),
        "n_batches": (batches.n_batches, AUDIT["n_batches"]),
        "zero_iat": (round(z_iat, 4), AUDIT["zero_iat"]),
        "mean_15m": (round(float(counts.values.mean()), 3), AUDIT["mean_15m"]),
        "idc_15m": (
            round(float(counts.values.var(ddof=1) / counts.values.mean()), 2),
            AUDIT["idc_15m"],
        ),
        "zero_frac_15m": (round(float((counts.values == 0).mean()), 4), AUDIT["zero_frac_15m"]),
        "max_15m": (int(counts.values.max()), AUDIT["max_15m"]),
    }
    print("audit verification (got, expected):")
    for k, (got, exp) in checks.items():
        flag = "OK " if np.isclose(float(got), float(exp), rtol=0.02) else "DIFF"
        print(f"  [{flag}] {k:<14} {got}  vs  {exp}")

    split = contiguous_split(counts)
    scheme = ActivityScheme.fit(split.train)
    profiles = segment_profiles(counts, 3)
    corr = profiles.corr().to_numpy()
    print(
        f"segment hour-of-week profile corr: {corr[0, 1]:.3f} {corr[0, 2]:.3f} {corr[1, 2]:.3f}"
        " (calendar gate 1 evidence)"
    )

    thr = scheme.busy_threshold
    env = bootstrap_metric_envelope(
        counts.values,
        lambda x: metric_vector(x, thr),
        method="stationary_24h",
        n_rep=150,
        seed=SEED,
    )
    # gate 3 grades IDC per scale against blocks long enough to carry it
    idc_env = scale_matched_idc_envelope(counts.values, n_rep=150, seed=SEED)
    print("scale-matched IDC envelope (gate 3):")
    print(idc_env.assign(trace=idc_curve(counts.values)).round(1).to_string())
    state.update(
        counts=counts,
        batches_sizes=batches.sizes,
        batch_times=batches.times,
        split_bounds=(len(split.train), len(split.train) + len(split.validation)),
        scheme=scheme,
        busy_threshold=float(thr),
        envelope=env,
        idc_envelope=idc_env,
        trace_marginals=base_marginals(counts.values),
        trace_idc=idc_curve(counts.values).to_dict(),
        n_bins=len(counts),
    )
    save_state(state)
    print(f"trace IDC curve: { {k: round(v, 1) for k, v in state['trace_idc'].items()} }")
    print(f"data stage done in {time.time() - t0:.0f}s; n_bins={len(counts)}, busy_thr={thr:.1f}")
