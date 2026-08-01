"""Invert completed-job durations into work units and freeze a law for them."""
from __future__ import annotations

import json
import time

import numpy as np

from ... import (
    EmpiricalJobSize,
    LogNormalMixture,
    ScipyJobSize,
    fit_jobsize,
    jobsize_table,
    jobsize_view,
    load_trace,
    roundtrip_error_s,
    step_time_s,
    write_jobsize_model,
)
from ...paths import RESULTS, TRACE_CSV
from ..config import SEED

# GPU counts the size summary is echoed at: a work unit is invariant, the
# duration it implies is not
_ECHO_GPUS = (1, 8, 16, 64, 256)


def stage_jobsize() -> None:
    """Write the work units, the candidate scoreboard and the frozen sampler.

    Reads the trace directly rather than the pipeline state: job sizes are a
    separate view of it (completed jobs only) and nothing here feeds the count
    models.
    """
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = jobsize_table(jobsize_view(load_trace(TRACE_CSV), "SFT"))
    sizes = table["job_size"].to_numpy()

    err = roundtrip_error_s(table)
    print(f"completed SFT jobs with a usable duration and GPU count: {len(table)}")
    print(f"inversion round-trip max abs error: {err:.3e} s")
    # the inversion is an identity: a large error means the latency coefficients
    # no longer match the ones the durations were read at
    if err > 1e-6:
        raise RuntimeError(f"work-unit inversion does not round-trip ({err:.3e} s)")

    print("\nwork units:")
    print(table["job_size"].describe().to_string())
    median = float(np.median(sizes))
    print(f"\nthe median job of {median:,.0f} units, replayed at f = 1 GHz:")
    for n in _ECHO_GPUS:
        secs = median * float(step_time_s(n, 1.0))
        print(f"  n = {n:4d}  ->  {secs:9.0f} s  ({secs / 60:7.1f} min)")

    board = fit_jobsize(sizes, seed=SEED)
    print("\ncandidate laws, best AIC first:")
    print(board.drop(columns="params").round(4).to_string(index=False))
    # params serialised as json so a figure can rebuild any row's law directly
    board.assign(params=board["params"].map(
        lambda p: json.dumps(list(p) if isinstance(p, tuple) else p)
    )).to_csv(RESULTS / "jobsize_candidates.csv", index=False)

    best = board.iloc[0]
    if best["kind"] == "mixture_lognormal":
        k = int(best["distribution"].rsplit("_", 1)[1])
        model = LogNormalMixture.fit(sizes, n_components=k, seed=SEED).freeze()
        print(f"\nfrozen law: {best['distribution']}  "
              f"AIC {model.report.aic:,.1f}  BIC {model.report.bic:,.1f}")
        for w, mu, sg in zip(model.weights, model.mu, model.sigma):
            print(f"  w {w:.4f}  mu {mu:8.4f}  sigma {sg:.4f}  "
                  f"(median {np.exp(mu):,.0f} units)")
    else:
        model = ScipyJobSize(name=str(best["distribution"]), params=best["params"])
        print(f"\nfrozen law: {best['distribution']} {model.params}")
    write_jobsize_model(model, RESULTS / "jobsize_model.json")

    pool = EmpiricalJobSize.from_table(table)
    pool.to_npz(RESULTS / "jobsize_pool.npz")
    print(f"\nwrote jobsize_candidates.csv, jobsize_model.json and a "
          f"{len(pool.sizes)}-job empirical pool to {RESULTS}")
    print(f"jobsize stage done in {time.time() - t0:.0f}s")
