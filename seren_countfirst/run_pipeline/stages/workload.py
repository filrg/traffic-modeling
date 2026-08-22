from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from ...data import ActivityScheme
from ...jobsize import assign_work, EmpiricalJobSize, read_two_part_jobsize
from ...reconstruct import make_iid_mark_sampler, PatternTemplates, reconstruct_events
from ...paths import RESULTS
from ..config import N_PATHS_RECONSTRUCT, RECONSTRUCT_ALPHAS, SEED
from ..state import fitted_models, load_state

# the load factor the exported workload is written at
EXPORT_ALPHA = 1.0


def _scheme(state: dict) -> ActivityScheme:
    scheme = state["scheme"]
    if isinstance(scheme, dict):
        return ActivityScheme(edges=tuple(scheme["edges"]), labels=tuple(scheme["labels"]),
                              busy_threshold=scheme["busy_threshold"],
                              quiet_threshold=scheme["quiet_threshold"],
                              source=scheme["source"])
    return scheme


def stage_workload() -> None:
    """Give every generated job a work unit, then read the exponent work follows.

    Sizes are fitted on the states whose runtime the cluster absorbed, then
    assigned to every arrival, including the preempted ones the size view leaves
    out; the ratio of the two job counts is reported beside the totals.
    """
    t0 = time.time()
    state = load_state()
    RESULTS.mkdir(parents=True, exist_ok=True)
    size_json, pool_npz = RESULTS / "jobsize_two_part.json", RESULTS / "jobsize_pool.npz"
    if not (size_json.exists() and pool_npz.exists()):
        raise SystemExit(f"no size model in {RESULTS}: run the `jobsize` stage first")
    sizes_model = read_two_part_jobsize(size_json, pool_npz)
    trace_sizes = EmpiricalJobSize.from_npz(pool_npz).sizes

    counts = state["counts"]
    bin_s = float(counts.bin_seconds)
    model = fitted_models(state)[state.get("winner", "gated_mm")]
    cal = state["trace_marginals"]["mean"] / float(model.params_["mean"])
    origin = counts.index[0]
    times_s = (pd.DatetimeIndex(state["batch_times"]) - origin).total_seconds().to_numpy(dtype=float)
    templates = PatternTemplates.from_trace(times_s, bin_s, counts.values,
                                            classes=_scheme(state).classify(counts.values))
    marks = make_iid_mark_sampler(np.asarray(state["batches_sizes"], dtype=int))

    print(f"size law: two-part over a {sizes_model.bulk_kind} bulk, cut "
          f"{sizes_model.cut:,.0f} units, k = {sizes_model.bulk.n_components}")
    print(f"trace, absorbed work: {len(trace_sizes):,} jobs, "
          f"{trace_sizes.sum():,.3e} work units, {trace_sizes.mean():,.0f} per job")

    rows = []
    for alpha, (reading, size_alpha) in itertools.product(
            RECONSTRUCT_ALPHAS,
            # what the counts mean to the size model: one busier stream, or that
            # many unit-load ones
            (("intensity", 1.0), ("superposition", None))):
        for rep in range(N_PATHS_RECONSTRUCT):
            # one stream per path index, not per alpha: a shared stream walked across
            # the sweep puts every alpha on a different draw and invents a trend
            syn = model.simulate(len(counts), np.random.default_rng(SEED + 4 + rep),
                                 alpha=cal * alpha)
            rng = np.random.default_rng(SEED + 7000 + rep)
            # sub-second placement throughout: dropping the paths that do not fit on
            # whole seconds would bias these totals toward the quieter draws
            res = reconstruct_events(syn, bin_s, rng, marks, offsets=templates,
                                     on_overflow="subsecond", alpha=alpha)
            work = assign_work(res.batch_times_s, res.batch_sizes, syn, bin_s,
                               sizes_model, rng,
                               alpha=alpha if size_alpha is None else size_alpha)
            big = work >= sizes_model.cut
            rows.append({"reading": reading, "alpha": alpha, "replicate": rep,
                         "n_batches": res.n_batches,
                         "n_jobs": res.n_jobs, "total_work": float(work.sum()),
                         "mean_work_per_job": float(work.mean()),
                         "large_job_frac": float(big.mean()),
                         "large_work_share": float(work[big].sum() / work.sum()),
                         "mean_activity": float(syn[syn > 0].mean())})
    per_path = pd.DataFrame(rows)
    per_path.to_csv(RESULTS / "workload_paths.csv", index=False)

    by_alpha = per_path.groupby(["reading", "alpha"]).agg(
        n_batches=("n_batches", "mean"), n_jobs=("n_jobs", "mean"),
        total_work=("total_work", "mean"), total_work_sd=("total_work", "std"),
        mean_work_per_job=("mean_work_per_job", "mean"),
        large_job_frac=("large_job_frac", "mean"),
        large_work_share=("large_work_share", "mean")).reset_index()
    exponents = []
    for reading, block in by_alpha.groupby("reading"):
        unit = block.loc[block["alpha"] == 1.0].iloc[0]
        log_a = np.log(block["alpha"].to_numpy(dtype=float))
        exponents.append({
            "reading": reading,
            "batch_volume": float(np.polyfit(log_a, np.log(block["n_batches"]), 1)[0]),
            "mean_work_per_job": float(np.polyfit(log_a, np.log(block["mean_work_per_job"]), 1)[0]),
            "offered_work": float(np.polyfit(log_a, np.log(block["total_work"]), 1)[0]),
            "work_at_alpha16_vs_linear":
                float(block["total_work"].iloc[-1] / (unit["total_work"] * 16.0)),
        })
    exponents = pd.DataFrame(exponents)

    print(f"\nworkload by load factor ({N_PATHS_RECONSTRUCT} paths each):")
    print(by_alpha.round(4).to_string(index=False))
    print("\nexponents in alpha:")
    print(exponents.round(4).to_string(index=False))
    print("\nbatch volume is linear under both readings; the difference is in the "
          "size model. Neither gives linear work, and which reading alpha carries "
          "for sizes is unsettled.")
    by_alpha.to_csv(RESULTS / "workload_by_alpha.csv", index=False)
    exponents.to_csv(RESULTS / "workload_exponents.csv", index=False)

    # the deliverable: one workload, arrival time paired with hardware-free demand
    path_rng = np.random.default_rng(SEED + 4)
    syn = model.simulate(len(counts), path_rng, alpha=cal * EXPORT_ALPHA)
    rng = np.random.default_rng(SEED + 7000)
    res = reconstruct_events(syn, bin_s, rng, marks, offsets=templates,
                             on_overflow="subsecond", alpha=EXPORT_ALPHA)
    work = assign_work(res.batch_times_s, res.batch_sizes, syn, bin_s, sizes_model,
                       rng, alpha=EXPORT_ALPHA)
    export = pd.DataFrame({"job_time_s": res.job_times_s, "work_units": work})
    export.to_csv(RESULTS / "workload_export.csv", index=False)
    print(f"\nexported {len(export):,} jobs at alpha = {EXPORT_ALPHA} over "
          f"{res.job_times_s.max() / 86400:.1f} days, {work.sum():,.3e} work units "
          f"({work.sum() / trace_sizes.sum():.3f} of the trace's absorbed work, "
          f"over {len(export) / len(trace_sizes):.3f} as many jobs)")
    print(f"wrote workload_paths.csv, workload_by_alpha.csv and workload_export.csv "
          f"to {RESULTS}")
    print(f"workload stage done in {time.time() - t0:.0f}s")
