from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from traffic_modeling.counts.series import ActivityScheme, CountSeries

from ...artifacts import write_csv
from ...jobsize import EmpiricalJobSize, TwoPartJobSize, assign_work, read_two_part_jobsize
from ...models import CountModel
from ...reconstruct import PatternTemplates, make_iid_mark_sampler, reconstruct_events
from ...state import PipelineState
from ..config import N_PATHS_RECONSTRUCT, RECONSTRUCT_ALPHAS
from ..context import PipelineContext, resolve_context
from ..randomness import SeedStream, generator
from ..state import fitted_models

EXPORT_ALPHA = 1.0


@dataclass(frozen=True)
class WorkloadInputs:
    counts: CountSeries
    model: CountModel
    calibration: float
    size_model: TwoPartJobSize
    trace_sizes: np.ndarray
    templates: PatternTemplates
    mark_sampler: Any
    bin_seconds: float
    base_seed: int


def _scheme(state: PipelineState) -> ActivityScheme:
    scheme = state["scheme"]
    return ActivityScheme.from_dict(scheme) if isinstance(scheme, dict) else scheme


def _load_inputs(context: PipelineContext) -> WorkloadInputs:
    state = context.load_state()
    size_json = context.result("jobsize_two_part.json")
    pool_npz = context.result("jobsize_pool.npz")
    if not (size_json.exists() and pool_npz.exists()):
        raise SystemExit(f"no size model in {context.results}: run the `jobsize` stage first")

    size_model = read_two_part_jobsize(size_json, pool_npz)
    trace_sizes = EmpiricalJobSize.from_npz(pool_npz).sizes
    counts = state["counts"]
    bin_seconds = float(counts.bin_seconds)
    model = fitted_models(state)[state.get("winner", "gated_mm")]
    calibration = state["trace_marginals"]["mean"] / float(model.params_["mean"])
    origin = counts.index[0]
    times_s = (
        (pd.DatetimeIndex(state["batch_times"]) - origin).total_seconds().to_numpy(dtype=float)
    )
    templates = PatternTemplates.from_trace(
        times_s,
        bin_seconds,
        counts.values,
        classes=_scheme(state).classify(counts.values),
    )
    mark_sampler = make_iid_mark_sampler(np.asarray(state["batches_sizes"], dtype=int))
    return WorkloadInputs(
        counts=counts,
        model=model,
        calibration=calibration,
        size_model=size_model,
        trace_sizes=trace_sizes,
        templates=templates,
        mark_sampler=mark_sampler,
        bin_seconds=bin_seconds,
        base_seed=context.seed,
    )


def _simulate_path(
    inputs: WorkloadInputs,
    alpha: float,
    reading: str,
    size_alpha: float | None,
    replicate: int,
) -> dict[str, float | int | str]:
    syn = inputs.model.simulate(
        len(inputs.counts),
        generator(SeedStream.LOAD_PATH, replicate, base=inputs.base_seed),
        alpha=inputs.calibration * alpha,
    )
    rng = generator(SeedStream.WORKLOAD_MARKS, replicate, base=inputs.base_seed)
    reconstructed = reconstruct_events(
        syn,
        inputs.bin_seconds,
        rng,
        inputs.mark_sampler,
        offsets=inputs.templates,
        on_overflow="subsecond",
        alpha=alpha,
    )
    work = assign_work(
        reconstructed.batch_times_s,
        reconstructed.batch_sizes,
        syn,
        inputs.bin_seconds,
        inputs.size_model,
        rng,
        alpha=alpha if size_alpha is None else size_alpha,
    )
    large = work >= inputs.size_model.cut
    return {
        "reading": reading,
        "alpha": alpha,
        "replicate": replicate,
        "n_batches": reconstructed.n_batches,
        "n_jobs": reconstructed.n_jobs,
        "total_work": float(work.sum()),
        "mean_work_per_job": float(work.mean()),
        "large_job_frac": float(large.mean()),
        "large_work_share": float(work[large].sum() / work.sum()),
        "mean_activity": float(syn[syn > 0].mean()),
    }


def _simulate_sweep(inputs: WorkloadInputs) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    readings = (("intensity", 1.0), ("superposition", None))
    for alpha, (reading, size_alpha) in itertools.product(RECONSTRUCT_ALPHAS, readings):
        rows.extend(
            _simulate_path(inputs, alpha, reading, size_alpha, replicate)
            for replicate in range(N_PATHS_RECONSTRUCT)
        )
    return pd.DataFrame(rows)


def _summarize_sweep(per_path: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_alpha = (
        per_path.groupby(["reading", "alpha"])
        .agg(
            n_batches=("n_batches", "mean"),
            n_jobs=("n_jobs", "mean"),
            total_work=("total_work", "mean"),
            total_work_sd=("total_work", "std"),
            mean_work_per_job=("mean_work_per_job", "mean"),
            large_job_frac=("large_job_frac", "mean"),
            large_work_share=("large_work_share", "mean"),
        )
        .reset_index()
    )
    rows = []
    for reading, block in by_alpha.groupby("reading"):
        unit = block.loc[block["alpha"] == 1.0].iloc[0]
        log_alpha = np.log(block["alpha"].to_numpy(dtype=float))
        rows.append(
            {
                "reading": reading,
                "batch_volume": float(np.polyfit(log_alpha, np.log(block["n_batches"]), 1)[0]),
                "mean_work_per_job": float(
                    np.polyfit(log_alpha, np.log(block["mean_work_per_job"]), 1)[0]
                ),
                "offered_work": float(np.polyfit(log_alpha, np.log(block["total_work"]), 1)[0]),
                "work_at_alpha16_vs_linear": float(
                    block["total_work"].iloc[-1] / (unit["total_work"] * 16.0)
                ),
            }
        )
    return by_alpha, pd.DataFrame(rows)


def _export_workload(inputs: WorkloadInputs) -> tuple[pd.DataFrame, np.ndarray, float]:
    syn = inputs.model.simulate(
        len(inputs.counts),
        generator(SeedStream.LOAD_PATH, base=inputs.base_seed),
        alpha=inputs.calibration * EXPORT_ALPHA,
    )
    rng = generator(SeedStream.WORKLOAD_MARKS, base=inputs.base_seed)
    reconstructed = reconstruct_events(
        syn,
        inputs.bin_seconds,
        rng,
        inputs.mark_sampler,
        offsets=inputs.templates,
        on_overflow="subsecond",
        alpha=EXPORT_ALPHA,
    )
    work = assign_work(
        reconstructed.batch_times_s,
        reconstructed.batch_sizes,
        syn,
        inputs.bin_seconds,
        inputs.size_model,
        rng,
        alpha=EXPORT_ALPHA,
    )
    export = pd.DataFrame({"job_time_s": reconstructed.job_times_s, "work_units": work})
    return export, work, float(reconstructed.job_times_s.max())


def stage_workload(context: PipelineContext | None = None) -> None:
    """Compose arrivals and job sizes into workload artifacts."""
    started = time.time()
    context = resolve_context(context)
    context.results.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(context)
    print(
        f"size law: two-part over a {inputs.size_model.bulk_kind} bulk, cut "
        f"{inputs.size_model.cut:,.0f} units, k = {inputs.size_model.bulk.n_components}"
    )
    print(
        f"trace, absorbed work: {len(inputs.trace_sizes):,} jobs, "
        f"{inputs.trace_sizes.sum():,.3e} work units, {inputs.trace_sizes.mean():,.0f} per job"
    )

    per_path = _simulate_sweep(inputs)
    by_alpha, exponents = _summarize_sweep(per_path)
    write_csv(per_path, context.result("workload_paths.csv"), index=False)
    write_csv(by_alpha, context.result("workload_by_alpha.csv"), index=False)
    write_csv(exponents, context.result("workload_exponents.csv"), index=False)
    print(f"\nworkload by load factor ({N_PATHS_RECONSTRUCT} paths each):")
    print(by_alpha.round(4).to_string(index=False))
    print("\nexponents in alpha:")
    print(exponents.round(4).to_string(index=False))
    print(
        "\nbatch volume is linear under both readings; the difference is in the "
        "size model. Neither gives linear work, and which reading alpha carries "
        "for sizes is unsettled."
    )

    export, work, last_job_time = _export_workload(inputs)
    write_csv(export, context.result("workload_export.csv"), index=False)
    print(
        f"\nexported {len(export):,} jobs at alpha = {EXPORT_ALPHA} over "
        f"{last_job_time / 86400:.1f} days, {work.sum():,.3e} work units "
        f"({work.sum() / inputs.trace_sizes.sum():.3f} of the trace's absorbed work, "
        f"over {len(export) / len(inputs.trace_sizes):.3f} as many jobs)"
    )
    print(
        "wrote workload_paths.csv, workload_by_alpha.csv and workload_export.csv "
        f"to {context.results}"
    )
    print(f"workload stage done in {time.time() - started:.0f}s")
