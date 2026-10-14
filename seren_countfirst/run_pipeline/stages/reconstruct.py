from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from traffic_modeling.counts.metrics import microstructure
from traffic_modeling.counts.series import ActivityScheme, CountSeries

from ...artifacts import write_csv
from ...models import CountModel
from ...reconstruct import (
    OffsetTemplates,
    PatternTemplates,
    collision_risk,
    export_check,
    make_conditional_mark_sampler,
    make_iid_mark_sampler,
    reconstruct_events,
)
from ...state import PipelineState
from ..config import N_PATHS_RECONSTRUCT, RECONSTRUCT_ALPHAS
from ..context import PipelineContext, resolve_context
from ..randomness import SeedStream, generator
from ..state import fitted_models

_MICROSTRUCTURE_STATS = (
    "zero_iat_fraction",
    "positive_gap_median_s",
    "positive_gap_q99_s",
    "zero_bin_fraction",
)


@dataclass(frozen=True)
class ReconstructionInputs:
    state: PipelineState
    counts: CountSeries
    model: CountModel
    calibration: float
    bin_seconds: float
    times_s: np.ndarray
    sizes: np.ndarray
    scheme: ActivityScheme
    placements: dict[str, Any]
    samplers: dict[str, Any]
    synthetic_paths: list[np.ndarray]
    trace_microstructure: pd.Series
    base_seed: int


def _trace_events(
    state: PipelineState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ActivityScheme]:
    counts = state["counts"]
    origin = counts.index[0]
    times_s = (
        (pd.DatetimeIndex(state["batch_times"]) - origin).total_seconds().to_numpy(dtype=float)
    )
    sizes = np.asarray(state["batches_sizes"], dtype=int)
    raw_scheme = state["scheme"]
    scheme = ActivityScheme.from_dict(raw_scheme) if isinstance(raw_scheme, dict) else raw_scheme
    return times_s, sizes, scheme.classify(counts.values), scheme


def _load_inputs(context: PipelineContext) -> ReconstructionInputs:
    state = context.load_state()
    counts = state["counts"]
    bin_seconds = float(counts.bin_seconds)
    model = fitted_models(state)[state.get("winner", "gated_mm")]
    calibration = state["trace_marginals"]["mean"] / float(model.params_["mean"])
    times_s, sizes, trace_classes, scheme = _trace_events(state)

    placements = {
        "pattern": PatternTemplates.from_trace(
            times_s, bin_seconds, counts.values, classes=trace_classes
        ),
        "iid_offsets": OffsetTemplates.from_trace(
            times_s, bin_seconds, counts.values, classes=trace_classes
        ),
    }
    bin_index = np.clip((times_s // bin_seconds).astype(int), 0, len(trace_classes) - 1)
    by_class = {
        str(label): sizes[trace_classes[bin_index] == label] for label in np.unique(trace_classes)
    }
    samplers = {
        "iid": make_iid_mark_sampler(sizes),
        "conditional": make_conditional_mark_sampler(by_class, fallback=sizes),
    }
    path_rng = generator(SeedStream.RECONSTRUCTION_POOL, base=context.seed)
    synthetic_paths = [
        model.simulate(len(counts), path_rng, alpha=calibration) for _ in range(N_PATHS_RECONSTRUCT)
    ]
    trace_jobs: np.ndarray = np.repeat(times_s, sizes)
    trace_microstructure = microstructure(trace_jobs, times_s, counts.values)
    return ReconstructionInputs(
        state=state,
        counts=counts,
        model=model,
        calibration=calibration,
        bin_seconds=bin_seconds,
        times_s=times_s,
        sizes=sizes,
        scheme=scheme,
        placements=placements,
        samplers=samplers,
        synthetic_paths=synthetic_paths,
        trace_microstructure=trace_microstructure,
        base_seed=context.seed,
    )


def _score_microstructure(
    inputs: ReconstructionInputs,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rows, checks, overflow = [], [], 0
    for placement, offsets in inputs.placements.items():
        for marks, sampler in inputs.samplers.items():
            for replicate, counts in enumerate(inputs.synthetic_paths):
                rng = generator(
                    SeedStream.RECONSTRUCTION_MARKS,
                    replicate,
                    base=inputs.base_seed,
                )
                try:
                    reconstructed = reconstruct_events(
                        counts,
                        inputs.bin_seconds,
                        rng,
                        sampler,
                        offsets=offsets,
                        classes=inputs.scheme.classify(counts),
                    )
                except ValueError:
                    overflow += 1
                    continue
                tag = {
                    "placement": placement,
                    "marks": marks,
                    "replicate": replicate,
                }
                rows.append(
                    {
                        **tag,
                        "n_batches": reconstructed.n_batches,
                        **microstructure(
                            reconstructed.job_times_s,
                            reconstructed.batch_times_s,
                            counts,
                        ).to_dict(),
                    }
                )
                checks.append({**tag, **export_check(reconstructed, counts).to_dict()})
    return pd.DataFrame(rows), pd.DataFrame(checks), overflow


def _summarize_microstructure(
    inputs: ReconstructionInputs,
    per_path: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = per_path.groupby(["placement", "marks"])[[*_MICROSTRUCTURE_STATS, "n_batches"]].agg(
        ["mean", "std"]
    )
    combined = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "placement": "TRACE",
                        "marks": "TRACE",
                        **inputs.trace_microstructure.to_dict(),
                        "n_batches": len(inputs.times_s),
                    }
                ]
            ),
            summary.xs("mean", axis=1, level=1).reset_index(),
        ],
        ignore_index=True,
    )
    return combined, summary


def _load_grid(inputs: ReconstructionInputs) -> pd.DataFrame:
    rows = []
    templates = inputs.placements["pattern"]
    sampler = inputs.samplers["iid"]
    for alpha in RECONSTRUCT_ALPHAS:
        path_rng = generator(SeedStream.LOAD_PATH, base=inputs.base_seed)
        risks, strict_ok, subsecond = [], [], []
        for _ in range(N_PATHS_RECONSTRUCT):
            counts = inputs.model.simulate(
                len(inputs.counts),
                path_rng,
                alpha=inputs.calibration * alpha,
            )
            risks.append(collision_risk(counts, inputs.bin_seconds))
            try:
                reconstruct_events(
                    counts,
                    inputs.bin_seconds,
                    generator(SeedStream.PLACEMENT_CHECK, base=inputs.base_seed),
                    sampler,
                    offsets=templates,
                    on_overflow="raise",
                )
                strict_ok.append(True)
            except ValueError:
                strict_ok.append(False)
            try:
                reconstructed = reconstruct_events(
                    counts,
                    inputs.bin_seconds,
                    generator(SeedStream.PLACEMENT_CHECK, base=inputs.base_seed),
                    sampler,
                    offsets=templates,
                    alpha=alpha,
                )
                subsecond.append(
                    reconstructed.collision_report.to_dict()
                    | {"n_batches": reconstructed.n_batches, "ok": 1.0}
                )
            except ValueError:
                subsecond.append(
                    {
                        "n_overflow_bins": np.nan,
                        "finest_resolution_s": np.nan,
                        "n_subsecond_placements": np.nan,
                        "n_batches": np.nan,
                        "ok": 0.0,
                    }
                )
        risk = pd.DataFrame(risks).mean()
        subsecond_frame = pd.DataFrame(subsecond)
        rows.append(
            {
                "alpha": alpha,
                "strict_1s_paths_ok": float(np.mean(strict_ok)),
                "default_paths_ok": float(subsecond_frame["ok"].mean()),
                "mean_max_count": risk["max_count"],
                "mean_max_fill_ratio": risk["max_fill_ratio"],
                "mean_refined_bins": subsecond_frame["n_overflow_bins"].mean(),
                "finest_resolution_s": subsecond_frame["finest_resolution_s"].min(),
                "frac_batches_off_whole_second": float(
                    subsecond_frame["n_subsecond_placements"].sum()
                    / subsecond_frame["n_batches"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def stage_reconstruct(context: PipelineContext | None = None) -> None:
    """Reconstruct events and report placement microstructure."""
    context = resolve_context(context)
    context.results.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(context)
    print(
        "trace microstructure:",
        {key: round(float(value), 4) for key, value in inputs.trace_microstructure.items()},
    )

    per_path, checks, overflow = _score_microstructure(inputs)
    if overflow:
        total = len(inputs.placements) * len(inputs.samplers) * N_PATHS_RECONSTRUCT
        print(
            f"note: {overflow} of {total} paths overflowed the "
            f"{int(inputs.bin_seconds)}-slot grid at alpha = 1 and are excluded from the "
            "microstructure summary (trace max bin 530; the model's tail runs past it)"
        )
    microstructure_frame, summary = _summarize_microstructure(inputs, per_path)
    write_csv(per_path, context.result("microstructure_paths.csv"), index=False)
    write_csv(checks, context.result("export_check.csv"), index=False)
    write_csv(microstructure_frame, context.result("microstructure.csv"), index=False)

    conserved = checks[["batch_conserved", "job_conserved", "distinct_timestamps"]].all().all()
    print(
        f"\nconservation over {len(checks)} reconstructions: "
        f"{'ok — every batch survives, every timestamp distinct' if conserved else 'FAILED'}"
    )
    print(f"\nmicrostructure, trace vs reconstruction (mean over {N_PATHS_RECONSTRUCT} paths):")
    print(microstructure_frame.round(4).to_string(index=False))
    print("\nper-path spread (sd):")
    print(summary.xs("std", axis=1, level=1).round(4).to_string())

    grid = _load_grid(inputs)
    write_csv(grid, context.result("reconstruction_by_alpha.csv"), index=False)
    print(
        f"\nexport grid by load factor ({N_PATHS_RECONSTRUCT} paths each; strict = whole "
        f"seconds only, {int(inputs.bin_seconds)} slots per bin):"
    )
    print(grid.round(4).to_string(index=False))
    inputs.state["microstructure"] = microstructure_frame
    inputs.state["reconstruction_by_alpha"] = grid
    context.save_state(inputs.state)
