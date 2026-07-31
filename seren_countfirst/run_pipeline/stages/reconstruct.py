"""Counts to batch timestamps, sizes and job arrivals."""
from __future__ import annotations


import numpy as np
import pandas as pd

from ... import (
    ActivityScheme,
    OffsetTemplates,
    collision_risk,
    export_check,
    make_conditional_mark_sampler,
    make_iid_mark_sampler,
    microstructure,
    reconstruct_events,
)
from ...paths import RESULTS
from ..config import N_PATHS_RECONSTRUCT, RECONSTRUCT_ALPHAS, SEED
from ..state import _load, _models, _save


def _trace_events(state: dict) -> tuple:
    """Trace batch times as seconds from the first bin edge, with sizes and per-bin classes."""
    counts = state["counts"]
    origin = counts.index[0]
    times_s = (pd.DatetimeIndex(state["batch_times"]) - origin).total_seconds().to_numpy(dtype=float)
    sizes = np.asarray(state["batches_sizes"], dtype=int)
    scheme = state["scheme"]
    if isinstance(scheme, dict):
        scheme = ActivityScheme(edges=tuple(scheme["edges"]), labels=tuple(scheme["labels"]),
                                busy_threshold=scheme["busy_threshold"],
                                quiet_threshold=scheme["quiet_threshold"], source=scheme["source"])
    return times_s, sizes, scheme.classify(counts.values), scheme


def stage_reconstruct() -> None:
    """Counts -> batch timestamps -> sizes -> jobs, scored against the trace's microstructure.

    This is the layer the count model is agnostic to: it decides *where inside a
    bin* the batches land, which is what produces the zero-inter-arrival
    structure that made a gap-first model awkward in the first place.  Reported,
    not gated: adding a ninth structural gate would invalidate the eight-gate
    certification, so that stays a decision rather than a side effect.
    """
    state = _load()
    RESULTS.mkdir(parents=True, exist_ok=True)
    counts = state["counts"]
    bin_s = float(counts.bin_seconds)
    model = _models(state)[state.get("winner", "gated_mm")]
    cal = state["trace_marginals"]["mean"] / float(model.params_["mean"])

    times_s, sizes, trace_classes, scheme = _trace_events(state)
    templates = OffsetTemplates.from_trace(times_s, bin_s, counts.values, classes=trace_classes)
    trace_jobs = np.repeat(times_s, sizes)
    trace_micro = microstructure(trace_jobs, times_s, counts.values)
    print("trace microstructure:", {k: round(float(v), 4) for k, v in trace_micro.items()})

    # the two mark samplers of guide eq. (14), as an ablation
    by_class = {str(lab): sizes[trace_classes[np.clip((times_s // bin_s).astype(int), 0,
                                                      len(trace_classes) - 1)] == lab]
                for lab in np.unique(trace_classes)}
    samplers = {
        "iid": make_iid_mark_sampler(sizes),
        "conditional": make_conditional_mark_sampler(by_class, fallback=sizes),
    }

    # the ablation must be paired: a mark sampler consumes randomness, so drawing
    # the count path from the same stream would put the two arms on different
    # paths and compare the load lottery instead of the marks
    path_rng = np.random.default_rng(SEED + 3)
    syn_paths = [model.simulate(len(counts), path_rng, alpha=cal)
                 for _ in range(N_PATHS_RECONSTRUCT)]

    rows, checks, overflow = [], [], 0
    for label, sampler in samplers.items():
        for rep, syn in enumerate(syn_paths):
            rng = np.random.default_rng(SEED + 5000 + rep)
            try:
                # thresholds stay frozen from the training interval (hard rule 4)
                res = reconstruct_events(syn, bin_s, rng, sampler, offsets=templates,
                                         classes=scheme.classify(syn))
            except ValueError:
                # hard rule 3: an overflowing bin is reported, never merged away.
                # It happens at alpha = 1 because the model's upper tail runs past
                # the trace's max of 530, so this is a result, not an error.
                overflow += 1
                continue
            rows.append({"marks": label, "replicate": rep, "n_batches": res.n_batches,
                         **microstructure(res.job_times_s, res.batch_times_s, syn).to_dict()})
            checks.append({"marks": label, "replicate": rep, **export_check(res, syn).to_dict()})
    if overflow:
        print(f"note: {overflow} of {2 * N_PATHS_RECONSTRUCT} paths overflowed the "
              f"{int(bin_s)}-slot grid at alpha = 1 and are excluded from the microstructure "
              f"summary (trace max bin 530; the model's tail runs past it)")
    per_path = pd.DataFrame(rows)
    check_df = pd.DataFrame(checks)
    per_path.to_csv(RESULTS / "microstructure_paths.csv", index=False)
    check_df.to_csv(RESULTS / "export_check.csv", index=False)

    stats = ["zero_iat_fraction", "positive_gap_median_s", "positive_gap_q99_s", "zero_bin_fraction"]
    summary = per_path.groupby("marks")[stats + ["n_batches"]].agg(["mean", "std"])
    micro_df = pd.concat([
        pd.DataFrame([{"marks": "TRACE", **trace_micro.to_dict(), "n_batches": len(times_s)}]),
        summary.xs("mean", axis=1, level=1).reset_index(),
    ], ignore_index=True)
    micro_df.to_csv(RESULTS / "microstructure.csv", index=False)

    conserved = check_df[["batch_conserved", "job_conserved", "distinct_timestamps"]].all().all()
    print(f"\nconservation over {len(check_df)} reconstructions: "
          f"{'ok — every batch survives, every timestamp distinct' if conserved else 'FAILED'}")
    print(f"\nmicrostructure, trace vs reconstruction (mean over {N_PATHS_RECONSTRUCT} paths):")
    print(micro_df.round(4).to_string(index=False))
    print("\nper-path spread (sd):")
    print(summary.xs("std", axis=1, level=1).round(4).to_string())

    # Export grid by load factor, scored both ways: strict whole-second placement,
    # which raises rather than merge (hard rule 3), and the adaptive sub-second
    # grid, which refines only the bins that cannot fit.
    grid = []
    for alpha in RECONSTRUCT_ALPHAS:
        path_rng = np.random.default_rng(SEED + 4)
        risks, strict_ok, sub = [], [], []
        for _ in range(N_PATHS_RECONSTRUCT):
            syn = model.simulate(len(counts), path_rng, alpha=cal * alpha)
            risks.append(collision_risk(syn, bin_s))
            try:
                # forced strict, to show what the whole-second grid alone can do
                reconstruct_events(syn, bin_s, np.random.default_rng(SEED + 6),
                                   samplers["iid"], offsets=templates, on_overflow="raise")
                strict_ok.append(True)
            except ValueError:
                strict_ok.append(False)
            # the default path: 'auto' resolves to subsecond above alpha = 1, and
            # stays strict at or below it, so this can still refuse at alpha <= 1
            try:
                res = reconstruct_events(syn, bin_s, np.random.default_rng(SEED + 6),
                                         samplers["iid"], offsets=templates, alpha=alpha)
                sub.append(res.collision_report.to_dict() | {"n_batches": res.n_batches, "ok": 1.0})
            except ValueError:
                sub.append({"n_overflow_bins": np.nan, "finest_resolution_s": np.nan,
                            "n_subsecond_placements": np.nan, "n_batches": np.nan, "ok": 0.0})
        risk, subs = pd.DataFrame(risks).mean(), pd.DataFrame(sub)
        grid.append({
            "alpha": alpha,
            "strict_1s_paths_ok": float(np.mean(strict_ok)),
            "default_paths_ok": float(subs["ok"].mean()),
            "mean_max_count": risk["max_count"],
            "mean_max_fill_ratio": risk["max_fill_ratio"],
            "mean_refined_bins": subs["n_overflow_bins"].mean(),
            "finest_resolution_s": subs["finest_resolution_s"].min(),
            "frac_batches_off_whole_second":
                float(subs["n_subsecond_placements"].sum() / subs["n_batches"].sum()),
        })
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv(RESULTS / "reconstruction_by_alpha.csv", index=False)
    print(f"\nexport grid by load factor ({N_PATHS_RECONSTRUCT} paths each; strict = whole "
          f"seconds only, {int(bin_s)} slots per bin):")
    print(grid_df.round(4).to_string(index=False))

    state["microstructure"] = micro_df
    state["reconstruction_by_alpha"] = grid_df
    _save(state)
