from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..data import Split
from ..metrics import d_acf, d_idc, d_idc_shape
from .folds import evaluate_on_folds
from .gates import GateEvidence, GateReport, evaluate_gates


def summary_scores(trace_counts: np.ndarray, syn_paths: Sequence[np.ndarray],
                   busy_threshold: float) -> pd.Series:
    """Return the ranking score; meaningful only among candidates that passed the gates."""
    trace = np.asarray(trace_counts, dtype=float)
    return pd.Series(
        {
            "D_IDC": float(np.mean([d_idc(p, trace) for p in syn_paths])),
            "D_IDC_shape": float(np.mean([d_idc_shape(p, trace) for p in syn_paths])),
            "D_ACF": float(np.mean([d_acf(p, trace) for p in syn_paths])),
            "zero_bin_abs_error": float(
                np.mean([abs((p == 0).mean() - (trace == 0).mean()) for p in syn_paths])
            ),
            "rate_abs_error": float(np.mean([abs(p.mean() / trace.mean() - 1.0) for p in syn_paths])),
        }
    )


def evaluate_candidate(
    model,
    trace_counts: np.ndarray,
    busy_threshold: float,
    envelope: pd.DataFrame,
    n_paths: int = 20,
    seed: int = 0,
    fold_n_paths: int | None = None,
    folds: Sequence[Split] | None = None,
    scaling: pd.DataFrame | None = None,
    idc_envelope: pd.DataFrame | None = None,
    evidence: GateEvidence | None = None,
    stop_at_first_failure: bool = True,
    **sim_kwargs,
) -> dict[str, object]:
    """Score one rung of the ladder with the shared harness.

    `folds` and `scaling` supply gates 7 and 8. The fold windows reuse the same
    `sim_kwargs`, including the global rate calibration.
    """
    rng = np.random.default_rng(seed)
    paths = [model.simulate(len(trace_counts), rng, **sim_kwargs) for _ in range(n_paths)]
    base = evidence or GateEvidence()
    fold_ev = base.fold_evidence
    if folds:
        fold_ev = evaluate_on_folds(
            model, folds, busy_threshold, seed=seed, **sim_kwargs,
            n_paths=fold_n_paths if fold_n_paths is not None else min(n_paths, 10),
        )
    ev = replace(base, fold_evidence=fold_ev,
                 scaling=base.scaling if scaling is None else scaling,
                 idc_envelope=base.idc_envelope if idc_envelope is None else idc_envelope)
    gates = evaluate_gates(trace_counts, paths, busy_threshold, envelope,
                           stop_at_first_failure=stop_at_first_failure, evidence=ev)
    scores = summary_scores(trace_counts, paths, busy_threshold)
    return {
        "model": model.name,
        "rung": model.rung,
        "role": model.role,
        "fit": model.report.to_dict() if model.report else None,
        "gates": gates,
        "scores": scores,
        "paths": paths,
        "fold_evidence": ev.fold_evidence,
    }


def benchmark_table(results: Sequence[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for r in results:
        gates: GateReport = r["gates"]
        fail = gates.first_failure
        rows.append(
            {
                "model": r["model"],
                "rung": r["rung"],
                "role": r.get("role", "candidate"),
                "loglik": (r["fit"] or {}).get("loglik", np.nan),
                "AIC": (r["fit"] or {}).get("AIC", np.nan),
                "gates_passed": gates.passed,
                "first_failed_gate": fail.name if fail else "-",
                **r["scores"].to_dict(),
                "boundary_warnings": "; ".join((r["fit"] or {}).get("boundary_warnings", [])) or "-",
            }
        )
    return pd.DataFrame(rows).sort_values(["gates_passed", "D_IDC"], ascending=[False, True])
