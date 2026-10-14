from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, NotRequired, TypedDict

import numpy as np
import pandas as pd

from traffic_modeling.counts.folds import evaluate_on_folds
from traffic_modeling.counts.gates import GateEvidence, GateReport, evaluate_gates
from traffic_modeling.counts.metrics import d_acf, d_idc, d_idc_shape
from traffic_modeling.counts.series import Split

from ..models import CountModel

_MODEL_RUNG = object()


class ModelEvaluation(TypedDict):
    model: str
    rung: object
    role: str
    fit: dict[str, Any] | None
    gates: GateReport
    scores: pd.Series
    paths: list[np.ndarray]
    fold_evidence: pd.DataFrame | None
    key: NotRequired[str]
    label: NotRequired[str]


def summary_scores(
    trace_counts: np.ndarray, syn_paths: Sequence[np.ndarray], busy_threshold: float
) -> pd.Series:
    """Return the ranking scores used after structural gate evaluation."""
    trace = np.asarray(trace_counts, dtype=float)
    return pd.Series(
        {
            "D_IDC": float(np.mean([d_idc(p, trace) for p in syn_paths])),
            "D_IDC_shape": float(np.mean([d_idc_shape(p, trace) for p in syn_paths])),
            "D_ACF": float(np.mean([d_acf(p, trace) for p in syn_paths])),
            "zero_bin_abs_error": float(
                np.mean([abs((p == 0).mean() - (trace == 0).mean()) for p in syn_paths])
            ),
            "rate_abs_error": float(
                np.mean([abs(p.mean() / trace.mean() - 1.0) for p in syn_paths])
            ),
        }
    )


def evaluate_model(
    model: CountModel,
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
    role: str | None = None,
    rung: object = _MODEL_RUNG,
    **sim_kwargs,
) -> ModelEvaluation:
    """Simulate and score any count model with the shared gate harness."""
    rng = np.random.default_rng(seed)
    paths = [model.simulate(len(trace_counts), rng, **sim_kwargs) for _ in range(n_paths)]
    base = evidence or GateEvidence()
    fold_evidence = base.fold_evidence
    if folds:
        fold_evidence = evaluate_on_folds(
            model,
            folds,
            busy_threshold,
            seed=seed,
            **sim_kwargs,
            n_paths=fold_n_paths if fold_n_paths is not None else min(n_paths, 10),
        )
    resolved_evidence = replace(
        base,
        fold_evidence=fold_evidence,
        scaling=base.scaling if scaling is None else scaling,
        idc_envelope=base.idc_envelope if idc_envelope is None else idc_envelope,
    )
    gates = evaluate_gates(
        trace_counts,
        paths,
        busy_threshold,
        envelope,
        stop_at_first_failure=stop_at_first_failure,
        evidence=resolved_evidence,
    )
    scores = summary_scores(trace_counts, paths, busy_threshold)
    return {
        "model": model.name,
        "rung": getattr(model, "rung", None) if rung is _MODEL_RUNG else rung,
        "role": getattr(model, "role", "candidate") if role is None else role,
        "fit": model.report.to_dict() if model.report else None,
        "gates": gates,
        "scores": scores,
        "paths": paths,
        "fold_evidence": resolved_evidence.fold_evidence,
    }


def benchmark_table(results: Sequence[ModelEvaluation]) -> pd.DataFrame:
    rows = []
    for result in results:
        gates: GateReport = result["gates"]
        failure = gates.first_failure
        fit = result["fit"] or {}
        rows.append(
            {
                "model": result["model"],
                "rung": result["rung"],
                "role": result["role"],
                "loglik": fit.get("loglik", np.nan),
                "AIC": fit.get("AIC", np.nan),
                "gates_passed": gates.passed,
                "first_failed_gate": failure.name if failure else "-",
                **result["scores"].to_dict(),
                "boundary_warnings": "; ".join(fit.get("boundary_warnings", [])) or "-",
            }
        )
    return pd.DataFrame(rows).sort_values(["gates_passed", "D_IDC"], ascending=[False, True])
