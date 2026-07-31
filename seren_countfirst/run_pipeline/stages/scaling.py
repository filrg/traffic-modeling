"""Load-scaling sweep and the full gate diagnostic."""
from __future__ import annotations


import numpy as np
import pandas as pd

from ... import expanding_window_folds, scaling_report
from ...evaluation import GateEvidence, evaluate_gates, evaluate_on_folds
from ...paths import RESULTS
from ..config import FOLD_KWARGS, N_PATHS_EVAL, N_PATHS_FOLD, N_PATHS_SCALING, SEED
from ..state import _load, _models, _save

def stage_scaling() -> None:
    state = _load()
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = state.get("winner", "gated_mm")
    model = _models(state)[name]
    rep = scaling_report(model, state["n_bins"], n_paths=N_PATHS_SCALING,
                         busy_threshold=state["busy_threshold"], seed=SEED)
    rep.to_csv(RESULTS / "scaling_report.csv", index=False)
    # the *_mean columns are the ones gate 8 grades; the single-path ones are
    # kept in the csv for continuity but are too noisy to read (they show a
    # non-zero drift at alpha = 1, where the drift is zero by construction)
    cols = ["kappa_rule", "alpha", "rate_error_vs_theory", "IDC_15m",
            "IDC_15m_drift_log_mean", "IDC_1d_drift_log_mean", "ACF_lag1_drift_mean",
            "p_exceed_slots", "max_fill_ratio"]
    print(rep[cols].round(4).to_string(index=False))
    state["scaling"] = rep
    _save(state)
    _full_gate_diagnostic(state, name, model, rep)


def _full_gate_diagnostic(state: dict, name: str, model, scaling: pd.DataFrame) -> None:
    """Every gate scored for the winner, gate 8 included.

    The sequential verdict in `gates_<winner>.csv` stands: this table runs the
    later gates even after an earlier failure, so a blocked candidate still
    shows what its fold stability and scaling drift look like.  It ranks
    nothing and unlocks nothing.
    """
    thr, env = state["busy_threshold"], state["envelope"]
    cal = state["trace_marginals"]["mean"] / float(model.params_["mean"])
    folds = expanding_window_folds(state["counts"], **FOLD_KWARGS)
    fold_ev = evaluate_on_folds(model, folds, thr, n_paths=N_PATHS_FOLD, seed=SEED + 1, alpha=cal)
    rng = np.random.default_rng(SEED + 1)
    paths = [model.simulate(state["n_bins"], rng, alpha=cal) for _ in range(N_PATHS_EVAL)]
    report = evaluate_gates(
        state["counts"].values, paths, thr, env, stop_at_first_failure=False,
        evidence=GateEvidence(fold_evidence=fold_ev, scaling=scaling,
                              idc_envelope=state.get("idc_envelope")),
    )
    frame = report.to_frame()
    frame.to_csv(RESULTS / f"gates_full_{name}.csv", index=False)
    print(f"\nall gates for {name} (diagnostic, non-sequential):")
    print(frame.to_string(index=False))
