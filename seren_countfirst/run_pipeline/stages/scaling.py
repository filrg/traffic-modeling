from __future__ import annotations


from ...artifacts import write_csv
from ...scaling import scaling_report
from ...paths import RESULTS
from ...state import PipelineState
from ..config import N_PATHS_SCALING, SEED
from ..gate_diagnostics import full_gate_frame
from ..state import fitted_models, load_state, save_state


def stage_scaling() -> None:
    state = load_state()
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = state.get("winner", "gated_mm")
    model = fitted_models(state)[name]
    rep = scaling_report(model, state["n_bins"], n_paths=N_PATHS_SCALING, seed=SEED)
    write_csv(rep, RESULTS / "scaling_report.csv", index=False)
    # gate 8 grades the *_mean columns; the single-path ones are kept for
    # continuity but are too noisy to read
    cols = [
        "kappa_rule",
        "alpha",
        "rate_error_vs_theory",
        "IDC_15m",
        "IDC_15m_drift_log_mean",
        "IDC_1d_drift_log_mean",
        "ACF_lag1_drift_mean",
        "p_exceed_slots",
        "max_fill_ratio",
    ]
    print(rep[cols].round(4).to_string(index=False))
    state["scaling"] = rep
    save_state(state)
    _full_gate_diagnostic(state, name, model, rep)


def _full_gate_diagnostic(state: PipelineState, name: str, model, scaling) -> None:
    """Score every gate for the winner, gate 8 included.

    Diagnostic only: it ranks nothing and unlocks nothing. The sequential
    verdict in `gates_<winner>.csv` stands; this table runs the later gates even
    after an earlier failure.
    """
    frame = full_gate_frame(state, model, scaling, SEED)
    write_csv(frame, RESULTS / f"gates_full_{name}.csv", index=False)
    print(f"\nall gates for {name} (diagnostic, non-sequential):")
    print(frame.to_string(index=False))
