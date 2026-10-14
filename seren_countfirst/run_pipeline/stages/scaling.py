from __future__ import annotations

from ...artifacts import write_csv
from ...scaling import scaling_report
from ...state import PipelineState
from ..config import N_PATHS_SCALING
from ..context import PipelineContext, resolve_context
from ..gate_diagnostics import full_gate_frame
from ..state import fitted_models


def stage_scaling(context: PipelineContext | None = None) -> None:
    context = resolve_context(context)
    state = context.load_state()
    name = state.get("winner", "gated_mm")
    model = fitted_models(state)[name]
    rep = scaling_report(model, state["n_bins"], n_paths=N_PATHS_SCALING, seed=context.seed)
    write_csv(rep, context.result("scaling_report.csv"), index=False)
    # gate 8 grades the path-averaged *_mean columns
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
    context.save_state(state)
    _full_gate_diagnostic(context, state, name, model, rep)


def _full_gate_diagnostic(
    context: PipelineContext, state: PipelineState, name: str, model, scaling
) -> None:
    """Score every gate for the winner, gate 8 included.

    Diagnostic only: it ranks nothing and unlocks nothing. The sequential
    verdict in `gates_<winner>.csv` stands; this table runs the later gates even
    after an earlier failure.
    """
    frame = full_gate_frame(state, model, scaling, context.seed)
    write_csv(frame, context.result(f"gates_full_{name}.csv"), index=False)
    print(f"\nall gates for {name} (diagnostic, non-sequential):")
    print(frame.to_string(index=False))
