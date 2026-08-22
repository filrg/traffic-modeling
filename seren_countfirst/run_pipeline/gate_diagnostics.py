from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import expanding_window_folds
from ..evaluation import GateEvidence, evaluate_gates, evaluate_on_folds
from ..state import PipelineState
from .config import FOLD_KWARGS, N_PATHS_EVAL, N_PATHS_FOLD


def full_gate_frame(
    state: PipelineState, model, scaling: pd.DataFrame, base_seed: int
) -> pd.DataFrame:
    """Evaluate every gate with the shared diagnostic seed schedule."""
    threshold = state["busy_threshold"]
    calibration = state["trace_marginals"]["mean"] / float(model.params_["mean"])
    folds = expanding_window_folds(state["counts"], **FOLD_KWARGS)
    evaluation_seed = base_seed + 1
    fold_evidence = evaluate_on_folds(
        model,
        folds,
        threshold,
        n_paths=N_PATHS_FOLD,
        seed=evaluation_seed,
        alpha=calibration,
    )
    rng = np.random.default_rng(evaluation_seed)
    paths = [model.simulate(state["n_bins"], rng, alpha=calibration) for _ in range(N_PATHS_EVAL)]
    return evaluate_gates(
        state["counts"].values,
        paths,
        threshold,
        state["envelope"],
        stop_at_first_failure=False,
        evidence=GateEvidence(
            fold_evidence=fold_evidence,
            scaling=scaling,
            idc_envelope=state.get("idc_envelope"),
        ),
    ).to_frame()
