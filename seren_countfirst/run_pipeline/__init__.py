"""Staged, resumable driver for the count-first pipeline."""
from .config import (
    AUDIT,
    FOLD_KWARGS,
    LONG_ACF_LAGS,
    MM4_KNOBS,
    N_PATHS_EVAL,
    N_PATHS_FOLD,
    N_PATHS_HORIZON,
    N_PATHS_RECONSTRUCT,
    N_PATHS_SCALING,
    RECONSTRUCT_ALPHAS,
    RUNG_KEYS,
    SEED,
)
from .stages.baselines import stage_baselines
from .stages.data import stage_data
from .stages.evaluate import stage_evaluate
from .stages.export import stage_export
from .stages.fit import stage_fit
from .stages.horizon import stage_horizon
from .stages.reconstruct import stage_reconstruct
from .stages.scaling import stage_scaling

#: stage name -> callable taking the remaining CLI arguments
STAGES = {
    "data": lambda args: stage_data(),
    "fit": stage_fit,
    "evaluate": lambda args: stage_evaluate(),
    "scaling": lambda args: stage_scaling(),
    "horizon": lambda args: stage_horizon(),
    "reconstruct": lambda args: stage_reconstruct(),
    "baselines": lambda args: stage_baselines(),
    "export": lambda args: stage_export(),
}

__all__ = [
    "AUDIT",
    "FOLD_KWARGS",
    "LONG_ACF_LAGS",
    "MM4_KNOBS",
    "N_PATHS_EVAL",
    "N_PATHS_FOLD",
    "N_PATHS_HORIZON",
    "N_PATHS_RECONSTRUCT",
    "N_PATHS_SCALING",
    "RECONSTRUCT_ALPHAS",
    "RUNG_KEYS",
    "SEED",
    "STAGES",
    "stage_baselines",
    "stage_data",
    "stage_evaluate",
    "stage_export",
    "stage_fit",
    "stage_horizon",
    "stage_reconstruct",
    "stage_scaling",
]
