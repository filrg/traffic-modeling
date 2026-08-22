from __future__ import annotations

from copy import deepcopy

from traffic_modeling.core import DatasetRef, ModelRef, RunSpec, config_fingerprint

from ..candidates import LONG_ACF_LAGS, MM4_TEXTURE
from ..data import BASE_FREQ
from .config import (
    AUDIT,
    CERTIFY_SEEDS,
    FOLD_KWARGS,
    N_PATHS_EVAL,
    N_PATHS_FOLD,
    N_PATHS_HORIZON,
    N_PATHS_RECONSTRUCT,
    N_PATHS_SCALING,
    RECONSTRUCT_ALPHAS,
    SEED,
)
from .model_registry import FIT_TARGETS

SEREN_TRACE_SHA256 = "747854c73ce8b3263b764aa9ff8d866d89903705d31fcd0e0520c33dd14af9a1"

_RESOLVED_CONFIG: dict[str, object] = {
    "audit": AUDIT,
    "base_frequency": BASE_FREQ,
    "certify_seeds": CERTIFY_SEEDS,
    "folds": FOLD_KWARGS,
    "long_acf_lags": LONG_ACF_LAGS,
    "mm4_knobs": MM4_TEXTURE.to_dict(),
    "n_paths": {
        "evaluation": N_PATHS_EVAL,
        "fold": N_PATHS_FOLD,
        "horizon": N_PATHS_HORIZON,
        "reconstruction": N_PATHS_RECONSTRUCT,
        "scaling": N_PATHS_SCALING,
    },
    "reconstruct_alphas": RECONSTRUCT_ALPHAS,
    # Keep the legacy field name so existing run fingerprints remain stable.
    "rung_keys": dict(FIT_TARGETS),
}


def resolved_config() -> dict[str, object]:
    """Return an isolated snapshot of the default run configuration."""
    return deepcopy(_RESOLVED_CONFIG)


DEFAULT_RUN_SPEC = RunSpec(
    dataset=DatasetRef(
        provider="internlm",
        dataset="acmetrace-seren",
        version="unversioned",
        subset="sft",
        split="full",
        content_hash=SEREN_TRACE_SHA256,
    ),
    model=ModelRef(
        framework="count-first",
        algorithm="gli-nb",
        variant="gated-mm4",
    ),
    evaluation_suite="structural-eight-gate-v1",
    seed=SEED,
    config_hash=config_fingerprint(_RESOLVED_CONFIG),
)
