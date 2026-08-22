from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from traffic_modeling.core import ModelSpec, fit_alias_registry

from ..baselines import BASELINE_BUILDERS, BASELINE_SPECS
from ..models import CountModel
from ..state import PipelineState
from .candidate_registry import (
    ABLATION_BUILDERS,
    ABLATION_SPECS,
    CANDIDATE_BUILDERS,
    CANDIDATE_KEYS,
    CANDIDATE_SPECS,
)

_ALL_SPECS: tuple[ModelSpec, ...] = (
    *CANDIDATE_SPECS.values(),
    *BASELINE_SPECS.values(),
    *ABLATION_SPECS.values(),
)

MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType({spec.key: spec for spec in _ALL_SPECS})
FIT_TARGETS = fit_alias_registry(_ALL_SPECS)

_BUILDERS = MappingProxyType(
    {
        **CANDIDATE_BUILDERS,
        **BASELINE_BUILDERS,
        **ABLATION_BUILDERS,
    }
)

if set(MODEL_SPECS) != set(_BUILDERS):
    raise RuntimeError("every model spec must have exactly one builder")
if len({spec.model.key for spec in _ALL_SPECS}) != len(_ALL_SPECS):
    raise RuntimeError("model identities must be unique")
if any(spec.parent_key not in CANDIDATE_SPECS for spec in ABLATION_SPECS.values()):
    raise RuntimeError("every ablation must name a registered candidate parent")


def model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"unregistered model key: {key}") from exc


def build_model(key: str, state: PipelineState, seed: int) -> CountModel:
    spec = model_spec(key)
    model = _BUILDERS[key](state, seed)
    model.role = spec.role.value
    model.rung = spec.ladder_rung
    return model


__all__ = [
    "CANDIDATE_KEYS",
    "CANDIDATE_SPECS",
    "FIT_TARGETS",
    "MODEL_SPECS",
    "build_model",
    "model_spec",
]
