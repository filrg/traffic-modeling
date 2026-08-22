from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from traffic_modeling.core import AblationSpec, CandidateSpec, model_spec_registry

from ..candidates import (
    GLI_NB_ABLATION_BUILDERS,
    GLI_NB_ABLATION_SPECS,
    GLI_NB_CANDIDATE_BUILDERS,
    GLI_NB_CANDIDATE_SPECS,
)
from ..models import CountModel
from ..state import PipelineState

ModelBuilder = Callable[[PipelineState, int], CountModel]

_CANDIDATES: tuple[CandidateSpec, ...] = tuple(GLI_NB_CANDIDATE_SPECS.values())
_ABLATIONS: tuple[AblationSpec, ...] = tuple(GLI_NB_ABLATION_SPECS.values())

CANDIDATE_SPECS = model_spec_registry(_CANDIDATES)
ABLATION_SPECS = model_spec_registry(_ABLATIONS)
CANDIDATE_KEYS = tuple(CANDIDATE_SPECS)

CANDIDATE_BUILDERS: Mapping[str, ModelBuilder] = MappingProxyType(dict(GLI_NB_CANDIDATE_BUILDERS))
ABLATION_BUILDERS: Mapping[str, ModelBuilder] = MappingProxyType(dict(GLI_NB_ABLATION_BUILDERS))
