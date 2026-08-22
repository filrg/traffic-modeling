from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, TypeVar

from .experiment import ModelRef, validate_identifier


class ModelRole(StrEnum):
    CANDIDATE = "candidate"
    BASELINE = "benchmark"
    ABLATION = "ablation"


class EstimationMethod(StrEnum):
    LIKELIHOOD = "likelihood"
    ANALYTIC_MOMENT = "analytic-moment"
    EMPIRICAL_MOMENT = "empirical-moment"
    FINITE_SAMPLE_MOMENT = "finite-sample-moment"


def _validate_spec(key: str, aliases: tuple[str, ...], model: ModelRef) -> None:
    validate_identifier(key, "key")
    if not isinstance(model, ModelRef):
        raise TypeError("model must be a ModelRef")
    if not isinstance(aliases, tuple):
        raise TypeError(f"{key}.aliases must be a tuple")
    if not aliases:
        raise ValueError(f"{key} must declare at least one CLI alias")
    for alias in aliases:
        validate_identifier(alias, f"{key}.aliases")
    if len(set(aliases)) != len(aliases):
        raise ValueError(f"{key} contains duplicate CLI aliases")


def _validate_rung(rung: int | str, key: str) -> None:
    if isinstance(rung, bool) or not isinstance(rung, (int, str)):
        raise TypeError(f"{key}.ladder_rung must be an integer or identifier")
    if isinstance(rung, int):
        if rung < 1:
            raise ValueError(f"{key}.ladder_rung must be positive")
    else:
        validate_identifier(rung, f"{key}.ladder_rung")


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    key: str
    aliases: tuple[str, ...]
    model: ModelRef
    ladder_rung: int | str
    estimation: EstimationMethod

    def __post_init__(self) -> None:
        _validate_spec(self.key, self.aliases, self.model)
        _validate_rung(self.ladder_rung, self.key)
        if not isinstance(self.estimation, EstimationMethod):
            raise TypeError("estimation must be an EstimationMethod")

    @property
    def role(self) -> ModelRole:
        return ModelRole.CANDIDATE

    @property
    def stop_at_first_failure(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    key: str
    aliases: tuple[str, ...]
    model: ModelRef
    estimation: EstimationMethod
    citation: str
    comparison_claim: str

    def __post_init__(self) -> None:
        _validate_spec(self.key, self.aliases, self.model)
        if not isinstance(self.estimation, EstimationMethod):
            raise TypeError("estimation must be an EstimationMethod")
        if not self.citation or not self.comparison_claim:
            raise ValueError(f"{self.key} must document its citation and comparison claim")

    @property
    def ladder_rung(self) -> None:
        return None

    @property
    def role(self) -> ModelRole:
        return ModelRole.BASELINE

    @property
    def stop_at_first_failure(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class AblationSpec:
    key: str
    aliases: tuple[str, ...]
    model: ModelRef
    ladder_rung: int | str
    estimation: EstimationMethod
    parent_key: str
    intervention: str

    def __post_init__(self) -> None:
        _validate_spec(self.key, self.aliases, self.model)
        _validate_rung(self.ladder_rung, self.key)
        validate_identifier(self.parent_key, "parent_key")
        if not isinstance(self.estimation, EstimationMethod):
            raise TypeError("estimation must be an EstimationMethod")
        if not self.intervention:
            raise ValueError(f"{self.key} must describe its intervention")

    @property
    def role(self) -> ModelRole:
        return ModelRole.ABLATION

    @property
    def stop_at_first_failure(self) -> bool:
        return False


ModelSpec = CandidateSpec | BaselineSpec | AblationSpec


class _IdentifiedSpec(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def aliases(self) -> tuple[str, ...]: ...


SpecT = TypeVar("SpecT", bound=_IdentifiedSpec)


def model_spec_registry(specs: Iterable[SpecT]) -> Mapping[str, SpecT]:
    registry: dict[str, SpecT] = {}
    for spec in specs:
        if spec.key in registry:
            raise ValueError(f"duplicate model key: {spec.key}")
        registry[spec.key] = spec
    return MappingProxyType(registry)


def fit_alias_registry(specs: Iterable[_IdentifiedSpec]) -> Mapping[str, str]:
    aliases: dict[str, str] = {}
    for spec in specs:
        for alias in spec.aliases:
            previous = aliases.get(alias)
            if previous is not None:
                raise ValueError(f"CLI alias {alias!r} maps to both {previous!r} and {spec.key!r}")
            aliases[alias] = spec.key
    return MappingProxyType(aliases)
