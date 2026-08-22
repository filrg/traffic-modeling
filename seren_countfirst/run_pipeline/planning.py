from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from traffic_modeling.core import StageInvocation

from .registry import StageSpec, plan_stages


@dataclass(frozen=True, slots=True)
class InvocationRecipe:
    name: str
    invocations: tuple[StageInvocation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("recipe name must be a non-empty string")
        if not isinstance(self.invocations, tuple) or any(
            not isinstance(invocation, StageInvocation) for invocation in self.invocations
        ):
            raise TypeError("recipe invocations must be a tuple of StageInvocation values")
        if len(set(self.invocations)) != len(self.invocations):
            raise ValueError(f"recipe `{self.name}` contains duplicate invocations")


@dataclass(frozen=True, slots=True)
class PlannedInvocation:
    invocation: StageInvocation
    requested: bool

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, StageInvocation):
            raise TypeError("planned invocation must contain a StageInvocation")
        if not isinstance(self.requested, bool):
            raise TypeError("planned invocation requested flag must be a bool")


def _index_recipe(
    recipe: InvocationRecipe,
    specs: Mapping[str, StageSpec],
) -> dict[str, tuple[StageInvocation, ...]]:
    grouped: dict[str, list[StageInvocation]] = {}
    for invocation in recipe.invocations:
        spec = specs.get(invocation.stage)
        if spec is None:
            raise ValueError(
                f"recipe `{recipe.name}` contains unknown stage `{invocation.stage}`"
            )
        try:
            spec.validate_arguments(invocation.arguments)
        except ValueError as exc:
            raise ValueError(f"invalid invocation in recipe `{recipe.name}`: {exc}") from exc
        grouped.setdefault(invocation.stage, []).append(invocation)
    return {stage: tuple(invocations) for stage, invocations in grouped.items()}


def plan_invocations(
    targets: Iterable[StageInvocation],
    specs: Mapping[str, StageSpec],
    recipe: InvocationRecipe,
) -> tuple[PlannedInvocation, ...]:
    """Resolve a stage DAG into dependency-first, concrete commands."""
    requested = tuple(targets)
    if not requested:
        raise ValueError("at least one target invocation is required")
    target_stages = tuple(invocation.stage for invocation in requested)
    if len(set(target_stages)) != len(target_stages):
        raise ValueError("target invocations must name distinct stages")
    for invocation in requested:
        spec = specs.get(invocation.stage)
        if spec is None:
            raise ValueError(f"unknown target stage `{invocation.stage}`")
        spec.validate_arguments(invocation.arguments)

    ordered_stages = plan_stages(target_stages, specs)
    recipe_by_stage = _index_recipe(recipe, specs)
    target_by_stage = {invocation.stage: invocation for invocation in requested}
    planned: list[PlannedInvocation] = []
    for stage in ordered_stages:
        target = target_by_stage.get(stage)
        if target is not None:
            planned.append(PlannedInvocation(target, requested=True))
            continue
        dependencies = recipe_by_stage.get(stage)
        if not dependencies:
            raise ValueError(
                f"recipe `{recipe.name}` has no concrete invocation for dependency `{stage}`"
            )
        planned.extend(
            PlannedInvocation(invocation, requested=False) for invocation in dependencies
        )
    return tuple(planned)
