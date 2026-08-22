from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from .context import PipelineContext
from .model_registry import FIT_TARGETS
from .stages.baselines import stage_baselines
from .stages.certify import stage_certify
from .stages.data import stage_data
from .stages.evaluate import stage_evaluate
from .stages.export import stage_export
from .stages.fit import stage_fit
from .stages.horizon import stage_horizon
from .stages.jobsize import stage_jobsize
from .stages.jobsize_folds import stage_jobsize_folds
from .stages.reconstruct import stage_reconstruct
from .stages.scaling import stage_scaling
from .stages.workload import stage_workload


class StageRunner(Protocol):
    def __call__(
        self,
        args: list[str],
        context: PipelineContext | None = None,
    ) -> None: ...


class _ContextStage(Protocol):
    def __call__(self, context: PipelineContext | None = None) -> None: ...


class ArgumentValidator(Protocol):
    def __call__(self, arguments: tuple[str, ...]) -> None: ...


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    run: StageRunner
    depends_on: tuple[str, ...] = ()
    reads_dataset: bool = False
    argument_validator: ArgumentValidator | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("stage name must be a non-empty string")
        if not callable(self.run):
            raise TypeError(f"{self.name}.run must be callable")
        if not isinstance(self.depends_on, tuple) or not all(
            isinstance(name, str) and name for name in self.depends_on
        ):
            raise TypeError(f"{self.name}.depends_on must be a tuple of stage names")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(f"{self.name} contains duplicate dependencies")
        if not isinstance(self.reads_dataset, bool):
            raise TypeError(f"{self.name}.reads_dataset must be a bool")
        if self.argument_validator is not None and not callable(self.argument_validator):
            raise TypeError(f"{self.name}.argument_validator must be callable")

    def validate_arguments(self, arguments: tuple[str, ...]) -> None:
        if not isinstance(arguments, tuple) or any(
            not isinstance(argument, str) or not argument for argument in arguments
        ):
            raise ValueError(f"stage `{self.name}` arguments must be non-empty strings")
        if self.argument_validator is None:
            if arguments:
                raise ValueError(f"stage `{self.name}` does not accept arguments: {list(arguments)}")
            return
        self.argument_validator(arguments)


@dataclass(frozen=True, slots=True)
class ReproductionStep:
    argv: tuple[str, ...]
    approximate_minutes: float
    skip_in_quick_run: bool = False


def _without_args(name: str, function: _ContextStage) -> StageRunner:
    def run(args: list[str], context: PipelineContext | None = None) -> None:
        if args:
            raise SystemExit(f"stage `{name}` does not accept arguments: {args}")
        function(context)

    return run


def _with_args(
    function: Callable[[list[str], PipelineContext | None], None],
) -> StageRunner:
    def run(args: list[str], context: PipelineContext | None = None) -> None:
        function(args, context)

    return run


def _validate_fit_arguments(arguments: tuple[str, ...]) -> None:
    if not arguments:
        raise ValueError("stage `fit` requires at least one model")
    if len(set(arguments)) != len(arguments):
        raise ValueError("stage `fit` model arguments must not contain duplicates")
    unknown = sorted(set(arguments) - set(FIT_TARGETS))
    if unknown:
        raise ValueError(f"unknown fit model(s) {unknown}; expected from {sorted(FIT_TARGETS)}")


def _resolve_stage_order(
    targets: tuple[str, ...], specs: Mapping[str, StageSpec]
) -> tuple[str, ...]:
    ordered: list[str] = []
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in active_set:
            start = active.index(name)
            cycle = (*active[start:], name)
            raise ValueError(f"stage dependency cycle: {' -> '.join(cycle)}")
        active.append(name)
        active_set.add(name)
        for dependency in specs[name].depends_on:
            visit(dependency)
        active.pop()
        active_set.remove(name)
        visited.add(name)
        ordered.append(name)

    for target in targets:
        visit(target)
    return tuple(ordered)


def validate_stage_graph(specs: Mapping[str, StageSpec]) -> None:
    """Reject inconsistent keys, unknown dependencies and dependency cycles."""
    for key, spec in specs.items():
        if key != spec.name:
            raise ValueError(f"stage registry key {key!r} does not match spec name {spec.name!r}")
        for dependency in spec.depends_on:
            if dependency not in specs:
                raise ValueError(f"stage `{spec.name}` depends on unknown stage `{dependency}`")
    _resolve_stage_order(tuple(specs), specs)


def plan_stages(targets: Iterable[str], specs: Mapping[str, StageSpec]) -> tuple[str, ...]:
    """Return the stable dependency-first closure of the requested stages."""
    requested = tuple(targets)
    if not requested:
        raise ValueError("at least one target stage is required")
    if len(set(requested)) != len(requested):
        raise ValueError("target stages must not contain duplicates")
    unknown = sorted(set(requested) - set(specs))
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; expected from {sorted(specs)}")
    validate_stage_graph(specs)
    return _resolve_stage_order(requested, specs)


def validate_stage_order(stages: Iterable[str], specs: Mapping[str, StageSpec]) -> tuple[str, ...]:
    """Accept omitted dependencies, but reject requested dependencies placed too late."""
    requested = tuple(stages)
    planned = plan_stages(requested, specs)
    requested_set = set(requested)
    expected = tuple(name for name in planned if name in requested_set)
    if requested != expected:
        raise ValueError(
            "requested stage order violates the dependency graph; use: " + " ".join(expected)
        )
    return requested


def _stage_spec_registry(specs: Iterable[StageSpec]) -> Mapping[str, StageSpec]:
    registry: dict[str, StageSpec] = {}
    for spec in specs:
        if spec.name in registry:
            raise ValueError(f"duplicate stage name: {spec.name}")
        registry[spec.name] = spec
    frozen = MappingProxyType(registry)
    validate_stage_graph(frozen)
    return frozen


_SPECS = (
    StageSpec("data", _without_args("data", stage_data), reads_dataset=True),
    StageSpec(
        "fit",
        _with_args(stage_fit),
        ("data",),
        argument_validator=_validate_fit_arguments,
    ),
    StageSpec("evaluate", _without_args("evaluate", stage_evaluate), ("data", "fit")),
    StageSpec("scaling", _without_args("scaling", stage_scaling), ("evaluate",)),
    StageSpec("certify", _without_args("certify", stage_certify), ("evaluate",)),
    StageSpec("horizon", _without_args("horizon", stage_horizon), ("evaluate",)),
    StageSpec("reconstruct", _without_args("reconstruct", stage_reconstruct), ("evaluate",)),
    StageSpec("baselines", _without_args("baselines", stage_baselines), ("data",)),
    StageSpec("jobsize", _without_args("jobsize", stage_jobsize), reads_dataset=True),
    StageSpec(
        "jobsize_folds",
        _without_args("jobsize_folds", stage_jobsize_folds),
        reads_dataset=True,
    ),
    StageSpec("export", _without_args("export", stage_export), ("evaluate",)),
    StageSpec("workload", _without_args("workload", stage_workload), ("evaluate", "jobsize")),
)

STAGE_SPECS = _stage_spec_registry(_SPECS)
# Compatibility map: callers historically treated STAGES as name -> callable.
STAGES = MappingProxyType({name: spec.run for name, spec in STAGE_SPECS.items()})

REPRODUCTION_STEPS = (
    ReproductionStep(("data",), 0.6),
    ReproductionStep(("fit", "r2", "r3", "r4"), 2.0),
    ReproductionStep(("fit", "gated"), 4.0),
    ReproductionStep(("fit", "mm", "mm2", "mm3", "mm4", "mm5"), 0.5),
    ReproductionStep(("fit", "onoff", "ingarch", "nogate"), 0.4),
    ReproductionStep(("evaluate",), 2.0),
    ReproductionStep(("scaling",), 3.0),
    ReproductionStep(("horizon",), 2.0),
    ReproductionStep(("reconstruct",), 2.0),
    ReproductionStep(("baselines",), 1.0),
    ReproductionStep(("jobsize",), 34.0, True),
    ReproductionStep(("jobsize_folds",), 11.0, True),
    ReproductionStep(("export",), 0.2),
    ReproductionStep(("workload",), 0.7, True),
    ReproductionStep(("certify",), 15.0, True),
)


def validate_reproduction_steps(
    steps: Iterable[ReproductionStep],
    specs: Mapping[str, StageSpec],
    *,
    quick: bool = False,
) -> None:
    """Require a fresh reproduction recipe to include dependencies first."""
    completed: set[str] = set()
    mode = "quick" if quick else "full"
    for step in steps:
        if quick and step.skip_in_quick_run:
            continue
        if not step.argv:
            raise ValueError(f"{mode} reproduction contains an empty command")
        name = step.argv[0]
        if name not in specs:
            raise ValueError(f"{mode} reproduction contains unknown stage `{name}`")
        try:
            specs[name].validate_arguments(tuple(step.argv[1:]))
        except ValueError as exc:
            raise ValueError(f"invalid {mode} reproduction invocation: {exc}") from exc
        missing = [
            dependency for dependency in specs[name].depends_on if dependency not in completed
        ]
        if missing:
            raise ValueError(
                f"{mode} reproduction stage `{name}` runs before dependencies {missing}"
            )
        completed.add(name)


validate_reproduction_steps(REPRODUCTION_STEPS, STAGE_SPECS)
validate_reproduction_steps(REPRODUCTION_STEPS, STAGE_SPECS, quick=True)
