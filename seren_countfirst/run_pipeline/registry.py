from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .context import PipelineContext
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


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    run: StageRunner
    depends_on: tuple[str, ...] = ()


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


_SPECS = (
    StageSpec("data", _without_args("data", stage_data)),
    StageSpec("fit", _with_args(stage_fit), ("data",)),
    StageSpec("evaluate", _without_args("evaluate", stage_evaluate), ("data", "fit")),
    StageSpec("scaling", _without_args("scaling", stage_scaling), ("evaluate",)),
    StageSpec("certify", _without_args("certify", stage_certify), ("evaluate",)),
    StageSpec("horizon", _without_args("horizon", stage_horizon), ("evaluate",)),
    StageSpec("reconstruct", _without_args("reconstruct", stage_reconstruct), ("evaluate",)),
    StageSpec("baselines", _without_args("baselines", stage_baselines), ("data",)),
    StageSpec("jobsize", _without_args("jobsize", stage_jobsize)),
    StageSpec("jobsize_folds", _without_args("jobsize_folds", stage_jobsize_folds), ("jobsize",)),
    StageSpec("export", _without_args("export", stage_export), ("evaluate",)),
    StageSpec("workload", _without_args("workload", stage_workload), ("evaluate", "jobsize")),
)

STAGE_SPECS = {spec.name: spec for spec in _SPECS}
# Compatibility map: callers historically treated STAGES as name -> callable.
STAGES = {name: spec.run for name, spec in STAGE_SPECS.items()}

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
    ReproductionStep(("workload",), 0.7),
    ReproductionStep(("certify",), 15.0, True),
)
