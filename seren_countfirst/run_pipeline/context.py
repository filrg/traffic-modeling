from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from traffic_modeling.core import RunSpec
from traffic_modeling.datasets import (
    DatasetAdapter,
    DatasetCapability,
    SerenDatasetAdapter,
    TraceDataset,
)

from ..paths import RESULTS, STATE_DIR, TRACE_CSV
from ..state import PipelineState, load_portable_state, save_portable_state
from .config import SEED
from .spec import DEFAULT_RUN_SPEC


class StateStore(Protocol):
    """Persistence boundary used by pipeline stages."""

    def load(self) -> PipelineState: ...

    def save(self, state: PipelineState) -> None: ...


@dataclass(frozen=True, slots=True)
class PortableStateStore:
    directory: Path

    def load(self) -> PipelineState:
        return load_portable_state(self.directory)

    def save(self, state: PipelineState) -> None:
        save_portable_state(state, self.directory)


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Runtime dependencies for one pipeline execution."""

    trace_csv: Path
    results: Path
    state_store: StateStore
    seed: int = SEED
    run_spec: RunSpec | None = None
    dataset_adapter: DatasetAdapter = field(default_factory=SerenDatasetAdapter)

    def __post_init__(self) -> None:
        if self.run_spec is not None and self.seed != self.run_spec.seed:
            raise ValueError(
                f"context seed {self.seed} does not match RunSpec seed {self.run_spec.seed}"
            )

    @classmethod
    def for_run(
        cls,
        *,
        trace_csv: Path,
        results: Path,
        state_store: StateStore,
        run_spec: RunSpec,
        dataset_adapter: DatasetAdapter | None = None,
    ) -> PipelineContext:
        return cls(
            trace_csv=trace_csv,
            results=results,
            state_store=state_store,
            seed=run_spec.seed,
            run_spec=run_spec,
            dataset_adapter=(
                dataset_adapter if dataset_adapter is not None else SerenDatasetAdapter()
            ),
        )

    def load_dataset(self, *required: DatasetCapability) -> TraceDataset:
        spec = self.run_spec or DEFAULT_RUN_SPEC
        dataset = self.dataset_adapter.load(self.trace_csv, spec.dataset)
        return dataset.require(*required)

    def result(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"result path must stay below the results directory: {relative}")
        return self.results / relative

    def load_state(self) -> PipelineState:
        return self.state_store.load()

    def save_state(self, state: PipelineState) -> None:
        self.results.mkdir(parents=True, exist_ok=True)
        self.state_store.save(state)


def default_context() -> PipelineContext:
    """Build the filesystem-backed context used by the CLI."""
    return PipelineContext(
        trace_csv=TRACE_CSV,
        results=RESULTS,
        state_store=PortableStateStore(STATE_DIR),
        run_spec=DEFAULT_RUN_SPEC,
    )


def resolve_context(context: PipelineContext | None) -> PipelineContext:
    return context if context is not None else default_context()
