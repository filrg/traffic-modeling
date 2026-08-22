from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..paths import RESULTS, STATE_DIR, TRACE_CSV
from ..state import PipelineState, load_portable_state, save_portable_state
from .config import SEED


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
    )


def resolve_context(context: PipelineContext | None) -> PipelineContext:
    return context if context is not None else default_context()
