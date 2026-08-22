from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from traffic_modeling.core import RunSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
_STATE_RELATIVE_PATH = Path("state_portable_data") / "state_portable"


def _from_env(var: str, default: Path) -> Path:
    return Path(os.environ[var]) if os.environ.get(var) else default


# raw trace; not tracked in the repo, verified by sha256 on load
TRACE_CSV = _from_env("SEREN_TRACE_CSV", REPO_ROOT / "trace_seren.csv")

# exported tables, gate reports and the frozen model card
RESULTS = _from_env("SEREN_RESULTS", REPO_ROOT / "results")

# resumable pipeline state as npy + csv + json; never pickle
STATE_DIR = _from_env("SEREN_STATE_DIR", RESULTS / "state_portable_data" / "state_portable")


class RunPathMode(StrEnum):
    LEGACY = "legacy"
    NAMESPACED = "namespaced"


@dataclass(frozen=True, slots=True)
class RunPaths:
    mode: RunPathMode
    results: Path
    state: Path


def namespaced_results(root: str | Path, run_spec: RunSpec) -> Path:
    """Map an immutable run identity to its isolated result directory."""
    if not isinstance(run_spec, RunSpec):
        raise TypeError("run_spec must be a RunSpec")
    return Path(root).joinpath(*run_spec.namespace)


def resolve_run_paths(
    run_spec: RunSpec,
    environ: Mapping[str, str] | None = None,
) -> RunPaths:
    """Resolve legacy or opt-in namespaced storage from one environment snapshot."""
    values = os.environ if environ is None else environ
    runs_dir = values.get("SEREN_RUNS_DIR")
    if runs_dir:
        conflicts = [
            name
            for name in ("SEREN_RESULTS", "SEREN_STATE_DIR")
            if values.get(name)
        ]
        if conflicts:
            joined = ", ".join(conflicts)
            raise ValueError(f"SEREN_RUNS_DIR cannot be combined with {joined}")
        results = namespaced_results(runs_dir, run_spec)
        return RunPaths(
            mode=RunPathMode.NAMESPACED,
            results=results,
            state=results / _STATE_RELATIVE_PATH,
        )

    results = (
        Path(values["SEREN_RESULTS"])
        if values.get("SEREN_RESULTS")
        else REPO_ROOT / "results"
    )
    state = (
        Path(values["SEREN_STATE_DIR"])
        if values.get("SEREN_STATE_DIR")
        else results / _STATE_RELATIVE_PATH
    )
    return RunPaths(mode=RunPathMode.LEGACY, results=results, state=state)
