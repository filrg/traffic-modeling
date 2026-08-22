from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import pandas as pd

from ..core import DatasetRef


class DatasetCapability(StrEnum):
    ARRIVAL_EVENTS = "arrival-events"
    JOB_STATES = "job-states"
    EXECUTION_INTERVALS = "execution-intervals"
    EXECUTION_DURATIONS = "execution-durations"
    GPU_COUNTS = "gpu-counts"


class DatasetCapabilityError(ValueError):
    pass


_CAPABILITY_COLUMNS = {
    DatasetCapability.ARRIVAL_EVENTS: frozenset({"job_id", "submitted_at"}),
    DatasetCapability.JOB_STATES: frozenset({"state"}),
    DatasetCapability.EXECUTION_INTERVALS: frozenset({"started_at", "ended_at"}),
    DatasetCapability.EXECUTION_DURATIONS: frozenset({"duration_s"}),
    DatasetCapability.GPU_COUNTS: frozenset({"gpu_count"}),
}


@dataclass(frozen=True, slots=True)
class TraceDataset:
    ref: DatasetRef
    jobs: pd.DataFrame
    capabilities: frozenset[DatasetCapability]

    def __post_init__(self) -> None:
        if not isinstance(self.ref, DatasetRef):
            raise TypeError("ref must be a DatasetRef")
        capabilities = frozenset(self.capabilities)
        invalid = [item for item in capabilities if not isinstance(item, DatasetCapability)]
        if invalid:
            raise TypeError(f"invalid dataset capabilities: {invalid}")
        required = set().union(*(_CAPABILITY_COLUMNS[item] for item in capabilities))
        missing = sorted(required - set(self.jobs.columns))
        if missing:
            raise ValueError(f"canonical dataset is missing columns: {missing}")
        if len(self.jobs):
            datetime_columns = required & {"submitted_at", "started_at", "ended_at"}
            invalid_datetimes = sorted(
                column
                for column in datetime_columns
                if not pd.api.types.is_datetime64_any_dtype(self.jobs[column].dtype)
            )
            if invalid_datetimes:
                raise TypeError(
                    f"canonical datetime columns have invalid dtype: {invalid_datetimes}"
                )
            numeric_columns = required & {"duration_s", "gpu_count"}
            invalid_numeric = sorted(
                column
                for column in numeric_columns
                if not pd.api.types.is_numeric_dtype(self.jobs[column].dtype)
            )
            if invalid_numeric:
                raise TypeError(f"canonical numeric columns have invalid dtype: {invalid_numeric}")
        object.__setattr__(self, "jobs", self.jobs.reset_index(drop=True).copy())
        object.__setattr__(self, "capabilities", capabilities)

    def require(self, *required: DatasetCapability) -> TraceDataset:
        missing = sorted(set(required) - self.capabilities)
        if missing:
            names = [item.value for item in missing]
            raise DatasetCapabilityError(f"dataset {self.ref.key} lacks capabilities: {names}")
        return self

    def jobs_in_states(self, states: Collection[str]) -> pd.DataFrame:
        self.require(DatasetCapability.JOB_STATES)
        return self.jobs.loc[self.jobs["state"].isin(states)].reset_index(drop=True)


class DatasetAdapter(Protocol):
    def load(self, source: Path, ref: DatasetRef) -> TraceDataset: ...
