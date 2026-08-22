from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core import DatasetRef, sha256_file
from .base import DatasetCapability, TraceDataset


class DatasetIntegrityError(ValueError):
    pass


class DatasetSchemaError(ValueError):
    pass


_RAW_COLUMNS = frozenset(
    {
        "job_id",
        "type",
        "state",
        "submit_time",
        "start_time",
        "end_time",
        "duration",
        "gpu_num",
    }
)

_CAPABILITIES = frozenset(
    {
        DatasetCapability.ARRIVAL_EVENTS,
        DatasetCapability.JOB_STATES,
        DatasetCapability.EXECUTION_INTERVALS,
        DatasetCapability.EXECUTION_DURATIONS,
        DatasetCapability.GPU_COUNTS,
    }
)


class SerenDatasetAdapter:
    def load(self, source: Path, ref: DatasetRef) -> TraceDataset:
        self._validate_ref(ref)
        actual_hash = sha256_file(source)
        if actual_hash != ref.content_hash:
            raise DatasetIntegrityError(
                f"trace SHA-256 mismatch: expected {ref.content_hash}, got {actual_hash}"
            )

        raw = pd.read_csv(source, low_memory=False)
        missing = sorted(_RAW_COLUMNS - set(raw.columns))
        if missing:
            raise DatasetSchemaError(f"Seren trace is missing columns: {missing}")

        take = raw["type"].eq("SFT")
        view = pd.DataFrame(
            {
                "job_id": raw.loc[take, "job_id"],
                "submitted_at": pd.to_datetime(raw.loc[take, "submit_time"], errors="coerce"),
                "started_at": pd.to_datetime(raw.loc[take, "start_time"], errors="coerce"),
                "ended_at": pd.to_datetime(raw.loc[take, "end_time"], errors="coerce"),
                "duration_s": pd.to_numeric(raw.loc[take, "duration"], errors="coerce"),
                "gpu_count": pd.to_numeric(raw.loc[take, "gpu_num"], errors="coerce"),
                "state": raw.loc[take, "state"],
            }
        )
        view = view.sort_values(["submitted_at", "job_id"], kind="stable").reset_index(drop=True)
        return TraceDataset(ref=ref, jobs=view, capabilities=_CAPABILITIES)

    @staticmethod
    def _validate_ref(ref: DatasetRef) -> None:
        identity = (ref.provider, ref.dataset, ref.version, ref.subset, ref.split)
        supported = ("internlm", "acmetrace-seren", "unversioned", "sft", "full")
        if identity != supported:
            raise ValueError(
                "SerenDatasetAdapter only supports internlm/acmetrace-seren/unversioned/sft/full"
            )
