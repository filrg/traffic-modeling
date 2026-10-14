from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from ..core import DatasetRef, config_fingerprint, sha256_file
from .base import (
    DatasetCapability,
    DatasetIntegrityError,
    DatasetSchemaError,
    TraceDataset,
)

# `Timestamp` is seconds from an unpublished calendar date, so this epoch is synthetic:
# hour-of-day survives, day-of-week labelling does not. Every subset shares it.
BURSTGPT_EPOCH = "1970-01-01T00:00:00+00:00"

BURSTGPT_SUBSETS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "combined-1-2": ("BurstGPT_1.csv", "BurstGPT_2.csv"),
        "period-3": ("BurstGPT_3.csv",),
    }
)

_IDENTITY = ("hpmll", "burstgpt", "v2.0")
_SPLIT = "full"

_REQUIRED_COLUMNS = ("Timestamp", "Model", "Request tokens", "Response tokens", "Log Type")
_OPTIONAL_CAPABILITY_COLUMNS = MappingProxyType(
    {
        DatasetCapability.SESSION_IDS: "Session ID",
        DatasetCapability.ELAPSED_TIMES: "Elapsed time",
    }
)
_BASE_CAPABILITIES = frozenset(
    {
        DatasetCapability.ARRIVAL_EVENTS,
        DatasetCapability.REQUEST_TOKENS,
        DatasetCapability.RESPONSE_TOKENS,
        DatasetCapability.REQUEST_TYPES,
    }
)


def subset_files(subset: str) -> tuple[str, ...]:
    """Return the release files a subset is defined over, in concatenation order."""
    try:
        return BURSTGPT_SUBSETS[subset]
    except KeyError:
        raise ValueError(
            f"unknown BurstGPT subset {subset!r}; expected from {sorted(BURSTGPT_SUBSETS)}"
        ) from None


def burstgpt_content_hash(source: Path, subset: str) -> str:
    """Digest a multi-file subset as one content hash over its per-file SHA-256 digests."""
    files = {}
    for name in subset_files(subset):
        path = Path(source) / name
        if not path.is_file():
            raise DatasetIntegrityError(f"BurstGPT subset {subset!r} is missing {path}")
        files[name] = sha256_file(path)
    return config_fingerprint({"subset": subset, "files": files})


def _numeric(frame: pd.DataFrame, column: str, source: Path) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if bool(values.isna().any()):
        raise DatasetSchemaError(f"{source.name} contains non-numeric {column!r} values")
    return values


@dataclass(frozen=True, slots=True)
class BurstGPTDatasetAdapter:
    epoch: str = BURSTGPT_EPOCH

    def load(self, source: Path, ref: DatasetRef) -> TraceDataset:
        self._validate_ref(ref)
        source = Path(source)
        if not source.is_dir():
            raise DatasetIntegrityError(f"BurstGPT source must be a release directory: {source}")
        actual_hash = burstgpt_content_hash(source, ref.subset)
        if actual_hash != ref.content_hash:
            raise DatasetIntegrityError(
                f"BurstGPT subset content hash mismatch: expected {ref.content_hash}, "
                f"got {actual_hash}"
            )

        paths = [source / name for name in subset_files(ref.subset)]
        headers = [set(pd.read_csv(path, nrows=0).columns) for path in paths]
        for path, columns in zip(paths, headers, strict=True):
            missing = sorted(set(_REQUIRED_COLUMNS) - columns)
            if missing:
                raise DatasetSchemaError(f"{path.name} is missing columns: {missing}")
        # a capability holds for a subset only if every file in it carries the column
        available = set.intersection(*headers)
        capabilities = set(_BASE_CAPABILITIES) | {
            capability
            for capability, column in _OPTIONAL_CAPABILITY_COLUMNS.items()
            if column in available
        }

        wanted = set(_REQUIRED_COLUMNS) | set(_OPTIONAL_CAPABILITY_COLUMNS.values())
        selected = sorted(available & wanted)
        frames = [self._read(path, selected) for path in paths]
        view = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        view = view.sort_values("submitted_at", kind="stable").reset_index(drop=True)
        view.insert(0, "job_id", np.arange(len(view), dtype=np.int64))
        return TraceDataset(ref=ref, jobs=view, capabilities=frozenset(capabilities))

    def _read(self, path: Path, columns: list[str]) -> pd.DataFrame:
        raw = pd.read_csv(path, usecols=columns, low_memory=False)
        seconds = _numeric(raw, "Timestamp", path)
        model = raw["Model"].astype("string").str.strip().str.lower()
        log_type = (
            raw["Log Type"].astype("string").str.strip().str.lower().str.removesuffix(" log")
        )
        view = pd.DataFrame(
            {
                "submitted_at": pd.Timestamp(self.epoch)
                + pd.to_timedelta(seconds.to_numpy(dtype=float), unit="s"),
                "request_tokens": _numeric(raw, "Request tokens", path).astype("int64"),
                "response_tokens": _numeric(raw, "Response tokens", path).astype("int64"),
                "request_type": (model + "/" + log_type).astype("string"),
            }
        )
        if "Session ID" in raw.columns:
            view["session_id"] = raw["Session ID"].astype("string")
        if "Elapsed time" in raw.columns:
            view["elapsed_s"] = pd.to_numeric(raw["Elapsed time"], errors="coerce")
        return view

    @staticmethod
    def _validate_ref(ref: DatasetRef) -> None:
        if (ref.provider, ref.dataset, ref.version) != _IDENTITY or ref.split != _SPLIT:
            supported = "/".join((*_IDENTITY, "<subset>", _SPLIT))
            raise ValueError(f"BurstGPTDatasetAdapter only supports {supported}")
        subset_files(ref.subset)
