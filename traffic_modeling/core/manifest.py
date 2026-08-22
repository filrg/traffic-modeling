from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, cast

from .experiment import RunSpec, config_fingerprint, validate_identifier, validate_sha256


def _require_keys(payload: Mapping[str, Any], expected: Collection[str], label: str) -> None:
    expected_keys = set(expected)
    actual = set(payload)
    if actual != expected_keys:
        raise ValueError(
            f"invalid {label} fields; missing={sorted(expected_keys - actual)}, "
            f"extra={sorted(actual - expected_keys)}"
        )


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_mapping(value: Mapping[str, object], field: str) -> Mapping[str, object]:
    try:
        config_fingerprint(value)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid {field}: {exc}") from exc
    return cast(Mapping[str, object], _freeze_json(value))


def _timestamp(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a UTC timestamp") from exc
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{field} must be a UTC timestamp")
    return value


class StageStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    scope: str
    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        validate_identifier(self.scope, "artifact scope")
        if not isinstance(self.relative_path, str):
            raise TypeError("artifact relative_path must be a string")
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or self.relative_path == "."
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in self.relative_path
            or path.as_posix() != self.relative_path
        ):
            raise ValueError("artifact relative_path must be a canonical relative POSIX path")
        validate_sha256(self.sha256, "artifact sha256")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("artifact size_bytes must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactRecord:
        _require_keys(
            payload,
            {"scope", "relative_path", "sha256", "size_bytes"},
            "ArtifactRecord",
        )
        try:
            return cls(
                scope=payload["scope"],
                relative_path=payload["relative_path"],
                sha256=payload["sha256"],
                size_bytes=payload["size_bytes"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid artifact record: {exc}") from exc


_STAGE_EXECUTION_V1_FIELDS = frozenset(
    {
        "invocation",
        "status",
        "started_utc",
        "elapsed_seconds",
        "environment",
        "code_revision",
        "working_tree_dirty",
        "error",
    }
)
_STAGE_EXECUTION_V2_FIELDS = _STAGE_EXECUTION_V1_FIELDS | {"outputs"}
_STAGE_EXECUTION_V3_FIELDS = _STAGE_EXECUTION_V2_FIELDS | {"inputs", "inputs_complete"}


@dataclass(frozen=True, slots=True)
class StageInvocation:
    stage: str
    arguments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.stage, "stage")
        if not isinstance(self.arguments, tuple):
            raise TypeError("arguments must be a tuple")
        if any(not isinstance(item, str) or not item for item in self.arguments):
            raise ValueError("arguments must contain non-empty strings")

    def to_dict(self) -> dict[str, object]:
        return {"stage": self.stage, "arguments": list(self.arguments)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StageInvocation:
        _require_keys(payload, {"stage", "arguments"}, "StageInvocation")
        stage = payload.get("stage")
        arguments = payload.get("arguments")
        if not isinstance(stage, str) or not isinstance(arguments, list):
            raise ValueError("invalid stage invocation")
        return cls(stage, tuple(arguments))


@dataclass(frozen=True, slots=True)
class StageExecution:
    invocation: StageInvocation
    status: StageStatus
    started_utc: str
    elapsed_seconds: float
    environment: Mapping[str, object]
    code_revision: str | None = None
    working_tree_dirty: bool | None = None
    error: str | None = None
    outputs: tuple[ArtifactRecord, ...] = ()
    inputs: tuple[ArtifactRecord, ...] = ()
    inputs_complete: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, StageInvocation):
            raise TypeError("invocation must be a StageInvocation")
        if not isinstance(self.status, StageStatus):
            raise TypeError("status must be a StageStatus")
        _timestamp(self.started_utc, "started_utc")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be a finite non-negative number")
        if self.code_revision is not None and not isinstance(self.code_revision, str):
            raise TypeError("code_revision must be a string or None")
        if self.working_tree_dirty is not None and not isinstance(self.working_tree_dirty, bool):
            raise TypeError("working_tree_dirty must be a bool or None")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        if self.status is StageStatus.SUCCEEDED and self.error is not None:
            raise ValueError("a successful execution cannot contain an error")
        if self.status is StageStatus.FAILED and not self.error:
            raise ValueError("a failed execution must contain an error")
        if not isinstance(self.inputs_complete, bool):
            raise TypeError("inputs_complete must be a bool")
        normalized_artifacts = {}
        for field, records in (("inputs", self.inputs), ("outputs", self.outputs)):
            if not isinstance(records, tuple) or any(
                not isinstance(item, ArtifactRecord) for item in records
            ):
                raise TypeError(f"{field} must be a tuple of ArtifactRecord values")
            keys = [(item.scope, item.relative_path) for item in records]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{field} cannot contain duplicate artifact paths")
            normalized_artifacts[field] = tuple(
                sorted(records, key=lambda item: (item.scope, item.relative_path))
            )
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "environment", _freeze_mapping(self.environment, "environment"))
        object.__setattr__(self, "inputs", normalized_artifacts["inputs"])
        object.__setattr__(self, "outputs", normalized_artifacts["outputs"])

    def to_dict(self) -> dict[str, object]:
        return {
            "invocation": self.invocation.to_dict(),
            "status": self.status.value,
            "started_utc": self.started_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "environment": _thaw_json(self.environment),
            "inputs": [item.to_dict() for item in self.inputs],
            "inputs_complete": self.inputs_complete,
            "outputs": [item.to_dict() for item in self.outputs],
            "code_revision": self.code_revision,
            "working_tree_dirty": self.working_tree_dirty,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StageExecution:
        _require_keys(payload, _STAGE_EXECUTION_V3_FIELDS, "StageExecution")
        invocation = payload.get("invocation")
        environment = payload.get("environment")
        inputs = payload.get("inputs")
        inputs_complete = payload.get("inputs_complete")
        outputs = payload.get("outputs")
        if (
            not isinstance(invocation, Mapping)
            or not isinstance(environment, Mapping)
            or not isinstance(inputs, list)
            or not isinstance(inputs_complete, bool)
            or not isinstance(outputs, list)
        ):
            raise ValueError("invalid stage execution")
        if any(not isinstance(item, Mapping) for item in (*inputs, *outputs)):
            raise ValueError("inputs and outputs must contain objects")
        status_value = payload.get("status")
        if not isinstance(status_value, str):
            raise ValueError("invalid stage execution status")
        try:
            status = StageStatus(status_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid stage execution status") from exc
        elapsed_seconds = payload.get("elapsed_seconds")
        code_revision = payload.get("code_revision")
        working_tree_dirty = payload.get("working_tree_dirty")
        error = payload.get("error")
        if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float)):
            raise ValueError("invalid elapsed_seconds")
        if code_revision is not None and not isinstance(code_revision, str):
            raise ValueError("invalid code_revision")
        if working_tree_dirty is not None and not isinstance(working_tree_dirty, bool):
            raise ValueError("invalid working_tree_dirty")
        if error is not None and not isinstance(error, str):
            raise ValueError("invalid execution error")
        return cls(
            invocation=StageInvocation.from_dict(invocation),
            status=status,
            started_utc=_timestamp(payload.get("started_utc"), "started_utc"),
            elapsed_seconds=elapsed_seconds,
            environment=environment,
            inputs=tuple(ArtifactRecord.from_dict(item) for item in inputs),
            inputs_complete=inputs_complete,
            outputs=tuple(ArtifactRecord.from_dict(item) for item in outputs),
            code_revision=code_revision,
            working_tree_dirty=working_tree_dirty,
            error=error,
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_spec: RunSpec
    resolved_config: Mapping[str, object]
    created_utc: str
    executions: tuple[StageExecution, ...] = ()
    schema_version: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.run_spec, RunSpec):
            raise TypeError("run_spec must be a RunSpec")
        if isinstance(self.schema_version, bool) or self.schema_version != 3:
            raise ValueError(f"unsupported RunManifest schema version: {self.schema_version}")
        _timestamp(self.created_utc, "created_utc")
        if not isinstance(self.executions, tuple) or any(
            not isinstance(item, StageExecution) for item in self.executions
        ):
            raise TypeError("executions must be a tuple of StageExecution values")
        config = _freeze_mapping(self.resolved_config, "resolved_config")
        if config_fingerprint(config) != self.run_spec.config_hash:
            raise ValueError("resolved_config does not match RunSpec config_hash")
        object.__setattr__(self, "resolved_config", config)

    def append(self, execution: StageExecution) -> RunManifest:
        if not isinstance(execution, StageExecution):
            raise TypeError("execution must be a StageExecution")
        return replace(self, executions=(*self.executions, execution))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_spec.run_id,
            "run_fingerprint": self.run_spec.fingerprint,
            "run_spec": self.run_spec.to_dict(),
            "resolved_config": _thaw_json(self.resolved_config),
            "created_utc": self.created_utc,
            "executions": [item.to_dict() for item in self.executions],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunManifest:
        _require_keys(
            payload,
            {
                "schema_version",
                "run_id",
                "run_fingerprint",
                "run_spec",
                "resolved_config",
                "created_utc",
                "executions",
            },
            "RunManifest",
        )
        run_spec = payload.get("run_spec")
        resolved_config = payload.get("resolved_config")
        executions = payload.get("executions")
        if (
            not isinstance(run_spec, Mapping)
            or not isinstance(resolved_config, Mapping)
            or not isinstance(executions, list)
        ):
            raise ValueError("invalid run manifest")
        schema_version = payload.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("schema_version must be an integer")
        if schema_version not in {1, 2, 3}:
            raise ValueError(f"unsupported RunManifest schema version: {schema_version}")
        normalized_executions: list[Mapping[str, Any]] = []
        for item in executions:
            if not isinstance(item, Mapping):
                raise ValueError("executions must contain objects")
            expected = {
                1: _STAGE_EXECUTION_V1_FIELDS,
                2: _STAGE_EXECUTION_V2_FIELDS,
                3: _STAGE_EXECUTION_V3_FIELDS,
            }[schema_version]
            _require_keys(item, expected, f"StageExecution v{schema_version}")
            upgraded = dict(item)
            if schema_version == 1:
                upgraded["outputs"] = []
            if schema_version <= 2:
                upgraded["inputs"] = []
                upgraded["inputs_complete"] = False
            normalized_executions.append(upgraded)
        manifest = cls(
            run_spec=RunSpec.from_dict(run_spec),
            resolved_config=resolved_config,
            created_utc=_timestamp(payload.get("created_utc"), "created_utc"),
            executions=tuple(StageExecution.from_dict(item) for item in normalized_executions),
            schema_version=3,
        )
        if payload.get("run_id") != manifest.run_spec.run_id:
            raise ValueError("run_id does not match RunSpec")
        if payload.get("run_fingerprint") != manifest.run_spec.fingerprint:
            raise ValueError("run_fingerprint does not match RunSpec")
        return manifest

    @classmethod
    def from_json(cls, payload: str) -> RunManifest:
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise ValueError("RunManifest JSON must contain an object")
        return cls.from_dict(decoded)
