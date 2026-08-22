from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_IDENTIFIER = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{field} must be a lowercase identifier containing letters, digits, '.', '_' or '-': "
            f"{value!r}"
        )


def _validate_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def _validate_json(value: object, path: str = "config") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key: {key!r}")
            _validate_json(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    _validate_json(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def config_fingerprint(config: Mapping[str, object]) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible configuration."""
    return hashlib.sha256(_canonical_json(dict(config)).encode("utf-8")).hexdigest()


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"invalid {label} fields; missing={missing}, extra={extra}")


@dataclass(frozen=True, slots=True)
class DatasetRef:
    provider: str
    dataset: str
    version: str
    subset: str
    split: str
    content_hash: str

    def __post_init__(self) -> None:
        for field in ("provider", "dataset", "version", "subset", "split"):
            validate_identifier(getattr(self, field), field)
        _validate_sha256(self.content_hash, "content_hash")

    @property
    def key(self) -> str:
        return "/".join((self.provider, self.dataset, self.version, self.subset, self.split))

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "dataset": self.dataset,
            "version": self.version,
            "subset": self.subset,
            "split": self.split,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetRef:
        _require_keys(
            payload,
            {"provider", "dataset", "version", "subset", "split", "content_hash"},
            "DatasetRef",
        )
        return cls(
            provider=_string(payload, "provider"),
            dataset=_string(payload, "dataset"),
            version=_string(payload, "version"),
            subset=_string(payload, "subset"),
            split=_string(payload, "split"),
            content_hash=_string(payload, "content_hash"),
        )


@dataclass(frozen=True, slots=True)
class ModelRef:
    framework: str
    algorithm: str
    variant: str

    def __post_init__(self) -> None:
        for field in ("framework", "algorithm", "variant"):
            validate_identifier(getattr(self, field), field)

    @property
    def key(self) -> str:
        return "/".join((self.framework, self.algorithm, self.variant))

    def to_dict(self) -> dict[str, str]:
        return {
            "framework": self.framework,
            "algorithm": self.algorithm,
            "variant": self.variant,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelRef:
        _require_keys(payload, {"framework", "algorithm", "variant"}, "ModelRef")
        return cls(
            framework=_string(payload, "framework"),
            algorithm=_string(payload, "algorithm"),
            variant=_string(payload, "variant"),
        )


@dataclass(frozen=True, slots=True)
class RunSpec:
    dataset: DatasetRef
    model: ModelRef
    evaluation_suite: str
    seed: int
    config_hash: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetRef):
            raise TypeError("dataset must be a DatasetRef")
        if not isinstance(self.model, ModelRef):
            raise TypeError("model must be a ModelRef")
        validate_identifier(self.evaluation_suite, "evaluation_suite")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32)")
        _validate_sha256(self.config_hash, "config_hash")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
        ):
            raise ValueError(f"unsupported RunSpec schema version: {self.schema_version}")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        return self.fingerprint[:16]

    @property
    def namespace(self) -> tuple[str, ...]:
        return (
            self.dataset.provider,
            self.dataset.dataset,
            self.dataset.version,
            self.dataset.subset,
            self.dataset.split,
            self.model.framework,
            self.model.algorithm,
            self.model.variant,
            self.run_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset.to_dict(),
            "model": self.model.to_dict(),
            "evaluation_suite": self.evaluation_suite,
            "seed": self.seed,
            "config_hash": self.config_hash,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunSpec:
        _require_keys(
            payload,
            {
                "schema_version",
                "dataset",
                "model",
                "evaluation_suite",
                "seed",
                "config_hash",
            },
            "RunSpec",
        )
        dataset = payload.get("dataset")
        model = payload.get("model")
        seed = payload.get("seed")
        schema_version = payload.get("schema_version")
        if not isinstance(dataset, Mapping):
            raise ValueError("dataset must be an object")
        if not isinstance(model, Mapping):
            raise ValueError("model must be an object")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("schema_version must be an integer")
        return cls(
            dataset=DatasetRef.from_dict(dataset),
            model=ModelRef.from_dict(model),
            evaluation_suite=_string(payload, "evaluation_suite"),
            seed=seed,
            config_hash=_string(payload, "config_hash"),
            schema_version=schema_version,
        )

    @classmethod
    def from_json(cls, payload: str) -> RunSpec:
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise ValueError("RunSpec JSON must contain an object")
        return cls.from_dict(decoded)
