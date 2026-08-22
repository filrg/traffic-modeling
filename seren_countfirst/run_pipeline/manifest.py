from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from traffic_modeling.core import RunManifest, StageExecution

from ..artifacts import write_json
from .context import PipelineContext

MANIFEST_NAME = "run_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest_path(context: PipelineContext) -> Path:
    return context.result(MANIFEST_NAME)


def load_run_manifest(path: Path) -> RunManifest:
    return RunManifest.from_json(path.read_text(encoding="utf-8"))


def save_run_manifest(manifest: RunManifest, path: Path) -> None:
    write_json(manifest.to_dict(), path)


def _manifest_for(context: PipelineContext) -> RunManifest | None:
    if context.run_spec is None or context.resolved_config is None:
        return None
    path = manifest_path(context)
    if not path.exists():
        return RunManifest(
            run_spec=context.run_spec,
            resolved_config=context.resolved_config,
            created_utc=_utc_now(),
        )
    manifest = load_run_manifest(path)
    if manifest.run_spec != context.run_spec:
        raise ValueError(
            f"existing manifest at {path} belongs to run {manifest.run_spec.run_id}, "
            f"not {context.run_spec.run_id}"
        )
    return manifest


def prepare_manifest(context: PipelineContext) -> RunManifest | None:
    manifest = _manifest_for(context)
    if manifest is not None and not manifest_path(context).exists():
        save_run_manifest(manifest, manifest_path(context))
    return manifest


def record_execution(context: PipelineContext, execution: StageExecution) -> RunManifest | None:
    manifest = _manifest_for(context)
    if manifest is None:
        return None
    updated = manifest.append(execution)
    save_run_manifest(updated, manifest_path(context))
    return updated
