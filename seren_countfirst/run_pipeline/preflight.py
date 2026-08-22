from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from traffic_modeling.core import (
    ArtifactRecord,
    RunManifest,
    StageExecution,
    StageInvocation,
    StageStatus,
    sha256_file,
)

from .artifact_tracking import current_artifact
from .context import PipelineContext
from .registry import StageSpec


@dataclass(frozen=True, slots=True)
class InputProvenance:
    records: tuple[ArtifactRecord, ...]
    complete: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResumeDecision:
    can_skip: bool
    reason: str
    execution: StageExecution | None = None


def _dataset_record(context: PipelineContext) -> ArtifactRecord:
    path = context.trace_csv
    if not path.is_file():
        raise FileNotFoundError(f"dataset source does not exist: {path}")
    stat = path.stat()
    digest = sha256_file(path)
    current = path.stat()
    if current.st_mtime_ns != stat.st_mtime_ns or current.st_size != stat.st_size:
        raise RuntimeError(f"dataset changed while it was being hashed: {path}")
    return ArtifactRecord("dataset", path.name, digest, stat.st_size)


def capture_inputs(
    context: PipelineContext,
    manifest: RunManifest | None,
    stage: StageSpec,
) -> InputProvenance:
    records: dict[tuple[str, str], ArtifactRecord] = {}
    issues: list[str] = []

    if stage.reads_dataset:
        try:
            dataset = _dataset_record(context)
            records[(dataset.scope, dataset.relative_path)] = dataset
            if context.run_spec is None or dataset.sha256 != context.run_spec.dataset.content_hash:
                issues.append("dataset content does not match RunSpec")
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append(str(exc))

    dependencies = set(stage.depends_on)
    owners: dict[tuple[str, str], tuple[ArtifactRecord, StageExecution]] = {}
    seen = {name: False for name in dependencies}
    had_outputs = {name: False for name in dependencies}
    if manifest is not None:
        for execution in manifest.executions:
            name = execution.invocation.stage
            if execution.status is not StageStatus.SUCCEEDED or name not in dependencies:
                continue
            seen[name] = True
            had_outputs[name] |= bool(execution.outputs)
            for output in execution.outputs:
                owners[(output.scope, output.relative_path)] = (output, execution)

    for dependency, found in seen.items():
        if not found:
            issues.append(f"dependency `{dependency}` has no successful execution")
        elif not had_outputs[dependency]:
            issues.append(f"dependency `{dependency}` has no recorded outputs")

    for key, (address, owner) in owners.items():
        if not owner.inputs_complete:
            issues.append(f"dependency `{owner.invocation.stage}` has incomplete input lineage")
        try:
            current = current_artifact(context, address)
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append(str(exc))
            continue
        if current is None:
            issues.append(f"dependency artifact is missing: {address.scope}/{address.relative_path}")
            continue
        if current.size_bytes != address.size_bytes or current.sha256 != address.sha256:
            issues.append(
                f"dependency artifact no longer matches its producer: "
                f"{address.scope}/{address.relative_path}"
            )
        records[key] = current

    unique_issues = tuple(dict.fromkeys(issues))
    return InputProvenance(
        records=tuple(sorted(records.values(), key=lambda item: (item.scope, item.relative_path))),
        complete=not unique_issues,
        issues=unique_issues,
    )


def _latest_success(
    manifest: RunManifest,
    invocation: StageInvocation,
) -> StageExecution | None:
    return next(
        (
            execution
            for execution in reversed(manifest.executions)
            if execution.status is StageStatus.SUCCEEDED
            and execution.invocation == invocation
        ),
        None,
    )


def _verify_record(
    context: PipelineContext,
    expected: ArtifactRecord,
    dataset: ArtifactRecord,
) -> str | None:
    if expected.scope == "dataset":
        if expected.relative_path != dataset.relative_path:
            return "dataset source name changed"
        current = dataset
    else:
        try:
            current = current_artifact(context, expected)
        except (OSError, RuntimeError, ValueError) as exc:
            return str(exc)
        if current is None:
            return f"artifact is missing: {expected.scope}/{expected.relative_path}"
    if current.size_bytes != expected.size_bytes:
        return f"artifact size changed: {expected.scope}/{expected.relative_path}"
    if current.sha256 != expected.sha256:
        return f"artifact hash changed: {expected.scope}/{expected.relative_path}"
    return None


def resume_decision(
    context: PipelineContext,
    manifest: RunManifest | None,
    invocation: StageInvocation,
    *,
    code_revision: str | None,
    working_tree_dirty: bool | None,
    environment: Mapping[str, object],
) -> ResumeDecision:
    if manifest is None:
        return ResumeDecision(False, "run manifest is unavailable")
    execution = _latest_success(manifest, invocation)
    if execution is None:
        return ResumeDecision(False, "no successful execution matches this invocation")
    if code_revision is None or execution.code_revision != code_revision:
        return ResumeDecision(False, "source revision changed or is unavailable", execution)
    if working_tree_dirty is not False or execution.working_tree_dirty is not False:
        return ResumeDecision(False, "current or recorded source tree is not clean", execution)
    if execution.environment != environment:
        return ResumeDecision(False, "runtime environment changed", execution)
    if not execution.inputs_complete:
        return ResumeDecision(False, "recorded input lineage is incomplete", execution)
    if not execution.outputs:
        return ResumeDecision(False, "execution has no recorded outputs", execution)
    if context.run_spec is None:
        return ResumeDecision(False, "RunSpec is unavailable", execution)

    try:
        dataset = _dataset_record(context)
    except (OSError, RuntimeError, ValueError) as exc:
        return ResumeDecision(False, str(exc), execution)
    if dataset.sha256 != context.run_spec.dataset.content_hash:
        return ResumeDecision(False, "dataset content does not match RunSpec", execution)

    output_keys = {(item.scope, item.relative_path) for item in execution.outputs}
    checked = [
        item
        for item in execution.inputs
        if (item.scope, item.relative_path) not in output_keys
    ]
    checked.extend(execution.outputs)
    for record in checked:
        reason = _verify_record(context, record, dataset)
        if reason is not None:
            return ResumeDecision(False, reason, execution)
    return ResumeDecision(True, "input and output provenance verified", execution)
