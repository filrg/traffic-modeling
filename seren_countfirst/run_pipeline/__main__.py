from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

from traffic_modeling.core import (
    ArtifactRecord,
    StageExecution,
    StageInvocation,
    StageStatus,
)

from ..provenance import runtime_environment, source_revision
from .artifact_tracking import ArtifactSnapshot, changed_artifacts, snapshot_artifacts
from .context import PipelineContext, resolve_context
from .manifest import prepare_manifest, record_execution
from .registry import STAGE_SPECS, plan_stages, validate_stage_order

USAGE = """usage:
    python -m seren_countfirst.run_pipeline <stage> [<stage> ...]
    python -m seren_countfirst.run_pipeline --plan <target-stage> [<target-stage> ...]

    data                      ingest the trace, verify the audit facts
    fit <model> [<model> ...] fit candidates, baselines or ablations
    evaluate                  score candidates, pick the winner
    scaling                   load-scaling sweep and the full gate table
    certify                   the full gate table over CERTIFY_SEEDS seeds
    horizon                   horizon distributions and scale support
    reconstruct               counts -> batch timestamps -> job arrivals
    baselines                 the bootstrap yardstick
    jobsize                   durations -> work units, and the law fitted to them
    jobsize_folds             the size law on held-out, rolling-origin windows
    export                    freeze the winner into the model card
    workload                  arrivals x sizes -> a workload, and work under load

`--plan` prints the stage-level dependency closure without executing any stage;
it accepts stage names, not `fit` model names.
Normal stage commands still use existing portable state for omitted dependencies.

Paths come from paths.py. SEREN_TRACE_CSV, SEREN_RESULTS and SEREN_STATE_DIR
preserve the legacy layout; SEREN_RUNS_DIR opts into RunSpec-based isolation.
"""


def _print_plan(targets: list[str]) -> None:
    try:
        planned = plan_stages(targets, STAGE_SPECS)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    requested = set(targets)
    print("dependency plan (no stages executed):")
    for index, name in enumerate(planned, start=1):
        kind = "target" if name in requested else "dependency"
        dependencies = ", ".join(STAGE_SPECS[name].depends_on) or "none"
        print(f"  {index:>2}. {name:<14} [{kind}; requires: {dependencies}]")


def _run_stage(name: str, arguments: list[str], context: PipelineContext) -> None:
    invocation = StageInvocation(name, tuple(arguments))
    manifest = prepare_manifest(context)
    before: ArtifactSnapshot | None = None
    if manifest is not None:
        before = snapshot_artifacts(context)
    revision, dirty = source_revision()
    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    started = time.perf_counter()
    try:
        STAGE_SPECS[name].run(arguments, context)
        outputs = changed_artifacts(context, before) if before is not None else ()
    except BaseException as exc:
        partial_outputs: tuple[ArtifactRecord, ...] = ()
        if before is not None:
            try:
                partial_outputs = changed_artifacts(context, before)
            except Exception as artifact_error:
                exc.add_note(f"could not collect output provenance: {artifact_error}")
        execution = StageExecution(
            invocation=invocation,
            status=StageStatus.FAILED,
            started_utc=started_utc,
            elapsed_seconds=time.perf_counter() - started,
            environment=runtime_environment(),
            outputs=partial_outputs,
            code_revision=revision,
            working_tree_dirty=dirty,
            error=f"{type(exc).__name__}: {exc}",
        )
        try:
            record_execution(context, execution)
        except Exception as manifest_error:
            exc.add_note(f"could not record failed execution: {manifest_error}")
        raise
    record_execution(
        context,
        StageExecution(
            invocation=invocation,
            status=StageStatus.SUCCEEDED,
            started_utc=started_utc,
            elapsed_seconds=time.perf_counter() - started,
            environment=runtime_environment(),
            outputs=outputs,
            code_revision=revision,
            working_tree_dirty=dirty,
        ),
    )


def main(argv: list[str], context: PipelineContext | None = None) -> None:
    if not argv:
        raise SystemExit(USAGE)
    if argv[0] == "--plan":
        _print_plan(argv[1:])
        return
    if argv[0] not in STAGE_SPECS:
        raise SystemExit(USAGE)
    runtime = resolve_context(context)
    if argv[0] == "fit":
        _run_stage("fit", argv[1:], runtime)
        return
    unknown = [s for s in argv if s not in STAGE_SPECS]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; expected from {sorted(STAGE_SPECS)}")
    if "fit" in argv[1:]:
        raise SystemExit("`fit` takes model arguments, so it must be run on its own")
    try:
        validate_stage_order(argv, STAGE_SPECS)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for name in argv:
        _run_stage(name, [], runtime)


def cli() -> None:
    """Console entry point; the `seren-pipeline` script declared in pyproject."""
    main(sys.argv[1:])


if __name__ == "__main__":
    cli()
