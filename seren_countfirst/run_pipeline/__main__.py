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
from .planning import PlannedInvocation, plan_invocations
from .preflight import capture_inputs, resume_decision
from .recipes import DEFAULT_INVOCATION_RECIPE
from .registry import STAGE_SPECS, plan_stages, validate_stage_order

USAGE = """usage:
    python -m seren_countfirst.run_pipeline <stage> [<stage> ...]
    python -m seren_countfirst.run_pipeline --resume <stage> [<stage> ...]
    python -m seren_countfirst.run_pipeline --with-deps <stage> [<stage> ...]
    python -m seren_countfirst.run_pipeline --resume --with-deps <stage> [<stage> ...]
    python -m seren_countfirst.run_pipeline --plan <target-stage> [<target-stage> ...]
    python -m seren_countfirst.run_pipeline --plan --with-deps <stage> [<stage> ...]

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
`--resume` skips only fully verified invocations; a verification miss runs them.
`--with-deps` resolves omitted dependencies through the concrete reproduction
recipe. Dependencies are resume-verified; targets run unless `--resume` is set.
Normal stage commands still use existing portable state for omitted dependencies.

Paths come from paths.py. SEREN_TRACE_CSV, SEREN_RESULTS and SEREN_STATE_DIR
relocate the default layout; SEREN_RUNS_DIR opts into RunSpec-based isolation.
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


def _print_invocation_plan(
    planned: tuple[PlannedInvocation, ...],
    *,
    resume_targets: bool,
) -> None:
    print(f"automatic dependency plan ({DEFAULT_INVOCATION_RECIPE.name}):")
    for index, item in enumerate(planned, start=1):
        role = "target" if item.requested else "dependency"
        action = "verify-or-run" if resume_targets or not item.requested else "run"
        invocation = item.invocation
        command = " ".join((invocation.stage, *invocation.arguments))
        print(f"  {index:>2}. {command:<44} [{role}; {action}]")


def _parse_execution_options(argv: list[str]) -> tuple[bool, bool, list[str]]:
    remaining = list(argv)
    resume = False
    with_dependencies = False
    while remaining and remaining[0] in {"--resume", "--with-deps"}:
        option = remaining.pop(0)
        if option == "--resume":
            if resume:
                raise SystemExit("duplicate option: --resume")
            resume = True
        else:
            if with_dependencies:
                raise SystemExit("duplicate option: --with-deps")
            with_dependencies = True
    if not remaining:
        raise SystemExit(USAGE)
    return resume, with_dependencies, remaining


def _target_invocations(argv: list[str]) -> tuple[StageInvocation, ...]:
    if not argv:
        raise SystemExit(USAGE)
    if argv[0] not in STAGE_SPECS:
        raise SystemExit(USAGE)
    if argv[0] == "fit":
        invocation = StageInvocation("fit", tuple(argv[1:]))
        try:
            STAGE_SPECS["fit"].validate_arguments(invocation.arguments)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return (invocation,)

    unknown = [stage for stage in argv if stage not in STAGE_SPECS]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; expected from {sorted(STAGE_SPECS)}")
    if "fit" in argv[1:]:
        raise SystemExit("`fit` takes model arguments, so it must be run on its own")
    try:
        ordered = validate_stage_order(argv, STAGE_SPECS)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return tuple(StageInvocation(stage) for stage in ordered)


def _concrete_plan(targets: tuple[StageInvocation, ...]) -> tuple[PlannedInvocation, ...]:
    try:
        return plan_invocations(targets, STAGE_SPECS, DEFAULT_INVOCATION_RECIPE)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _run_stage(
    name: str,
    arguments: list[str],
    context: PipelineContext,
    *,
    resume: bool = False,
) -> None:
    invocation = StageInvocation(name, tuple(arguments))
    manifest = prepare_manifest(context)
    revision, dirty = source_revision()
    environment = runtime_environment()
    command = " ".join((name, *arguments))
    if resume:
        decision = resume_decision(
            context,
            manifest,
            invocation,
            code_revision=revision,
            working_tree_dirty=dirty,
            environment=environment,
        )
        if decision.can_skip:
            print(f"resume hit: {command} ({decision.reason})")
            return
        print(f"resume miss: {command} ({decision.reason}); running")
    inputs = capture_inputs(context, manifest, STAGE_SPECS[name])
    if not inputs.complete:
        print(f"provenance warning: {command} ({'; '.join(inputs.issues)})")
    before: ArtifactSnapshot | None = None
    if manifest is not None:
        before = snapshot_artifacts(context)
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
            environment=environment,
            inputs=inputs.records,
            inputs_complete=inputs.complete,
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
            environment=environment,
            inputs=inputs.records,
            inputs_complete=inputs.complete,
            outputs=outputs,
            code_revision=revision,
            working_tree_dirty=dirty,
        ),
    )


def main(argv: list[str], context: PipelineContext | None = None) -> None:
    if not argv:
        raise SystemExit(USAGE)
    if argv[0] == "--plan":
        plan_arguments = argv[1:]
        if plan_arguments[:1] == ["--with-deps"]:
            planned = _concrete_plan(_target_invocations(plan_arguments[1:]))
            _print_invocation_plan(planned, resume_targets=False)
        else:
            _print_plan(plan_arguments)
        return
    resume, with_dependencies, argv = _parse_execution_options(argv)
    targets = _target_invocations(argv)
    if with_dependencies:
        planned = _concrete_plan(targets)
        _print_invocation_plan(planned, resume_targets=resume)
    else:
        planned = tuple(PlannedInvocation(invocation, requested=True) for invocation in targets)

    runtime = resolve_context(context)
    for item in planned:
        invocation = item.invocation
        _run_stage(
            invocation.stage,
            list(invocation.arguments),
            runtime,
            resume=resume or not item.requested,
        )


def cli() -> None:
    """Console entry point; the `seren-pipeline` script declared in pyproject."""
    main(sys.argv[1:])


if __name__ == "__main__":
    cli()
