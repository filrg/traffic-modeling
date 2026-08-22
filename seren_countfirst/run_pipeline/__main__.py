from __future__ import annotations

import sys

from .context import PipelineContext
from .registry import STAGE_SPECS, STAGES, plan_stages, validate_stage_order

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

Paths come from paths.py and are overridable by SEREN_TRACE_CSV,
SEREN_RESULTS and SEREN_STATE_DIR.
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


def main(argv: list[str], context: PipelineContext | None = None) -> None:
    if not argv:
        raise SystemExit(USAGE)
    if argv[0] == "--plan":
        _print_plan(argv[1:])
        return
    if argv[0] not in STAGES:
        raise SystemExit(USAGE)
    if argv[0] == "fit":
        STAGES["fit"](argv[1:], context)
        return
    unknown = [s for s in argv if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; expected from {sorted(STAGES)}")
    if "fit" in argv[1:]:
        raise SystemExit("`fit` takes model arguments, so it must be run on its own")
    try:
        validate_stage_order(argv, STAGE_SPECS)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for name in argv:
        STAGES[name]([], context)


def cli() -> None:
    """Console entry point; the `seren-pipeline` script declared in pyproject."""
    main(sys.argv[1:])


if __name__ == "__main__":
    cli()
