from __future__ import annotations

import sys

from . import STAGES
from .context import PipelineContext

USAGE = """usage: python -m seren_countfirst.run_pipeline <stage> [<stage> ...]

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

Paths come from paths.py and are overridable by SEREN_TRACE_CSV,
SEREN_RESULTS and SEREN_STATE_DIR.
"""


def main(argv: list[str], context: PipelineContext | None = None) -> None:
    if not argv or argv[0] not in STAGES:
        raise SystemExit(USAGE)
    if argv[0] == "fit":
        STAGES["fit"](argv[1:], context)
        return
    unknown = [s for s in argv if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; expected from {sorted(STAGES)}")
    if "fit" in argv[1:]:
        raise SystemExit("`fit` takes model arguments, so it must be run on its own")
    for name in argv:
        STAGES[name]([], context)


def cli() -> None:
    """Console entry point; the `seren-pipeline` script declared in pyproject."""
    main(sys.argv[1:])


if __name__ == "__main__":
    cli()
