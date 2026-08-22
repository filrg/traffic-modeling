from __future__ import annotations

import sys

from . import STAGES
from .stages.fit import stage_fit

USAGE = """usage: python -m seren_countfirst.run_pipeline <stage> [<stage> ...]

    data                      ingest the trace, verify the audit facts
    fit <rung> [<rung> ...]   fit or calibrate ladder rungs
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


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in STAGES:
        raise SystemExit(USAGE)
    if argv[0] == "fit":
        stage_fit(argv[1:])
        return
    unknown = [s for s in argv if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; expected from {sorted(STAGES)}")
    if "fit" in argv[1:]:
        raise SystemExit("`fit` takes rung arguments, so it must be run on its own")
    for name in argv:
        STAGES[name]([])


def cli() -> None:
    """Console entry point; the `seren-pipeline` script declared in pyproject."""
    main(sys.argv[1:])


if __name__ == "__main__":
    cli()
