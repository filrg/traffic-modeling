from __future__ import annotations

import argparse
import filecmp
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from seren_countfirst.provenance import environment  # noqa: E402
from seren_countfirst.artifacts import write_json  # noqa: E402
from seren_countfirst.run_pipeline import REPRODUCTION_STEPS  # noqa: E402

# (argv, approximate minutes, whether --quick skips it). `fit` consumes the rest
# of its line as rung names, so it cannot share an invocation with another stage.
STAGES: list[tuple[list[str], float, bool]] = [
    (list(step.argv), step.approximate_minutes, step.skip_in_quick_run)
    for step in REPRODUCTION_STEPS
]


def _stage_environment(target: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("SEREN_RUNS_DIR", None)
    env.pop("SEREN_STATE_DIR", None)
    env["SEREN_RESULTS"] = str(target)
    return env


def run(target: Path, quick: bool) -> list[dict]:
    env = _stage_environment(target)
    target.mkdir(parents=True, exist_ok=True)
    log = []
    for argv, minutes, skippable in STAGES:
        if quick and skippable:
            print(f"--- skip {' '.join(argv)} (~{minutes:.0f} min)")
            continue
        print(f"--- {' '.join(argv)} (~{minutes:.0f} min)", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "seren_countfirst.run_pipeline", *argv], cwd=REPO, env=env
        )
        dt = time.time() - t0
        log.append(
            {"stage": " ".join(argv), "seconds": round(dt, 1), "returncode": proc.returncode}
        )
        print(f"    {dt:.0f}s, exit {proc.returncode}", flush=True)
        if proc.returncode:
            raise SystemExit(f"stage `{' '.join(argv)}` failed")
    return log


def compare(target: Path, published: Path) -> int:
    """Diff a rebuild against the published tree, file by file."""
    same = differ = missing = extra = 0
    for p in sorted(published.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(published)
        q = target / rel
        if not q.exists():
            print(f"  MISSING  {rel}")
            missing += 1
        elif filecmp.cmp(p, q, shallow=False):
            same += 1
        else:
            print(f"  DIFFERS  {rel}")
            differ += 1
    for p in sorted(target.rglob("*")):
        if p.is_file() and not (published / p.relative_to(target)).exists():
            print(f"  EXTRA    {p.relative_to(target)}")
            extra += 1
    print(f"\n{same} identical, {differ} differing, {missing} missing, {extra} extra")
    return differ + missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rebuild every published artifact from the trace, and optionally "
        "diff the rebuild against the published one."
    )
    ap.add_argument(
        "--into",
        type=Path,
        default=REPO / "results_repro",
        help="where to write the rebuild (default: results_repro/)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="skip jobsize, jobsize_folds, dependent workload, and certify "
        "(~60 of the ~80 minutes)",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="after the run, diff it against results/ and exit non-zero on any difference",
    )
    args = ap.parse_args()

    t0 = time.time()
    log = run(args.into, args.quick)
    report = {
        "environment": environment(),
        "quick": args.quick,
        "total_seconds": round(time.time() - t0, 1),
        "stages": log,
        "run_manifest": "run_manifest.json",
    }
    report_path = args.into / "reproduction_report.json"
    write_json(report, report_path)
    print(f"\nwrote {report_path}; {report['total_seconds'] / 60:.0f} min total")

    if args.verify:
        print(f"\ndiffing {args.into} against {REPO / 'results'}:")
        raise SystemExit(1 if compare(args.into, REPO / "results") else 0)


if __name__ == "__main__":
    main()
