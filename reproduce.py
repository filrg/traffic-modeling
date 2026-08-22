from __future__ import annotations

import argparse
import filecmp
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from seren_countfirst.provenance import environment  # noqa: E402

# (argv, approximate minutes, whether --quick skips it). `fit` consumes the rest
# of its line as rung names, so it cannot share an invocation with another stage.
STAGES: list[tuple[list[str], float, bool]] = [
    (["data"], 0.6, False),
    (["fit", "r2", "r3", "r4"], 2.0, False),
    (["fit", "gated"], 4.0, False),
    (["fit", "mm", "mm2", "mm3", "mm4", "mm5"], 0.5, False),
    (["fit", "onoff", "ingarch", "nogate"], 0.4, False),
    (["evaluate"], 2.0, False),
    (["scaling"], 3.0, False),
    (["horizon"], 2.0, False),
    (["reconstruct"], 2.0, False),
    (["baselines"], 1.0, False),
    (["jobsize"], 34.0, True),
    (["jobsize_folds"], 11.0, True),
    (["export"], 0.2, False),
    (["workload"], 0.7, False),
    (["certify"], 15.0, True),
]


def run(target: Path, quick: bool) -> list[dict]:
    env = dict(os.environ, SEREN_RESULTS=str(target))
    target.mkdir(parents=True, exist_ok=True)
    log = []
    for argv, minutes, skippable in STAGES:
        if quick and skippable:
            print(f"--- skip {' '.join(argv)} (~{minutes:.0f} min)")
            continue
        print(f"--- {' '.join(argv)} (~{minutes:.0f} min)", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "seren_countfirst.run_pipeline", *argv],
            cwd=REPO, env=env)
        dt = time.time() - t0
        log.append({"stage": " ".join(argv), "seconds": round(dt, 1),
                    "returncode": proc.returncode})
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
                    "diff the rebuild against the published one.")
    ap.add_argument("--into", type=Path, default=REPO / "results_repro",
                    help="where to write the rebuild (default: results_repro/)")
    ap.add_argument("--quick", action="store_true",
                    help="skip jobsize, jobsize_folds and certify (~60 of the ~80 minutes)")
    ap.add_argument("--verify", action="store_true",
                    help="after the run, diff it against results/ and exit non-zero on any difference")
    args = ap.parse_args()

    t0 = time.time()
    log = run(args.into, args.quick)
    manifest = {"environment": environment(), "quick": args.quick,
                "total_seconds": round(time.time() - t0, 1), "stages": log}
    (args.into / "run_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                 encoding="utf-8")
    print(f"\nwrote {args.into / 'run_manifest.json'}; {manifest['total_seconds'] / 60:.0f} min total")

    if args.verify:
        print(f"\ndiffing {args.into} against {REPO / 'results'}:")
        raise SystemExit(1 if compare(args.into, REPO / "results") else 0)


if __name__ == "__main__":
    main()
