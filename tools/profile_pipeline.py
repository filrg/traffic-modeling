from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PUBLISHED_RESULTS = (ROOT / "results").resolve()


def _resolve_targets(runs_dir: Path, output: Path) -> tuple[Path, Path]:
    runs = runs_dir.resolve()
    profile = output.resolve()
    if runs == ROOT or runs == PUBLISHED_RESULTS or PUBLISHED_RESULTS in runs.parents:
        raise ValueError("profile runs directory must not be the repository or published results")
    if profile == runs or runs not in profile.parents:
        raise ValueError("profile output must be a file below the profile runs directory")
    return runs, profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile a pipeline command in an isolated SEREN_RUNS_DIR."
    )
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace", type=Path, help="optional raw trace override")
    parser.add_argument("--limit", type=int, default=40, help="number of cumulative rows to print")
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    pipeline_args = arguments.pipeline_args
    if pipeline_args[:1] == ["--"]:
        pipeline_args = pipeline_args[1:]
    if not pipeline_args:
        parser.error("a pipeline command is required after the profiler options")
    if arguments.limit < 1:
        parser.error("--limit must be positive")
    try:
        runs, output = _resolve_targets(arguments.runs_dir, arguments.output)
    except ValueError as exc:
        parser.error(str(exc))

    runs.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["SEREN_RUNS_DIR"] = str(runs)
    os.environ.pop("SEREN_RESULTS", None)
    os.environ.pop("SEREN_STATE_DIR", None)
    if arguments.trace is not None:
        os.environ["SEREN_TRACE_CSV"] = str(arguments.trace.resolve())
    sys.path.insert(0, str(ROOT))

    from seren_countfirst.run_pipeline.__main__ import main as pipeline_main

    profiler = cProfile.Profile()
    try:
        profiler.runcall(pipeline_main, pipeline_args)
    finally:
        profiler.dump_stats(output)
        print(f"\nprofile written to {output}")
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(arguments.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
