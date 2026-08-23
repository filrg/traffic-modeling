from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_TARGETS = (
    "traffic_modeling",
    "seren_countfirst",
    "figures/code",
    "reproduce.py",
    "tools",
)


def _run(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    rendered = subprocess.list2cmdline(list(command))
    print(f"\n> {rendered}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _timed(label: str, action: Callable[[], None]) -> None:
    started = time.perf_counter()
    print(f"\n== {label} ==", flush=True)
    action()
    print(f"== {label}: {time.perf_counter() - started:.2f}s ==", flush=True)


def _source_checks() -> None:
    _run((sys.executable, "-m", "ruff", "check", *SOURCE_TARGETS))
    _run((sys.executable, "-m", "mypy"))
    _run((sys.executable, "-m", "compileall", "-q", *SOURCE_TARGETS))


def _copy_build_tree(destination: Path) -> None:
    destination.mkdir()
    for filename in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / filename, destination / filename)
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc")
    for package in ("seren_countfirst", "traffic_modeling"):
        shutil.copytree(ROOT / package, destination / package, ignore=ignored)


def _assert_wheel_contract(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
    required = ("seren_countfirst/__init__.py", "traffic_modeling/__init__.py")
    missing = [path for path in required if path not in members]
    if missing:
        raise RuntimeError(f"wheel is missing packages: {missing}")
    private = [path for path in members if path.startswith("seren_countfirst/tests/")]
    if private:
        raise RuntimeError("wheel contains internal seren_countfirst/tests")


def _package_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="seren-quality-") as temporary:
        work = Path(temporary)
        source = work / "source"
        wheels = work / "wheels"
        site = work / "site"
        outside = work / "outside"
        _copy_build_tree(source)
        wheels.mkdir()
        outside.mkdir()
        _run(
            (
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheels),
                str(source),
            )
        )
        built = tuple(wheels.glob("*.whl"))
        if len(built) != 1:
            raise RuntimeError(f"expected one wheel, found {len(built)}")
        _assert_wheel_contract(built[0])
        _run(
            (
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-index",
                "--no-compile",
                "--target",
                str(site),
                str(built[0]),
            )
        )
        smoke = f"""
import sys
from importlib.metadata import distribution
from pathlib import Path

site = Path({str(site)!r}).resolve()
sys.path.insert(0, str(site))
import seren_countfirst
import traffic_modeling
from seren_countfirst.run_pipeline.__main__ import main

for package in (seren_countfirst, traffic_modeling):
    location = Path(package.__file__).resolve()
    assert location.is_relative_to(site), location
entry_points = distribution("seren-countfirst").entry_points
assert any(
    item.name == "seren-pipeline"
    and item.value == "seren_countfirst.run_pipeline.__main__:cli"
    for item in entry_points
)
main(["--plan", "workload"])
"""
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        _run((sys.executable, "-I", "-c", smoke), cwd=outside, env=environment)


def _tests(*, include_slow: bool) -> None:
    tests = ROOT / "seren_countfirst" / "tests"
    if not tests.is_dir():
        raise RuntimeError("internal tests are unavailable; run `quality.py source` instead")
    with tempfile.TemporaryDirectory(prefix="seren-pytest-") as temporary:
        command = [
            sys.executable,
            "-m",
            "pytest",
            str(tests),
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(Path(temporary) / "work"),
        ]
        if not include_slow:
            command.extend(("-m", "not slow"))
        _run(command)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the repository quality ratchet.")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("source", "fast", "full", "package"),
        default="fast",
        help="source checks tracked code; fast/full add internal tests; package runs wheel smoke",
    )
    mode = parser.parse_args(argv).mode
    started = time.perf_counter()
    try:
        if mode == "package":
            _timed("wheel build/install smoke", _package_smoke)
        else:
            _timed("lint, type ratchet and compile", _source_checks)
            if mode in {"fast", "full"}:
                _timed(
                    "full tests" if mode == "full" else "fast tests",
                    lambda: _tests(include_slow=mode == "full"),
                )
            _timed("wheel build/install smoke", _package_smoke)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nQUALITY FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"\nQUALITY PASSED in {time.perf_counter() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
