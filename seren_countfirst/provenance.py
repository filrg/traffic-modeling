from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def environment() -> dict[str, str]:
    """Return the interpreter and library versions a result was produced under."""
    import numpy
    import pandas
    import scipy

    from . import __version__

    return {
        "package": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "platform": f"{platform.system()} {platform.release()}",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": " ".join(sys.argv[1:]),
    }


def runtime_environment() -> dict[str, str]:
    """Return stable runtime facts suitable for an individual execution record."""
    facts = environment()
    facts.pop("argv")
    facts.pop("written_utc")
    return facts


def source_revision(root: Path | None = None) -> tuple[str | None, bool | None]:
    """Return the Git revision and tracked-worktree status when Git metadata is available."""
    repository = (root or Path(__file__).resolve().parent.parent).resolve()
    prefix = ["git", "-c", f"safe.directory={repository.as_posix()}"]

    def run(*arguments: str) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                [*prefix, *arguments],
                cwd=repository,
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError:
            return None

    revision = run("rev-parse", "HEAD")
    if revision is None or revision.returncode:
        return None, None
    status = run("status", "--porcelain", "--untracked-files=no")
    dirty = None if status is None or status.returncode else bool(status.stdout.strip())
    return revision.stdout.strip(), dirty
