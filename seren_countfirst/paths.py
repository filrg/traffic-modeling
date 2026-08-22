from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _from_env(var: str, default: Path) -> Path:
    return Path(os.environ[var]) if os.environ.get(var) else default


# raw trace; not tracked in the repo, verified by sha256 on load
TRACE_CSV = _from_env("SEREN_TRACE_CSV", REPO_ROOT / "trace_seren.csv")

# exported tables, gate reports and the frozen model card
RESULTS = _from_env("SEREN_RESULTS", REPO_ROOT / "results")

# resumable pipeline state as npy + csv + json; never pickle
STATE_DIR = _from_env("SEREN_STATE_DIR", RESULTS / "state_portable_data" / "state_portable")
