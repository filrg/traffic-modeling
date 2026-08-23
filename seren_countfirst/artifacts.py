from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _atomic_write(path: str | Path, writer: Callable[[Path], None]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(frame: pd.DataFrame | pd.Series, path: str | Path, *, index: bool) -> None:
    """Write a CSV atomically, preserving pandas' default representation."""
    _atomic_write(path, lambda temporary: frame.to_csv(temporary, index=index))


def write_json(
    payload: Any,
    path: str | Path,
    *,
    default: Callable[[Any], Any] | None = None,
) -> None:
    """Write indented UTF-8 JSON atomically."""
    text = json.dumps(payload, indent=2, default=default)

    def save(temporary: Path) -> None:
        temporary.write_text(text, encoding="utf-8")

    _atomic_write(path, save)


def write_npy(array: np.ndarray, path: str | Path) -> None:
    """Write a NumPy array atomically without changing its binary encoding."""

    def save(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.save(handle, array)

    _atomic_write(path, save)
