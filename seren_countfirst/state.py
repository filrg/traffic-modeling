from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import pandas as pd

from .artifacts import write_csv, write_json, write_npy
from .paths import STATE_DIR

if TYPE_CHECKING:
    from .data import ActivityScheme, CountSeries
    from .models import CountModel

OUT = STATE_DIR


ModelEntry = TypedDict(
    "ModelEntry",
    {
        "class": str,
        "name": str,
        "rung": object,
        "role": str,
        "params": dict[str, Any],
        "fit": dict[str, Any] | None,
    },
    total=False,
)


class PipelineState(TypedDict, total=False):
    trace_sha256: str
    n_bins: int
    busy_threshold: float
    winner: str
    trace_marginals: dict[str, float]
    trace_idc: dict[str, float]
    counts: CountSeries
    scheme: ActivityScheme
    split_bounds: tuple[int, ...]
    batches_sizes: np.ndarray
    batch_times: pd.DatetimeIndex
    models: dict[str, CountModel]
    models_json: dict[str, ModelEntry]
    envelope: pd.DataFrame
    idc_envelope: pd.DataFrame
    scaling: pd.DataFrame
    horizon: pd.DataFrame
    horizon_scale_support: pd.DataFrame
    bootstrap_baselines: pd.DataFrame
    microstructure: pd.DataFrame
    reconstruction_by_alpha: pd.DataFrame


# sections whose index carries meaning and must survive the round trip
_INDEXED_TABLES = ("envelope", "idc_envelope")
# sections with a throwaway RangeIndex. A section absent from both lists is
# dropped on save, so anything a stage puts in `state` must be listed here.
_FLAT_TABLES = (
    "scaling",
    "horizon",
    "horizon_scale_support",
    "bootstrap_baselines",
    "microstructure",
    "reconstruction_by_alpha",
)
_META_SCALARS = (
    "trace_sha256",
    "n_bins",
    "busy_threshold",
    "winner",
    "trace_marginals",
    "trace_idc",
)


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def load_portable_state(out: Path = OUT) -> PipelineState:
    """Read the portable state back on any pandas version.

    Returns an empty dict when no state has been written, so the first pipeline
    stage can start from nothing.
    """
    from .data import ActivityScheme, CountSeries

    meta_path = out / "meta.json"
    if not meta_path.exists():
        return {}

    meta = json.loads(meta_path.read_text())
    state = cast(PipelineState, {k: meta[k] for k in _META_SCALARS if k in meta})
    mutable_state = cast(dict[str, Any], state)

    if (out / "counts.npy").exists():
        idx = pd.DatetimeIndex(pd.read_csv(out / "counts_index.csv")["timestamp"])
        state["counts"] = CountSeries(np.load(out / "counts.npy"), idx)
    if "activity_scheme" in meta:
        state["scheme"] = ActivityScheme.from_dict(meta["activity_scheme"])
    if "split_bounds" in meta:
        state["split_bounds"] = tuple(meta["split_bounds"])
    if (out / "batch_sizes.npy").exists():
        state["batches_sizes"] = np.load(out / "batch_sizes.npy")
    if (out / "batch_times.csv").exists():
        state["batch_times"] = pd.DatetimeIndex(pd.read_csv(out / "batch_times.csv")["batch_time"])

    models_path = out / "models.json"
    state["models_json"] = json.loads(models_path.read_text()) if models_path.exists() else {}

    for key in _INDEXED_TABLES + _FLAT_TABLES:
        path = out / f"{key}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if key in _INDEXED_TABLES:
            mutable_state[key] = df.set_index(df.columns[0])
        else:
            mutable_state[key] = df
    return state


def _model_entry(model: CountModel, previous: ModelEntry | None = None) -> ModelEntry:
    """Serialise one model, keeping the recorded fit when the object was rebuilt."""
    report = getattr(model, "report", None)
    fit = report.to_dict() if report is not None else None
    if fit is None and previous is not None:
        fit = previous.get("fit")
    return {
        "class": type(model).__name__,
        "name": model.name,
        "rung": _jsonable(getattr(model, "rung", None)),
        # Preserve the published table schema; selection now comes from the registry.
        "role": getattr(model, "role", "candidate"),
        "params": {k: _jsonable(v) for k, v in model.params_.items()},
        "fit": fit,
    }


def save_portable_state(state: PipelineState, out: Path = OUT) -> None:
    """Write `state` back to the portable layout, updating only the keys present.

    Saving is incremental because each stage mutates one section; sections
    absent from `state` keep whatever is on disk.
    """
    out.mkdir(parents=True, exist_ok=True)
    raw_state = cast(dict[str, Any], state)

    counts = state.get("counts")
    if counts is not None:
        write_npy(np.asarray(counts.values, dtype=float), out / "counts.npy")
        write_csv(
            pd.Series(counts.index.astype(str), name="timestamp"),
            out / "counts_index.csv",
            index=False,
        )
    if "batches_sizes" in state:
        write_npy(np.asarray(state["batches_sizes"], dtype=int), out / "batch_sizes.npy")
    if "batch_times" in state:
        write_csv(
            pd.Series(pd.DatetimeIndex(state["batch_times"]).astype(str), name="batch_time"),
            out / "batch_times.csv",
            index=False,
        )

    for key in _INDEXED_TABLES + _FLAT_TABLES:
        df = raw_state.get(key)
        if not isinstance(df, pd.DataFrame):
            continue
        frame = df.reset_index() if key in _INDEXED_TABLES else df
        frame.columns = [str(c) for c in frame.columns]
        write_csv(frame, out / f"{key}.csv", index=False)

    models_path = out / "models.json"
    previous = json.loads(models_path.read_text()) if models_path.exists() else {}
    entries = dict(state.get("models_json", previous))
    for name, model in state.get("models", {}).items():
        entries[name] = _model_entry(model, previous.get(name))
    write_json(entries, models_path, default=_jsonable)

    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for key in _META_SCALARS:
        if key in raw_state:
            meta[key] = _jsonable(raw_state[key])
    if "n_bins" in meta and meta["n_bins"] is not None:
        meta["n_bins"] = int(meta["n_bins"])
    if "busy_threshold" in meta and meta["busy_threshold"] is not None:
        meta["busy_threshold"] = float(meta["busy_threshold"])
    if "split_bounds" in state:
        meta["split_bounds"] = [int(x) for x in state["split_bounds"]]
    if "scheme" in state:
        meta["activity_scheme"] = state["scheme"].to_dict()
    write_json(meta, meta_path, default=_jsonable)


def _rebuild_report(fit: dict[str, Any] | None):
    """Recreate a FitReport from its serialised form; None if the model was never fitted."""
    from .models import FitReport

    if not fit:
        return None
    return FitReport.from_dict(fit)


def rebuild_model(entry: ModelEntry) -> CountModel:
    """Recreate a model object from its params json; no pickle is involved."""
    from .models import IIDNegBin, NBINGARCH, NBLatentAR, NBRegimeSwitch, ONOFFCounts
    from .models import NBGatedLatent

    cls = {
        "IIDNegBin": IIDNegBin,
        "NBLatentAR": NBLatentAR,
        "NBRegimeSwitch": NBRegimeSwitch,
        "NBGatedLatent": NBGatedLatent,
        "ONOFFCounts": ONOFFCounts,
        "NBINGARCH": NBINGARCH,
    }[entry["class"]]
    if cls is NBLatentAR:
        model = cls(n_components=1 if entry["name"].startswith("nb_ar1") else 2)
    elif cls is NBGatedLatent:
        model = cls(sharpness=float(entry["params"].get("a", 4.0)))
    elif cls is ONOFFCounts:
        model = cls(
            emission="nb" if np.isfinite(entry["params"].get("kappa", np.inf)) else "poisson"
        )
    else:
        model = cls()
    model.params_ = {
        k: (np.asarray(v) if isinstance(v, list) else v) for k, v in entry["params"].items()
    }
    model.name = entry["name"]
    model.role = entry.get("role", type(model).role)
    model.report = _rebuild_report(entry.get("fit"))
    return model
