from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import STATE_DIR

OUT = STATE_DIR

# sections whose index carries meaning and must survive the round trip
_INDEXED_TABLES = ("envelope", "idc_envelope")
# sections with a throwaway RangeIndex. A section absent from both lists is
# dropped on save, so anything a stage puts in `state` must be listed here.
_FLAT_TABLES = ("scaling", "horizon", "horizon_scale_support", "bootstrap_baselines",
                "microstructure", "reconstruction_by_alpha")
_META_SCALARS = ("trace_sha256", "n_bins", "busy_threshold", "winner",
                 "trace_marginals", "trace_idc")


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def load_portable_state(out: Path = OUT) -> dict:
    """Read the portable state back on any pandas version.

    Returns an empty dict when no state has been written, so the first pipeline
    stage can start from nothing.
    """
    from .data import ActivityScheme, CountSeries

    meta_path = out / "meta.json"
    if not meta_path.exists():
        return {}

    meta = json.loads(meta_path.read_text())
    state: dict = {k: meta[k] for k in _META_SCALARS if k in meta}

    if (out / "counts.npy").exists():
        idx = pd.DatetimeIndex(pd.read_csv(out / "counts_index.csv")["timestamp"])
        state["counts"] = CountSeries(np.load(out / "counts.npy"), idx)
    if "activity_scheme" in meta:
        a = meta["activity_scheme"]
        state["scheme"] = ActivityScheme(
            edges=tuple(a["edges"]), labels=tuple(a["labels"]),
            busy_threshold=a["busy_threshold"], quiet_threshold=a["quiet_threshold"], source=a["source"],
        )
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
            state[key] = df.set_index(df.columns[0])
        else:
            state[key] = df
    return state


def _model_entry(model, previous: dict | None = None) -> dict:
    """Serialise one model, keeping the recorded fit when the object was rebuilt."""
    report = getattr(model, "report", None)
    fit = report.to_dict() if report is not None else None
    if fit is None and previous is not None:
        fit = previous.get("fit")
    return {
        "class": type(model).__name__,
        "name": model.name,
        "rung": _jsonable(getattr(model, "rung", None)),
        # role must round-trip: an ablation rebuilt as a candidate could win selection
        "role": getattr(model, "role", "candidate"),
        "params": {k: _jsonable(v) for k, v in model.params_.items()},
        "fit": fit,
    }


def save_portable_state(state: dict, out: Path = OUT) -> None:
    """Write `state` back to the portable layout, updating only the keys present.

    Saving is incremental because each stage mutates one section; sections
    absent from `state` keep whatever is on disk.
    """
    out.mkdir(parents=True, exist_ok=True)

    counts = state.get("counts")
    if counts is not None:
        np.save(out / "counts.npy", np.asarray(counts.values, dtype=float))
        pd.Series(counts.index.astype(str),
                  name="timestamp").to_csv(out / "counts_index.csv", index=False)
    if "batches_sizes" in state:
        np.save(out / "batch_sizes.npy", np.asarray(state["batches_sizes"], dtype=int))
    if "batch_times" in state:
        pd.Series(pd.DatetimeIndex(state["batch_times"]).astype(str), name="batch_time").to_csv(
            out / "batch_times.csv", index=False
        )

    for key in _INDEXED_TABLES + _FLAT_TABLES:
        df = state.get(key)
        if not isinstance(df, pd.DataFrame):
            continue
        frame = df.reset_index() if key in _INDEXED_TABLES else df
        frame.columns = [str(c) for c in frame.columns]
        frame.to_csv(out / f"{key}.csv", index=False)

    models_path = out / "models.json"
    previous = json.loads(models_path.read_text()) if models_path.exists() else {}
    entries = dict(state.get("models_json", previous))
    for name, model in state.get("models", {}).items():
        entries[name] = _model_entry(model, previous.get(name))
    models_path.write_text(json.dumps(entries, indent=2, default=_jsonable))

    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for key in _META_SCALARS:
        if key in state:
            meta[key] = _jsonable(state[key])
    if "n_bins" in meta and meta["n_bins"] is not None:
        meta["n_bins"] = int(meta["n_bins"])
    if "busy_threshold" in meta and meta["busy_threshold"] is not None:
        meta["busy_threshold"] = float(meta["busy_threshold"])
    if "split_bounds" in state:
        meta["split_bounds"] = [int(x) for x in state["split_bounds"]]
    if "scheme" in state:
        meta["activity_scheme"] = state["scheme"].to_dict()
    meta_path.write_text(json.dumps(meta, indent=2, default=_jsonable))


def _rebuild_report(fit: dict | None):
    """Recreate a FitReport from its serialised form; None if the model was never fitted."""
    from .models import FitReport

    if not fit:
        return None
    n_obs = fit.get("n_obs")
    if n_obs is None:
        # legacy entries stored only the derived criteria: invert BIC for n_obs
        bic, ll, k = fit.get("BIC"), fit.get("loglik"), fit.get("n_params")
        n_obs = int(round(float(np.exp((bic + 2.0 * ll) / k)))) if None not in (bic, ll,
                    k) and k else 0
    return FitReport(
        converged=bool(fit.get("converged", False)),
        n_iter=int(fit.get("n_iter", 0)),
        grad_norm=float(fit.get("grad_norm", np.nan)),
        loglik=float(fit.get("loglik", np.nan)),
        n_params=int(fit.get("n_params", 0)),
        n_obs=int(n_obs),
        boundary_warnings=list(fit.get("boundary_warnings", [])),
        restarts=int(fit.get("restarts", 1)),
    )


def rebuild_model(entry: dict):
    """Recreate a model object from its params json; no pickle is involved."""
    from .models import IIDNegBin, NBINGARCH, NBLatentAR, NBRegimeSwitch, ONOFFCounts
    from .models import NBGatedLatent

    cls = {"IIDNegBin": IIDNegBin, "NBLatentAR": NBLatentAR,
           "NBRegimeSwitch": NBRegimeSwitch, "NBGatedLatent": NBGatedLatent,
           "ONOFFCounts": ONOFFCounts, "NBINGARCH": NBINGARCH}[entry["class"]]
    if cls is NBLatentAR:
        model = cls(n_components=1 if entry["name"].startswith("nb_ar1") else 2)
    elif cls is NBGatedLatent:
        model = cls(sharpness=float(entry["params"].get("a", 4.0)))
    elif cls is ONOFFCounts:
        model = cls(emission="nb" if np.isfinite(entry["params"].get("kappa", np.inf)) else "poisson")
    else:
        model = cls()
    model.params_ = {
        k: (np.asarray(v) if isinstance(v, list) else v) for k, v in entry["params"].items()
    }
    model.name = entry["name"]
    model.role = entry.get("role", type(model).role)
    model.report = _rebuild_report(entry.get("fit"))
    return model
