from __future__ import annotations

import numpy as np
import pandas as pd

from ...data import expanding_window_folds
from ...scaling import scaling_report
from ...evaluation import GateEvidence, evaluate_gates, evaluate_on_folds
from ...paths import RESULTS
from ..config import (
    CERTIFY_SEEDS,
    FOLD_KWARGS,
    N_PATHS_EVAL,
    N_PATHS_FOLD,
    N_PATHS_SCALING,
    SEED,
)
from ..state import fitted_models, load_state

# Gate 3 scores the fraction of IDC scales inside the envelope, so it passes
# upward; every other gate passes downward against its tolerance.
HIGHER_IS_BETTER = {"3_idc_envelope"}


def _margin(name: str, value: float, threshold: float) -> float:
    """Return the gate value as a ratio oriented so that > 1 always means failure."""
    if not np.isfinite(value) or not np.isfinite(threshold):
        return np.inf
    if name in HIGHER_IS_BETTER:
        return np.inf if value <= 0 else threshold / value
    return np.inf if threshold <= 0 else value / threshold


def _score_seed(state: dict, model, folds, thr: float, env, cal: float, seed: int) -> pd.DataFrame:
    """Score all eight gates at one seed, reseeding every stochastic input.

    Mirrors `scaling._full_gate_diagnostic` exactly, including its seed offsets,
    so that seed `SEED` reproduces `gates_full_<winner>.csv` row for row.
    """
    scaling = scaling_report(model, state["n_bins"], n_paths=N_PATHS_SCALING,
                             seed=seed)
    fold_ev = evaluate_on_folds(model, folds, thr, n_paths=N_PATHS_FOLD,
                                seed=seed + 1, alpha=cal)
    rng = np.random.default_rng(seed + 1)
    paths = [model.simulate(state["n_bins"], rng, alpha=cal) for _ in range(N_PATHS_EVAL)]
    report = evaluate_gates(
        state["counts"].values, paths, thr, env, stop_at_first_failure=False,
        evidence=GateEvidence(fold_evidence=fold_ev, scaling=scaling,
                              idc_envelope=state.get("idc_envelope")),
    )
    frame = report.to_frame()
    frame.insert(0, "seed", seed)
    frame["margin"] = [_margin(n, v, t) for n, v, t
                       in zip(frame["name"], frame["value"], frame["threshold"])]
    return frame


def _summarise(rows: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """Reduce the per-seed table to one worst-case row per gate."""
    out = []
    for name, grp in rows.groupby("name", sort=True):
        worst = grp.loc[grp["margin"].idxmax()]
        out.append({
            "name": name,
            "n_pass": int(grp["passed"].sum()),
            "n_seeds": n_seeds,
            "worst_value": float(worst["value"]),
            "threshold": float(worst["threshold"]),
            "worst_margin": float(worst["margin"]),
            "worst_seed": int(worst["seed"]),
            "median_value": float(grp["value"].median()),
            "worst_detail": str(worst["detail"]),
        })
    return pd.DataFrame(out)


def stage_certify() -> None:
    state = load_state()
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = state.get("winner", "gated_mm")
    model = fitted_models(state)[name]
    thr, env = state["busy_threshold"], state["envelope"]
    cal = state["trace_marginals"]["mean"] / float(model.params_["mean"])
    folds = expanding_window_folds(state["counts"], **FOLD_KWARGS)
    seeds = [SEED + 1000 * i for i in range(CERTIFY_SEEDS)]

    frames: list[pd.DataFrame] = []
    for i, seed in enumerate(seeds, start=1):
        frame = _score_seed(state, model, folds, thr, env, cal, seed)
        frames.append(frame)
        # written every seed so a long run stays inspectable and interruptible
        pd.concat(frames, ignore_index=True).to_csv(
            RESULTS / f"certify_seeds_{name}.csv", index=False)
        failed = frame.loc[~frame["passed"], "name"].tolist()
        print(f"[{i}/{len(seeds)}] seed {seed}: "
              f"{int(frame['passed'].sum())}/8 gates" + (f", failed {failed}" if failed else ""))

    rows = pd.concat(frames, ignore_index=True)
    summary = _summarise(rows, len(seeds))
    summary.to_csv(RESULTS / f"certify_summary_{name}.csv", index=False)

    # `passed` from the gate machinery is authoritative; `margin` only ranks.
    disagree = rows.loc[rows["passed"] != (rows["margin"] <= 1.0 + 1e-9)]
    if len(disagree):
        print("\nWARNING: passed flag disagrees with value/threshold on "
              f"{len(disagree)} row(s); the margin column is misoriented for "
              f"{sorted(set(disagree['name']))}")

    n_full = int(sum(grp["passed"].all() for _, grp in rows.groupby("seed")))
    print(f"\ncertification of {name} over {len(seeds)} seeds "
          f"at ({N_PATHS_EVAL}, {N_PATHS_FOLD}, {N_PATHS_SCALING}):")
    print(summary.drop(columns=["worst_detail"]).round(4).to_string(index=False))
    binding = summary.loc[summary["worst_margin"].idxmax()]
    print(f"\nall eight gates pass on {n_full} of {len(seeds)} seeds; "
          f"binding gate {binding['name']} at {binding['worst_margin']:.3f} of tolerance "
          f"(seed {int(binding['worst_seed'])})")
    print(f"certify done -> results/certify_summary_{name}.csv")
