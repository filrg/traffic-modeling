from __future__ import annotations

import numpy as np
import pandas as pd

from ...artifacts import write_csv
from ...scaling import scaling_report
from ...paths import RESULTS
from ..config import (
    CERTIFY_SEEDS,
    N_PATHS_EVAL,
    N_PATHS_FOLD,
    N_PATHS_SCALING,
    SEED,
)
from ..gate_diagnostics import full_gate_frame
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


def _score_seed(state: dict, model, seed: int) -> pd.DataFrame:
    """Score all eight gates at one seed."""
    scaling = scaling_report(model, state["n_bins"], n_paths=N_PATHS_SCALING, seed=seed)
    frame = full_gate_frame(state, model, scaling, seed)
    frame.insert(0, "seed", seed)
    frame["margin"] = [
        _margin(n, v, t) for n, v, t in zip(frame["name"], frame["value"], frame["threshold"])
    ]
    return frame


def _summarise(rows: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """Reduce the per-seed table to one worst-case row per gate."""
    out = []
    for name, grp in rows.groupby("name", sort=True):
        worst = grp.loc[grp["margin"].idxmax()]
        out.append(
            {
                "name": name,
                "n_pass": int(grp["passed"].sum()),
                "n_seeds": n_seeds,
                "worst_value": float(worst["value"]),
                "threshold": float(worst["threshold"]),
                "worst_margin": float(worst["margin"]),
                "worst_seed": int(worst["seed"]),
                "median_value": float(grp["value"].median()),
                "worst_detail": str(worst["detail"]),
            }
        )
    return pd.DataFrame(out)


def stage_certify() -> None:
    state = load_state()
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = state.get("winner", "gated_mm")
    model = fitted_models(state)[name]
    seeds = [SEED + 1000 * i for i in range(CERTIFY_SEEDS)]

    frames: list[pd.DataFrame] = []
    for i, seed in enumerate(seeds, start=1):
        frame = _score_seed(state, model, seed)
        frames.append(frame)
        # written every seed so a long run stays inspectable and interruptible
        write_csv(
            pd.concat(frames, ignore_index=True), RESULTS / f"certify_seeds_{name}.csv", index=False
        )
        failed = frame.loc[~frame["passed"], "name"].tolist()
        print(
            f"[{i}/{len(seeds)}] seed {seed}: "
            f"{int(frame['passed'].sum())}/8 gates" + (f", failed {failed}" if failed else "")
        )

    rows = pd.concat(frames, ignore_index=True)
    summary = _summarise(rows, len(seeds))
    write_csv(summary, RESULTS / f"certify_summary_{name}.csv", index=False)

    # `passed` from the gate machinery is authoritative; `margin` only ranks.
    disagree = rows.loc[rows["passed"] != (rows["margin"] <= 1.0 + 1e-9)]
    if len(disagree):
        print(
            "\nWARNING: passed flag disagrees with value/threshold on "
            f"{len(disagree)} row(s); the margin column is misoriented for "
            f"{sorted(set(disagree['name']))}"
        )

    n_full = int(sum(grp["passed"].all() for _, grp in rows.groupby("seed")))
    print(
        f"\ncertification of {name} over {len(seeds)} seeds "
        f"at ({N_PATHS_EVAL}, {N_PATHS_FOLD}, {N_PATHS_SCALING}):"
    )
    print(summary.drop(columns=["worst_detail"]).round(4).to_string(index=False))
    binding = summary.loc[summary["worst_margin"].idxmax()]
    print(
        f"\nall eight gates pass on {n_full} of {len(seeds)} seeds; "
        f"binding gate {binding['name']} at {binding['worst_margin']:.3f} of tolerance "
        f"(seed {int(binding['worst_seed'])})"
    )
    print(f"certify done -> results/certify_summary_{name}.csv")
