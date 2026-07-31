"""Horizon distributions and their scale support."""
from __future__ import annotations


import pandas as pd

from ... import horizon_distribution, scale_support
from ...paths import RESULTS
from ..config import N_PATHS_HORIZON, SEED
from ..state import _load, _models, _save

def stage_horizon() -> None:
    state = _load()
    RESULTS.mkdir(parents=True, exist_ok=True)
    model = _models(state)[state.get("winner", "gated_mm")]
    hz = horizon_distribution(model, n_paths=N_PATHS_HORIZON,
                              busy_threshold=state["busy_threshold"], seed=SEED)
    # a horizon holding fewer than 8 aggregation windows cannot support a
    # fixed-scale statistic, so say which scales each one can carry rather than
    # printing day-scale numbers off a one-day window
    support = pd.concat(
        [scale_support(int(d * 96)).assign(horizon_days=d) for d in sorted(hz["horizon_days"].unique())]
    ).reset_index()
    hz.to_csv(RESULTS / "horizon_report.csv", index=False)
    support.to_csv(RESULTS / "horizon_scale_support.csv", index=False)
    show = hz[hz["statistic"].isin(["total_batches", "max_daily", "longest_quiet_run"])]
    print(show.round(2).to_string(index=False))
    print("\nreportable scales per horizon (>= 8 effective windows):")
    print(support.pivot(index="horizon_days", columns="scale", values="reportable").to_string())
    state["horizon"] = hz
    state["horizon_scale_support"] = support
    _save(state)
