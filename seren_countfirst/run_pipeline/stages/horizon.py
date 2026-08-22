from __future__ import annotations


import pandas as pd

from ...artifacts import write_csv
from ...evaluation import horizon_distribution, scale_support
from ...paths import RESULTS
from ..config import N_PATHS_HORIZON, SEED
from ..state import fitted_models, load_state, save_state


def stage_horizon() -> None:
    state = load_state()
    RESULTS.mkdir(parents=True, exist_ok=True)
    model = fitted_models(state)[state.get("winner", "gated_mm")]
    hz = horizon_distribution(
        model, n_paths=N_PATHS_HORIZON, busy_threshold=state["busy_threshold"], seed=SEED
    )
    # a horizon holding fewer than 8 aggregation windows cannot support a
    # fixed-scale statistic, so record which scales each one can carry
    support = pd.concat(
        [
            scale_support(int(d * 96)).assign(horizon_days=d)
            for d in sorted(hz["horizon_days"].unique())
        ]
    ).reset_index()
    write_csv(hz, RESULTS / "horizon_report.csv", index=False)
    write_csv(support, RESULTS / "horizon_scale_support.csv", index=False)
    show = hz[hz["statistic"].isin(["total_batches", "max_daily", "longest_quiet_run"])]
    print(show.round(2).to_string(index=False))
    print("\nreportable scales per horizon (>= 8 effective windows):")
    print(support.pivot(index="horizon_days", columns="scale", values="reportable").to_string())
    state["horizon"] = hz
    state["horizon_scale_support"] = support
    save_state(state)
