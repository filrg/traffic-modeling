from __future__ import annotations


import pandas as pd

from ...artifacts import write_csv
from ...evaluation import horizon_distribution, scale_support
from ..config import N_PATHS_HORIZON
from ..context import PipelineContext, resolve_context
from ..state import fitted_models


def stage_horizon(context: PipelineContext | None = None) -> None:
    context = resolve_context(context)
    state = context.load_state()
    model = fitted_models(state)[state.get("winner", "gated_mm")]
    hz = horizon_distribution(
        model,
        n_paths=N_PATHS_HORIZON,
        busy_threshold=state["busy_threshold"],
        seed=context.seed,
    )
    # a horizon holding fewer than 8 aggregation windows cannot support a
    # fixed-scale statistic, so record which scales each one can carry
    support = pd.concat(
        [
            scale_support(int(d * 96)).assign(horizon_days=d)
            for d in sorted(hz["horizon_days"].unique())
        ]
    ).reset_index()
    write_csv(hz, context.result("horizon_report.csv"), index=False)
    write_csv(support, context.result("horizon_scale_support.csv"), index=False)
    show = hz[hz["statistic"].isin(["total_batches", "max_daily", "longest_quiet_run"])]
    print(show.round(2).to_string(index=False))
    print("\nreportable scales per horizon (>= 8 effective windows):")
    print(support.pivot(index="horizon_days", columns="scale", values="reportable").to_string())
    state["horizon"] = hz
    state["horizon_scale_support"] = support
    context.save_state(state)
