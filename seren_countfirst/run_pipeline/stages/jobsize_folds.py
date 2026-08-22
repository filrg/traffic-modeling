from __future__ import annotations

import time

import pandas as pd

from ...artifacts import write_csv
from ...data import arrival_view, jobsize_view, load_trace
from ...jobsize import attach_activity, fold_report, jobsize_table
from ..config import FOLD_KWARGS
from ..context import PipelineContext, resolve_context

# fewer restarts than the frozen fit: these runs measure stability, not a law
N_INIT_FOLD = 4
N_REP_FOLD = 20

# statistics carried into the printed comparison, in the order they are read
_SHOWN = [
    "cv",
    "q99_over_median",
    "spearman_jobs_work",
    "top5pct_share",
    "ks_job_marginal",
    "mean_work_ratio",
    "median_ratio",
]


def stage_jobsize_folds(context: PipelineContext | None = None) -> None:
    """Report parameter stability and held-out fidelity over rolling-origin folds.

    Splits are contiguous and training always precedes validation, so no bin's
    neighbours leak across the boundary. The tail cut and every fitted parameter
    are frozen from the training window and carried into the one that follows.
    """
    t0 = time.time()
    context = resolve_context(context)
    context.results.mkdir(parents=True, exist_ok=True)
    raw = load_trace(context.trace_csv)
    jobs = attach_activity(jobsize_table(jobsize_view(raw, "SFT")), arrival_view(raw, "SFT"))
    print(
        f"{len(jobs):,} jobs over "
        f"{(jobs['bin'].max() - jobs['bin'].min()).total_seconds() / 86400:.1f} days"
    )
    print(f"folds: {FOLD_KWARGS}")

    table = fold_report(
        jobs,
        n_init=N_INIT_FOLD,
        seed=context.seed,
        n_rep=N_REP_FOLD,
        **FOLD_KWARGS,
    )
    if table.empty:
        raise RuntimeError("no fold fits in the trace's span")
    write_csv(table, context.result("jobsize_folds.csv"), index=False)

    fitted = table.loc[table["arm"] != "observed"]
    print("\nfitted parameters by window (out_of_sample = the training window):")
    print(
        fitted[["fold", "arm", "n_train", "n_val", "k", "beta", "escape", "mu", "cut", "rate_b"]]
        .round(4)
        .to_string(index=False)
    )

    train_arm = table.loc[table["arm"] == "out_of_sample"]
    if train_arm["beta"].notna().any():
        b = train_arm["beta"].dropna()
        print(
            f"\nbeta across training windows: {b.min():+.4f} to {b.max():+.4f}  "
            f"(spread {b.max() - b.min():.4f})"
        )

    print("\nheld-out fidelity, per fold:")
    for fold, block in table.groupby("fold"):
        print(f"  fold {int(fold)}  n_val {int(block['n_val'].iloc[0]):,}")
        print(block.set_index("arm")[_SHOWN].round(4).to_string())

    oos = table.loc[table["arm"] == "out_of_sample", _SHOWN].mean()
    win = table.loc[table["arm"] == "in_window", _SHOWN].mean()
    obs = table.loc[table["arm"] == "observed", _SHOWN].mean()
    summary = pd.DataFrame({"observed": obs, "in_window": win, "out_of_sample": oos})
    # the transfer cost is the gap the fold design exists to measure
    summary["transfer_cost"] = summary["out_of_sample"] - summary["in_window"]
    print("\nmean over folds:")
    print(summary.round(4).to_string())
    write_csv(summary, context.result("jobsize_folds_summary.csv"), index=True)

    print(f"\nwrote jobsize_folds.csv and jobsize_folds_summary.csv to {context.results}")
    print(f"jobsize_folds stage done in {time.time() - t0:.0f}s")
