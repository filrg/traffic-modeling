from __future__ import annotations


from ...artifacts import write_csv
from ...metrics import metric_vector
from ...bootstrap import bootstrap_baseline_table
from ..context import PipelineContext, resolve_context


def stage_baselines(context: PipelineContext | None = None) -> None:
    """Score every resampler with the shared harness, giving the candidates a yardstick."""
    context = resolve_context(context)
    state = context.load_state()
    counts, thr = state["counts"].values, state["busy_threshold"]
    table = bootstrap_baseline_table(
        counts, lambda x: metric_vector(x, thr), n_rep=150, seed=context.seed
    )
    write_csv(table, context.result("bootstrap_baselines.csv"), index=False)
    coverage = table.groupby("method")["inside_envelope"].mean().sort_values(ascending=False)
    print("fraction of metrics whose trace value lies inside each bootstrap's own envelope:")
    print(coverage.round(3).to_string())
    print(
        "\n(a resampler covering ~all of its own metrics is self-consistent; one that "
        "does not cannot be used as a yardstick at those scales)"
    )
    state["bootstrap_baselines"] = table
    context.save_state(state)
