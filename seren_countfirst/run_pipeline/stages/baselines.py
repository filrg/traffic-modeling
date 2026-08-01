"""The bootstrap yardstick candidates are read against."""
from __future__ import annotations


from ... import bootstrap_baseline_table, metric_vector
from ...paths import RESULTS
from ..config import SEED
from ..state import _load, _save


def stage_baselines() -> None:
    """Score every resampler with the shared harness, giving the candidates a yardstick."""
    state = _load()
    RESULTS.mkdir(parents=True, exist_ok=True)
    counts, thr = state["counts"].values, state["busy_threshold"]
    table = bootstrap_baseline_table(counts, lambda x: metric_vector(x, thr),
                                     n_rep=150, seed=SEED)
    table.to_csv(RESULTS / "bootstrap_baselines.csv", index=False)
    coverage = (table.groupby("method")["inside_envelope"].mean().sort_values(ascending=False))
    print("fraction of metrics whose trace value lies inside each bootstrap's own envelope:")
    print(coverage.round(3).to_string())
    print("\n(a resampler covering ~all of its own metrics is self-consistent; one that "
          "does not cannot be used as a yardstick at those scales)")
    state["bootstrap_baselines"] = table
    _save(state)
