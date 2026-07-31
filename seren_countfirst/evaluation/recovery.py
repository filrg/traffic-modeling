"""Parameter recovery: the estimator itself must be tested, not only the metrics."""
from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np
import pandas as pd

# parameter recovery (added gate: the estimator itself must be tested)
def parameter_recovery(
    model_factory: Callable[[], object],
    truth_model,
    n_bins: int,
    n_replicates: int = 5,
    seed: int = 0,
    param_keys: Sequence[str] = ("mean", "kappa"),
    array_keys: Sequence[str] = ("sigma", "half_life_bins"),
    **fit_kwargs,
) -> pd.DataFrame:
    """Simulate from a known model, refit, and check the parameters come back.

    Metric unit tests check the yardstick; this checks the estimator, and in
    particular the kappa versus fast-latent-variance confound.
    """
    rows: List[dict] = []
    for rep in range(n_replicates):
        rng = np.random.default_rng(seed + rep)
        synthetic = truth_model.simulate(n_bins, rng)
        est = model_factory()
        est.fit(synthetic, **fit_kwargs)
        for key in param_keys:
            if key in truth_model.params_ and key in est.params_:
                rows.append(
                    {
                        "replicate": rep,
                        "parameter": key,
                        "true": float(truth_model.params_[key]),
                        "estimated": float(est.params_[key]),
                        "log_ratio": float(np.log(float(est.params_[key]) / float(truth_model.params_[key]))),
                    }
                )
        for key in array_keys:
            if key in truth_model.params_ and key in est.params_:
                tvals = np.atleast_1d(truth_model.params_[key])
                evals = np.atleast_1d(est.params_[key])
                for j in range(min(len(tvals), len(evals))):
                    rows.append(
                        {
                            "replicate": rep,
                            "parameter": f"{key}[{j}]",
                            "true": float(tvals[j]),
                            "estimated": float(evals[j]),
                            "log_ratio": float(np.log(float(evals[j]) / float(tvals[j]))),
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summary = (
        frame.groupby("parameter")
        .agg(true=("true", "first"), mean_estimate=("estimated", "mean"),
             sd_estimate=("estimated", "std"), mean_log_ratio=("log_ratio", "mean"),
             max_abs_log_ratio=("log_ratio", lambda s: float(np.max(np.abs(s)))))
        .reset_index()
    )
    summary["recovered"] = summary["max_abs_log_ratio"] < np.log(2.0)
    return summary
