"""Frozen settings for the staged pipeline."""
from __future__ import annotations

SEED = 20260729

#: gate-7 folds: rolling origin, 21-day validation windows (guide section 1.10)
FOLD_KWARGS = {"validation_days": 21.0, "n_folds": 4, "min_train_days": 30.0}

#: Simulation counts for the scoring harness. Measurement settings, not gate
#: tolerances: raising one relaxes nothing. Each is sized by the gate it binds.
N_PATHS_EVAL = 1200        # gate 1, the expected-rate tolerance
N_PATHS_FOLD = 120         # gate 7, per rolling-origin fold
N_PATHS_SCALING = 300      # gate 8, per alpha per kappa rule
N_PATHS_HORIZON = 400      # sized for usable quantiles, not for tight means

#: CLI rung name -> state key
RUNG_KEYS = {"r2": "r2", "r3": "r3", "r4": "r4", "gated": "gated", "mm": "gated_mm",
             "gated_mm": "gated_mm", "mm2": "gated_mm2", "gated_mm2": "gated_mm2",
             "mm3": "gated_mm3", "gated_mm3": "gated_mm3",
             "mm4": "gated_mm4", "gated_mm4": "gated_mm4",
             "mm5": "gated_mm5", "gated_mm5": "gated_mm5"}

#: rung 4b knobs controlling burst texture at fixed marginals. Tuned against
#: gate 5, which is therefore in-sample for `mm4`; gate 7 is not.
MM4_KNOBS = {"kappa": 40.0, "sigma_fast": 0.60, "tau_fast_bins": 0.5}

#: base-bin lags the two-scale calibration targets. Deliberately many and
#: spread, so the long scale is fitted to the ACF envelope rather than to the
#: trace's lag-96 diurnal spike, which the model has no calendar to reproduce.
LONG_ACF_LAGS = (24, 48, 96, 192, 288, 384, 672)

#: hard verification targets for the data stage; a mismatch stops the run
AUDIT = {
    "n_sft_jobs": 66971,
    "n_batches": 49747,
    "zero_iat": 0.2572,
    "mean_15m": 3.748,
    "idc_15m": 73.16,
    "zero_frac_15m": 0.7747,
    "max_15m": 530,
}

#: load factors at which event reconstruction is attempted, starting at the
#: declared operating-range lower bound rather than at nominal load
RECONSTRUCT_ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
N_PATHS_RECONSTRUCT = 8
