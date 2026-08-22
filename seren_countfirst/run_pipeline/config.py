from __future__ import annotations

SEED = 20260729

# gate-7 folds: rolling origin, 21-day validation windows
FOLD_KWARGS = {"validation_days": 21.0, "n_folds": 4, "min_train_days": 30.0}

# Simulation counts for the scoring harness. Measurement settings, not gate
# tolerances: raising one relaxes nothing. Each is sized by the gate it binds.
N_PATHS_EVAL = 1200  # gate 1, the expected-rate tolerance
N_PATHS_FOLD = 120  # gate 7, per rolling-origin fold
N_PATHS_SCALING = 300  # gate 8, per alpha per kappa rule
N_PATHS_HORIZON = 400  # sized for usable quantiles, not for tight means

# Scoring seeds for the `certify` stage; a verdict is the worst case over them,
# never a single seed.
CERTIFY_SEEDS = 10

# hard verification targets for the data stage; a mismatch stops the run
AUDIT = {
    "n_sft_jobs": 66971,
    "n_batches": 49747,
    "zero_iat": 0.2572,
    "mean_15m": 3.748,
    "idc_15m": 73.16,
    "zero_frac_15m": 0.7747,
    "max_15m": 530,
}

# load factors at which event reconstruction is attempted, starting at the
# declared operating-range lower bound rather than at nominal load
RECONSTRUCT_ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
N_PATHS_RECONSTRUCT = 8
