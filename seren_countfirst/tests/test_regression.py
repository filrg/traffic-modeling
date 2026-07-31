"""Numerical baseline: refactoring must not move a single fitted number.

The pins below were captured from the frozen session-2 results before any
cleanup. `baseline/models_baseline.json` is an immutable snapshot of the
fitted parameters, so a later pipeline re-run cannot quietly move the target.

A failure here means a change altered simulation, metrics or scaling — not
that the pins are stale. Update them only with a recorded reason.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from seren_countfirst.metrics import metric_vector
from seren_countfirst.metrics import quiet_run_vs_geometric
from seren_countfirst.paths import RESULTS, STATE_DIR
from seren_countfirst.scaling import scaling_report
from seren_countfirst.tools.migrate_state import load_portable_state, rebuild_model

SEED = 20260729
SIM_BINS = 4000
BASELINE = Path(__file__).parent / "baseline" / "models_baseline.json"

#: sha256 prefix of simulate() output, plus total and max, per fitted model
SIM_CHECKSUMS = {
    "r2": ("2dc8e955460e4166", 14864.0, 316.0),
    "r3": ("da230c11d360917b", 47018.0, 9300.0),
    "r4": ("9b6d1a04231ed8ea", 246741.0, 54093.0),
    "gated": ("5b1ed73f0733a9d6", 10275.0, 544.0),
    "gated_mm": ("911cb0bdedc9e204", 20236.0, 549.0),
}

#: published likelihood results (results/benchmark_table.csv, session 2)
PUBLISHED_FITS = {
    "r2": (-17350.326982993513, 34704.653965987025, 2),
    "r3": (-13012.282197140079, 26032.564394280158, 4),
    "r4": (-12981.853168163103, 25975.706336326206, 6),
    "gated": (-12864.699522832618, 25743.399045665235, 7),
}
N_OBS = 13272

#: metric_vector over the real trace counts
TRACE_METRIC_VECTOR_SHA = "fb820b6844aa442a"
TRACE_METRIC_N = 42

#: quiet-run fingerprint of the real trace (the MMPP-separating texture)
TRACE_QUIET_RUN = {
    "n_quiet_runs": 687.0,
    "mean_run": 14.965065502183407,
    "q95_empirical": 69.69999999999993,
    "q95_geometric_same_mean": 44.0,
    "q99_empirical": 147.41999999999996,
    "q99_geometric_same_mean": 67.0,
    "max_empirical": 703.0,
    "P_exceed_geoQ99_empirical": 0.05240174672489083,
    "P_exceed_geoQ99_geometric": 0.01,
}

#: scaling_report for the frozen winner: alpha -> (mean_total, IDC_15m, ACF_lag1)
SCALING_BINS = 2000
SCALING_PINS = {
    1.0: (7204.0, 84.30679425009923, 0.6337010815959208),
    2.0: (14678.0, 165.0813565627579, 0.6589472179343068),
}


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=float).tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def baseline_models() -> dict:
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="module")
def trace_counts() -> np.ndarray:
    state = load_portable_state(STATE_DIR)
    if "counts" not in state:
        pytest.skip(f"no portable state at {STATE_DIR}")
    return state["counts"].values


@pytest.fixture(scope="module")
def busy_threshold() -> float:
    state = load_portable_state(STATE_DIR)
    if "busy_threshold" not in state:
        pytest.skip(f"no portable state at {STATE_DIR}")
    return float(state["busy_threshold"])


@pytest.mark.parametrize("key", sorted(SIM_CHECKSUMS))
def test_simulate_is_bit_for_bit_unchanged(baseline_models, key):
    """Same params, same seed, same draws — this catches any RNG reordering."""
    model = rebuild_model(baseline_models[key])
    path = model.simulate(SIM_BINS, np.random.default_rng(SEED))
    expected_sha, expected_total, expected_max = SIM_CHECKSUMS[key]
    assert float(path.sum()) == expected_total, f"{key}: total moved"
    assert float(path.max()) == expected_max, f"{key}: max moved"
    assert _sha(path) == expected_sha, f"{key}: simulated path checksum moved"


@pytest.mark.parametrize("key", sorted(PUBLISHED_FITS))
def test_fit_reports_survive_the_state_round_trip(baseline_models, key):
    """Rebuilt models must carry their loglik/AIC, or benchmark_table goes blank."""
    report = rebuild_model(baseline_models[key]).report
    loglik, aic, n_params = PUBLISHED_FITS[key]
    assert report is not None, f"{key}: fit report lost on rebuild"
    assert report.loglik == pytest.approx(loglik, rel=1e-12)
    assert report.aic == pytest.approx(aic, rel=1e-12)
    assert report.n_params == n_params
    assert report.n_obs == N_OBS


def test_moment_calibrated_winner_has_no_likelihood_fit(baseline_models):
    """gated_mm is calibrated, not fitted; the model card depends on that."""
    model = rebuild_model(baseline_models["gated_mm"])
    assert model.report is None
    assert model.name == "nb_gated_mm"


def test_trace_metric_vector_unchanged(trace_counts, busy_threshold):
    """Guards aggregate / idc / acf / run-length / wavelet against refactors."""
    mv = metric_vector(trace_counts, busy_threshold)
    assert len(mv) == TRACE_METRIC_N
    assert _sha(mv.to_numpy()) == TRACE_METRIC_VECTOR_SHA


def test_trace_quiet_run_fingerprint_unchanged(trace_counts):
    got = quiet_run_vs_geometric(trace_counts)
    for name, expected in TRACE_QUIET_RUN.items():
        assert float(got[name]) == pytest.approx(expected, rel=1e-12), name


def test_scaling_report_unchanged(baseline_models):
    """Pins simulate() under load scaling plus the drift/collision columns."""
    winner = rebuild_model(baseline_models["gated_mm"])
    report = scaling_report(
        winner, SCALING_BINS, alphas=tuple(SCALING_PINS), n_paths=2,
        kappa_rules=("constant",), seed=SEED,
    )
    for _, row in report.iterrows():
        total, idc_15m, acf_lag1 = SCALING_PINS[float(row["alpha"])]
        assert float(row["mean_total"]) == pytest.approx(total, rel=1e-12)
        assert float(row["IDC_15m"]) == pytest.approx(idc_15m, rel=1e-12)
        assert float(row["ACF_lag1"]) == pytest.approx(acf_lag1, rel=1e-12)


def test_published_benchmark_table_still_on_disk():
    """The exported table must keep agreeing with the fits it was built from."""
    import pandas as pd

    path = RESULTS / "benchmark_table.csv"
    if not path.exists():
        pytest.skip("benchmark_table.csv not exported yet")
    table = pd.read_csv(path).set_index("model")
    names = {"r2": "iid_nb", "r3": "nb_ar1", "r4": "nb_ar2", "gated": "nb_gated"}
    for key, (loglik, aic, _) in PUBLISHED_FITS.items():
        row = table.loc[names[key]]
        assert float(row["loglik"]) == pytest.approx(loglik, rel=1e-9)
        assert float(row["AIC"]) == pytest.approx(aic, rel=1e-9)
