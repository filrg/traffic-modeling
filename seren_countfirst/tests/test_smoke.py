"""Reachability checks: every model fits, simulates and freezes on a short series."""
from __future__ import annotations

import numpy as np
import pytest

from seren_countfirst import (
    IIDNegBin,
    NBGatedLatent,
    NBINGARCH,
    NBLatentAR,
    NBRegimeSwitch,
    ONOFFCounts,
    marginal_feasibility,
)
from seren_countfirst.models.ingarch import b_max

N_BINS = 600


def _series(seed: int = 0) -> np.ndarray:
    """Return overdispersed counts with persistence and zeros, so no rung is degenerate."""
    rng = np.random.default_rng(seed)
    x = np.zeros(N_BINS)
    for t in range(1, N_BINS):
        x[t] = 0.9 * x[t - 1] + rng.normal(0.0, 0.5)
    lam = np.exp(0.6 + x - 0.5 * np.var(x))
    return rng.poisson(lam * rng.gamma(2.0, 0.5, N_BINS)).astype(float)


# (label, factory, fit kwargs); budgets are minimal, since only reachability is checked
FITTABLE = [
    ("rung 2 iid", lambda: IIDNegBin(), {}),
    ("rung 3 latent AR(1)", lambda: NBLatentAR(1), {"n_restarts": 1, "maxiter": 20}),
    ("rung 4 latent AR(2)", lambda: NBLatentAR(2), {"n_restarts": 1, "maxiter": 20}),
    ("benchmark regime", lambda: NBRegimeSwitch(n_states=2, fast_ar=False), {"n_restarts": 1}),
    ("benchmark regime + AR", lambda: NBRegimeSwitch(n_states=2, fast_ar=True), {"n_restarts": 1}),
    ("rung 4b gated", lambda: NBGatedLatent(), {"n_restarts": 1, "maxiter": 5, "seed": 0}),
    ("benchmark ingarch", lambda: NBINGARCH(), {"n_restarts": 1, "maxiter": 30, "seed": 0}),
]


@pytest.mark.parametrize("label,factory,kwargs", FITTABLE, ids=[f[0] for f in FITTABLE])
def test_every_model_fits_simulates_and_freezes(label, factory, kwargs):
    counts = _series()
    model = factory().fit(counts, **kwargs)

    assert model.params_, f"{label}: no parameters after fit"
    assert model.report is not None, f"{label}: no fit report"
    assert np.isfinite(model.report.loglik), f"{label}: non-finite loglik"

    path = model.simulate(200, np.random.default_rng(1))
    assert path.shape == (200,)
    assert np.all(np.isfinite(path)) and np.all(path >= 0)

    model.freeze()
    with pytest.raises(RuntimeError):
        model.fit(counts, **kwargs)


# the calibration routes; no likelihood fit exercises these
CALIBRATORS = ["single scale", "two scale", "debiased", "debiased marginals"]


@pytest.mark.parametrize("route", CALIBRATORS)
def test_gated_calibration_routes_run(route):
    counts = _series()
    targets = dict(mean=float(counts.mean()),
                   variance=float(counts.var(ddof=1)),
                   zero_fraction=float((counts == 0).mean()),
                   acf_1h=0.5)
    acf_long = {24: 0.20, 48: 0.12, 96: 0.08}
    common = dict(n_bins=N_BINS, seed=0, n_paths=2, n_rounds=1)

    if route == "single scale":
        model = NBGatedLatent.calibrate(**targets)
    elif route == "two scale":
        model = NBGatedLatent.calibrate(**targets, acf_long=acf_long)
    elif route == "debiased":
        model = NBGatedLatent.calibrate_debiased(**targets, acf_long=acf_long, **common)
    else:
        model = NBGatedLatent.calibrate_debiased(**targets, acf_long=acf_long,
                                                 marginals=True, marginal_paths=4, **common)

    assert model.report is None, "moment calibration must not report a likelihood"
    implied = model.implied_marginals()
    assert np.isfinite(implied["variance"]) and implied["variance"] > 0
    assert 0.0 <= float(implied["P_zero"]) <= 1.0
    path = model.simulate(200, np.random.default_rng(1))
    assert path.shape == (200,) and np.all(path >= 0)


@pytest.mark.parametrize("emission", ["nb", "poisson"])
def test_onoff_benchmark_calibrates_and_simulates(emission):
    counts = _series()
    model = ONOFFCounts.calibrate(counts, emission=emission)

    assert model.role == "benchmark" and model.rung is None
    assert model.report is None, "moment calibration must not report a likelihood"
    # the chain is read off the indicator, so mean run lengths come back exactly
    assert model.params_["mean_off_run_bins"] == pytest.approx(
        model.params_["observed_off_run_mean"], rel=0.05)
    implied = model.implied_marginals()
    assert implied["mean"] == pytest.approx(float(counts.mean()), rel=1e-9)
    assert 0.0 <= float(implied["P_zero"]) <= 1.0

    path = model.simulate(400, np.random.default_rng(1))
    assert path.shape == (400,) and np.all(path >= 0)
    # OFF is silent, so the zero fraction cannot fall below P(OFF)
    assert (path == 0).mean() >= 1.0 - model.params_["pi_on"] - 0.1


def test_ingarch_stays_second_order_stationary():
    """Check that every reachable parameter has a finite stationary variance."""
    counts = _series()
    model = NBINGARCH().fit(counts, n_restarts=1, maxiter=30, seed=0)
    p = model.params_

    assert p["a"] >= 0.0 and p["b"] >= 0.0 and p["persistence"] < 1.0
    assert p["b"] <= b_max(p["persistence"], p["kappa"]) + 1e-9
    assert np.isfinite(p["variance"]) and p["variance"] > 0.0


def test_ingarch_moment_route_hits_its_own_targets():
    """Check that the four analytic targets are matched by the four free parameters."""
    model = NBINGARCH.calibrate(mean=3.75, variance=274.2, acf_lag1=0.686, acf_lag4=0.438, seed=0)

    assert model.report is None, "moment calibration must not report a likelihood"
    for key in ("resid_mean", "resid_IDC", "resid_acf_lag1", "resid_acf_lag4"):
        assert abs(model.params_[key]) < 1e-3, f"{key} not matched: {model.params_[key]}"
    path = model.simulate(400, np.random.default_rng(1))
    assert path.shape == (400,) and np.all(path >= 0)


def test_zero_sharpness_removes_the_gate_exactly():
    """Check that sharpness 0 makes the gate constant, which is the `nogate` ablation."""
    targets = dict(mean=3.75, variance=274.2, zero_fraction=0.775, acf_1h=0.5)
    model = NBGatedLatent.calibrate(**targets, sharpness=0.0)

    assert model.params_["a"] == 0.0
    # sigmoid(0) = 1/2 everywhere, so exactly half the gate mass is active
    assert float(model.implied_marginals()["active_fraction_gate"]) == pytest.approx(0.5)
    # `c` is then inert: the same model comes back whatever it is set to
    shifted = NBGatedLatent.calibrate(**targets, sharpness=0.0)
    shifted.params_["c"] += 3.0
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    np.testing.assert_allclose(model.simulate(200, rng_a), shifted.simulate(200, rng_b))


def test_marginal_feasibility_reports_both_targets():
    table = marginal_feasibility(3.75, 274.2, 0.775, 0.665)
    assert {"zero_ok", "acf_ok"} <= set(table.columns)
    assert "feasible" in table.attrs
