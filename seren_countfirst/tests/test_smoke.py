"""Every model must fit, simulate and freeze on a short synthetic series.

Reachability rather than accuracy. These paths are otherwise only exercised by
the staged pipeline and the standalone experiments, so a missing import or a
changed signature can survive the rest of the suite untouched — as happened to
`NBRegimeSwitch.fit`, which was broken by a file split while all other tests
passed. Optimiser budgets are deliberately tiny: nothing here checks that a fit
is *good*, only that it runs and returns a usable model.
"""
from __future__ import annotations

import numpy as np
import pytest

from seren_countfirst import (
    IIDNegBin,
    NBGatedLatent,
    NBLatentAR,
    NBRegimeSwitch,
    marginal_feasibility,
)

N_BINS = 600


def _series(seed: int = 0) -> np.ndarray:
    """Overdispersed counts with persistence and zeros, so no rung is degenerate."""
    rng = np.random.default_rng(seed)
    x = np.zeros(N_BINS)
    for t in range(1, N_BINS):
        x[t] = 0.9 * x[t - 1] + rng.normal(0.0, 0.5)
    lam = np.exp(0.6 + x - 0.5 * np.var(x))
    return rng.poisson(lam * rng.gamma(2.0, 0.5, N_BINS)).astype(float)


#: (label, factory, fit kwargs) — budgets kept minimal on purpose
FITTABLE = [
    ("rung 2 iid", lambda: IIDNegBin(), {}),
    ("rung 3 latent AR(1)", lambda: NBLatentAR(1), {"n_restarts": 1, "maxiter": 20}),
    ("rung 4 latent AR(2)", lambda: NBLatentAR(2), {"n_restarts": 1, "maxiter": 20}),
    ("benchmark regime", lambda: NBRegimeSwitch(n_states=2, fast_ar=False), {"n_restarts": 1}),
    ("benchmark regime + AR", lambda: NBRegimeSwitch(n_states=2, fast_ar=True), {"n_restarts": 1}),
    ("rung 4b gated", lambda: NBGatedLatent(), {"n_restarts": 1, "maxiter": 5, "seed": 0}),
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


#: the calibration routes, which are how the production candidate is built and
#: which no likelihood fit exercises
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


def test_marginal_feasibility_reports_both_targets():
    table = marginal_feasibility(3.75, 274.2, 0.775, 0.665)
    assert {"zero_ok", "acf_ok"} <= set(table.columns)
    assert "feasible" in table.attrs
