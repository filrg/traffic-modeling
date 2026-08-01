"""Invariants of the job-size pipeline: the inversion, the law, and the pool."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seren_countfirst import (
    EmpiricalJobSize,
    F_MAX,
    LogNormalMixture,
    ScipyJobSize,
    fit_jobsize,
    jobsize_table,
    roundtrip_error_s,
    step_time_s,
    work_units,
)


def _synthetic_view(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Return a completed-jobs view containing two unusable rows."""
    rng = np.random.default_rng(seed)
    gpus = rng.choice([1, 2, 4, 8, 16], size=n)
    sizes = np.exp(rng.normal(10.0, 0.8, size=n))
    view = pd.DataFrame({
        "gpu_num": gpus,
        "duration": sizes * step_time_s(gpus, F_MAX),
        "state": "COMPLETED",
    })
    view.loc[0, "duration"] = 0.0     # zero-length job
    view.loc[1, "gpu_num"] = 0        # missing GPU count
    return view


# the inversion
def test_step_time_single_gpu_has_no_parallel_term():
    """Check that n = 1 does not pay gamma_t, making one GPU the reference."""
    assert step_time_s(1, 1.0) == pytest.approx(0.0050 + 0.038)


def test_step_time_speedup_is_sublinear():
    """Check that doubling GPUs helps by less than a factor of two."""
    one, two = float(step_time_s(1, 1.0)), float(step_time_s(2, 1.0))
    assert one / 2.0 < two < one


def test_work_units_round_trip_to_machine_precision():
    """Check that the inversion is an identity at the job's own operating point."""
    table = jobsize_table(_synthetic_view())
    assert roundtrip_error_s(table) < 1e-6


def test_work_units_are_invariant_to_the_gpu_count():
    """Check that the same work on more GPUs gives a shorter duration, same size."""
    size = 50_000.0
    for n in (1, 8, 64):
        duration = size * float(step_time_s(n, F_MAX))
        assert work_units(duration, n) == pytest.approx(size)


def test_jobsize_table_drops_unusable_rows():
    view = _synthetic_view()
    assert len(jobsize_table(view)) == len(view) - 2


# the fitted law
def test_mixture_recovers_a_known_two_component_law():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(8.0, 0.3, 4000), rng.normal(12.0, 0.5, 6000)])
    model = LogNormalMixture.fit(np.exp(z), n_components=2, n_init=5, seed=1)
    assert model.mu == pytest.approx([8.0, 12.0], abs=0.05)
    assert model.sigma == pytest.approx([0.3, 0.5], abs=0.05)
    assert model.weights == pytest.approx([0.4, 0.6], abs=0.02)


def test_mixture_components_come_back_sorted_by_mu():
    """Check that serialisation does not depend on which restart won."""
    rng = np.random.default_rng(3)
    x = np.exp(np.concatenate([rng.normal(11.0, 0.4, 2000), rng.normal(8.0, 0.4, 2000)]))
    model = LogNormalMixture.fit(x, n_components=2, n_init=4, seed=2)
    assert np.all(np.diff(model.mu) > 0)


def test_mixture_loglik_is_on_the_size_scale():
    """Check that the report's loglik includes the Jacobian, so its AIC is comparable."""
    rng = np.random.default_rng(11)
    x = np.exp(rng.normal(9.0, 0.7, 3000))
    model = LogNormalMixture.fit(x, n_components=2, n_init=3, seed=5)
    assert model.report.loglik == pytest.approx(float(model.logpdf(x).sum()))


def test_mixture_density_integrates_through_its_cdf():
    rng = np.random.default_rng(4)
    model = LogNormalMixture.fit(np.exp(rng.normal(9.0, 0.6, 2000)), n_components=2,
                                 n_init=3, seed=4)
    grid = np.linspace(1.0, 1e6, 400_000)
    assert np.trapezoid(model.pdf(grid), grid) == pytest.approx(
        float(model.cdf(grid[-1]) - model.cdf(grid[0])), abs=2e-3)


def test_mixture_samples_match_its_own_cdf():
    rng = np.random.default_rng(6)
    model = LogNormalMixture(weights=np.array([0.3, 0.7]), mu=np.array([8.0, 11.0]),
                             sigma=np.array([0.4, 0.6]))
    draws = model.rvs(40_000, rng)
    for q in (0.1, 0.5, 0.9):
        assert model.cdf(np.quantile(draws, q)) == pytest.approx(q, abs=0.01)


def test_mixture_round_trips_through_json():
    rng = np.random.default_rng(8)
    model = LogNormalMixture.fit(np.exp(rng.normal(9.5, 0.5, 1500)), n_components=2,
                                 n_init=3, seed=9)
    back = LogNormalMixture.from_dict(model.to_dict())
    assert back.weights == pytest.approx(model.weights)
    assert back.mu == pytest.approx(model.mu)
    assert back.sigma == pytest.approx(model.sigma)


def test_n_params_is_three_k_minus_one():
    m = LogNormalMixture(weights=np.full(3, 1 / 3), mu=np.arange(3.0), sigma=np.ones(3))
    assert m.n_params == 8


# the scoreboard
def test_scoreboard_ranks_by_aic_and_finds_a_planted_mixture():
    rng = np.random.default_rng(12)
    x = np.exp(np.concatenate([rng.normal(8.0, 0.3, 3000), rng.normal(12.0, 0.4, 3000)]))
    board = fit_jobsize(x, mixture_components=(2,), n_init=3, seed=13)
    assert board["AIC"].is_monotonic_increasing
    assert board.iloc[0]["distribution"] == "mix_lognormal_2"


def test_scoreboard_prefers_the_single_family_that_generated_the_data():
    rng = np.random.default_rng(14)
    board = fit_jobsize(np.exp(rng.normal(9.0, 0.7, 4000)), mixture_components=(),
                        seed=15)
    assert board.iloc[0]["distribution"] == "lognormal"


def test_scipy_winner_is_usable_as_a_sampler():
    rng = np.random.default_rng(16)
    model = ScipyJobSize(name="lognormal", params=(0.7, 0.0, np.exp(9.0)))
    draws = model.rvs(20_000, rng)
    assert np.median(draws) == pytest.approx(np.exp(9.0), rel=0.05)


# the empirical pool
def test_empirical_replay_preserves_submission_order():
    table = jobsize_table(_synthetic_view())
    pool = EmpiricalJobSize.from_table(table)
    sizes, gpus = pool.replay()
    assert sizes == pytest.approx(table["job_size"].to_numpy())
    assert gpus == pytest.approx(table["gpu_num"].to_numpy(dtype=float))


def test_empirical_bootstrap_only_returns_sizes_the_trace_contained():
    table = jobsize_table(_synthetic_view())
    pool = EmpiricalJobSize.from_table(table)
    draws, gpus = pool.rvs(500, np.random.default_rng(17), with_gpu=True)
    assert set(np.unique(draws)).issubset(set(pool.sizes))
    assert len(gpus) == 500


def test_empirical_pool_round_trips_through_npz(tmp_path):
    pool = EmpiricalJobSize.from_table(jobsize_table(_synthetic_view()))
    pool.to_npz(tmp_path / "pool.npz")
    back = EmpiricalJobSize.from_npz(tmp_path / "pool.npz")
    assert back.sizes == pytest.approx(pool.sizes)
    assert back.gpus == pytest.approx(pool.gpus)
