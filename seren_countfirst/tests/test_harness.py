"""The methodology rules as executable invariants (guide 1.14).

Metric unit tests run on processes whose IDC and ACF are known in closed form,
so a failure localises to the metric rather than to the model. A failing test
here is the specification working, not a test to relax.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seren_countfirst import (
    ActivityScheme,
    CountSeries,
    GateEvidence,
    IIDNegBin,
    NBGatedLatent,
    NBLatentAR,
    acf,
    aggregate,
    bootstrap_metric_envelope,
    build_counts,
    collision_risk,
    contiguous_split,
    episode_stats,
    evaluate_gates,
    evaluate_on_folds,
    expanding_window_folds,
    fold_stability_gate,
    idc,
    idc_curve,
    make_iid_mark_sampler,
    marginal_feasibility,
    metric_vector,
    nb2_sample,
    reconstruct_events,
    scale_matched_idc_envelope,
    scaling_drift_gate,
    theoretical_acf_two_ar,
    theoretical_idc_two_ar,
)
from seren_countfirst.models import ar1_logdet_precision, ar1_precision, half_life_to_phi
from seren_countfirst.reconstruct import OffsetTemplates, export_check


# metric unit tests on processes with known answers
def test_idc_poisson_is_one_at_every_scale():
    rng = np.random.default_rng(0)
    x = rng.poisson(4.0, size=200_000).astype(float)
    curve = idc_curve(x)
    for scale in ("15m", "1h", "6h", "1d"):
        assert abs(curve[scale] - 1.0) < 0.12, f"Poisson IDC at {scale} = {curve[scale]}"


def test_idc_iid_nb_matches_closed_form():
    rng = np.random.default_rng(1)
    lam, kappa = 4.0, 2.0
    x = nb2_sample(np.full(300_000, lam), kappa, rng).astype(float)
    expected = 1.0 + lam / kappa
    assert abs(idc(x) / expected - 1.0) < 0.05
    # aggregation of an iid sequence multiplies both mean and variance by q
    assert abs(idc(aggregate(x, 4)) / expected - 1.0) < 0.08


def test_acf_of_iid_sequence_is_zero():
    rng = np.random.default_rng(2)
    x = nb2_sample(np.full(100_000, 3.0), 1.5, rng).astype(float)
    a = acf(x, [1, 5, 50])
    assert np.all(np.abs(a.to_numpy()) < 0.02)


def test_idc_and_acf_match_two_ar_theory():
    """Known-dependence process: the guide's own eq. (37) and (41)."""
    model = NBLatentAR(2)
    sigma = np.array([0.8, 0.5])
    phi = np.array([half_life_to_phi(40.0), half_life_to_phi(3.0)])
    model.params_ = {
        "mu": np.log(5.0), "mean": 5.0, "sigma": sigma, "phi": phi,
        "half_life_bins": np.array([40.0, 3.0]), "kappa": 3.0, "V": float(np.sum(sigma ** 2)),
    }
    rng = np.random.default_rng(3)
    x = np.concatenate([model.simulate(120_000, rng) for _ in range(4)])

    for q in (1, 4, 24):
        emp = idc(aggregate(x, q))
        th = theoretical_idc_two_ar(5.0, sigma, phi, 3.0, q)
        assert abs(emp / th - 1.0) < 0.15, f"IDC(q={q}): empirical {emp:.3f} vs theory {th:.3f}"

    for lag in (1, 5, 20):
        emp = acf(x, [lag])[lag]
        th = theoretical_acf_two_ar(5.0, sigma, phi, 3.0, lag)
        assert abs(emp - th) < 0.03, f"ACF(lag={lag}): empirical {emp:.3f} vs theory {th:.3f}"


def test_run_length_metric_on_deterministic_pattern():
    x = np.array([0, 0, 0, 5, 5, 0, 7, 0, 0, 0, 0, 9, 9, 9], dtype=float)
    ep = episode_stats(x, busy_threshold=1.0, quiet_threshold=0.0)
    assert ep["n_busy_runs"] == 3
    assert ep["busy_run_max"] == 3
    assert ep["quiet_run_max"] == 4


def test_aggregate_drops_only_the_incomplete_tail():
    x = np.arange(10, dtype=float)
    agg = aggregate(x, 4)
    assert len(agg) == 2 and agg[0] == 0 + 1 + 2 + 3 and agg[1] == 4 + 5 + 6 + 7


# latent simulation invariants
def test_ar1_precision_gives_the_requested_stationary_variance():
    n, sigma, phi = 8, 1.4, 0.7
    cov = np.linalg.inv(ar1_precision(n, sigma, phi).toarray())
    assert np.allclose(np.diag(cov), sigma ** 2, rtol=1e-10)
    assert np.isclose(cov[3, 4] / sigma ** 2, phi, rtol=1e-10)
    assert np.isclose(ar1_logdet_precision(n, sigma, phi), np.linalg.slogdet(ar1_precision(n, sigma, phi).toarray())[1])


def test_stationary_initialisation_hits_the_theoretical_mean():
    """No warm-up, no post-hoc rescaling: the mean must be right on its own."""
    model = NBLatentAR(2)
    sigma = np.array([1.1, 0.7])
    model.params_ = {
        "mu": np.log(3.75), "mean": 3.75, "sigma": sigma,
        "phi": np.array([half_life_to_phi(30.0), half_life_to_phi(2.0)]),
        "half_life_bins": np.array([30.0, 2.0]), "kappa": 2.0, "V": float(np.sum(sigma ** 2)),
    }
    rng = np.random.default_rng(4)
    means = np.array([model.simulate(4000, rng).mean() for _ in range(150)])
    se = means.std(ddof=1) / np.sqrt(len(means))
    assert abs(means.mean() - 3.75) < 4 * se, f"mean {means.mean():.3f} vs 3.75, se {se:.3f}"


def test_first_bin_is_already_stationary():
    """The very first bin must have the stationary mean: no burn-in artefact."""
    model = NBLatentAR(1)
    model.params_ = {
        "mu": np.log(6.0), "mean": 6.0, "sigma": np.array([0.9]),
        "phi": np.array([half_life_to_phi(50.0)]), "half_life_bins": np.array([50.0]),
        "kappa": 4.0, "V": 0.81,
    }
    rng = np.random.default_rng(5)
    firsts = np.array([model.simulate(3, rng)[0] for _ in range(4000)])
    se = firsts.std(ddof=1) / np.sqrt(len(firsts))
    assert abs(firsts.mean() - 6.0) < 4 * se


def test_no_pathwise_normalisation():
    """Independent paths must differ in total count; equal totals would mean
    the generator secretly rescaled each path."""
    model = NBLatentAR(1)
    model.params_ = {
        "mu": np.log(4.0), "mean": 4.0, "sigma": np.array([0.8]),
        "phi": np.array([half_life_to_phi(20.0)]), "half_life_bins": np.array([20.0]),
        "kappa": 2.0, "V": 0.64,
    }
    rng = np.random.default_rng(6)
    totals = np.array([model.simulate(2000, rng).sum() for _ in range(30)])
    assert totals.std(ddof=1) > 0.02 * totals.mean()


# load scaling invariants
def test_intensity_multiplication_scales_the_mean():
    model = NBLatentAR(1)
    model.params_ = {
        "mu": np.log(2.0), "mean": 2.0, "sigma": np.array([0.6]),
        "phi": np.array([half_life_to_phi(10.0)]), "half_life_bins": np.array([10.0]),
        "kappa": 3.0, "V": 0.36,
    }
    rng = np.random.default_rng(7)
    for alpha in (0.5, 2.0, 4.0):
        means = np.array([model.simulate(3000, rng, alpha=alpha).mean() for _ in range(40)])
        se = means.std(ddof=1) / np.sqrt(len(means))
        assert abs(means.mean() - alpha * 2.0) < 4 * se + 0.02 * alpha


def test_both_kappa_rules_are_available_and_differ():
    model = IIDNegBin()
    model.params_ = {"mu": np.log(3.0), "mean": 3.0, "kappa": 1.0}
    assert model.scaled_kappa(4.0, "constant") == 1.0
    assert model.scaled_kappa(4.0, "proportional") == 4.0
    rng = np.random.default_rng(8)
    a = np.mean([idc(model.simulate(20000, rng, alpha=4.0, kappa_rule="constant")) for _ in range(5)])
    b = np.mean([idc(model.simulate(20000, rng, alpha=4.0, kappa_rule="proportional")) for _ in range(5)])
    assert a > b, "constant kappa must inflate dispersion more than the proportional rule"


# event reconstruction invariants
def test_reconstruction_conserves_counts_and_jobs():
    counts = np.array([0, 3, 0, 0, 12, 1, 250, 0], dtype=float)
    rng = np.random.default_rng(9)
    sampler = make_iid_mark_sampler(np.array([1, 1, 1, 2, 3, 6]))
    res = reconstruct_events(counts, 900.0, rng, sampler)
    chk = export_check(res, counts)
    assert chk["batch_conserved"] and chk["job_conserved"] and chk["distinct_timestamps"]
    assert res.n_batches == int(counts.sum())


def test_reconstruction_refuses_to_silently_merge():
    counts = np.array([1500.0])
    rng = np.random.default_rng(10)
    sampler = make_iid_mark_sampler(np.array([1]))
    with pytest.raises(ValueError, match="distinct slots"):
        reconstruct_events(counts, 900.0, rng, sampler)


def test_collision_risk_is_reported_not_hidden():
    counts = np.array([100, 950, 20], dtype=float)
    rep = collision_risk(counts, 900.0)
    assert rep["n_bins_exceeding"] == 1
    assert rep["max_fill_ratio"] > 1.0


def test_offsets_are_distinct_within_a_bin():
    rng = np.random.default_rng(11)
    templates = OffsetTemplates(pooled=np.array([1.0, 1.0, 1.0, 2.0, 899.0]), by_class={})
    draw = templates.draw(5, rng, 900.0)
    assert len(np.unique(draw)) == 5


# data-split invariants
def _toy_counts(n=4000, seed=12):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-04-01", periods=n, freq="15min", tz="Asia/Shanghai")
    return CountSeries(rng.poisson(3.0, n).astype(float), idx)


def test_contiguous_split_has_no_leakage():
    cs = _toy_counts()
    split = contiguous_split(cs)
    assert split.train.index[-1] < split.validation.index[0]
    assert split.validation.index[-1] < split.test.index[0]
    assert len(split.train) + len(split.validation) + len(split.test) == len(cs)


def test_expanding_folds_always_train_before_validation():
    cs = _toy_counts(n=96 * 120)
    folds = expanding_window_folds(cs, validation_days=14, n_folds=4, min_train_days=30)
    assert len(folds) >= 2
    for fold in folds:
        assert fold.train.index[-1] < fold.validation.index[0]


def test_activity_thresholds_are_frozen_after_training():
    cs = _toy_counts()
    scheme = ActivityScheme.fit(cs.slice_bins(0, 2000))
    before = scheme.to_dict()
    scheme.classify(cs.values)
    assert scheme.to_dict() == before


def test_build_counts_preserves_zero_bins():
    idx = pd.to_datetime(
        ["2023-04-01 00:00:00+08:00", "2023-04-01 00:02:00+08:00", "2023-04-01 03:40:00+08:00"]
    )
    cs = build_counts(pd.DatetimeIndex(idx))
    assert cs.values.sum() == 3
    assert (cs.values == 0).sum() > 0, "empty bins between events must be preserved"


# gate 3: the IDC envelope must be able to carry the scale it grades
def _persistent_counts(n=96 * 140, seed=41, tau=400.0):
    """Counts with genuine multi-day persistence, so long-scale IDC is real."""
    from seren_countfirst.models import ar1_path, half_life_to_phi, nb2_sample

    rng = np.random.default_rng(seed)
    u = ar1_path(n, 1.0, half_life_to_phi(tau), rng)
    return nb2_sample(np.exp(1.0 + u), 5.0, rng).astype(float)


def test_scale_matched_envelope_uses_longer_blocks_for_longer_scales():
    x = _persistent_counts(n=96 * 140)
    env = scale_matched_idc_envelope(x, n_rep=40, seed=1)
    assert env.loc["15m", "block_bins"] <= env.loc["1d", "block_bins"] < env.loc["7d", "block_bins"]
    assert env.loc["7d", "n_blocks"] < env.loc["1d", "n_blocks"]
    assert not bool(env.loc["7d", "supported"]), "138 days cannot support a 7-day IDC envelope"
    assert bool(env.loc["1d", "supported"])


def test_shared_block_envelope_understates_long_scale_dispersion():
    """The defect being fixed: 24h blocks cannot carry multi-day dispersion."""
    x = _persistent_counts(n=96 * 140)
    shared = bootstrap_metric_envelope(x, lambda v: idc_curve(v).add_prefix("IDC_"),
                                       method="stationary_24h", n_rep=40, seed=1)
    matched = scale_matched_idc_envelope(x, n_rep=40, seed=1)
    truth = idc_curve(x)
    assert shared.loc["IDC_4d", "median"] < 0.75 * truth["4d"], "the shared envelope should sag"
    assert matched.loc["4d", "median"] > shared.loc["IDC_4d", "median"], "matched blocks must carry more"
    assert matched.loc["4d", "q975"] > shared.loc["IDC_4d", "q975"]


def test_gate3_no_longer_punishes_a_model_that_matches_long_scales():
    """A synthetic 'model' equal to the trace must not be scored down."""
    x = _persistent_counts(n=96 * 140)
    thr = float(np.quantile(x, 0.8))
    env = bootstrap_metric_envelope(x, lambda v: metric_vector(v, thr), method="stationary_24h",
                                    n_rep=40, seed=1)
    matched = scale_matched_idc_envelope(x, n_rep=40, seed=1)

    shared_gate = evaluate_gates(x, [x], thr, env, stop_at_first_failure=False)
    matched_gate = evaluate_gates(x, [x], thr, env, stop_at_first_failure=False,
                                  evidence=GateEvidence(idc_envelope=matched))
    shared3 = next(g for g in shared_gate.results if g.name == "3_idc_envelope")
    matched3 = next(g for g in matched_gate.results if g.name == "3_idc_envelope")
    assert matched3.value >= shared3.value, "a perfect match must not score worse under the fix"
    assert matched3.passed and "ungraded" in matched3.detail


def test_gate3_falls_back_to_the_shared_envelope_when_none_is_given():
    x = _persistent_counts(n=96 * 60)
    thr = float(np.quantile(x, 0.8))
    env = bootstrap_metric_envelope(x, lambda v: metric_vector(v, thr), n_rep=30, seed=2)
    report = evaluate_gates(x, [x], thr, env, stop_at_first_failure=False)
    gate3 = next(g for g in report.results if g.name == "3_idc_envelope")
    assert "single shared envelope" in gate3.detail


# rung 4b: the second slow scale in the gate latent
_MM_ARGS = dict(mean=3.747985, variance=274.206902, zero_fraction=0.774655, acf_1h=0.665)


def test_second_scale_off_reproduces_the_single_scale_model_exactly():
    """w_long = 0 must be the old model bit for bit, RNG stream included."""
    model = NBGatedLatent.calibrate(**_MM_ARGS)
    assert model.params_["w_long"] == 0.0
    two_scale = NBGatedLatent.calibrate(**_MM_ARGS, tau_long_bins=500.0, w_long=0.0)
    assert two_scale.params_["tau_slow_bins"] == pytest.approx(
        model.params_["tau_slow_bins"], rel=1e-15
    ), "an inert long scale must not move tau_slow"

    a = model.simulate(3000, np.random.default_rng(31))
    b = two_scale.simulate(3000, np.random.default_rng(31))
    assert np.array_equal(a, b), "an inert long scale must not consume RNG draws"


def test_second_scale_leaves_the_marginals_alone():
    """u stays N(0,1), so mean / variance / zero fraction cannot move with w."""
    one = NBGatedLatent.calibrate(**_MM_ARGS)
    two = NBGatedLatent.calibrate(**_MM_ARGS, tau_long_bins=400.0, w_long=0.6)
    for key in ("mean", "variance", "IDC_base", "P_zero"):
        assert two.implied_marginals()[key] == pytest.approx(one.implied_marginals()[key], rel=1e-12), key
    # and the hourly ACF is re-pinned exactly, by re-solving tau_slow
    assert two.implied_marginals()["acf_1h_lag1"] == pytest.approx(_MM_ARGS["acf_1h"], abs=1e-6)
    assert two.params_["tau_slow_bins"] != one.params_["tau_slow_bins"]


def test_second_scale_carries_persistence_to_long_lags():
    one = NBGatedLatent.calibrate(**_MM_ARGS)
    two = NBGatedLatent.calibrate(**_MM_ARGS, tau_long_bins=400.0, w_long=0.6)
    assert one.implied_marginals()["acf_lag96"] < 0.02, "single scale is dead by one day"
    assert two.implied_marginals()["acf_lag96"] > 5 * one.implied_marginals()["acf_lag96"]


def test_simulated_two_scale_latent_keeps_unit_variance():
    """The sqrt(1-w^2)/w mixing is what protects the marginals; check the path."""
    model = NBGatedLatent.calibrate(**_MM_ARGS, tau_long_bins=200.0, w_long=0.5)
    rng = np.random.default_rng(32)
    counts, lam = model.simulate(60_000, rng, return_intensity=True)
    assert float(counts.mean()) == pytest.approx(model.params_["mean"], rel=0.12)
    assert float(np.mean(counts == 0)) == pytest.approx(
        float(model.implied_marginals()["P_zero"]), abs=0.05
    )


def test_debiased_calibration_aims_higher_than_the_analytic_one():
    """Matching at finite length must ask the process for more persistence."""
    targets = {96: 0.19, 192: 0.12, 384: 0.07}
    plain = NBGatedLatent.calibrate(**_MM_ARGS, acf_long=targets)
    debiased = NBGatedLatent.calibrate_debiased(**_MM_ARGS, acf_long=targets, n_bins=6000,
                                                seed=3, n_paths=4, n_rounds=1)
    assert debiased.params_["w_long"] > plain.params_["w_long"]
    assert debiased.implied_marginals()["acf_lag96"] > plain.implied_marginals()["acf_lag96"]
    assert 0.0 < debiased.params_["acf_attenuation"] <= 1.0
    # the marginals are still algebraically untouched
    for key in ("mean", "variance", "P_zero"):
        assert debiased.implied_marginals()[key] == pytest.approx(
            plain.implied_marginals()[key], rel=1e-12
        ), key


def test_debiased_calibration_is_deterministic():
    kw = dict(acf_long={96: 0.19, 192: 0.12}, n_bins=4000, seed=7, n_paths=3, n_rounds=1)
    a = NBGatedLatent.calibrate_debiased(**_MM_ARGS, **kw)
    b = NBGatedLatent.calibrate_debiased(**_MM_ARGS, **kw)
    assert a.params_["w_long"] == pytest.approx(b.params_["w_long"], rel=1e-12)
    assert a.params_["tau_long_bins"] == pytest.approx(b.params_["tau_long_bins"], rel=1e-12)


def test_long_scale_calibration_hits_the_requested_lags():
    targets = {24: 0.21, 96: 0.19, 192: 0.12, 384: 0.07}
    model = NBGatedLatent.calibrate(**_MM_ARGS, acf_long=targets)
    assert 0.0 < model.params_["w_long"] < 1.0
    assert model.params_["tau_long_bins"] > model.params_["tau_slow_bins"]
    got = model.implied_marginals()["acf_lag96"]
    assert abs(got - targets[96]) < 0.10, f"lag-96 ACF {got:.3f} nowhere near the target"
    assert model.implied_marginals()["acf_1h_lag1"] == pytest.approx(_MM_ARGS["acf_1h"], abs=1e-6)


# gate 7: rolling-origin fold stability
def _fold_frame(z_by_fold, folds=("expanding_1", "expanding_2")) -> pd.DataFrame:
    rows = []
    for fold, zs in zip(folds, z_by_fold):
        for metric, z in zip(("mean", "IDC_15m"), zs):
            rows.append({"fold": fold, "n_bins": 1344, "metric": metric,
                         "validation": 1.0, "synthetic": 1.0, "boot_se": 1.0, "Z": float(z)})
    return pd.DataFrame(rows)


def test_fold_stability_gate_grades_the_worst_fold():
    ev = _fold_frame([[0.4, 1.2], [0.9, 2.1]])
    passed, _, value, tol = fold_stability_gate(GateEvidence(fold_evidence=ev))
    assert passed and value == pytest.approx(2.1) and tol == 3.0

    ev.loc[ev.index[-1], "Z"] = 7.4
    passed, detail, value, _ = fold_stability_gate(GateEvidence(fold_evidence=ev))
    assert not passed and value == pytest.approx(7.4)
    assert "expanding_2" in detail and "IDC_15m" in detail


def test_fold_stability_needs_more_than_one_origin():
    ev = _fold_frame([[0.1, 0.2]], folds=("expanding_1",))
    passed, detail, _, _ = fold_stability_gate(GateEvidence(fold_evidence=ev))
    assert not passed and "rolling-origin" in detail


def test_fold_evidence_separates_a_matched_model_from_a_mis_scaled_one():
    """The same model at twice the rate must score worse on every window."""
    cs = _toy_counts(n=96 * 120, seed=21)
    folds = expanding_window_folds(cs, validation_days=14, n_folds=3, min_train_days=30)
    model = IIDNegBin().fit(cs.values)
    thr = float(np.quantile(cs.values, 0.8))

    matched = evaluate_on_folds(model, folds, thr, n_paths=5, seed=1, boot_rep=40)
    mis_scaled = evaluate_on_folds(model, folds, thr, n_paths=5, seed=1, boot_rep=40, alpha=2.0)

    assert matched["fold"].nunique() >= 2
    assert mis_scaled["Z"].max() > matched["Z"].max()
    assert not fold_stability_gate(GateEvidence(fold_evidence=mis_scaled))[0]


def test_short_folds_are_not_graded_on_unsupported_scales():
    """A 14-day window holds two 7-day blocks: `scale_support` must drop that IDC."""
    cs = _toy_counts(n=96 * 120, seed=22)
    folds = expanding_window_folds(cs, validation_days=14, n_folds=3, min_train_days=30)
    model = IIDNegBin().fit(cs.values)
    graded = set(evaluate_on_folds(model, folds, 5.0, n_paths=3, seed=2, boot_rep=20)["metric"])
    assert "IDC_1d" in graded
    assert "IDC_4d" not in graded and "IDC_7d" not in graded


# gate 8: load-scaling drift
def _scaling_frame(**overrides) -> pd.DataFrame:
    rows = []
    for alpha in (0.5, 1.0, 2.0, 4.0, 8.0):
        expected = 1000.0 * alpha
        row = {
            # track the frozen rule: gate 8 selects the half of the report
            # matching GateEvidence.kappa_rule, so a hardcoded label here would
            # break this whole block if the rule were ever refrozen
            "kappa_rule": GateEvidence().kappa_rule, "alpha": alpha, "n_paths": 10,
            "expected_total": expected, "mean_total": expected, "sd_total": 0.2 * expected,
            "IDC_15m_drift_log": 0.1, "IDC_1d_drift_log": 0.1,
            "ACF_lag1_drift": 0.01, "ACF_lag96_drift": 0.01, "p_exceed_slots": 0.0,
        }
        row.update({k: (v(alpha) if callable(v) else v) for k, v in overrides.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def test_scaling_drift_gate_passes_a_flat_report():
    passed, detail, ratio, _ = scaling_drift_gate(GateEvidence(scaling=_scaling_frame()))
    assert passed and ratio < 1.0 and GateEvidence().kappa_rule in detail


def test_scaling_drift_gate_fails_inside_the_operating_range():
    report = _scaling_frame(IDC_1d_drift_log=lambda a: 2.5 if a == 2.0 else 0.1)
    passed, detail, ratio, threshold = scaling_drift_gate(GateEvidence(scaling=report))
    assert not passed and ratio == pytest.approx(2.5) and threshold == 1.0
    assert "IDC_1d_drift_log" in detail and "alpha=2" in detail


def test_scaling_drift_outside_the_declared_range_is_documented_not_failed():
    """A.2 predicts drift once the latent dominates; alpha = 8 is out of range."""
    report = _scaling_frame(IDC_15m_drift_log=lambda a: 3.0 if a == 8.0 else 0.1)
    passed, detail, _, _ = scaling_drift_gate(GateEvidence(scaling=report))
    assert passed, "drift beyond the operating range must not be silently graded"
    assert "full sweep 3.00" in detail, "but it must still be reported"


def test_scaling_drift_gate_judges_rate_error_against_monte_carlo_noise():
    within_noise = _scaling_frame(mean_total=lambda a: 1000.0 * a * 1.05)
    assert scaling_drift_gate(GateEvidence(scaling=within_noise))[0]
    biased = _scaling_frame(mean_total=lambda a: 1000.0 * a * 1.30, sd_total=lambda a: 20.0 * a)
    passed, detail, _, _ = scaling_drift_gate(GateEvidence(scaling=biased))
    assert not passed and "rate_Z" in detail


def test_scaling_drift_gate_prefers_the_path_averaged_drift():
    """One path's day-scale IDC is mostly sampling error; the mean is graded."""
    report = _scaling_frame(IDC_1d_drift_log=lambda a: 2.5 if a == 2.0 else 0.1)
    report["IDC_1d_drift_log_mean"] = 0.2
    passed, detail, ratio, _ = scaling_drift_gate(GateEvidence(scaling=report))
    assert passed and ratio < 1.0
    assert "IDC_1d_drift_log_mean" in detail


def test_scaling_drift_gate_refuses_a_report_without_the_error_column():
    report = _scaling_frame().drop(columns=["n_paths"])
    passed, detail, _, _ = scaling_drift_gate(GateEvidence(scaling=report))
    assert not passed and "regenerate" in detail


def test_gates_7_and_8_never_pass_without_evidence():
    """The stub they replace passed unconditionally; absence must now fail."""
    cs = _toy_counts(n=96 * 60, seed=23)
    model = IIDNegBin().fit(cs.values)
    rng = np.random.default_rng(24)
    paths = [model.simulate(len(cs), rng) for _ in range(5)]
    env = bootstrap_metric_envelope(cs.values, lambda x: metric_vector(x, 5.0), n_rep=30, seed=25)

    report = evaluate_gates(cs.values, paths, 5.0, env, stop_at_first_failure=False)
    graded = {g.name: g for g in report.results}
    assert len(graded) == 8, "every gate must be reported when nothing stops the sequence"
    for name in ("7_fold_stability", "8_scaling_drift"):
        assert not graded[name].passed, f"{name} passed with no evidence behind it"
        assert "not evaluated" in graded[name].detail
    assert not report.passed


# appendix A.4: the marginal feasibility result
def test_trace_marginals_are_infeasible_for_a_gaussian_log_latent():
    table = marginal_feasibility(mean=3.747985, variance=274.206902, zero_fraction=0.774655, target_acf=0.665)
    assert not table.attrs["feasible"], "no (V, kappa) should satisfy both the zero fraction and the ACF"
    # the two requirements pull in opposite directions along the constraint curve
    assert table.loc[table["zero_ok"], "max_acf"].max() < 0.05
    assert table.loc[table["acf_ok"], "P_zero"].max() < 0.55


def test_feasibility_accepts_a_case_that_really_is_feasible():
    """Guard against a diagnostic that always says 'infeasible'."""
    rng = np.random.default_rng(13)
    model = NBLatentAR(1)
    sigma = np.array([1.0])
    model.params_ = {
        "mu": np.log(3.0), "mean": 3.0, "sigma": sigma, "phi": np.array([half_life_to_phi(20.0)]),
        "half_life_bins": np.array([20.0]), "kappa": 5.0, "V": 1.0,
    }
    x = np.concatenate([model.simulate(60_000, rng) for _ in range(3)])
    table = marginal_feasibility(
        mean=float(x.mean()), variance=float(x.var(ddof=1)),
        zero_fraction=float((x == 0).mean()), target_acf=float(acf(x, [1])[1]),
    )
    assert table.attrs["feasible"], "data generated by the family must be judged feasible"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
