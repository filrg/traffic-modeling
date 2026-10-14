from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import DEFAULT_METRICS, MetricSpec, metric_vector, z_table


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    value: float = np.nan
    threshold: float = np.nan


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.results)

    @property
    def first_failure(self) -> GateResult | None:
        for g in self.results:
            if not g.passed:
                return g
        return None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([g.__dict__ for g in self.results])


@dataclass
class GateEvidence:
    """Evidence and tolerances the gates need beyond one metric comparison.

    Missing evidence fails its gate rather than passing it. `idc_envelope` is the
    exception: gate 3 falls back to the single shared envelope when none is given.

    The tolerances are declared requirements, not fitted quantities.
    `alpha_range` is the load range the model card claims; `idc_drift_tol` bounds
    second-order distortion across that range.
    """

    idc_envelope: pd.DataFrame | None = None
    fold_evidence: pd.DataFrame | None = None
    scaling: pd.DataFrame | None = None
    min_folds: int = 2
    kappa_rule: str = "proportional"
    alpha_range: tuple[float, float] = (0.5, 2.0)
    idc_drift_tol: float = 1.0
    acf_drift_tol: float = 0.10
    rate_z_tol: float = 3.0
    collision_tol: float = 1e-3


def fold_stability_gate(
    evidence: GateEvidence, z_tol: float = 3.0
) -> tuple[bool, str, float, float]:
    """Grade gate 7: no validation window is reproduced materially worse than the rest."""
    ev = evidence.fold_evidence
    if ev is None or len(ev) == 0:
        return False, "not evaluated: no fold evidence (see evaluate_on_folds)", np.nan, z_tol
    n_folds = int(ev["fold"].nunique())
    if n_folds < evidence.min_folds:
        return (
            False,
            f"{n_folds} fold(s), rolling-origin stability needs >= {evidence.min_folds}",
            np.nan,
            z_tol,
        )
    finite = ev.loc[np.isfinite(ev["Z"].to_numpy(dtype=float))].reset_index(drop=True)
    if finite.empty:
        return (
            False,
            f"{n_folds} folds but no finite Z: metrics undefined on these windows",
            np.nan,
            z_tol,
        )
    worst = finite.loc[finite["Z"].idxmax()]
    z = float(worst["Z"])
    detail = f"{n_folds} folds, worst Z {z:.2f} ({worst['metric']} on {worst['fold']})"
    return z <= z_tol, detail, z, z_tol


def scaling_drift_gate(evidence: GateEvidence) -> tuple[bool, str, float, float]:
    """Grade gate 8: load scaling stays linear in rate and bounded in second-order drift.

    Rate linearity is judged against the report's own Monte Carlo error. IDC/ACF
    drift and collision risk are judged inside `alpha_range` only; the detail
    string reports the full sweep regardless.
    """
    rep = evidence.scaling
    if rep is None or len(rep) == 0:
        return False, "not evaluated: no scaling report (see scaling.scaling_report)", np.nan, 1.0
    sub = rep.loc[rep["kappa_rule"] == evidence.kappa_rule]
    if sub.empty:
        return False, f"scaling report carries no kappa_rule '{evidence.kappa_rule}'", np.nan, 1.0
    if "n_paths" not in sub.columns:
        return False, "scaling report predates the n_paths column; regenerate it", np.nan, 1.0

    lo, hi = evidence.alpha_range
    band = sub.loc[(sub["alpha"] >= lo) & (sub["alpha"] <= hi)]
    if band.empty:
        return False, f"no alpha in the operating range [{lo}, {hi}]", np.nan, 1.0

    checks: list[tuple[str, float, float, float]] = []
    se = sub["sd_total"].to_numpy(dtype=float) / np.sqrt(sub["n_paths"].to_numpy(dtype=float))
    z_rate = np.abs(
        sub["mean_total"].to_numpy(dtype=float) - sub["expected_total"].to_numpy(dtype=float)
    )
    z_rate = z_rate / np.maximum(se, 1e-12)
    # rate linearity is required over the whole sweep, not only `alpha_range`
    checks.append(
        (
            "rate_Z",
            float(np.nanmax(z_rate)),
            evidence.rate_z_tol,
            float(sub["alpha"].to_numpy()[int(np.nanargmax(z_rate))]),
        )
    )
    for base_col, tol in (
        ("IDC_15m_drift_log", evidence.idc_drift_tol),
        ("IDC_1d_drift_log", evidence.idc_drift_tol),
        ("ACF_lag1_drift", evidence.acf_drift_tol),
        ("ACF_lag96_drift", evidence.acf_drift_tol),
        ("p_exceed_slots", evidence.collision_tol),
    ):
        # prefer the path-averaged column; the single-path one is mostly noise
        col = f"{base_col}_mean" if f"{base_col}_mean" in band.columns else base_col
        if col not in band.columns:
            continue
        vals = np.abs(band[col].to_numpy(dtype=float))
        if not np.isfinite(vals).any():
            continue
        j = int(np.nanargmax(vals))
        checks.append((col, float(vals[j]), float(tol), float(band["alpha"].to_numpy()[j])))

    ratios = [value / tol if tol > 0 else np.inf for _, value, tol, _ in checks]
    worst_i = int(np.argmax(ratios))
    name, value, tol, alpha = checks[worst_i]
    full = float(np.nanmax(np.abs(sub["IDC_15m_drift_log"].to_numpy(dtype=float))))
    detail = (
        f"rule={evidence.kappa_rule}, alpha in [{lo}, {hi}]: worst {name} {value:.3g} "
        f"vs tol {tol:.3g} at alpha={alpha:g}; |IDC_15m drift| over the full sweep {full:.2f}"
    )
    return ratios[worst_i] <= 1.0, detail, float(ratios[worst_i]), 1.0


def _result(
    name: str,
    passed: bool,
    detail: str,
    value: float = np.nan,
    threshold: float = np.nan,
) -> GateResult:
    return GateResult(name, bool(passed), detail, float(value), float(threshold))


def _expected_rate(
    synthetic: pd.Series,
    trace: pd.Series,
    tolerance: float,
) -> GateResult:
    error = abs(synthetic["mean"] / trace["mean"] - 1.0)
    return _result(
        "1_expected_rate",
        error <= tolerance,
        f"relative rate error {error:.4f}",
        error,
        tolerance,
    )


def _realised_rate(
    paths: Sequence[np.ndarray],
    trace: pd.Series,
    envelope: pd.DataFrame,
    z_tolerance: float,
) -> GateResult:
    means = np.array([path.mean() for path in paths])
    standard_error = (
        float(np.asarray(envelope.loc["mean", "se"], dtype=float))
        if "mean" in envelope.index
        else np.nan
    )
    z_score = abs(means.mean() - trace["mean"]) / max(standard_error, 1e-12)
    return _result(
        "2_realised_rate_distribution",
        z_score <= z_tolerance,
        f"Z = {z_score:.2f} against bootstrap SE",
        z_score,
        z_tolerance,
    )


def _idc_envelope(
    synthetic: pd.Series,
    envelope: pd.DataFrame,
    z_scores: pd.DataFrame,
    evidence: GateEvidence,
) -> GateResult:
    scale_matched = evidence.idc_envelope
    if scale_matched is not None and len(scale_matched):
        graded = [
            scale
            for scale in scale_matched.index
            if bool(scale_matched.loc[scale, "supported"]) and f"IDC_{scale}" in synthetic.index
        ]
        inside = [
            scale_matched.loc[scale, "q025"]
            <= synthetic[f"IDC_{scale}"]
            <= scale_matched.loc[scale, "q975"]
            for scale in graded
        ]
        skipped = [
            scale
            for scale in scale_matched.index
            if not bool(scale_matched.loc[scale, "supported"])
        ]
        note = f" (scale-matched blocks; ungraded: {', '.join(skipped) or 'none'})"
    else:
        rows = [metric for metric in z_scores.index if metric.startswith("IDC_")]
        inside = [
            envelope.loc[metric, "q025"] <= synthetic[metric] <= envelope.loc[metric, "q975"]
            for metric in rows
            if metric in envelope.index
        ]
        note = " (single shared envelope)"
    fraction = float(np.mean(inside)) if inside else 0.0
    return _result(
        "3_idc_envelope",
        fraction >= 0.6,
        f"{fraction:.0%} of {len(inside)} scales inside envelope{note}",
        fraction,
        0.6,
    )


def _acf_bias(
    synthetic: pd.Series,
    trace: pd.Series,
    z_scores: pd.DataFrame,
    z_tolerance: float,
) -> GateResult:
    rows = [metric for metric in z_scores.index if metric.startswith("ACF_")]
    signed = np.array([synthetic[metric] - trace[metric] for metric in rows])
    systematic = (
        bool(np.all(signed > 0) or np.all(signed < 0)) and float(np.mean(np.abs(signed))) > 0.05
    )
    worst = float(z_scores.loc[rows, "Z"].max()) if rows else np.nan
    return _result(
        "4_acf_no_systematic_bias",
        (not systematic) and worst <= z_tolerance,
        f"worst ACF Z {worst:.2f}, systematic={systematic}",
        worst,
        z_tolerance,
    )


def _episodes(z_scores: pd.DataFrame, z_tolerance: float) -> GateResult:
    rows = [
        metric
        for metric in z_scores.index
        if metric.startswith(("busy_", "quiet_", "roll", "zero_bin"))
    ]
    worst = float(z_scores.loc[rows, "Z"].max()) if rows else np.nan
    return _result(
        "5_episodes_and_tails",
        worst <= z_tolerance,
        f"worst episode/tail Z {worst:.2f}",
        worst,
        z_tolerance,
    )


def _spectrum(z_scores: pd.DataFrame, z_tolerance: float) -> GateResult:
    rows = [metric for metric in z_scores.index if metric.startswith("wv_level")]
    worst = float(z_scores.loc[rows, "Z"].max()) if rows else np.nan
    threshold = 2 * z_tolerance
    return _result(
        "6_spectrum_and_totals",
        worst <= threshold,
        f"worst wavelet-variance Z {worst:.2f}",
        worst,
        threshold,
    )


def _measurer(busy_threshold: float, metrics: MetricSpec) -> Callable[[np.ndarray], pd.Series]:
    def measure(x: np.ndarray) -> pd.Series:
        return metric_vector(
            x,
            busy_threshold,
            quiet_threshold=metrics.quiet_threshold,
            acf_lags=metrics.acf_lags,
            rolling_windows=metrics.rolling_windows,
            scales=metrics.scales,
        )

    return measure


def evaluate_gates(
    trace_counts: np.ndarray,
    syn_paths: Sequence[np.ndarray],
    busy_threshold: float,
    envelope: pd.DataFrame,
    z_tol: float = 3.0,
    rate_tol: float = 0.05,
    stop_at_first_failure: bool = True,
    evidence: GateEvidence | None = None,
    metrics: MetricSpec = DEFAULT_METRICS,
) -> GateReport:
    """Run the eight structural gates in order."""
    evidence = evidence or GateEvidence()
    trace_counts = np.asarray(trace_counts, dtype=float)
    paths = [np.asarray(path, dtype=float) for path in syn_paths]
    measure = _measurer(busy_threshold, metrics)
    synthetic = pd.DataFrame([measure(path) for path in paths]).mean()
    trace = measure(trace_counts)
    z_scores = z_table(synthetic, trace, envelope["se"])

    first_six = (
        _expected_rate(synthetic, trace, rate_tol),
        _realised_rate(paths, trace, envelope, z_tol),
        _idc_envelope(synthetic, envelope, z_scores, evidence),
        _acf_bias(synthetic, trace, z_scores, z_tol),
        _episodes(z_scores, z_tol),
        _spectrum(z_scores, z_tol),
    )
    report = GateReport()
    for result in first_six:
        report.results.append(result)
        if not result.passed and stop_at_first_failure:
            return report

    report.results.append(_result("7_fold_stability", *fold_stability_gate(evidence, z_tol)))
    if not report.results[-1].passed and stop_at_first_failure:
        return report
    report.results.append(_result("8_scaling_drift", *scaling_drift_gate(evidence)))
    return report
