from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..metrics import metric_vector, z_table


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


def fold_stability_gate(evidence: GateEvidence,
                        z_tol: float = 3.0) -> tuple[bool, str, float, float]:
    """Grade gate 7: no validation window is reproduced materially worse than the rest."""
    ev = evidence.fold_evidence
    if ev is None or len(ev) == 0:
        return False, "not evaluated: no fold evidence (see evaluate_on_folds)", np.nan, z_tol
    n_folds = int(ev["fold"].nunique())
    if n_folds < evidence.min_folds:
        return False, f"{n_folds} fold(s), rolling-origin stability needs >= {evidence.min_folds}", np.nan, z_tol
    finite = ev.loc[np.isfinite(ev["Z"].to_numpy(dtype=float))].reset_index(drop=True)
    if finite.empty:
        return False, f"{n_folds} folds but no finite Z: metrics undefined on these windows", np.nan, z_tol
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

    checks: list[tuple] = []
    se = sub["sd_total"].to_numpy(dtype=float) / np.sqrt(sub["n_paths"].to_numpy(dtype=float))
    z_rate = np.abs(sub["mean_total"].to_numpy(dtype=float) - sub["expected_total"].to_numpy(dtype=float))
    z_rate = z_rate / np.maximum(se, 1e-12)
    # rate linearity is required over the whole sweep, not only `alpha_range`
    checks.append(("rate_Z", float(np.nanmax(z_rate)), evidence.rate_z_tol,
                   float(sub["alpha"].to_numpy()[int(np.nanargmax(z_rate))])))
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


def evaluate_gates(
    trace_counts: np.ndarray,
    syn_paths: Sequence[np.ndarray],
    busy_threshold: float,
    envelope: pd.DataFrame,
    z_tol: float = 3.0,
    rate_tol: float = 0.05,
    stop_at_first_failure: bool = True,
    evidence: GateEvidence | None = None,
) -> GateReport:
    """Run gates 1-8 in order; the scalar rate calibration unlocks only after all pass.

    Gates 7 and 8 are graded from `evidence`; without it they fail as
    unevaluated, so a report can never pass on gates that were never run.
    """
    evidence = evidence or GateEvidence()
    trace = np.asarray(trace_counts, dtype=float)
    syn = [np.asarray(p, dtype=float) for p in syn_paths]
    syn_metrics = pd.DataFrame([metric_vector(p, busy_threshold) for p in syn]).mean()
    trace_metrics = metric_vector(trace, busy_threshold)
    zt = z_table(syn_metrics, trace_metrics, envelope["se"])
    report = GateReport()

    def add(name: str, passed: bool, detail: str, value=np.nan, threshold=np.nan) -> bool:
        report.results.append(GateResult(name, bool(passed), detail, float(value), float(threshold)))
        return bool(passed)

    # 1 expected rate
    err = abs(syn_metrics["mean"] / trace_metrics["mean"] - 1.0)
    ok = add("1_expected_rate", err <= rate_tol, f"relative rate error {err:.4f}", err, rate_tol)
    if not ok and stop_at_first_failure:
        return report

    # 2 realised-rate distribution against finite-horizon uncertainty
    means = np.array([p.mean() for p in syn])
    se = envelope.loc["mean", "se"] if "mean" in envelope.index else np.nan
    z = abs(means.mean() - trace_metrics["mean"]) / max(se, 1e-12)
    ok = add("2_realised_rate_distribution", z <= z_tol,
             f"Z = {z:.2f} against bootstrap SE", z, z_tol)
    if not ok and stop_at_first_failure:
        return report

    # 3 IDC curve inside the trace-bootstrap envelope
    sm = evidence.idc_envelope
    if sm is not None and len(sm):
        # scale-matched blocks; scales the trace cannot support are not graded
        graded = [s for s in sm.index if bool(sm.loc[s, "supported"]) and f"IDC_{s}" in syn_metrics.index]
        inside = [sm.loc[s, "q025"] <= syn_metrics[f"IDC_{s}"] <= sm.loc[s, "q975"] for s in graded]
        skipped = [s for s in sm.index if not bool(sm.loc[s, "supported"])]
        note = f" (scale-matched blocks; ungraded: {', '.join(skipped) or 'none'})"
    else:
        idc_rows = [m for m in zt.index if m.startswith("IDC_")]
        inside = [envelope.loc[m, "q025"] <= syn_metrics[m] <= envelope.loc[m, "q975"] for m in idc_rows
                  if m in envelope.index]
        note = " (single shared envelope)"
    frac = float(np.mean(inside)) if inside else 0.0
    ok = add("3_idc_envelope", frac >= 0.6,
             f"{frac:.0%} of {len(inside)} scales inside envelope{note}",
             frac, 0.6)
    if not ok and stop_at_first_failure:
        return report

    # 4 raw ACF without systematic bias
    acf_rows = [m for m in zt.index if m.startswith("ACF_")]
    signed = np.array([syn_metrics[m] - trace_metrics[m] for m in acf_rows])
    systematic = bool(np.all(signed > 0) or np.all(signed < 0)) and float(np.mean(np.abs(signed))) > 0.05
    worst_z = float(zt.loc[acf_rows, "Z"].max()) if acf_rows else np.nan
    ok = add("4_acf_no_systematic_bias", (not systematic) and worst_z <= z_tol,
             f"worst ACF Z {worst_z:.2f}, systematic={systematic}", worst_z, z_tol)
    if not ok and stop_at_first_failure:
        return report

    # 5 episodes and tails
    ep_rows = [m for m in zt.index if m.startswith(("busy_", "quiet_", "roll", "zero_bin"))]
    worst_ep = float(zt.loc[ep_rows, "Z"].max()) if ep_rows else np.nan
    ok = add("5_episodes_and_tails", worst_ep <= z_tol,
             f"worst episode/tail Z {worst_ep:.2f}", worst_ep, z_tol)
    if not ok and stop_at_first_failure:
        return report

    # 6 spectrum and horizon totals
    wv_rows = [m for m in zt.index if m.startswith("wv_level")]
    worst_wv = float(zt.loc[wv_rows, "Z"].max()) if wv_rows else np.nan
    ok = add("6_spectrum_and_totals", worst_wv <= 2 * z_tol,
             f"worst wavelet-variance Z {worst_wv:.2f}",
             worst_wv, 2 * z_tol)
    if not ok and stop_at_first_failure:
        return report

    # 7 stability across rolling-origin folds
    passed, detail, value, threshold = fold_stability_gate(evidence, z_tol)
    ok = add("7_fold_stability", passed, detail, value, threshold)
    if not ok and stop_at_first_failure:
        return report

    # 8 load-scaling drift
    add("8_scaling_drift", *scaling_drift_gate(evidence))
    return report
