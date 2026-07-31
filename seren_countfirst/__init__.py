"""Count-first workload generator for the Seren SFT trace.

Pipeline (guide eq. 1):
    SFT submissions -> exact-timestamp batches -> regular batch-count series
    -> dynamic count model -> batch timestamps -> batch sizes -> job arrivals

Default configuration: no calendar, no residual layer, no conditional marks.
Each is an ablation that must earn its place on a contiguous holdout.
"""
from .data import (
    ActivityScheme,
    BASE_BIN_S,
    BASE_FREQ,
    BatchTable,
    CountSeries,
    SCALES,
    Split,
    arrival_view,
    build_batches,
    build_counts,
    contiguous_split,
    expanding_window_folds,
    hour_of_week,
    jobsize_view,
    load_trace,
    segment_profiles,
    sha256_file,
    zero_iat_fraction,
)
from .metrics import (
    HOURLY_ACF_LAG_WEIGHTS,
    acf,
    aggregate,
    base_marginals,
    d_acf,
    d_idc,
    d_idc_shape,
    episode_stats,
    idc,
    idc_curve,
    hourly_acf1_from_autocov,
    metric_vector,
    microstructure,
    quiet_run_vs_geometric,
    rolling_tails,
    run_lengths,
    theoretical_acf_two_ar,
    theoretical_idc_two_ar,
    wavelet_variance,
    z_table,
)
from .bootstrap import (
    BOOTSTRAPS,
    bootstrap_baseline_table,
    bootstrap_metric_envelope,
    scale_matched_idc_envelope,
)
from .models import (
    CalendarTerm,
    CountModel,
    IIDNegBin,
    NBLatentAR,
    NBRegimeSwitch,
    marginal_feasibility,
    nb2_sample,
)
from .models import NBGatedLatent
from .reconstruct import (
    OffsetTemplates,
    collision_risk,
    export_check,
    make_conditional_mark_sampler,
    make_iid_mark_sampler,
    reconstruct_events,
)
from .scaling import scaling_report, superposition_comparison, calibrate_intercept
from .evaluation import (
    FOLD_CORE_METRICS,
    GateEvidence,
    benchmark_table,
    evaluate_candidate,
    evaluate_gates,
    evaluate_on_folds,
    fold_stability_gate,
    horizon_distribution,
    parameter_recovery,
    scale_support,
    scaling_drift_gate,
)

__all__ = [
    # data
    "ActivityScheme",
    "BASE_BIN_S",
    "BASE_FREQ",
    "BatchTable",
    "CountSeries",
    "SCALES",
    "Split",
    "arrival_view",
    "build_batches",
    "build_counts",
    "contiguous_split",
    "expanding_window_folds",
    "hour_of_week",
    "jobsize_view",
    "load_trace",
    "segment_profiles",
    "sha256_file",
    "zero_iat_fraction",
    # metrics
    "HOURLY_ACF_LAG_WEIGHTS",
    "acf",
    "aggregate",
    "base_marginals",
    "d_acf",
    "d_idc",
    "d_idc_shape",
    "episode_stats",
    "hourly_acf1_from_autocov",
    "idc",
    "idc_curve",
    "metric_vector",
    "microstructure",
    "rolling_tails",
    "run_lengths",
    "theoretical_acf_two_ar",
    "theoretical_idc_two_ar",
    "wavelet_variance",
    "z_table",
    # bootstrap
    "BOOTSTRAPS",
    "bootstrap_baseline_table",
    "bootstrap_metric_envelope",
    "scale_matched_idc_envelope",
    # models
    "CalendarTerm",
    "CountModel",
    "IIDNegBin",
    "NBGatedLatent",
    "NBLatentAR",
    "NBRegimeSwitch",
    "marginal_feasibility",
    "nb2_sample",
    "quiet_run_vs_geometric",
    # reconstruction
    "OffsetTemplates",
    "collision_risk",
    "export_check",
    "make_conditional_mark_sampler",
    "make_iid_mark_sampler",
    "reconstruct_events",
    # scaling and evaluation
    "calibrate_intercept",
    "scaling_report",
    "superposition_comparison",
    "FOLD_CORE_METRICS",
    "GateEvidence",
    "benchmark_table",
    "evaluate_candidate",
    "evaluate_gates",
    "evaluate_on_folds",
    "fold_stability_gate",
    "horizon_distribution",
    "parameter_recovery",
    "scale_support",
    "scaling_drift_gate",
]
__version__ = "0.1.0"
