from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType

from traffic_modeling.core import (
    AblationSpec,
    CandidateSpec,
    EstimationMethod,
    ModelRef,
    model_spec_registry,
)

from ..metrics import acf
from ..models import CountModel, IIDNegBin, NBGatedLatent, NBLatentAR
from ..state import PipelineState
from .config import LONG_ACF_LAGS, MM4_KNOBS

ModelBuilder = Callable[[PipelineState, int], CountModel]

_CANDIDATES = (
    CandidateSpec(
        "r2",
        ("r2",),
        ModelRef("count-first", "gli-nb", "iid-nb"),
        2,
        EstimationMethod.LIKELIHOOD,
    ),
    CandidateSpec(
        "r3",
        ("r3",),
        ModelRef("count-first", "gli-nb", "latent-ar1"),
        3,
        EstimationMethod.LIKELIHOOD,
    ),
    CandidateSpec(
        "r4",
        ("r4",),
        ModelRef("count-first", "gli-nb", "latent-ar2"),
        4,
        EstimationMethod.LIKELIHOOD,
    ),
    CandidateSpec(
        "gated",
        ("gated",),
        ModelRef("count-first", "gli-nb", "gated-likelihood"),
        "4b",
        EstimationMethod.LIKELIHOOD,
    ),
    CandidateSpec(
        "gated_mm",
        ("mm", "gated_mm"),
        ModelRef("count-first", "gli-nb", "gated-mm"),
        "4b",
        EstimationMethod.ANALYTIC_MOMENT,
    ),
    CandidateSpec(
        "gated_mm2",
        ("mm2", "gated_mm2"),
        ModelRef("count-first", "gli-nb", "gated-mm2"),
        "4b",
        EstimationMethod.ANALYTIC_MOMENT,
    ),
    CandidateSpec(
        "gated_mm3",
        ("mm3", "gated_mm3"),
        ModelRef("count-first", "gli-nb", "gated-mm3"),
        "4b",
        EstimationMethod.FINITE_SAMPLE_MOMENT,
    ),
    CandidateSpec(
        "gated_mm4",
        ("mm4", "gated_mm4"),
        ModelRef("count-first", "gli-nb", "gated-mm4"),
        "4b",
        EstimationMethod.FINITE_SAMPLE_MOMENT,
    ),
    CandidateSpec(
        "gated_mm5",
        ("mm5", "gated_mm5"),
        ModelRef("count-first", "gli-nb", "gated-mm5"),
        "4b",
        EstimationMethod.FINITE_SAMPLE_MOMENT,
    ),
)

_ABLATIONS = (
    AblationSpec(
        "gated_nogate",
        ("nogate",),
        ModelRef("count-first", "gli-nb", "gated-mm4-no-gate"),
        "4b",
        EstimationMethod.FINITE_SAMPLE_MOMENT,
        parent_key="gated_mm4",
        intervention="Set gate sharpness to zero while preserving the mm4 calibration.",
    ),
)

CANDIDATE_SPECS = model_spec_registry(_CANDIDATES)
ABLATION_SPECS = model_spec_registry(_ABLATIONS)
CANDIDATE_KEYS = tuple(CANDIDATE_SPECS)

_GATED_NAMES = {
    "gated_mm": "nb_gated_mm",
    "gated_mm2": "nb_gated_mm2",
    "gated_mm3": "nb_gated_mm3",
    "gated_mm4": "nb_gated_mm4",
    "gated_mm5": "nb_gated_mm5",
    "gated_nogate": "nb_gated_mm4_nogate",
}


def _fit_r2(state: PipelineState, _seed: int) -> CountModel:
    return IIDNegBin().fit(state["counts"].values)


def _fit_r3(state: PipelineState, _seed: int) -> CountModel:
    return NBLatentAR(1).fit(state["counts"].values, n_restarts=2, maxiter=250)


def _fit_r4(state: PipelineState, _seed: int) -> CountModel:
    return NBLatentAR(2).fit(state["counts"].values, n_restarts=2, maxiter=250)


def _fit_gated(state: PipelineState, seed: int) -> CountModel:
    return NBGatedLatent().fit(state["counts"].values, n_restarts=1, maxiter=80, seed=seed)


def _calibrate_gated(state: PipelineState, seed: int, *, variant: str) -> CountModel:
    counts = state["counts"].values
    marginals = state["trace_marginals"]
    mean = marginals["mean"]
    variance = marginals["IDC_15m"] * mean
    zero_fraction = marginals["zero_frac"]
    acf_1h = marginals["acf_1h_lag1"]
    if variant == "gated_mm":
        model = NBGatedLatent.calibrate(mean, variance, zero_fraction, acf_1h)
    else:
        long_acf = {int(h): float(v) for h, v in acf(counts, LONG_ACF_LAGS).items()}
        print("   long-lag ACF targets:", {k: round(v, 4) for k, v in long_acf.items()})
        if variant == "gated_mm2":
            model = NBGatedLatent.calibrate(
                mean, variance, zero_fraction, acf_1h, acf_long=long_acf
            )
        else:
            debias_args = (mean, variance, zero_fraction, acf_1h, long_acf, state["n_bins"])
            if variant in {"gated_mm4", "gated_mm5", "gated_nogate"}:
                model = NBGatedLatent.calibrate_debiased(
                    *debias_args,
                    seed=seed,
                    marginals=(variant == "gated_mm5"),
                    sharpness=0.0 if variant == "gated_nogate" else 4.0,
                    kappa=MM4_KNOBS["kappa"],
                    sigma_fast=MM4_KNOBS["sigma_fast"],
                    tau_fast_bins=MM4_KNOBS["tau_fast_bins"],
                )
            else:
                model = NBGatedLatent.calibrate_debiased(*debias_args, seed=seed)
    model.name = _GATED_NAMES[variant]
    return model


CANDIDATE_BUILDERS: Mapping[str, ModelBuilder] = MappingProxyType(
    {
        "r2": _fit_r2,
        "r3": _fit_r3,
        "r4": _fit_r4,
        "gated": _fit_gated,
        "gated_mm": partial(_calibrate_gated, variant="gated_mm"),
        "gated_mm2": partial(_calibrate_gated, variant="gated_mm2"),
        "gated_mm3": partial(_calibrate_gated, variant="gated_mm3"),
        "gated_mm4": partial(_calibrate_gated, variant="gated_mm4"),
        "gated_mm5": partial(_calibrate_gated, variant="gated_mm5"),
    }
)

ABLATION_BUILDERS: Mapping[str, ModelBuilder] = MappingProxyType(
    {"gated_nogate": partial(_calibrate_gated, variant="gated_nogate")}
)
