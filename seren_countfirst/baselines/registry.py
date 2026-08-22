from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType

from traffic_modeling.core import BaselineSpec, EstimationMethod, ModelRef, model_spec_registry

from ..metrics import acf
from ..models import CountModel, NBINGARCH, ONOFFCounts
from ..state import PipelineState

ModelBuilder = Callable[[PipelineState, int], CountModel]

_BASELINES = (
    BaselineSpec(
        "onoff",
        ("onoff",),
        ModelRef("count-first", "onoff", "negative-binomial"),
        EstimationMethod.EMPIRICAL_MOMENT,
        citation="Classical two-state ON/OFF source model.",
        comparison_claim="Tests a discrete activity gate against the continuous latent gate.",
    ),
    BaselineSpec(
        "onoff_pois",
        ("onoff_pois",),
        ModelRef("count-first", "onoff", "poisson"),
        EstimationMethod.EMPIRICAL_MOMENT,
        citation="Classical two-state ON/OFF source model.",
        comparison_claim="Separates the activity chain from overdispersed ON-state emissions.",
    ),
    BaselineSpec(
        "ingarch",
        ("ingarch",),
        ModelRef("count-first", "nb-ingarch", "moment-calibrated"),
        EstimationMethod.ANALYTIC_MOMENT,
        citation="Ferland et al. (2006); Zhu (2011); Christou and Fokianos (2014).",
        comparison_claim="Tests observation-driven persistence against latent persistence.",
    ),
    BaselineSpec(
        "ingarch_mle",
        ("ingarch_mle",),
        ModelRef("count-first", "nb-ingarch", "likelihood-fitted"),
        EstimationMethod.LIKELIHOOD,
        citation="Ferland et al. (2006); Zhu (2011); Christou and Fokianos (2014).",
        comparison_claim="Provides the exact-likelihood INGARCH comparison and its AIC.",
    ),
)

BASELINE_SPECS = model_spec_registry(_BASELINES)


def _calibrate_onoff(state: PipelineState, _seed: int, *, emission: str) -> CountModel:
    return ONOFFCounts.calibrate(state["counts"].values, emission=emission)


def _calibrate_ingarch(state: PipelineState, seed: int) -> CountModel:
    counts = state["counts"].values
    marginals = state["trace_marginals"]
    trace_acf = acf(counts, (1, 4))
    return NBINGARCH.calibrate(
        mean=marginals["mean"],
        variance=marginals["IDC_15m"] * marginals["mean"],
        acf_lag1=float(trace_acf[1]),
        acf_lag4=float(trace_acf[4]),
        seed=seed,
    )


def _fit_ingarch(state: PipelineState, seed: int) -> CountModel:
    return NBINGARCH().fit(state["counts"].values, n_restarts=3, maxiter=400, seed=seed)


BASELINE_BUILDERS: Mapping[str, ModelBuilder] = MappingProxyType(
    {
        "onoff": partial(_calibrate_onoff, emission="nb"),
        "onoff_pois": partial(_calibrate_onoff, emission="poisson"),
        "ingarch": _calibrate_ingarch,
        "ingarch_mle": _fit_ingarch,
    }
)
