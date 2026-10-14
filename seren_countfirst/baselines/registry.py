from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType

from traffic_modeling.core import BaselineSpec, EstimationMethod, ModelRef, model_spec_registry
from traffic_modeling.counts.metrics import acf

from ..candidates import LONG_ACF_LAGS
from ..models.base import CountModel
from ..state import PipelineState
from .ingarch import NBINGARCH
from .onoff import ONOFFCounts

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
        "onoff_long",
        ("onoff_long",),
        ModelRef("count-first", "onoff", "negative-binomial-long-lag"),
        EstimationMethod.FINITE_SAMPLE_MOMENT,
        citation="Classical two-state ON/OFF source model.",
        comparison_claim="Gives the discrete activity gate the GLI-NB calibration targets.",
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
        "ingarch_long",
        ("ingarch_long",),
        ModelRef("count-first", "nb-ingarch", "long-lag-calibrated"),
        EstimationMethod.FINITE_SAMPLE_MOMENT,
        citation="Ferland et al. (2006); Zhu (2011); Christou and Fokianos (2014).",
        comparison_claim="Gives observation-driven persistence the GLI-NB calibration targets.",
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


def _calibrate_long_lag(
    state: PipelineState, seed: int, *, model: type[ONOFFCounts] | type[NBINGARCH]
) -> CountModel:
    """Calibrate to the GLI-NB targets, long-lag correction and record length included."""
    counts = state["counts"].values
    marginals = state["trace_marginals"]
    mean = marginals["mean"]
    acf_long = {int(lag): float(value) for lag, value in acf(counts, LONG_ACF_LAGS).items()}
    return model.calibrate_debiased(
        mean,
        marginals["IDC_15m"] * mean,
        marginals["zero_frac"],
        marginals["acf_1h_lag1"],
        acf_long,
        state["n_bins"],
        seed=seed,
    )


BASELINE_BUILDERS: Mapping[str, ModelBuilder] = MappingProxyType(
    {
        "onoff": partial(_calibrate_onoff, emission="nb"),
        "onoff_pois": partial(_calibrate_onoff, emission="poisson"),
        "onoff_long": partial(_calibrate_long_lag, model=ONOFFCounts),
        "ingarch": _calibrate_ingarch,
        "ingarch_mle": _fit_ingarch,
        "ingarch_long": partial(_calibrate_long_lag, model=NBINGARCH),
    }
)
