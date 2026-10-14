from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from math import isfinite
from types import MappingProxyType

import numpy as np

from traffic_modeling.core import (
    AblationSpec,
    CandidateSpec,
    EstimationMethod,
    ModelRef,
    model_spec_registry,
)
from traffic_modeling.counts.metrics import acf

from ..models import CountModel, IIDNegBin, NBGatedLatent, NBLatentAR
from ..state import PipelineState

ModelBuilder = Callable[[PipelineState, int], CountModel]

LONG_ACF_LAGS = (24, 48, 96, 192, 288, 384, 672)


class GLINBImplementation(StrEnum):
    """Concrete model structure used by a GLI-NB variant."""

    IID = "iid"
    LATENT_AR = "latent-ar"
    GATED_LATENT = "gated-latent"


class CalibrationRoute(StrEnum):
    """Supported parameter-estimation routes for GLI-NB."""

    LIKELIHOOD = "likelihood"
    ANALYTIC_MOMENT = "analytic-moment"
    ANALYTIC_LONG_ACF = "analytic-long-acf"
    FINITE_SAMPLE_MOMENT = "finite-sample-moment"

    @property
    def estimation(self) -> EstimationMethod:
        if self is CalibrationRoute.LIKELIHOOD:
            return EstimationMethod.LIKELIHOOD
        if self in {CalibrationRoute.ANALYTIC_MOMENT, CalibrationRoute.ANALYTIC_LONG_ACF}:
            return EstimationMethod.ANALYTIC_MOMENT
        return EstimationMethod.FINITE_SAMPLE_MOMENT


@dataclass(frozen=True, slots=True)
class BurstTexture:
    """Fixed fast-scale parameters used by finite-sample calibration."""

    kappa: float
    sigma_fast: float
    tau_fast_bins: float

    def __post_init__(self) -> None:
        for field in ("kappa", "sigma_fast", "tau_fast_bins"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field} must be a positive number")

    def to_dict(self) -> dict[str, float]:
        return {
            "kappa": float(self.kappa),
            "sigma_fast": float(self.sigma_fast),
            "tau_fast_bins": float(self.tau_fast_bins),
        }


MM4_TEXTURE = BurstTexture(kappa=40.0, sigma_fast=0.60, tau_fast_bins=0.5)


@dataclass(frozen=True, slots=True)
class GLINBVariantConfig:
    """Validated immutable build configuration for one GLI-NB variant."""

    key: str
    aliases: tuple[str, ...]
    reference_variant: str
    model_name: str
    ladder_rung: int | str
    implementation: GLINBImplementation
    calibration: CalibrationRoute
    n_components: int | None = None
    fit_restarts: int | None = None
    fit_maxiter: int | None = None
    sharpness: float | None = None
    texture: BurstTexture | None = None
    debias_marginals: bool = False
    parent_key: str | None = None
    intervention: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, GLINBImplementation):
            raise TypeError("implementation must be a GLINBImplementation")
        if not isinstance(self.calibration, CalibrationRoute):
            raise TypeError("calibration must be a CalibrationRoute")
        if self.texture is not None and not isinstance(self.texture, BurstTexture):
            raise TypeError("texture must be a BurstTexture")
        if not isinstance(self.debias_marginals, bool):
            raise TypeError("debias_marginals must be a boolean")
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if (self.parent_key is None) != (self.intervention is None):
            raise ValueError("parent_key and intervention must be declared together")
        self._validate_implementation()

    @property
    def is_ablation(self) -> bool:
        return self.parent_key is not None

    @property
    def estimation(self) -> EstimationMethod:
        return self.calibration.estimation

    def _validate_implementation(self) -> None:
        has_budget = self.fit_restarts is not None or self.fit_maxiter is not None
        if has_budget and (self.fit_restarts is None or self.fit_maxiter is None):
            raise ValueError("fit_restarts and fit_maxiter must be declared together")
        for field in ("fit_restarts", "fit_maxiter"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{field} must be a positive integer")

        if self.implementation is GLINBImplementation.IID:
            if self.calibration is not CalibrationRoute.LIKELIHOOD:
                raise ValueError("IID GLI-NB must use likelihood estimation")
            if (
                any(
                    value is not None
                    for value in (
                        self.n_components,
                        self.fit_restarts,
                        self.fit_maxiter,
                        self.sharpness,
                        self.texture,
                    )
                )
                or self.debias_marginals
            ):
                raise ValueError("IID GLI-NB does not accept latent or gated options")
            return

        if self.implementation is GLINBImplementation.LATENT_AR:
            if self.calibration is not CalibrationRoute.LIKELIHOOD:
                raise ValueError("latent-AR GLI-NB must use likelihood estimation")
            if (
                self.n_components is None
                or isinstance(self.n_components, bool)
                or not isinstance(self.n_components, int)
                or self.n_components < 1
            ):
                raise ValueError("latent-AR GLI-NB requires n_components >= 1")
            if not has_budget:
                raise ValueError("latent-AR GLI-NB requires a fit budget")
            if self.sharpness is not None or self.texture is not None or self.debias_marginals:
                raise ValueError("latent-AR GLI-NB does not accept gated options")
            return

        if self.n_components is not None:
            raise ValueError("gated GLI-NB does not accept n_components")
        if (
            self.sharpness is None
            or isinstance(self.sharpness, bool)
            or not isinstance(self.sharpness, (int, float))
            or not isfinite(self.sharpness)
            or self.sharpness < 0
        ):
            raise ValueError("gated GLI-NB requires non-negative sharpness")
        if self.calibration is CalibrationRoute.LIKELIHOOD:
            if not has_budget:
                raise ValueError("likelihood-fitted gated GLI-NB requires a fit budget")
        elif has_budget:
            raise ValueError("moment-calibrated gated GLI-NB does not accept a fit budget")
        if (
            self.texture is not None
            and self.calibration is not CalibrationRoute.FINITE_SAMPLE_MOMENT
        ):
            raise ValueError("texture overrides require finite-sample calibration")
        if self.debias_marginals and self.calibration is not CalibrationRoute.FINITE_SAMPLE_MOMENT:
            raise ValueError("marginal debiasing requires finite-sample calibration")


_VARIANTS = (
    GLINBVariantConfig(
        "r2",
        ("r2",),
        "iid-nb",
        "iid_nb",
        2,
        GLINBImplementation.IID,
        CalibrationRoute.LIKELIHOOD,
    ),
    GLINBVariantConfig(
        "r3",
        ("r3",),
        "latent-ar1",
        "nb_ar1",
        3,
        GLINBImplementation.LATENT_AR,
        CalibrationRoute.LIKELIHOOD,
        n_components=1,
        fit_restarts=2,
        fit_maxiter=250,
    ),
    GLINBVariantConfig(
        "r4",
        ("r4",),
        "latent-ar2",
        "nb_ar2",
        4,
        GLINBImplementation.LATENT_AR,
        CalibrationRoute.LIKELIHOOD,
        n_components=2,
        fit_restarts=2,
        fit_maxiter=250,
    ),
    GLINBVariantConfig(
        "gated",
        ("gated",),
        "gated-likelihood",
        "nb_gated",
        "4b",
        GLINBImplementation.GATED_LATENT,
        CalibrationRoute.LIKELIHOOD,
        fit_restarts=1,
        fit_maxiter=80,
        sharpness=4.0,
    ),
    GLINBVariantConfig(
        "gated_mm",
        ("mm", "gated_mm"),
        "gated-mm",
        "nb_gated_mm",
        "4b",
        GLINBImplementation.GATED_LATENT,
        CalibrationRoute.ANALYTIC_MOMENT,
        sharpness=4.0,
    ),
    GLINBVariantConfig(
        "gated_mm2",
        ("mm2", "gated_mm2"),
        "gated-mm2",
        "nb_gated_mm2",
        "4b",
        GLINBImplementation.GATED_LATENT,
        CalibrationRoute.ANALYTIC_LONG_ACF,
        sharpness=4.0,
    ),
    GLINBVariantConfig(
        "gated_mm3",
        ("mm3", "gated_mm3"),
        "gated-mm3",
        "nb_gated_mm3",
        "4b",
        GLINBImplementation.GATED_LATENT,
        CalibrationRoute.FINITE_SAMPLE_MOMENT,
        sharpness=4.0,
    ),
    GLINBVariantConfig(
        "gated_mm4",
        ("mm4", "gated_mm4"),
        "gated-mm4",
        "nb_gated_mm4",
        "4b",
        GLINBImplementation.GATED_LATENT,
        CalibrationRoute.FINITE_SAMPLE_MOMENT,
        sharpness=4.0,
        texture=MM4_TEXTURE,
    ),
    GLINBVariantConfig(
        "gated_mm5",
        ("mm5", "gated_mm5"),
        "gated-mm5",
        "nb_gated_mm5",
        "4b",
        GLINBImplementation.GATED_LATENT,
        CalibrationRoute.FINITE_SAMPLE_MOMENT,
        sharpness=4.0,
        texture=MM4_TEXTURE,
        debias_marginals=True,
    ),
    GLINBVariantConfig(
        "gated_nogate",
        ("nogate",),
        "gated-mm4-no-gate",
        "nb_gated_mm4_nogate",
        "4b",
        GLINBImplementation.GATED_LATENT,
        CalibrationRoute.FINITE_SAMPLE_MOMENT,
        sharpness=0.0,
        texture=MM4_TEXTURE,
        parent_key="gated_mm4",
        intervention="Set gate sharpness to zero while preserving the mm4 calibration.",
    ),
)

GLI_NB_VARIANTS: Mapping[str, GLINBVariantConfig] = MappingProxyType(
    {variant.key: variant for variant in _VARIANTS}
)
if len(GLI_NB_VARIANTS) != len(_VARIANTS):
    raise RuntimeError("GLI-NB variant keys must be unique")


def _candidate_spec(config: GLINBVariantConfig) -> CandidateSpec:
    return CandidateSpec(
        config.key,
        config.aliases,
        ModelRef("count-first", "gli-nb", config.reference_variant),
        config.ladder_rung,
        config.estimation,
    )


def _ablation_spec(config: GLINBVariantConfig) -> AblationSpec:
    parent_key, intervention = config.parent_key, config.intervention
    if parent_key is None or intervention is None:
        raise RuntimeError(f"{config.key} is not a validated ablation")
    return AblationSpec(
        config.key,
        config.aliases,
        ModelRef("count-first", "gli-nb", config.reference_variant),
        config.ladder_rung,
        config.estimation,
        parent_key=parent_key,
        intervention=intervention,
    )


GLI_NB_CANDIDATE_SPECS = model_spec_registry(
    _candidate_spec(config) for config in _VARIANTS if not config.is_ablation
)
GLI_NB_ABLATION_SPECS = model_spec_registry(
    _ablation_spec(config) for config in _VARIANTS if config.is_ablation
)


def _long_acf_targets(counts: np.ndarray) -> dict[int, float]:
    targets = {int(lag): float(value) for lag, value in acf(counts, LONG_ACF_LAGS).items()}
    print("   long-lag ACF targets:", {lag: round(value, 4) for lag, value in targets.items()})
    return targets


def _build_gated(config: GLINBVariantConfig, state: PipelineState, seed: int) -> NBGatedLatent:
    if config.sharpness is None:
        raise RuntimeError(f"{config.key} is not a validated gated variant")
    counts = state["counts"].values
    if config.calibration is CalibrationRoute.LIKELIHOOD:
        if config.fit_restarts is None or config.fit_maxiter is None:
            raise RuntimeError(f"{config.key} has no validated fit budget")
        return NBGatedLatent(sharpness=config.sharpness).fit(
            counts,
            n_restarts=config.fit_restarts,
            maxiter=config.fit_maxiter,
            seed=seed,
        )

    marginals = state["trace_marginals"]
    mean = marginals["mean"]
    variance = marginals["IDC_15m"] * mean
    zero_fraction = marginals["zero_frac"]
    acf_1h = marginals["acf_1h_lag1"]
    if config.calibration is CalibrationRoute.ANALYTIC_MOMENT:
        return NBGatedLatent.calibrate(
            mean,
            variance,
            zero_fraction,
            acf_1h,
            sharpness=config.sharpness,
        )

    long_acf = _long_acf_targets(counts)
    if config.calibration is CalibrationRoute.ANALYTIC_LONG_ACF:
        return NBGatedLatent.calibrate(
            mean,
            variance,
            zero_fraction,
            acf_1h,
            sharpness=config.sharpness,
            acf_long=long_acf,
        )

    kwargs = config.texture.to_dict() if config.texture is not None else {}
    return NBGatedLatent.calibrate_debiased(
        mean,
        variance,
        zero_fraction,
        acf_1h,
        long_acf,
        state["n_bins"],
        seed=seed,
        marginals=config.debias_marginals,
        sharpness=config.sharpness,
        **kwargs,
    )


def build_gli_nb_variant(
    state: PipelineState, seed: int, *, config: GLINBVariantConfig
) -> CountModel:
    """Fit or calibrate the model selected by a validated variant config."""
    if config.implementation is GLINBImplementation.IID:
        model = IIDNegBin().fit(state["counts"].values)
    elif config.implementation is GLINBImplementation.LATENT_AR:
        if config.n_components is None or config.fit_restarts is None or config.fit_maxiter is None:
            raise RuntimeError(f"{config.key} is not a validated latent-AR variant")
        model = NBLatentAR(config.n_components).fit(
            state["counts"].values,
            n_restarts=config.fit_restarts,
            maxiter=config.fit_maxiter,
        )
    else:
        model = _build_gated(config, state, seed)
    model.name = config.model_name
    return model


GLI_NB_CANDIDATE_BUILDERS: Mapping[str, ModelBuilder] = MappingProxyType(
    {
        key: partial(build_gli_nb_variant, config=config)
        for key, config in GLI_NB_VARIANTS.items()
        if not config.is_ablation
    }
)
GLI_NB_ABLATION_BUILDERS: Mapping[str, ModelBuilder] = MappingProxyType(
    {
        key: partial(build_gli_nb_variant, config=config)
        for key, config in GLI_NB_VARIANTS.items()
        if config.is_ablation
    }
)
