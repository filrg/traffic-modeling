"""The model ladder: rungs 2-4b, plus the regime-switch benchmark.

Fitting is MAP for the latent path with a Laplace approximation for the
marginal likelihood.
"""
from .ar1 import (
    ar1_logdet_precision,
    ar1_path,
    ar1_precision,
    half_life_to_phi,
    phi_to_half_life,
    sparse_logdet,
)
from .base import CalendarTerm, CountModel, FitReport
from .feasibility import marginal_feasibility
from .gated import NBGatedLatent
from .iid import IIDNegBin
from .latent_ar import NBLatentAR
from .nb import nb2_info_eta, nb2_loglik, nb2_sample, nb2_score_eta
from .regime import NBRegimeSwitch

__all__ = [
    "CalendarTerm",
    "CountModel",
    "FitReport",
    "IIDNegBin",
    "NBGatedLatent",
    "NBLatentAR",
    "NBRegimeSwitch",
    "ar1_logdet_precision",
    "ar1_path",
    "ar1_precision",
    "half_life_to_phi",
    "marginal_feasibility",
    "nb2_info_eta",
    "nb2_loglik",
    "nb2_sample",
    "nb2_score_eta",
    "phi_to_half_life",
    "sparse_logdet",
]
