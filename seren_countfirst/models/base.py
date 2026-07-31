"""The model interface every rung implements, and its fit report."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .nb import nb2_info_eta, nb2_score_eta

# calendar offset (guide section 1.4) -- ablation only
@dataclass
class CalendarTerm:
    """Normalised low-order Fourier hour-of-week offset, eq. (9)-(10)."""

    beta: np.ndarray
    n_harmonics: int
    offset_by_how: np.ndarray  # length 168, already normalised

    @staticmethod
    def design(hour_of_week: np.ndarray, n_harmonics: int) -> np.ndarray:
        x = 2.0 * np.pi * np.asarray(hour_of_week, dtype=float) / 168.0
        cols = [np.ones_like(x)]
        for k in range(1, n_harmonics + 1):
            cols += [np.cos(k * x), np.sin(k * x)]
        return np.column_stack(cols)

    @classmethod
    def fit(cls, counts: np.ndarray, hour_of_week: np.ndarray, n_harmonics: int = 3, kappa: float = 1.0,
            max_iter: int = 100) -> "CalendarTerm":
        X = cls.design(hour_of_week, n_harmonics)
        beta = np.zeros(X.shape[1])
        beta[0] = np.log(max(np.mean(counts), 1e-6))
        for _ in range(max_iter):
            eta = X @ beta
            lam = np.exp(np.clip(eta, -30, 30))
            w = nb2_info_eta(counts, lam, kappa)
            z = eta + nb2_score_eta(counts, lam, kappa) / np.maximum(w, 1e-12)
            wx = X * w[:, None]
            new = np.linalg.solve(X.T @ wx + 1e-8 * np.eye(X.shape[1]), wx.T @ z)
            if np.max(np.abs(new - beta)) < 1e-9:
                beta = new
                break
            beta = new
        grid = cls.design(np.arange(168), n_harmonics) @ beta
        grid = grid - np.log(np.mean(np.exp(grid)))  # eq. (9): weekly mean of exp is 1
        return cls(beta=beta, n_harmonics=n_harmonics, offset_by_how=grid)

    def offset(self, hour_of_week: np.ndarray) -> np.ndarray:
        return self.offset_by_how[np.asarray(hour_of_week, dtype=int) % 168]

    def to_dict(self) -> dict:
        return {"n_harmonics": self.n_harmonics, "beta": self.beta.tolist()}


@dataclass
class FitReport:
    converged: bool
    n_iter: int
    grad_norm: float
    loglik: float
    n_params: int
    n_obs: int
    boundary_warnings: List[str] = field(default_factory=list)
    restarts: int = 1

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.n_params * np.log(self.n_obs) - 2 * self.loglik

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "n_iter": self.n_iter,
            "grad_norm": self.grad_norm,
            "loglik": self.loglik,
            "n_params": self.n_params,
            "n_obs": self.n_obs,
            "AIC": self.aic,
            "BIC": self.bic,
            "boundary_warnings": list(self.boundary_warnings),
            "restarts": self.restarts,
        }


class CountModel:
    """Interface shared by every rung of the ladder."""

    name = "base"
    rung: object = -1
    #: "candidate" competes for selection; "benchmark" is reported but never chosen
    role = "candidate"

    def __init__(self) -> None:
        self.report: Optional[FitReport] = None
        self.params_: Dict[str, Any] = {}
        self._frozen = False

    def fit(self, counts: np.ndarray, **kwargs) -> "CountModel":  # pragma: no cover
        raise NotImplementedError

    def simulate(self, n_bins: int, rng: np.random.Generator, alpha: float = 1.0,
                 kappa_rule: str = "proportional", **kwargs) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def freeze(self) -> "CountModel":
        self._frozen = True
        return self

    def _check_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError(f"{self.name} is frozen: refit is not allowed after export")

    def scaled_kappa(self, alpha: float, rule: str) -> float:
        """Dispersion under load scaling; 'proportional' is the frozen rule.

        Summing alpha iid NB2(mu, kappa) gives NB2(alpha*mu, alpha*kappa)
        exactly, so 'proportional' states that alpha superposes alpha
        independent submitters onto one shared latent driver. The two rules
        coincide at alpha = 1.
        """
        kappa = float(self.params_["kappa"])
        if rule == "constant":
            return kappa
        if rule == "proportional":
            return alpha * kappa
        raise ValueError("kappa_rule must be 'constant' or 'proportional'")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rung": self.rung,
            "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.params_.items()},
            "fit": self.report.to_dict() if self.report else None,
        }
