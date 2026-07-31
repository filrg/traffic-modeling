"""Benchmark, not a rung: Markov regime-switching level.

Kept for comparison only. The current-paper model carries no discrete regime
chain; that direction is a separate study.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .ar1 import ar1_path, half_life_to_phi
from .base import CountModel, FitReport
from .latent_ar import NBLatentAR
from .nb import nb2_sample

# benchmark (not a rung): Markov regime-switching level
class NBRegimeSwitch(CountModel):
    """Two-state Markov level with NB2 emissions, plus an optional fast AR(1).

    Motivation: on this trace the marginal targets {mean, IDC_15m, zero
    fraction, ACF} are jointly infeasible for a Gaussian log-latent, and the
    zero bins arrive in long runs rather than independently, so the quiet
    state has to be persistent.

    The optional fast AR(1) is fitted conditionally on the Viterbi regime
    path.  That is an approximation to the exact switching state-space
    likelihood and is reported as such in the model card.
    """

    name = "nb_regime"
    rung = None  # not a rung of the ladder: MMPP/ON-OFF benchmark only (decision D3)
    role = "benchmark"

    def __init__(self, n_states: int = 2, fast_ar: bool = True) -> None:
        super().__init__()
        self.n_states = n_states
        self.fast_ar = fast_ar
        self.name = f"nb_regime{n_states}" + ("_fast" if fast_ar else "")

    # forward / backward
    @staticmethod
    def _emission_loglik(n: np.ndarray, lam_states: np.ndarray, kappa: float) -> np.ndarray:
        n_col = n[:, None]
        lam = lam_states[None, :]
        return (
            gammaln(n_col + kappa)
            - gammaln(kappa)
            - gammaln(n_col + 1.0)
            + kappa * (np.log(kappa) - np.log(kappa + lam))
            + n_col * (np.log(lam) - np.log(kappa + lam))
        )

    @staticmethod
    def _stationary(P: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eig(P.T)
        v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        v = np.abs(v)
        return v / v.sum()

    def _forward_backward(self, logB: np.ndarray, P: np.ndarray, pi: np.ndarray):
        T, K = logB.shape
        alpha = np.zeros((T, K))
        scale = np.zeros(T)
        a = np.log(pi + 1e-300) + logB[0]
        m = a.max()
        w = np.exp(a - m)
        scale[0] = m + np.log(w.sum())
        alpha[0] = w / w.sum()
        for t in range(1, T):
            pred = alpha[t - 1] @ P
            a = np.log(pred + 1e-300) + logB[t]
            m = a.max()
            w = np.exp(a - m)
            scale[t] = m + np.log(w.sum())
            alpha[t] = w / w.sum()
        beta = np.zeros((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            b = P @ (np.exp(np.clip(logB[t + 1] - logB[t + 1].max(), -700, 0)) * beta[t + 1])
            beta[t] = b / max(b.sum(), 1e-300)
        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        xi = np.zeros((K, K))
        for t in range(T - 1):
            shifted = np.exp(np.clip(logB[t + 1] - logB[t + 1].max(), -700, 0)) * beta[t + 1]
            num = (alpha[t][:, None] * P) * shifted[None, :]
            xi += num / max(num.sum(), 1e-300)
        return gamma, xi, float(scale.sum())

    def fit(self, counts: np.ndarray, max_iter: int = 200, tol: float = 1e-6, seed: int = 0,
            n_restarts: int = 3, **kwargs) -> "NBRegimeSwitch":
        self._check_mutable()
        n = np.asarray(counts, dtype=float)
        T, K = len(n), self.n_states
        rng = np.random.default_rng(seed)
        best = None

        for restart in range(n_restarts):
            qs = np.linspace(0.2, 0.9, K) if restart == 0 else np.sort(rng.uniform(0.1, 0.95, K))
            lam = np.maximum(np.quantile(n, qs), 1e-3)
            lam = np.sort(lam)
            kappa = 1.0
            P = np.full((K, K), 0.05 / max(K - 1, 1))
            np.fill_diagonal(P, 0.95)
            pi = self._stationary(P)
            prev_ll, converged, it = -np.inf, False, 0

            for it in range(1, max_iter + 1):
                logB = self._emission_loglik(n, lam, kappa)
                gamma, xi, ll = self._forward_backward(logB, P, pi)
                P = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
                pi = self._stationary(P)

                def nll(theta):
                    lam_k = np.exp(np.clip(theta[:K], -30, 30))
                    kap = np.exp(np.clip(theta[K], -20, 20))
                    lb = self._emission_loglik(n, lam_k, kap)
                    return -float(np.sum(gamma * lb))

                res = minimize(nll, np.r_[np.log(lam), np.log(kappa)], method="Nelder-Mead",
                               options={"maxiter": 1200, "xatol": 1e-7, "fatol": 1e-7})
                lam = np.sort(np.exp(np.clip(res.x[:K], -30, 30)))
                kappa = float(np.exp(np.clip(res.x[K], -20, 20)))
                if abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
                    converged = True
                    prev_ll = ll
                    break
                prev_ll = ll

            if best is None or prev_ll > best["ll"]:
                best = {"ll": prev_ll, "lam": lam, "kappa": kappa, "P": P, "pi": pi,
                        "converged": converged, "n_iter": it}

        lam, kappa, P, pi = best["lam"], best["kappa"], best["P"], best["pi"]
        n_params = K + 1 + K * (K - 1)
        warnings: List[str] = []
        if np.any(np.diag(P) > 1 - 1e-4):
            warnings.append("absorbing regime: transition probability at boundary")
        if np.max(lam) / max(np.min(lam), 1e-9) < 1.5:
            warnings.append("regime means nearly equal: prefer the simpler rung")

        self.params_ = {
            "lam_states": lam,
            "mu_states": np.log(lam),
            "kappa": float(kappa),
            "P": P,
            "pi": pi,
            "mean": float(np.dot(pi, lam)),
            "expected_dwell_bins": 1.0 / np.maximum(1.0 - np.diag(P), 1e-12),
        }

        loglik = best["ll"]
        if self.fast_ar:
            fast = self._fit_fast_ar(n, lam, kappa, P, pi)
            self.params_.update(fast)
            n_params += 2
            loglik = fast.pop("_loglik_conditional", loglik)
            warnings.append("fast AR(1) fitted conditionally on the Viterbi path (approximate likelihood)")

        self.report = FitReport(bool(best["converged"]), int(best["n_iter"]), 0.0, float(loglik),
                                n_params, T, warnings, restarts=n_restarts)
        return self

    def _viterbi(self, n: np.ndarray, lam: np.ndarray, kappa: float, P: np.ndarray, pi: np.ndarray) -> np.ndarray:
        logB = self._emission_loglik(n, lam, kappa)
        T, K = logB.shape
        delta = np.log(pi + 1e-300) + logB[0]
        psi = np.zeros((T, K), dtype=int)
        logP = np.log(P + 1e-300)
        for t in range(1, T):
            m = delta[:, None] + logP
            psi[t] = np.argmax(m, axis=0)
            delta = m.max(axis=0) + logB[t]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def _fit_fast_ar(self, n: np.ndarray, lam, kappa, P, pi) -> dict:
        """Conditional Laplace fit of one fast AR(1) given the MAP regime path."""
        path = self._viterbi(n, lam, kappa, P, pi)
        base = np.log(lam)[path]
        helper = NBLatentAR(n_components=1)

        def negll(theta):
            sigma = np.exp(np.clip(theta[0], -8, 3))
            tau = np.exp(np.clip(theta[1], -3, 6))
            kap = np.exp(np.clip(theta[2], -20, 20))
            ll, _, _ = helper._laplace_loglik(
                n, np.array([0.0, np.log(sigma), np.log(tau), np.log(kap)]), offset=base + 0.5 * sigma ** 2
            )
            return -ll if np.isfinite(ll) else 1e12

        res = minimize(negll, [np.log(0.5), np.log(2.0), np.log(kappa)], method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-2})
        sigma_fast = float(np.exp(np.clip(res.x[0], -8, 3)))
        tau_fast = float(np.exp(np.clip(res.x[1], -3, 6)))
        return {
            "sigma_fast": sigma_fast,
            "phi_fast": half_life_to_phi(tau_fast),
            "half_life_fast_bins": tau_fast,
            "kappa": float(np.exp(np.clip(res.x[2], -20, 20))),
            "regime_path_viterbi": path,
            "_loglik_conditional": float(-res.fun),
        }

    def simulate(self, n_bins: int, rng: np.random.Generator, alpha: float = 1.0,
                 kappa_rule: str = "proportional", return_intensity: bool = False, **kwargs):
        p = self.params_
        P, pi = np.asarray(p["P"]), np.asarray(p["pi"])
        mu_states = np.asarray(p["mu_states"])
        states = np.zeros(n_bins, dtype=int)
        states[0] = rng.choice(len(pi), p=pi)  # stationary initialisation
        for t in range(1, n_bins):
            states[t] = rng.choice(len(pi), p=P[states[t - 1]])
        eta = mu_states[states] + np.log(alpha)
        if self.fast_ar and "sigma_fast" in p:
            s, phi = float(p["sigma_fast"]), float(p["phi_fast"])
            eta = eta + ar1_path(n_bins, s, phi, rng) - 0.5 * s ** 2
        lam = np.exp(np.clip(eta, -30, 30))
        counts = nb2_sample(lam, self.scaled_kappa(alpha, kappa_rule), rng).astype(float)
        return (counts, lam) if return_intensity else counts
