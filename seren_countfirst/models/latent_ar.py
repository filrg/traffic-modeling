from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

from .ar1 import (
    ar1_logdet_precision,
    ar1_path,
    ar1_precision,
    half_life_to_phi,
    phi_to_half_life,
    sparse_logdet,
)
from .base import CalendarTerm, CountModel, FitReport
from .nb import nb2_info_eta, nb2_loglik, nb2_sample, nb2_score_eta


class NBLatentAR(CountModel):
    """NB counts over a sum of stationary AR(1) latents.

    `log lambda_t = mu + sum_j x_t^(j) - V/2`, with V the total latent variance.
    """

    def __init__(self, n_components: int = 2, calendar: CalendarTerm | None = None) -> None:
        super().__init__()
        if n_components not in (1, 2):
            raise ValueError("n_components must be 1 or 2")
        self.n_components = n_components
        self.calendar = calendar
        self.name = f"nb_ar{n_components}" + ("_cal" if calendar is not None else "")
        self.rung = 3 if n_components == 1 else 4
        self.latent_mode_: np.ndarray | None = None

    def _unpack(self, theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
        if self.n_components == 1:
            mu, log_sigma, log_tau, log_kappa = theta
            sigma = np.array([np.exp(log_sigma)])
            tau = np.array([np.exp(log_tau)])
        else:
            mu, log_s_slow, log_s_fast, log_tau_fast, log_gap, log_kappa = theta
            sigma = np.array([np.exp(log_s_slow), np.exp(log_s_fast)])
            tau_fast = np.exp(log_tau_fast)
            tau = np.array(
                [tau_fast + np.exp(log_gap), tau_fast]
            )  # gap parameterisation identifies the order
        return float(mu), sigma, tau, float(np.exp(np.clip(log_kappa, -20, 20)))

    def _eta(
        self, mu: float, sigma: np.ndarray, x: np.ndarray, offset: np.ndarray | None
    ) -> np.ndarray:
        eta = mu - 0.5 * float(np.sum(sigma**2)) + x.reshape(self.n_components, -1).sum(axis=0)
        if offset is not None:
            eta = eta + offset
        return np.clip(eta, -30.0, 30.0)

    def _mode(
        self,
        n: np.ndarray,
        mu: float,
        sigma: np.ndarray,
        tau: np.ndarray,
        kappa: float,
        offset: np.ndarray | None,
        x0: np.ndarray | None,
        max_iter: int = 60,
        tol: float = 1e-7,
    ) -> tuple[np.ndarray, sp.csc_matrix, float, float]:
        T = len(n)
        J = self.n_components
        phi = np.array([half_life_to_phi(t) for t in tau])
        Q = [ar1_precision(T, sigma[j], phi[j]) for j in range(J)]
        x = np.zeros(J * T) if x0 is None else x0.copy()

        def joint(xv):
            eta = self._eta(mu, sigma, xv, offset)
            lam = np.exp(eta)
            val = nb2_loglik(n, lam, kappa)
            for j in range(J):
                xj = xv[j * T : (j + 1) * T]
                val -= 0.5 * float(xj @ (Q[j] @ xj))
            return val

        cur = joint(x)
        grad_norm = np.inf
        it = 0
        for it in range(1, max_iter + 1):
            eta = self._eta(mu, sigma, x, offset)
            lam = np.exp(eta)
            s = nb2_score_eta(n, lam, kappa)
            w = np.maximum(nb2_info_eta(n, lam, kappa), 1e-10)
            grad = np.concatenate([s - Q[j] @ x[j * T : (j + 1) * T] for j in range(J)])
            grad_norm = float(np.linalg.norm(grad) / np.sqrt(grad.size))
            if grad_norm < tol:
                break
            W = sp.diags(w, format="csc")
            if J == 1:
                H = (Q[0] + W).tocsc()
            else:
                H = sp.bmat([[Q[0] + W, W], [W, Q[1] + W]], format="csc")
            try:
                step = spla.spsolve(H, grad)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                H = H + sp.identity(H.shape[0], format="csc") * 1e-8
                step = spla.spsolve(H, grad)
            scale = 1.0
            for _ in range(30):
                cand = x + scale * step
                val = joint(cand)
                if np.isfinite(val) and val >= cur - 1e-12:
                    x, cur = cand, val
                    break
                scale *= 0.5
            else:
                break
        eta = self._eta(mu, sigma, x, offset)
        lam = np.exp(eta)
        w = np.maximum(nb2_info_eta(n, lam, kappa), 1e-10)
        W = sp.diags(w, format="csc")
        H = (Q[0] + W).tocsc() if J == 1 else sp.bmat([[Q[0] + W, W], [W, Q[1] + W]], format="csc")
        return x, H, cur, grad_norm

    def _laplace_loglik(
        self,
        n: np.ndarray,
        theta: np.ndarray,
        offset: np.ndarray | None,
        x0: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, float]:
        mu, sigma, tau, kappa = self._unpack(theta)
        if not np.all(np.isfinite([mu, kappa])) or np.any(sigma <= 0) or np.any(tau <= 0):
            return -np.inf, np.zeros(self.n_components * len(n)), np.inf
        T = len(n)
        phi = np.array([half_life_to_phi(t) for t in tau])
        if np.any(phi >= 1 - 1e-10):
            return -np.inf, np.zeros(self.n_components * len(n)), np.inf
        x_hat, H, joint_at_mode, gnorm = self._mode(n, mu, sigma, tau, kappa, offset, x0)
        try:
            logdet_H = sparse_logdet(H)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return -np.inf, x_hat, gnorm
        logdet_Q = sum(ar1_logdet_precision(T, sigma[j], phi[j]) for j in range(self.n_components))
        ll = joint_at_mode + 0.5 * logdet_Q - 0.5 * logdet_H
        return float(ll), x_hat, gnorm

    def moment_start(self, n: np.ndarray, max_lag: int = 240) -> np.ndarray:
        """Return a method-of-moments starting point read off the empirical ACF.

        The count autocovariance is gamma_h = m^2 (e^{C_h} - 1) with
        C_h = sum_j sigma_j^2 phi_j^h, so the latent variances and half-lives
        follow without evaluating the Laplace likelihood.
        """
        from scipy.optimize import least_squares

        n = np.asarray(n, dtype=float)
        m = max(n.mean(), 1e-6)
        nc = n - m
        H = int(min(max_lag, len(n) // 4))
        lags = np.arange(1, max(H, 2))
        gamma = np.array([float(np.sum(nc[:-h] * nc[h:])) / len(n) for h in lags])
        C_hat = np.log(np.clip(1.0 + gamma / m**2, 1e-6, None))
        ok = np.isfinite(C_hat) & (C_hat > 0)
        w = 1.0 / np.sqrt(1.0 + lags)

        if self.n_components == 1:

            def resid(p):
                a, phi = np.exp(p[0]), 1.0 / (1.0 + np.exp(-p[1]))
                return w[ok] * (a * phi ** lags[ok] - C_hat[ok])

            p0 = [np.log(max(C_hat[ok][0], 1e-3)) if ok.any() else 0.0, 1.0]
            sol = least_squares(resid, p0, max_nfev=400)
            a = np.array([np.exp(sol.x[0])])
            phi = np.array([1.0 / (1.0 + np.exp(-sol.x[1]))])
        else:

            def resid(p):
                a_s, a_f = np.exp(p[0]), np.exp(p[1])
                phi_f = 1.0 / (1.0 + np.exp(-p[2]))
                phi_s = phi_f + (1.0 - phi_f) / (1.0 + np.exp(-p[3]))  # enforces phi_s > phi_f
                return w[ok] * (a_s * phi_s ** lags[ok] + a_f * phi_f ** lags[ok] - C_hat[ok])

            c0 = float(C_hat[ok][0]) if ok.any() else 0.5
            sol = least_squares(
                resid,
                [np.log(max(0.5 * c0, 1e-3)), np.log(max(0.5 * c0, 1e-3)), 0.0, 1.5],
                max_nfev=800,
            )
            a_s, a_f = np.exp(sol.x[0]), np.exp(sol.x[1])
            phi_f = 1.0 / (1.0 + np.exp(-sol.x[2]))
            phi_s = phi_f + (1.0 - phi_f) / (1.0 + np.exp(-sol.x[3]))
            a, phi = np.array([a_s, a_f]), np.array([phi_s, phi_f])

        a = np.clip(a, 1e-4, 8.0)
        phi = np.clip(phi, 1e-4, 1 - 1e-6)
        V = float(a.sum())
        gamma0 = float(n.var(ddof=1))
        resid_var = gamma0 - m - m**2 * (np.exp(V) - 1.0)
        kappa = float(np.clip(m**2 * np.exp(V) / resid_var, 1e-3, 1e4)) if resid_var > 0 else 1e3
        tau = np.array([phi_to_half_life(p) for p in phi])

        if self.n_components == 1:
            return np.array([np.log(m), 0.5 * np.log(a[0]), np.log(tau[0]), np.log(kappa)])
        gap = max(tau[0] - tau[1], 1e-2)
        return np.array(
            [
                np.log(m),
                0.5 * np.log(a[0]),
                0.5 * np.log(a[1]),
                np.log(max(tau[1], 1e-2)),
                np.log(gap),
                np.log(kappa),
            ]
        )

    def fit(
        self,
        counts: np.ndarray,
        hour_of_week: np.ndarray | None = None,
        n_restarts: int = 3,
        maxiter: int = 900,
        verbose: bool = False,
        seed: int = 0,
    ) -> "NBLatentAR":
        self._check_mutable()
        n = np.asarray(counts, dtype=float)
        offset = None
        if self.calendar is not None:
            if hour_of_week is None:
                raise ValueError("calendar model requires hour_of_week")
            offset = self.calendar.offset(hour_of_week)

        rng = np.random.default_rng(seed)

        # restart 0 is the moment start; the rest jitter around it
        base = self.moment_start(n)
        starts = [base]
        for _ in range(max(n_restarts, 1) - 1):
            starts.append(base + rng.normal(0.0, 0.45, size=base.shape))

        cache: dict[bytes, tuple[float, np.ndarray, float]] = {}
        warm: dict[str, np.ndarray | None] = {"x": None}

        def negll(theta: np.ndarray) -> float:
            key = np.round(theta, 8).tobytes()
            if key in cache:
                return -cache[key][0]
            ll, x_hat, gnorm = self._laplace_loglik(
                n, np.asarray(theta, dtype=float), offset, warm["x"]
            )
            if np.isfinite(ll):
                warm["x"] = x_hat
            cache[key] = (ll, x_hat, gnorm)
            return -ll if np.isfinite(ll) else 1e12

        best, best_theta, best_res = np.inf, None, None
        for theta0 in starts:
            warm["x"] = None
            res = minimize(
                negll,
                theta0,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "maxfun": 6 * maxiter, "eps": 1e-4, "ftol": 1e-10},
            )
            res = minimize(
                negll,
                res.x,
                method="Nelder-Mead",
                options={
                    "maxiter": max(maxiter // 4, 60),
                    "xatol": 1e-4,
                    "fatol": 1e-3,
                    "adaptive": True,
                },
            )
            if verbose:
                print(f"  restart -> nll {res.fun:.2f} converged={res.success}")
            if res.fun < best:
                best, best_theta, best_res = res.fun, res.x, res

        mu, sigma, tau, kappa = self._unpack(best_theta)
        ll, x_hat, gnorm = self._laplace_loglik(n, best_theta, offset, None)
        self.latent_mode_ = x_hat
        phi = np.array([half_life_to_phi(t) for t in tau])

        warnings: list[str] = []
        for j, s in enumerate(sigma):
            if s < 1e-2:
                warnings.append(f"sigma[{j}]={s:.2e} at lower boundary: prefer the simpler rung")
        if self.n_components == 2 and abs(np.log(tau[0] / tau[1])) < np.log(1.5):
            warnings.append("half-lives collapsed to one scale: prefer NBLatentAR(1)")
        if np.any(phi > 1 - 1e-6):
            warnings.append(
                "phi at unit-root boundary: slow scale is not identified on this horizon"
            )
        if kappa > 1e4:
            warnings.append("kappa very large: observation overdispersion vanished into the latent")

        self.params_ = {
            "mu": float(mu),
            "mean": float(np.exp(mu)),
            "sigma": sigma,
            "phi": phi,
            "half_life_bins": tau,
            "kappa": float(kappa),
            "V": float(np.sum(sigma**2)),
        }
        self.report = FitReport(
            bool(best_res.success),
            int(best_res.nit),
            float(gnorm),
            float(ll),
            len(best_theta),
            len(n),
            warnings,
            restarts=len(starts),
        )
        return self

    def simulate(
        self,
        n_bins: int,
        rng: np.random.Generator,
        alpha: float = 1.0,
        kappa_rule: str = "proportional",
        hour_of_week: np.ndarray | None = None,
        return_intensity: bool = False,
    ):
        p = self.params_
        sigma, phi = np.asarray(p["sigma"]), np.asarray(p["phi"])
        x = np.zeros((self.n_components, n_bins))
        for j in range(self.n_components):
            x[j] = ar1_path(n_bins, sigma[j], phi[j], rng)
        eta = p["mu"] + np.log(alpha) - 0.5 * p["V"] + x.sum(axis=0)
        if self.calendar is not None:
            if hour_of_week is None:
                raise ValueError("calendar model requires hour_of_week for simulation")
            eta = eta + self.calendar.offset(hour_of_week)
        lam = np.exp(np.clip(eta, -30, 30))
        counts = nb2_sample(lam, self.scaled_kappa(alpha, kappa_rule), rng).astype(float)
        return (counts, lam) if return_intensity else counts
