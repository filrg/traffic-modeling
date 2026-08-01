"""Gated latent-intensity NB, rung 4b of the ladder."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq, minimize
from scipy.special import expit, roots_hermitenorm

from ..metrics import (
    HOURLY_ACF_LAG_WEIGHTS,
    acf as _acf,
    aggregate as _aggregate,
    hourly_acf1_from_autocov,
)
from .ar1 import (
    ar1_logdet_precision,
    ar1_path,
    ar1_precision,
    half_life_to_phi,
    sparse_logdet,
)
from .base import CountModel, FitReport
from .nb import nb2_info_eta, nb2_loglik, nb2_sample, nb2_score_eta

_N1, _W1 = roots_hermitenorm(80)
_W1 = _W1 / _W1.sum()
_N2, _W2 = roots_hermitenorm(60)
_W2 = _W2 / _W2.sum()
_NX, _WX = roots_hermitenorm(40)
_WX = _WX / _WX.sum()


# quadrature moments: deterministic, shared by calibration and fitting
def _gate(u: np.ndarray, a: float, c: float) -> np.ndarray:
    return expit(a * (u - c))


def _z1(a: float, b: float, c: float) -> float:
    return float(np.sum(_W1 * _gate(_N1, a, c) * np.exp(b * _N1)))


def _z2(a: float, b: float, c: float) -> float:
    return float(np.sum(_W1 * _gate(_N1, a, c) ** 2 * np.exp(2.0 * b * _N1)))


def _second_moment_ratio(a: float, b: float, c: float, sigma_f: float) -> float:
    """Return E[lambda^2] / m^2."""
    z1 = _z1(a, b, c)
    return _z2(a, b, c) / max(z1 * z1, 1e-300) * float(np.exp(sigma_f ** 2))


def _zero_prob(m: float, a: float, b: float, c: float, sigma_f: float, kappa: float) -> float:
    z1 = _z1(a, b, c)
    U = _N1[:, None]
    X = _NX[None, :]
    lam = m / z1 * _gate(U, a, c) * np.exp(b * U) * np.exp(sigma_f * X - 0.5 * sigma_f ** 2)
    p0 = (1.0 + lam / kappa) ** (-kappa)
    return float(_W1 @ p0 @ _WX)


def _pair_moment(a: float, b: float, c: float, r: float) -> float:
    """Return E[g(U) g(U') e^{b(U+U')}] with corr(U, U') = r."""
    U = _N2[:, None]
    V = _N2[None, :]
    Up = r * U + np.sqrt(max(1.0 - r * r, 0.0)) * V
    val = _gate(U, a, c) * _gate(Up, a, c) * np.exp(b * (U + Up))
    return float(_W2 @ val @ _W2)


def _corr_u(h: int, phi_slow: float, phi_long: float, w_long: float) -> float:
    """Return corr(u_t, u_t+h) for the two-scale gate latent; w_long = 0 gives phi_slow^h."""
    w2 = w_long * w_long
    return (1.0 - w2) * phi_slow ** h + w2 * phi_long ** h


def _autocov(m, a, b, c, sigma_f, corr_u, phi_f, h) -> float:
    z1 = _z1(a, b, c)
    pair = _pair_moment(a, b, c, corr_u)
    return m * m * (pair / (z1 * z1) * np.exp(sigma_f ** 2 * phi_f ** h) - 1.0)


def _gamma0(m, a, b, c, sigma_f, kappa) -> float:
    e2 = m * m * _second_moment_ratio(a, b, c, sigma_f)
    return m + e2 / kappa + (e2 - m * m)


def _hourly_acf1(m, a, b, c, sigma_f, kappa, phi_s, phi_f, phi_l=None, w_long=0.0) -> float:
    g0 = _gamma0(m, a, b, c, sigma_f, kappa)
    phi_l = phi_s if phi_l is None else phi_l
    gh = {h: _autocov(m, a, b, c, sigma_f, _corr_u(h, phi_s, phi_l, w_long), phi_f, h)
          for h in HOURLY_ACF_LAG_WEIGHTS}
    return hourly_acf1_from_autocov(g0, gh)


def _base_acf(m, a, b, c, sigma_f, kappa, phi_s, phi_f, phi_l, w_long, h) -> float:
    """Return the ACF of the base-resolution counts at lag h."""
    g0 = _gamma0(m, a, b, c, sigma_f, kappa)
    return _autocov(m, a, b, c, sigma_f, _corr_u(h, phi_s, phi_l, w_long), phi_f, h) / g0


class NBGatedLatent(CountModel):
    """NB counts driven by a soft gate on a latent, with no discrete regime chain.

    The gate latent may carry two timescales as
    `u_t = sqrt(1 - w^2) u_slow,t + w u_long,t`, both unit-variance AR(1). The
    marginal of u stays N(0, 1), so the quadrature moments — and with them the
    mean, variance and zero fraction — do not depend on w; only corr(u_t, u_t+h)
    does. `w_long = 0` reduces to the single-scale model, RNG stream included.
    """

    name = "nb_gated"
    rung = "4b"

    def __init__(self, sharpness: float = 4.0) -> None:
        super().__init__()
        self.a = float(sharpness)  # profiled, not free-fitted
        self.latent_mode_: Optional[np.ndarray] = None

    @classmethod
    def calibrate(cls, mean: float, variance: float, zero_fraction: float, acf_1h: float,
                  sharpness: float = 4.0, kappa: float = 8.0, sigma_fast: float = 0.6,
                  tau_fast_bins: float = 2.0,
                  acf_long: Optional[Mapping[int, float]] = None,
                  tau_long_bins: Optional[float] = None, w_long: float = 0.0) -> "NBGatedLatent":
        """Calibrate to the trace moments; no likelihood is evaluated.

        `acf_long` maps base-bin lags to target ACF values and enables the second
        slow scale: `w_long` and `tau_long_bins` are solved by least squares over
        those lags, with `tau_slow_bins` re-solved at each step to keep the hourly
        lag-1 ACF exact. Multiple long lags are required because the trace's
        lag-96 spike is diurnal echo, which a calendar-free model can only match
        in envelope. Omitting `acf_long` gives the single-scale calibration.
        """
        from scipy import stats

        m, a = float(mean), float(sharpness)
        target_ratio = (float(variance) - m + m * m) / (1.0 + 1.0 / kappa) / (m * m)

        def loss(theta):
            c, b = theta
            r = _second_moment_ratio(a, b, c, sigma_fast)
            p0 = _zero_prob(m, a, b, c, sigma_fast, kappa)
            return np.log(r / target_ratio) ** 2 + 25.0 * (p0 - zero_fraction) ** 2

        best = None
        for c0, b0 in [(stats.norm.ppf(zero_fraction), 1.0), (0.5, 0.6), (1.2, 1.4)]:
            res = minimize(loss, [c0, b0], method="Nelder-Mead",
                           options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 2000})
            if best is None or res.fun < best.fun:
                best = res
        c, b = float(best.x[0]), float(best.x[1])

        phi_f = half_life_to_phi(tau_fast_bins)

        def solve_tau_slow(phi_l: float, w: float) -> float:
            """Solve for the tau_slow that puts the hourly lag-1 ACF on target."""
            def acf_gap(tau_s):
                return _hourly_acf1(m, a, b, c, sigma_fast, kappa,
                                    half_life_to_phi(tau_s), phi_f, phi_l, w) - acf_1h

            lo, hi = 0.5, 5000.0
            if acf_gap(lo) * acf_gap(hi) < 0:
                return float(brentq(acf_gap, lo, hi, xtol=1e-6))
            return hi if acf_gap(hi) < 0 else lo

        w = float(w_long)
        tau_long = float(tau_long_bins) if tau_long_bins else 0.0
        if acf_long:
            lags = np.array(sorted(acf_long), dtype=int)
            targets = np.array([float(acf_long[int(h)]) for h in lags])

            def long_loss(theta):
                w_try = float(expit(theta[0]))
                # same bracket as the tau_slow solve, to keep Nelder-Mead in range
                tau_try = float(np.exp(np.clip(theta[1], np.log(2.0), np.log(5000.0))))
                phi_l = half_life_to_phi(tau_try)
                tau_s = solve_tau_slow(phi_l, w_try)
                model_acf = np.array([
                    _base_acf(m, a, b, c, sigma_fast, kappa, half_life_to_phi(tau_s),
                              phi_f, phi_l, w_try, int(h)) for h in lags
                ])
                return float(np.sum((model_acf - targets) ** 2))

            best_long = None
            for w0, t0 in [(0.5, 200.0), (0.3, 500.0), (0.7, 96.0)]:
                res = minimize(long_loss, [float(np.log(w0 / (1 - w0))), float(np.log(t0))],
                               method="Nelder-Mead",
                               options={"xatol": 1e-6, "fatol": 1e-14, "maxiter": 800})
                if best_long is None or res.fun < best_long.fun:
                    best_long = res
            w = float(expit(best_long.x[0]))
            tau_long = float(np.exp(np.clip(best_long.x[1], np.log(2.0), np.log(5000.0))))

        phi_l = half_life_to_phi(tau_long) if tau_long > 0 else None
        tau_slow = solve_tau_slow(phi_l if phi_l is not None else 0.0, w)
        phi_s = half_life_to_phi(tau_slow)

        model = cls(sharpness=a)
        model.params_ = {
            "mean": m, "b": b, "c": c, "a": a,
            "tau_slow_bins": float(tau_slow), "phi_slow": phi_s,
            "tau_long_bins": float(tau_long), "phi_long": float(phi_l if phi_l is not None else phi_s),
            "w_long": w,
            "sigma_fast": float(sigma_fast), "tau_fast_bins": float(tau_fast_bins),
            "phi_fast": phi_f, "kappa": float(kappa), "Z1": _z1(a, b, c),
        }
        return model

    @classmethod
    def calibrate_debiased(cls, mean: float, variance: float, zero_fraction: float, acf_1h: float,
                           acf_long: Mapping[int, float], n_bins: int, seed: int = 0,
                           n_paths: int = 8, n_rounds: int = 2, max_inflation: float = 3.0,
                           marginals: bool = False, marginal_paths: int = 200,
                           **kwargs) -> "NBGatedLatent":
        """Calibrate so the model's statistics at length `n_bins` hit the target.

        `calibrate` matches an analytic ACF to a target measured on one finite
        path. At 13272 bins the empirical ACF of this process runs 15-25% below
        its true value, so that comparison is biased. Here the attenuation is
        measured — simulate at `n_bins`, compare with the analytic value, and
        re-solve against targets divided by that ratio. Seeded and deterministic;
        `calibrate` itself stays analytic.

        `marginals` extends the correction to the variance, zero fraction and
        `acf_1h`, which carry the same bias. The mean is never corrected: the
        sample mean is unbiased and anchors the requested rate. Off by default,
        so frozen candidates keep their published values and RNG stream.
        """
        targets = {int(h): float(v) for h, v in acf_long.items()}
        lags = sorted(targets)
        adjusted = dict(targets)
        m_targets = {"variance": float(variance), "zero_fraction": float(zero_fraction),
                     "acf_1h": float(acf_1h)}
        m_adj = dict(m_targets)
        model = cls.calibrate(mean, m_adj["variance"], m_adj["zero_fraction"], m_adj["acf_1h"],
                              acf_long=adjusted, **kwargs)

        for _ in range(max(int(n_rounds), 1)):
            rng = np.random.default_rng(seed)
            paths = [model.simulate(int(n_bins), rng) for _ in range(n_paths)]
            sim = pd.DataFrame([_acf(x, lags) for x in paths]).mean()
            p = model.params_
            attenuation = {}
            for h in lags:
                ana = _base_acf(p["mean"], p["a"], p["b"], p["c"], p["sigma_fast"], p["kappa"],
                                p["phi_slow"], p["phi_fast"], p["phi_long"], p["w_long"], h)
                r = float(sim[h]) / ana if ana > 1e-9 else 1.0
                # one-sided: the correction may only undo attenuation
                attenuation[h] = float(np.clip(r, 1.0 / max_inflation, 1.0))
            adjusted = {h: targets[h] / attenuation[h] for h in lags}
            if marginals:
                imp = model.implied_marginals()
                # each ratio is measured on the functional the pipeline reads off
                # the trace: dispersion is per-path IDC, not the raw variance,
                # because E[var]/E[mean] != E[var/mean] here
                mrng = np.random.default_rng(seed + 104729)
                mpaths = [model.simulate(int(n_bins), mrng) for _ in range(int(marginal_paths))]
                measured = {
                    "variance": float(np.mean([x.var(ddof=1) / x.mean() for x in mpaths])),
                    "zero_fraction": float(np.mean([(x == 0).mean() for x in mpaths])),
                    "acf_1h": float(np.mean([_acf(_aggregate(x, 4), [1])[1] for x in mpaths])),
                }
                analytic = {"variance": float(imp["IDC_base"]),
                            "zero_fraction": float(imp["P_zero"]),
                            "acf_1h": float(imp["acf_1h_lag1"])}
                for k in m_targets:
                    r = measured[k] / analytic[k] if analytic[k] > 1e-12 else 1.0
                    # two-sided: the zero fraction is an indicator average and can
                    # bias either way, unlike the mean-subtracted statistics
                    m_adj[k] = m_targets[k] / float(np.clip(r, 1.0 / max_inflation, max_inflation))
            model = cls.calibrate(mean, m_adj["variance"], m_adj["zero_fraction"], m_adj["acf_1h"],
                                  acf_long=adjusted, **kwargs)

        model.params_["acf_attenuation"] = float(np.mean(list(attenuation.values())))
        return model

    def _slow_scales(self) -> tuple:
        """Return (phi_slow, phi_long, w_long), tolerating params saved before the second scale."""
        p = self.params_
        w = float(p.get("w_long", 0.0))
        return float(p["phi_slow"]), float(p.get("phi_long", p["phi_slow"])), w

    def implied_marginals(self) -> pd.Series:
        p = self.params_
        m, a, b, c, sf, kap = p["mean"], p["a"], p["b"], p["c"], p["sigma_fast"], p["kappa"]
        phi_s, phi_l, w = self._slow_scales()
        e2 = m * m * _second_moment_ratio(a, b, c, sf)
        var = m + e2 / kap + e2 - m * m
        return pd.Series(
            {
                "mean": m,
                "variance": var,
                "IDC_base": var / m,
                "P_zero": _zero_prob(m, a, b, c, sf, kap),
                "acf_1h_lag1": _hourly_acf1(m, a, b, c, sf, kap, phi_s, p["phi_fast"], phi_l, w),
                "acf_lag96": _base_acf(m, a, b, c, sf, kap, phi_s, p["phi_fast"], phi_l, w, 96),
                "active_fraction_gate": float(np.sum(_W1 * _gate(_N1, a, c))),
            }
        )

    def _eta_from_states(self, u: np.ndarray, x: np.ndarray, alpha: float, gate_shift: float) -> np.ndarray:
        p = self.params_
        base = np.log(alpha * p["mean"]) - np.log(p["Z1"]) - 0.5 * p["sigma_fast"] ** 2
        return np.clip(
            base + np.log(_gate(u, p["a"], p["c"] - gate_shift) + 1e-300) + p["b"] * u + x, -30.0, 30.0
        )

    def simulate(self, n_bins: int, rng: np.random.Generator, alpha: float = 1.0,
                 kappa_rule: str = "proportional", gate_shift: float = 0.0,
                 return_intensity: bool = False, **kwargs):
        p = self.params_
        phi_s, phi_l, w = self._slow_scales()
        u = ar1_path(n_bins, 1.0, phi_s, rng)
        if w > 0.0:
            # drawn only when the second scale is live, so w_long = 0 preserves the RNG stream
            u = np.sqrt(1.0 - w * w) * u + w * ar1_path(n_bins, 1.0, phi_l, rng)
        x = ar1_path(n_bins, p["sigma_fast"], p["phi_fast"], rng)
        lam = np.exp(self._eta_from_states(u, x, alpha, gate_shift))
        counts = nb2_sample(lam, self.scaled_kappa(alpha, kappa_rule), rng).astype(float)
        return (counts, lam) if return_intensity else counts

    def _mode(self, n, m, b, c, tau_s, sf, tau_f, kappa, x0=None, max_iter=60, tol=1e-7):
        T = len(n)
        a = self.a
        phi_s, phi_f = half_life_to_phi(tau_s), half_life_to_phi(tau_f)
        Qu = ar1_precision(T, 1.0, phi_s)
        Qx = ar1_precision(T, sf, phi_f)
        base = np.log(m) - np.log(max(_z1(a, b, c), 1e-300)) - 0.5 * sf ** 2
        z = np.zeros(2 * T) if x0 is None else x0.copy()

        def eta_of(zv):
            u, x = zv[:T], zv[T:]
            return np.clip(base + np.log(_gate(u, a, c) + 1e-300) + b * u + x, -30.0, 30.0), u, x

        def joint(zv):
            eta, u, x = eta_of(zv)
            return (nb2_loglik(n, np.exp(eta), kappa)
                    - 0.5 * float(u @ (Qu @ u)) - 0.5 * float(x @ (Qx @ x)))

        cur, gnorm = joint(z), np.inf
        for _ in range(max_iter):
            eta, u, x = eta_of(z)
            lam = np.exp(eta)
            s = nb2_score_eta(n, lam, kappa)
            w = np.maximum(nb2_info_eta(n, lam, kappa), 1e-10)
            g = _gate(u, a, c)
            Ju = b + a * (1.0 - g)                      # d eta / d u
            grad = np.concatenate([Ju * s - Qu @ u, s - Qx @ x])
            gnorm = float(np.linalg.norm(grad) / np.sqrt(grad.size))
            if gnorm < tol:
                break
            # Fisher scoring: per-bin outer([Ju, 1]) * w gives a PSD observation block
            H = sp.bmat(
                [[Qu + sp.diags(Ju * Ju * w), sp.diags(Ju * w)],
                 [sp.diags(Ju * w), Qx + sp.diags(w)]], format="csc"
            )
            try:
                step = spla.spsolve(H, grad)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                step = spla.spsolve(H + 1e-8 * sp.identity(2 * T, format="csc"), grad)
            scale = 1.0
            for _ in range(30):
                cand = z + scale * step
                val = joint(cand)
                if np.isfinite(val) and val >= cur - 1e-12:
                    z, cur = cand, val
                    break
                scale *= 0.5
            else:
                break
        eta, u, x = eta_of(z)
        lam = np.exp(eta)
        w = np.maximum(nb2_info_eta(n, lam, kappa), 1e-10)
        Ju = b + a * (1.0 - _gate(u, a, c))
        H = sp.bmat(
            [[Qu + sp.diags(Ju * Ju * w), sp.diags(Ju * w)],
             [sp.diags(Ju * w), Qx + sp.diags(w)]], format="csc"
        )
        return z, H, cur, gnorm, (phi_s, phi_f)

    def _laplace_loglik(self, n, theta, x0=None):
        log_m, b, c, log_tau_s, log_sf, log_tau_f, log_kappa = theta
        m, tau_s = np.exp(log_m), np.exp(log_tau_s)
        sf, tau_f = np.exp(log_sf), np.exp(log_tau_f)
        kappa = np.exp(np.clip(log_kappa, -20, 20))
        if not np.all(np.isfinite([m, b, c, tau_s, sf, tau_f, kappa])):
            return -np.inf, None, np.inf
        T = len(n)
        z, H, joint_at_mode, gnorm, (phi_s, phi_f) = self._mode(n, m, b, c, tau_s, sf, tau_f, kappa, x0)
        try:
            logdet_H = sparse_logdet(H)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return -np.inf, z, gnorm
        logdet_Q = ar1_logdet_precision(T, 1.0, phi_s) + ar1_logdet_precision(T, sf, phi_f)
        return float(joint_at_mode + 0.5 * logdet_Q - 0.5 * logdet_H), z, gnorm

    def fit(self, counts: np.ndarray, n_restarts: int = 2, maxiter: int = 200,
            seed: int = 0, verbose: bool = False, **kwargs) -> "NBGatedLatent":
        """Fit the single-scale submodel (`w_long = 0`) by Laplace MLE.

        The second slow scale is calibrated by moments, not fitted. At
        `n_restarts=3, maxiter=300` this estimator converges and restarts agree,
        but it drives `tau_slow` shorter (12.3 bins against the moment-calibrated
        17.3) and pushes overdispersion into kappa.
        """
        self._check_mutable()
        n = np.asarray(counts, dtype=float)
        hourly = _aggregate(n, 4)
        start_model = NBGatedLatent.calibrate(
            mean=float(n.mean()), variance=float(n.var(ddof=1)),
            zero_fraction=float((n == 0).mean()),
            acf_1h=float(np.nan_to_num(_acf(hourly, [1])[1], nan=0.3)),
            sharpness=self.a,
        )
        p0 = start_model.params_
        base = np.array([np.log(p0["mean"]), p0["b"], p0["c"], np.log(p0["tau_slow_bins"]),
                         np.log(p0["sigma_fast"]), np.log(p0["tau_fast_bins"]), np.log(p0["kappa"])])

        rng = np.random.default_rng(seed)
        starts = [base] + [base + rng.normal(0, 0.3, size=base.shape) for _ in range(max(n_restarts, 1) - 1)]

        cache: Dict[bytes, float] = {}
        warm = {"z": None}

        def negll(theta):
            key = np.round(theta, 8).tobytes()
            if key in cache:
                return cache[key]
            ll, z, _ = self._laplace_loglik(n, np.asarray(theta, float), warm["z"])
            if np.isfinite(ll):
                warm["z"] = z
            out = -ll if np.isfinite(ll) else 1e12
            cache[key] = out
            return out

        best, best_theta, best_res = np.inf, None, None
        for theta0 in starts:
            warm["z"] = None
            res = minimize(negll, theta0, method="L-BFGS-B",
                           options={"maxiter": maxiter, "maxfun": 5 * maxiter, "eps": 1e-4, "ftol": 1e-10})
            if verbose:
                print(f"  restart -> nll {res.fun:.2f}")
            if res.fun < best:
                best, best_theta, best_res = res.fun, res.x, res

        log_m, b, c, log_ts, log_sf, log_tf, log_k = best_theta
        ll, z, gnorm = self._laplace_loglik(n, best_theta, None)
        self.latent_mode_ = z
        tau_s, sf, tau_f = float(np.exp(log_ts)), float(np.exp(log_sf)), float(np.exp(log_tf))

        warnings: List[str] = []
        if tau_s * 4 > len(n):
            warnings.append("tau_slow within a factor 4 of the horizon: slow scale weakly identified")
        if abs(b) < 5e-2:
            warnings.append("b near zero: level decoupled from the gate latent")
        if sf < 1e-2:
            warnings.append("sigma_fast at boundary: prefer dropping the fast component")
        if float(np.exp(log_k)) > 1e3:
            warnings.append("kappa very large: overdispersion pushed into the latent (kappa/sigma_f confound)")

        self.params_ = {
            "mean": float(np.exp(log_m)), "b": float(b), "c": float(c), "a": self.a,
            "tau_slow_bins": tau_s, "phi_slow": half_life_to_phi(tau_s),
            # second scale inert, not absent: keys stay present for round-tripping
            "tau_long_bins": tau_s, "phi_long": half_life_to_phi(tau_s), "w_long": 0.0,
            "sigma_fast": sf, "tau_fast_bins": tau_f, "phi_fast": half_life_to_phi(tau_f),
            "kappa": float(np.exp(log_k)), "Z1": _z1(self.a, float(b), float(c)),
        }
        self.report = FitReport(bool(best_res.success), int(best_res.nit), float(gnorm), float(ll),
                                7, len(n), warnings, restarts=len(starts))
        return self
