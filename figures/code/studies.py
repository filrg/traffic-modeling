from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

SEED = C.SEED
BINS_PER_DAY = C.BINS_PER_DAY

# --- rate-error -----------------------------------------------------------
RATE_N_PATHS = 400
# run lengths in days; 138.25 d is the trace's own length
RATE_HORIZON_D = (1, 2, 4, 7, 14, 30, 60, 138.25, 180, 365, 730, 1095)

# --- horizon --------------------------------------------------------------
HZ_N_PATHS_ACF = 300
HZ_N_PATHS_EXT = 200
HZ_LAGS = (1, 4, 96, 384, 672)
HZ_SCALES = (
    (1, "15 min"),
    (4, "1 h"),
    (24, "6 h"),
    (96, "1 d"),
    (192, "2 d"),
    (384, "4 d"),
    (672, "7 d"),
    (2880, "30 d"),
)
HZ_HORIZON_D = (1, 7, 30, 138.25, 365, 1095)

# --- load-scaling ---------------------------------------------------------
LS_N_PATHS = 200
LS_ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
LS_LAGS = (1, 4, 96)


def analytic_rho(params: dict, max_lag: int) -> np.ndarray:
    """Return rho(h) for h = 1..max_lag from the model's own moment machinery."""
    from seren_countfirst.models.gated import _autocov, _corr_u, _gamma0

    p = params
    g0 = _gamma0(p["mean"], p["a"], p["b"], p["c"], p["sigma_fast"], p["kappa"])
    rho = np.empty(max_lag + 1)
    rho[0] = 1.0
    for h in range(1, max_lag + 1):
        r = _corr_u(h, p["phi_slow"], p["phi_long"], p["w_long"])
        rho[h] = (
            _autocov(p["mean"], p["a"], p["b"], p["c"], p["sigma_fast"], r, p["phi_fast"], h) / g0
        )
    return rho


def tau_int(rho: np.ndarray, n_bins: int) -> float:
    """Return tau_int(T) = 1 + 2 sum_{h<T} (1 - h/T) rho(h).

    The finite-T triangular weighting, not the T -> inf sum. The mean rate over
    T bins then has Var = gamma(0) tau_int(T) / T.
    """
    h = np.arange(1, min(len(rho), n_bins))
    return 1.0 + 2.0 * float(np.sum((1.0 - h / n_bins) * rho[h]))


def _longest_zero_run(x: np.ndarray) -> int:
    best = cur = 0
    for v in x == 0:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def rate_error() -> None:
    """How the realised-rate error shrinks with run length, analytically and measured.

    The bar on a single run scales as 1/sqrt(T/tau_int), not 1/sqrt(T).
    """
    from seren_countfirst.models.gated import _gamma0

    st = C.state()
    model, cal = C.winner(st)
    p = dict(model.params_)
    n_trace = int(st["n_bins"])
    g0 = _gamma0(p["mean"], p["a"], p["b"], p["c"], p["sigma_fast"], p["kappa"])
    mu = float(p["mean"])
    # truncated where phi_long^h has decayed below the summation floor
    rho = analytic_rho(p, min(int(max(RATE_HORIZON_D) * BINS_PER_DAY), 20000))
    tau_inf = 1.0 + 2.0 * float(np.sum(rho[1:]))

    rows = []
    for d in RATE_HORIZON_D:
        n_bins = n_trace if abs(d - n_trace / BINS_PER_DAY) < 1e-6 else int(round(d * BINS_PER_DAY))
        t_i = tau_int(rho, n_bins)
        rates = np.array(
            [
                model.simulate(n_bins, np.random.default_rng(SEED + 100 + i), alpha=cal).mean()
                for i in range(RATE_N_PATHS)
            ]
        )
        cv_obs = float(rates.std(ddof=1) / rates.mean())
        rows.append(
            {
                "days": d,
                "n_bins": n_bins,
                "tau_int_bins": t_i,
                "n_effective_stretches": n_bins / t_i,
                "cv_predicted": float(np.sqrt(g0 * t_i / n_bins) / mu),
                "cv_measured": cv_obs,
                # the ensemble bar the load figure draws: +-1.96 CV / sqrt(N)
                "ensemble_ci95_pct": 100 * 1.959964 * cv_obs / np.sqrt(RATE_N_PATHS),
                "mean_rate": float(rates.mean()),
                "q025": float(np.quantile(rates, 0.025)),
                "q975": float(np.quantile(rates, 0.975)),
            }
        )
    df = pd.DataFrame(rows)

    # local log-log slope; -0.5 is the asymptote
    lt, lc = np.log(df["n_bins"].to_numpy()), np.log(df["cv_measured"].to_numpy())
    df["slope_measured"] = np.r_[np.nan, np.diff(lc) / np.diff(lt)]
    lp = np.log(df["cv_predicted"].to_numpy())
    df["slope_predicted"] = np.r_[np.nan, np.diff(lp) / np.diff(lt)]
    C.write_csv(df, "analysis_rate_error_horizon", index=False)

    print(f"model {st['winner']}, {RATE_N_PATHS} paths per horizon")
    print(f"gamma(0) = {g0:.4f}, mean = {mu:.4f}, per-bin CV = {np.sqrt(g0) / mu:.3f}")
    print(
        f"tau_long = {p['tau_long_bins']:.1f} bins = "
        f"{p['tau_long_bins'] / BINS_PER_DAY:.2f} d, w_long = {p['w_long']:.3f}"
    )
    print(f"tau_inf  = {tau_inf:.1f} bins = {tau_inf / BINS_PER_DAY:.2f} d\n")
    cols = [
        "days",
        "n_bins",
        "tau_int_bins",
        "n_effective_stretches",
        "cv_predicted",
        "cv_measured",
        "slope_predicted",
        "slope_measured",
        "ensemble_ci95_pct",
        "mean_rate",
    ]
    print(df[cols].round(4).to_string(index=False))

    # the run length needed for a given rate error, from the analytic constant
    # rather than the fit
    tail = df[df["n_bins"] >= 4 * tau_inf]
    const = float(np.sqrt(g0 * tau_inf) / mu)
    fitted = float(np.mean(tail["cv_measured"] * np.sqrt(tail["n_bins"])))
    print(
        f"\nasymptote  CV(T) = {const:.2f} / sqrt(T_bins) analytic, "
        f"{fitted:.2f} fitted past 4 tau_inf"
    )
    print(f"           CV(T) = {const / np.sqrt(BINS_PER_DAY):.2f} / sqrt(T_days)")
    for target in (0.20, 0.10, 0.05):
        n = (const / target) ** 2
        print(
            f"  CV = {target:.0%} needs T = {n:,.0f} bins = {n / BINS_PER_DAY:,.0f} d"
            f"   (or {(const / np.sqrt(n_trace) / target) ** 2:,.1f} runs of 138 d)"
        )


def horizon() -> None:
    """What depends on the horizon, analytically and measured.

    The model is stationary at every length, so no population statistic depends
    on T; the tau curve, the finite-T estimator bias and the extensive
    statistics do.
    """
    from seren_countfirst.metrics import acf, idc
    from seren_countfirst.models.gated import _gamma0

    st = C.state()
    model, _ = C.winner(st)
    p = dict(model.params_)
    m = float(p["mean"])
    g0 = _gamma0(m, p["a"], p["b"], p["c"], p["sigma_fast"], p["kappa"])
    n_trace = int(st["n_bins"])
    rho = analytic_rho(p, 20000)
    tau_inf = 1.0 + 2.0 * float(np.sum(rho[1:]))

    # one tau function serves both dispersion by scale and reliability by length
    print(f"model {st['winner']}")
    print(
        f"IDC(1) = {g0 / m:.3f}, tau_inf = {tau_inf:.1f} bins, "
        f"IDC ceiling = IDC(1) tau_inf = {g0 / m * tau_inf:,.0f}\n"
    )
    scale = pd.DataFrame(
        [
            {
                "q": q,
                "window": name,
                "tau_q": tau_int(rho, q),
                "idc_q": g0 / m * tau_int(rho, q),
                "frac_of_ceiling": tau_int(rho, q) / tau_inf,
            }
            for q, name in (*HZ_SCALES, (n_trace, "138 d"))
        ]
    )
    # the identity CV^2(T) = IDC(T)/(m T) must reproduce the tau-based error
    scale["cv_from_idc"] = np.sqrt(scale["idc_q"] / (m * scale["q"]))
    scale["cv_from_tau"] = np.sqrt(g0 * scale["tau_q"] / scale["q"]) / m
    C.write_csv(scale, "analysis_horizon_scale", index=False)
    print("dispersion by scale, and the same curve read as run-to-run error:")
    print(scale.round(4).to_string(index=False))

    # sample-ACF bias against horizon
    rows = []
    for T in (n_trace // 4, n_trace, n_trace * 4):
        paths = [
            model.simulate(T, np.random.default_rng(SEED + 900 + i)) for i in range(HZ_N_PATHS_ACF)
        ]
        est = np.array([acf(x, HZ_LAGS).to_numpy() for x in paths])
        t_T = tau_int(rho, T)
        for j, h in enumerate(HZ_LAGS):
            pred = (1 - h / T) * (rho[h] - t_T / T) / (1 - t_T / T)
            rows.append(
                {
                    "T": T,
                    "T_over_tau": T / t_T,
                    "lag": h,
                    "rho_analytic": rho[h],
                    "atten_predicted": pred / rho[h],
                    "atten_measured": est[:, j].mean() / rho[h],
                    "mean_hat": est[:, j].mean(),
                    "median_hat": float(np.median(est[:, j])),
                }
            )
    bias = pd.DataFrame(rows)
    C.write_csv(bias, "analysis_horizon_acf_bias", index=False)
    print("\nsample-ACF attenuation: mean substitution is a lower bound on it")
    print(bias.round(4).to_string(index=False))

    # extensive statistics, and the estimator of IDC itself
    long_path = model.simulate(400 * BINS_PER_DAY, np.random.default_rng(SEED + 7))
    rows = []
    for d in HZ_HORIZON_D:
        T = n_trace if d == 138.25 else int(round(d * BINS_PER_DAY))
        paths = [
            model.simulate(T, np.random.default_rng(SEED + 1300 + i)) for i in range(HZ_N_PATHS_EXT)
        ]
        mx = np.array([x.max() for x in paths])
        rows.append(
            {
                "days": d,
                "T": T,
                "median_max": float(np.median(mx)),
                "mean_max": float(mx.mean()),
                # the marginal's 1 - 1/T quantile: an iid series of this length
                "marginal_q": float(np.quantile(long_path, 1 - 1 / T)),
                "median_longest_quiet": float(np.median([_longest_zero_run(x) for x in paths])),
                "idc_measured": float(np.mean([idc(x) for x in paths])),
                "idc_analytic": g0 / m,
            }
        )
    ext = pd.DataFrame(rows)
    ext["idc_attenuation"] = ext["idc_measured"] / ext["idc_analytic"]
    C.write_csv(ext, "analysis_horizon_extensive", index=False)
    print("\nextensive statistics grow with T; base-bin IDC converges from below")
    print(ext.round(3).to_string(index=False))


def load_scaling() -> None:
    """How each base-bin statistic moves with the load factor, closed form and measured.

    Under the proportional kappa rule all four follow from one variance
    decomposition with R = E[lambda^2]/m^2 at alpha = 1: the mean is linear, IDC
    affine, the ACF saturating, and P(N=0) takes alpha inside the expectation.
    """
    from scipy.optimize import brentq
    from scipy.stats import norm

    from seren_countfirst.metrics import acf, idc
    from seren_countfirst.models.gated import (
        _base_acf,
        _gamma0,
        _second_moment_ratio,
        _zero_prob,
    )

    st = C.state()
    model, cal = C.winner(st)
    p = dict(model.params_)
    n_bins = int(st["n_bins"])
    m, kappa = float(p["mean"]), float(p["kappa"])
    R = _second_moment_ratio(p["a"], p["b"], p["c"], p["sigma_fast"])
    A, B = 1.0 + m * R / kappa, m * (R - 1.0)

    def analytic(alpha: float, rule: str = "proportional") -> dict:
        m_a = alpha * p["mean"]
        k_a = alpha * p["kappa"] if rule == "proportional" else p["kappa"]
        args = (m_a, p["a"], p["b"], p["c"], p["sigma_fast"])
        var = _gamma0(*args, k_a)
        row = {
            "alpha": alpha,
            "mean": m_a,
            "var": var,
            "idc": var / m_a,
            "zero_frac": _zero_prob(*args, k_a),
        }
        for h in LS_LAGS:
            row[f"acf_{h}"] = _base_acf(
                *args, k_a, p["phi_slow"], p["phi_fast"], p["phi_long"], p["w_long"], h
            )
        return row

    print(f"model {st['winner']}, kappa_rule = proportional, {LS_N_PATHS} paths per alpha")
    print(f"m = {m:.4f}, kappa = {kappa:.1f}, R = E[lam^2]/m^2 = {R:.3f}")
    print(f"IDC(alpha) = {A:.3f} + {B:.3f} * alpha      (A = 1 + mR/kappa, B = m(R-1))")
    print(
        f"ACF saturates: alpha_half = A/B = {A / B:.4f}, so alpha=1 is at "
        f"{1 / (1 + A / B):.3f} of the ceiling\n"
    )

    # rho_alpha(h) = rho_inf(h) * s(alpha) exactly: one alpha-dependent scalar
    # times an alpha-independent shape, so tau_int at load alpha follows from
    # tau_int at alpha = 1 without re-summing 13k lags per load factor
    tau1 = tau_int(analytic_rho(p, min(n_bins, 20000)), n_bins)

    def s(al: float) -> float:
        return al / (al + A / B)

    rows = []
    for a in LS_ALPHAS:
        row = analytic(a)
        row["idc_affine"] = A + B * a
        row["tau_int_bins"] = 1.0 + (s(a) / s(1.0)) * (tau1 - 1.0)
        row["cv_predicted"] = float(
            np.sqrt(row["var"] * row["tau_int_bins"] / n_bins) / row["mean"]
        )
        row["idc_kappa_constant"] = analytic(a, rule="constant")["idc"]
        paths = [
            model.simulate(n_bins, np.random.default_rng(SEED + 200 + i), alpha=cal * a)
            for i in range(LS_N_PATHS)
        ]
        row["idc_realised"] = float(np.median([idc(x) for x in paths]))
        row["zero_frac_realised"] = float(np.median([(x == 0).mean() for x in paths]))
        ac = np.array([acf(x, LS_LAGS).to_numpy() for x in paths])
        for j, h in enumerate(LS_LAGS):
            row[f"acf_{h}_realised"] = float(np.median(ac[:, j]))
        totals = np.array([x.sum() for x in paths])
        row["path_cv"] = float(totals.std(ddof=1) / totals.mean())
        row["max_bin_median"] = float(np.median([x.max() for x in paths]))
        rows.append(row)
    df = pd.DataFrame(rows)
    C.write_csv(df, "analysis_load_scaling_stats", index=False)

    print("dispersion - affine in alpha, and the kappa rule barely matters:")
    print(
        df[["alpha", "idc", "idc_affine", "idc_realised", "idc_kappa_constant"]]
        .round(2)
        .to_string(index=False)
    )
    print("\ncorrelation - nearly alpha-invariant above alpha = 1:")
    cols = ["alpha"] + [f"acf_{h}" for h in LS_LAGS] + [f"acf_{h}_realised" for h in LS_LAGS]
    print(df[cols].round(4).to_string(index=False))
    print("\nemptiness, run-to-run rate error, busiest bin:")
    print(
        df[
            [
                "alpha",
                "zero_frac",
                "zero_frac_realised",
                "tau_int_bins",
                "cv_predicted",
                "path_cv",
                "max_bin_median",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    # tests whether P(0 | alpha) is probit-linear in log alpha with slope -1/(a+b)
    z = norm.ppf(df["zero_frac"].to_numpy())
    slope, intercept = np.polyfit(np.log(df["alpha"].to_numpy()), z, 1)
    resid = float(np.max(np.abs(z - (intercept + slope * np.log(df["alpha"])))))
    print(
        f"\nP(0|alpha) = Phi({intercept:.4f} {slope:+.4f} log alpha), "
        f"max residual {resid:.4f} in probit units"
    )
    print(
        f"  gate+ramp exponent a+b = {p['a'] + p['b']:.3f}, -1/(a+b) = {-1 / (p['a'] + p['b']):.4f}"
    )

    # the load at which idleness ends, against the Poisson reference
    def p0_gap(la: float, target: float) -> float:
        return analytic(float(np.exp(la)))["zero_frac"] - target

    print("\nload needed to reach a given idle fraction:")
    for target in (0.50, 0.25, 0.10):
        a_gli = float(np.exp(brentq(p0_gap, np.log(0.1), np.log(1e9), args=(target,))))
        print(
            f"  {target:.0%} empty bins: alpha = {a_gli:>12,.1f}   "
            f"Poisson needs alpha = {-np.log(target) / m:.3f}"
        )


def latent_frontier() -> None:
    """Whether an ungated latent NB can reach the trace's three targets at once.

    Matching the dispersion confines (V, kappa) to one curve; this scans it for
    the largest zero fraction whose ACF ceiling still clears the target.
    """
    from seren_countfirst.metrics import hourly_acf1_from_autocov

    # Gauss-Hermite nodes for the lognormal expectations
    z_nodes, w = np.polynomial.hermite_e.hermegauss(200)
    w = w / w.sum()

    def zero_frac(m: float, V: float, kappa: float) -> float:
        lam = m * np.exp(np.sqrt(V) * z_nodes - 0.5 * V)
        return float(w @ (1.0 + lam / kappa) ** (-kappa))

    def acf_ceiling(m: float, V: float, gamma0: float) -> float:
        gamma = {h: m * m * (np.exp(V) - 1.0) for h in range(1, 8)}
        return hourly_acf1_from_autocov(gamma0, gamma)

    tm = C.state()["trace_marginals"]
    m, idc_v = float(tm["mean"]), float(tm["IDC_15m"])
    p0_target, acf_target = float(tm["zero_frac"]), float(tm["acf_1h_lag1"])
    gamma0 = idc_v * m
    const = 1.0 + (idc_v - 1.0) / m

    print(
        f"trace: mean {m:.4f}, IDC {idc_v:.3f}, P(0) {p0_target:.4f}, "
        f"hourly lag-1 ACF {acf_target:.4f}"
    )
    print(f"variance constraint: exp(V)(1 + 1/kappa) = {const:.4f}, so V <= {np.log(const):.4f}\n")

    rows = []
    for kappa in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 40.0, 100.0, 1e3, 1e6):
        E = const / (1.0 + 1.0 / kappa)
        if E <= 1.0:
            continue
        V = float(np.log(E))
        rows.append(
            {
                "kappa": kappa,
                "V": V,
                "zero_frac": zero_frac(m, V, kappa),
                "acf_ceiling": acf_ceiling(m, V, gamma0),
            }
        )
    table = pd.DataFrame(rows)
    table["acf_reachable"] = table["acf_ceiling"] >= acf_target
    table["zero_frac_reached"] = table["zero_frac"] >= p0_target
    C.write_csv(table, "analysis_plain_latent_frontier", index=False)
    print("along the variance curve, the two targets pull opposite ways:")
    print(table.round(4).to_string(index=False))

    best = None
    for kappa in np.exp(np.linspace(np.log(0.05), np.log(1e7), 4000)):
        E = const / (1.0 + 1.0 / kappa)
        if E <= 1.0:
            continue
        V = float(np.log(E))
        if acf_ceiling(m, V, gamma0) >= acf_target:
            p0 = zero_frac(m, V, kappa)
            if best is None or p0 > best[0]:
                best = (p0, kappa, V)
    print(
        f"\nlargest P(0) attainable with the mean, the dispersion and the ACF target: {best[0]:.4f}"
    )
    print(f"  at kappa {best[1]:.3g}, V {best[2]:.4f}, and only as phi -> 1")
    print(
        f"trace requires {p0_target:.4f}: short by {p0_target - best[0]:.4f}. "
        f"No admissible parameter set reproduces the three together."
    )


INPUTS = ("state", "counts", "models")

VIEWS = {
    "rate-error": rate_error,
    "horizon": horizon,
    "load-scaling": load_scaling,
    "latent-frontier": latent_frontier,
}

if __name__ == "__main__":
    C.run(VIEWS, "rate-error", "Supporting numeric studies: tables and csv, no figures.", INPUTS)
