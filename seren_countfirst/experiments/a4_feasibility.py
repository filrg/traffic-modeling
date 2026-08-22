from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ..bootstrap import bootstrap_metric_envelope
from ..evaluation import benchmark_table, evaluate_candidate
from ..metrics import (
    HOURLY_ACF_LAG_WEIGHTS,
    acf,
    base_marginals,
    hourly_acf1_from_autocov,
    metric_vector,
)
from ..models import IIDNegBin, NBLatentAR, NBRegimeSwitch, marginal_feasibility
from ..models import NBGatedLatent

# Seren SFT subset, 15-minute batch counts
TRACE = {
    "mean": 3.747985,
    "variance": 274.206902,
    "zero_fraction": 0.774655,
    "acf_1h_lag1": 0.664880,
    "n_bins": 13273,
}


def calibrate_surrogate(mean, variance, zero_fraction, acf_1h_target, kappa=8.0,
                        tau_fast_bins=2.0, dwell_off_bins=32.0, lam_off=0.01):
    """Calibrate a regime level with continuous spread inside the active state.

    A bare two-state Markov level cannot reach the trace marginals, so the excess
    variance sits in a persistent latent:

        log lambda_t = mu_{z_t} + x_t - V/2,   z_t a two-state Markov chain,
                                               x_t a stationary AR(1)

    with kappa fixed. The base-bin variance pins the latent variance given
    `pi_off`, leaving the zero fraction to determine `pi_off`.
    """
    from scipy import integrate, stats

    m, g0, kappa = float(mean), float(variance), float(kappa)
    D = g0 - m + m ** 2  # = E[lam^2] (1 + 1/kappa)

    def pieces(p):
        lam_on = (m - p * lam_off) / (1.0 - p)
        second_raw = p * lam_off ** 2 + (1.0 - p) * lam_on ** 2
        eV = D / ((1.0 + 1.0 / kappa) * second_raw)
        return lam_on, second_raw, eV

    def zero_gap(p):
        lam_on, _, eV = pieces(p)
        if eV <= 1.0:
            return 1.0
        V, s = np.log(eV), np.sqrt(np.log(eV))

        def state_zero(lam):
            return integrate.quad(
                lambda z: stats.norm.pdf(z) * (1.0 + lam * np.exp(s * z - V / 2) / kappa) ** (-kappa),
                -8, 8, limit=200,
            )[0]

        return p * state_zero(lam_off) + (1.0 - p) * state_zero(lam_on) - zero_fraction

    p = brentq(zero_gap, 0.05, 0.95, xtol=1e-12)
    lam_on, second_raw, eV = pieces(p)
    V = float(np.log(eV))
    sigma_fast = float(np.sqrt(V))
    lam = np.array([lam_off, lam_on])

    # regime persistence is pinned by `dwell_off_bins`; the latent half-life is
    # then solved against the hourly ACF
    p_off_on = 1.0 / float(dwell_off_bins)
    p_on_off = p * p_off_on / (1.0 - p)
    rho = 1.0 - p_off_on - p_on_off
    P = np.array([[1 - p_off_on, p_off_on], [p_on_off, 1 - p_on_off]])

    gamma0 = float(m + second_raw * eV / kappa + second_raw * eV - m ** 2)

    def gamma(h, phi):
        pi = np.array([p, 1 - p])
        swing = p * (1 - p) * rho ** h
        joint = np.outer(pi, pi) + swing * np.array([[1.0, -1.0], [-1.0, 1.0]])
        return float(np.sum(joint * np.outer(lam, lam)) * np.exp(V * phi ** h) - m ** 2)

    def hourly_acf1(tau):
        phi = 2.0 ** (-1.0 / max(tau, 1e-6))
        return hourly_acf1_from_autocov(gamma0, {h: gamma(h, phi) for h in HOURLY_ACF_LAG_WEIGHTS})

    lo, hi = 0.05, 5000.0
    if (hourly_acf1(lo) - acf_1h_target) * (hourly_acf1(hi) - acf_1h_target) < 0:
        tau_fast_bins = brentq(lambda t: hourly_acf1(t) - acf_1h_target, lo, hi, xtol=1e-8)
    phi = float(2.0 ** (-1.0 / tau_fast_bins))

    model = NBRegimeSwitch(n_states=2, fast_ar=True)
    model.params_ = {
        "lam_states": lam,
        "mu_states": np.log(lam),
        "kappa": kappa,
        "P": P,
        "pi": np.array([p, 1 - p]),
        "mean": m,
        "expected_dwell_bins": 1.0 / np.maximum(1.0 - np.diag(P), 1e-12),
        "sigma_fast": sigma_fast,
        "phi_fast": phi,
        "half_life_fast_bins": float(tau_fast_bins),
    }
    model.freeze()
    return model, {"pi_off": p, "lam_on": lam_on, "kappa": kappa, "V_on": V,
                   "sigma_spread": sigma_fast, "rho_regime": rho, "tau_fast_bins": float(tau_fast_bins),
                   "dwell_off_bins": 1 / p_off_on, "dwell_on_bins": 1 / p_on_off}


def _gated_moment(x: np.ndarray):
    """Build rung 4b by moments, matching how the production model is calibrated."""
    from ..run_pipeline import LONG_ACF_LAGS, MM4_KNOBS, SEED

    bm = base_marginals(x)
    return NBGatedLatent.calibrate_debiased(
        mean=bm["mean"], variance=bm["IDC_15m"] * bm["mean"],
        zero_fraction=bm["zero_frac"], acf_1h=bm["acf_1h_lag1"],
        acf_long={int(h): float(v) for h, v in acf(x, LONG_ACF_LAGS).items()},
        n_bins=len(x), seed=SEED, **MM4_KNOBS)


# sized by gate 1, as in the pipeline; same value as N_PATHS_EVAL
N_PATHS_LADDER = 1200


def main(seed: int = 20240728, n_paths: int = N_PATHS_LADDER, n_surrogate_paths: int = 200,
         verbose: bool = True) -> dict:
    t0 = time.time()
    print("=" * 78)
    print("A.4  analytic sweep along the constraint  e^V (1 + 1/kappa) = C")
    print("=" * 78)
    table = marginal_feasibility(TRACE["mean"], TRACE["variance"],
                                 TRACE["zero_fraction"], TRACE["acf_1h_lag1"])
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nC = {table.attrs['C']:.3f}   any (V, kappa) satisfying both targets: {table.attrs['feasible']}")

    print("\n" + "=" * 78)
    print("Surrogate with the trace's four base-bin statistics")
    print("=" * 78)
    truth, info = calibrate_surrogate(TRACE["mean"], TRACE["variance"],
                                      TRACE["zero_fraction"], TRACE["acf_1h_lag1"])
    print({k: round(float(v), 4) for k, v in info.items()})
    targets = {
        "mean": TRACE["mean"],
        "IDC_15m": TRACE["variance"] / TRACE["mean"],
        "zero_frac": TRACE["zero_fraction"],
        "acf_1h_lag1": TRACE["acf_1h_lag1"],
    }

    # the construction pins the targets in expectation only, so it is checked
    # against the ensemble rather than against one path
    rng = np.random.default_rng(seed)
    draws = [truth.simulate(TRACE["n_bins"], rng) for _ in range(n_surrogate_paths)]
    ens = pd.DataFrame([base_marginals(x) for x in draws])
    print(f"\ntarget vs surrogate ensemble ({n_surrogate_paths} paths):")
    for k, tgt in targets.items():
        print(f"  {k:>14}: target {tgt:9.4f}   ensemble mean {ens[k].mean():9.4f}"
              f"   median {ens[k].median():9.4f}   per-path CV {ens[k].std(ddof=1)/ens[k].mean():.3f}")

    # one realisation, picked by total log-distance from the ensemble median
    # across all four facts; the mean alone does not identify a typical path
    dev = sum((np.log(ens[k] / ens[k].median())).abs() for k in targets)
    pick = int(dev.idxmin())
    surrogate = draws[pick]
    got = base_marginals(surrogate)
    pct = {k: float((ens[k] < got[k]).mean()) for k in targets}
    print(f"\nladder runs on the most typical path (#{pick}, closest to the ensemble median "
          f"on all four facts); its percentile within the ensemble:")
    for k in targets:
        print(f"  {k:>14}: {got[k]:9.4f}   ({pct[k]:.0%} of paths below)")

    print("\n" + "=" * 78)
    print("Ladder on the surrogate (same harness, same data view)")
    print("=" * 78)
    busy_threshold = float(np.quantile(surrogate[surrogate > 0], 0.80))
    envelope = bootstrap_metric_envelope(
        surrogate, lambda x: metric_vector(x, busy_threshold), method="stationary_24h", n_rep=120, seed=seed
    )

    # each entry builds its own estimate: the rungs are not all fitted the same
    # way, and 4b is moment-calibrated in production
    candidates = [
        ("rung 2  iid NB", lambda x: IIDNegBin().fit(x)),
        ("rung 3  NB + 1 AR", lambda x: NBLatentAR(1).fit(x, n_restarts=2, maxiter=300)),
        ("rung 4  NB + 2 AR", lambda x: NBLatentAR(2).fit(x, n_restarts=2, maxiter=300)),
        ("rung 4b gated (moments)", _gated_moment),
        ("bench.  NB + regime", lambda x: NBRegimeSwitch(n_states=2, fast_ar=False).fit(x)),
        ("bench.  NB + regime + AR", lambda x: NBRegimeSwitch(n_states=2, fast_ar=True).fit(x)),
    ]
    results = []
    for label, build in candidates:
        t = time.time()
        model = build(surrogate)
        res = evaluate_candidate(model, surrogate, busy_threshold, envelope,
                                 n_paths=n_paths, seed=seed + 1)
        res["label"] = label
        results.append(res)
        gates = res["gates"]
        fail = gates.first_failure
        print(f"  {label:<26} fitted in {time.time()-t:6.1f}s   "
              f"gates {'PASS' if gates.passed else 'FAIL at ' + fail.name}")
        if verbose and not gates.passed:
            print(f"      -> {fail.detail}")

    print("\nPer-candidate reproduction of the four target statistics:")
    rows = []
    for res in results:
        stats = pd.DataFrame([base_marginals(p) for p in res["paths"]]).mean()
        rows.append({"model": res["label"], **stats.to_dict(),
                     "gates_passed": res["gates"].passed,
                     "first_failed_gate": (res["gates"].first_failure.name if res["gates"].first_failure else "-")})
    summary = pd.DataFrame(rows)
    ref = pd.DataFrame([{"model": "SURROGATE (target)", **base_marginals(surrogate),
                         "gates_passed": True, "first_failed_gate": "-"}])
    print(pd.concat([ref, summary], ignore_index=True).to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))

    print("\nRanking scores (lower is better):")
    print(benchmark_table(results).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\ntotal {time.time()-t0:.0f}s")
    return {"surrogate": surrogate, "results": results, "feasibility": table}


if __name__ == "__main__":
    main()
