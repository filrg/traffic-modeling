"""Fit or calibrate the rungs of the ladder, and the rows that are not rungs."""
from __future__ import annotations

import time

import numpy as np

from ... import IIDNegBin, NBGatedLatent, NBINGARCH, NBLatentAR, ONOFFCounts, acf
from ..config import LONG_ACF_LAGS, MM4_KNOBS, RUNG_KEYS, SEED
from ..state import _load, _save

# state key -> published model name for the gated family
_GATED_NAMES = {"gated_mm": "nb_gated_mm", "gated_mm2": "nb_gated_mm2",
                "gated_mm3": "nb_gated_mm3", "gated_mm4": "nb_gated_mm4",
                "gated_mm5": "nb_gated_mm5", "gated_nogate": "nb_gated_mm4_nogate"}


def _fit_one(name: str, state: dict):
    """Fit or calibrate one rung on the trace counts."""
    n = state["counts"].values
    if name == "r2":
        return IIDNegBin().fit(n)
    if name == "r3":
        return NBLatentAR(1).fit(n, n_restarts=2, maxiter=250)
    if name == "r4":
        return NBLatentAR(2).fit(n, n_restarts=2, maxiter=250)
    if name == "gated":
        return NBGatedLatent().fit(n, n_restarts=1, maxiter=80, seed=SEED)
    if name in ("onoff", "onoff_pois"):
        return ONOFFCounts.calibrate(n, emission="nb" if name == "onoff" else "poisson")
    if name in ("ingarch", "ingarch_mle"):
        # moment-calibrated on the same footing as the mm* rungs. `ingarch_mle`
        # keeps the likelihood route: it is the one exactly comparable AIC, and
        # it records that the MLE leaves the second-order stationary region.
        if name == "ingarch_mle":
            return NBINGARCH().fit(n, n_restarts=3, maxiter=400, seed=SEED)
        tm, a_tr = state["trace_marginals"], acf(n, (1, 4))
        model = NBINGARCH.calibrate(mean=tm["mean"], variance=tm["IDC_15m"] * tm["mean"],
                                    acf_lag1=float(a_tr[1]), acf_lag4=float(a_tr[4]), seed=SEED)
        return model
    if name in _GATED_NAMES:
        # moment calibration against the frozen trace facts, never a likelihood fit
        tm = state["trace_marginals"]
        args = dict(mean=tm["mean"], variance=tm["IDC_15m"] * tm["mean"],
                    zero_fraction=tm["zero_frac"], acf_1h=tm["acf_1h_lag1"])
        if name == "gated_mm":
            model = NBGatedLatent.calibrate(**args)
        else:
            # second slow scale calibrated to the trace's own long-lag ACF
            long_targets = {int(h): float(v) for h, v in acf(n, LONG_ACF_LAGS).items()}
            print("   long-lag ACF targets:", {k: round(v, 4) for k, v in long_targets.items()})
            if name == "gated_mm2":
                model = NBGatedLatent.calibrate(**args, acf_long=long_targets)
            else:
                # mm3/mm4 match at the trace's own length rather than in the limit
                knobs = MM4_KNOBS if name in ("gated_mm4", "gated_mm5", "gated_nogate") else {}
                if name == "gated_nogate":
                    # sharpness 0 makes the gate a constant 1/2, absorbed by log Z:
                    # mm4 with the gate removed, everything else untouched
                    knobs = {**knobs, "sharpness": 0.0}
                # mm5 debiases all four calibration targets, not just acf_long
                model = NBGatedLatent.calibrate_debiased(
                    **args, acf_long=long_targets, n_bins=state["n_bins"], seed=SEED,
                    marginals=(name == "gated_mm5"), **knobs
                )
        model.name = _GATED_NAMES[name]
        if name == "gated_nogate":
            model.role = "ablation"  # reported beside mm4, but never selectable
        return model
    raise SystemExit(f"unknown rung {name}")


def stage_fit(which: list[str]) -> None:
    state = _load()
    models = state.get("models", {})
    for arg in which:
        if arg not in RUNG_KEYS:
            raise SystemExit(f"unknown rung {arg}; expected one of {sorted(RUNG_KEYS)}")
        key = RUNG_KEYS[arg]
        t = time.time()
        m = _fit_one(key, state)
        models[key] = m
        rep = m.report
        if rep is None:
            implied = m.implied_marginals()
            print(f"[{key}] {m.name}: {time.time()-t:.0f}s  moment-calibrated (no likelihood fit)")
            print("   implied:", {k: round(float(v), 3) for k, v in implied.items()})
        else:
            print(f"[{key}] {m.name}: {time.time()-t:.0f}s  ll={rep.loglik:.1f}  AIC={rep.aic:.1f}  "
                  f"converged={rep.converged}  warnings={rep.boundary_warnings or '-'}")
        print("   params:", {k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else round(float(v), 3))
                             for k, v in m.params_.items() if not isinstance(v, np.ndarray) or v.size <= 4})
    state["models"] = models
    _save(state)
