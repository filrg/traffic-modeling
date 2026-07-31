"""Marginal feasibility diagnostic (appendix A.4)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# marginal feasibility diagnostic (appendix A.4)
def marginal_feasibility(mean: float, variance: float, zero_fraction: float, target_acf: float,
                         kappa_grid: Sequence[float] | None = None) -> pd.DataFrame:
    """Sweep the constraint e^V (1 + 1/kappa) = C and report what is reachable.

    Any Gaussian log-latent + NB2 model that matches `mean` and `variance` at
    the base bin lies on this one-dimensional curve, so if no point on it
    reaches both the zero fraction and the ACF, the family is infeasible.
    """
    from scipy import integrate, stats

    m, C = float(mean), 1.0 + (float(variance) - float(mean)) / float(mean) ** 2
    grid = kappa_grid or [50.0, 10.0, 6.5, 3.0, 1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.055]
    rows = []
    for kappa in grid:
        eV = C / (1.0 + 1.0 / kappa)
        if eV <= 1.0:
            continue
        V = float(np.log(eV))
        mu, s = np.log(m) - V / 2.0, np.sqrt(V)
        p0 = integrate.quad(
            lambda z: stats.norm.pdf(z) * (1.0 + np.exp(mu + s * z) / kappa) ** (-kappa), -8, 8, limit=200
        )[0]
        rows.append(
            {
                "kappa": kappa,
                "V": V,
                "sigma_total": s,
                "P_zero": float(p0),
                "max_acf": float(m * m * (np.exp(V) - 1.0) / variance),
                "zero_ok": bool(abs(p0 - zero_fraction) < 0.02),
                "acf_ok": bool(m * m * (np.exp(V) - 1.0) / variance >= target_acf),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["C"] = C
    out.attrs["feasible"] = bool((out["zero_ok"] & out["acf_ok"]).any())
    return out
