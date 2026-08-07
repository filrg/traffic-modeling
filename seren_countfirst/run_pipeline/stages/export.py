"""Freeze the winner into the model card."""
from __future__ import annotations

import json

import numpy as np

from ... import microstructure
from ...paths import RESULTS
from ..config import N_PATHS_SCALING, SEED
from ..state import _load, _models


def stage_export() -> None:
    state = _load()
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = state.get("winner", "gated_mm")
    model = _models(state)[name]
    model.freeze()
    card = {
        "model": model.name,
        "semantics": "single SFT workload population; alpha = intensity scaling of the active level; "
                     "gate_shift = participation scaling (different semantics, never mixed with alpha); "
                     "superposition = independent populations",
        "provenance": {"trace_sha256": state["trace_sha256"], "n_bins": state["n_bins"],
                       "resolution": "15min", "seed": SEED},
        "params": {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else float(v))
                   for k, v in model.params_.items()},
        "fit": (model.report.to_dict() if model.report else {"estimator": "moment-calibrated; likelihood refit with larger budget pending"}),
        "trace_marginals": state["trace_marginals"],
        "trace_idc": state["trace_idc"],
        "handle": "GLI-NB (gated latent-intensity negative binomial)",
        "known_failure_modes": [
            "all eight gates pass on 8 of 10 scoring seeds on the 2026-08-07 "
            "certification at (1200, 120, 300): gate 1 fails on one seed (0.0591 vs "
            "0.05) and gate 8 on another (1.170 vs 1.0), both on scoring noise with "
            "the model fixed. Quote the worst case over seeds, never a single seed",
            "gates 5 and 7 clear by 2.8% (2.917) and 3.4% (2.898); the binding gates "
            "are 1 and 8, not 5",
            "gate 5 is in-sample (sigma_fast / tau_fast / kappa were tuned against it); "
            "gate 7 is the out-of-sample evidence",
            "realised IDC_15m and hourly ACF sit below the trace because the finite-length "
            "correction is applied to the long-lag ACF targets only",
            "IDC at 7d is ungraded: 138 days supports fewer than 8 blocks at that scale",
            "whole-second export saturates from alpha = 2 (5 of 8 paths) and fails by alpha = 4; "
            "reconstruct with on_overflow='subsecond', which refines only the bins that cannot fit "
            "(alpha = 16 needs 1/11 s in 1.6% of bins) and is an exact no-op at alpha <= 1",
            "job-size model out of scope here (separate absorbed-work pipeline; "
            "compose the two with the `workload` stage)",
        ],
        "frozen_decisions": [
            "kappa_alpha = 'proportional' (frozen 2026-07-30). Sum of alpha iid NB2(mu, kappa) "
            "is NB2(alpha*mu, alpha*kappa) exactly, so the rule states that alpha superposes "
            "alpha independent submitters onto one shared latent driver; 'constant' would let "
            "per-batch clumping grow with load, a size-like effect, and job size is modelled "
            "elsewhere. The rules coincide at alpha = 1 and differ by under 2% on IDC, since "
            "the NB layer carries 2.56% of the dispersion (IDC(1) = 74.92 = 1 + 1.89 + 72.03)",
            "GateEvidence.idc_drift_tol = 1.0 is a declared engineering tolerance, not a "
            "measurement; gate 8's worst over 10 seeds at N_PATHS_SCALING = 300 is 1.170, "
            "a failure on 1 seed. The binding sub-check flips between IDC_1d drift "
            "(0.70-0.79) and rate_Z (2.72-3.51), and the failure is a rate_Z at alpha = 8, "
            "outside the alpha_range that bounds gate 8's IDC and ACF checks but not its "
            "rate check",
        ],
        "open_decisions": [
            "microstructure gates in evaluate: a ninth structural gate would invalidate the "
            "eight-gate certification, so it is deliberately not added",
        ],
    }
    (RESULTS / "model_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps({k: card[k] for k in ("model", "params", "open_decisions")}, indent=2)[:1200])
    print("export done -> results/model_card.json")
