from __future__ import annotations

import time

import numpy as np

from ..context import PipelineContext, resolve_context
from ..model_registry import FIT_TARGETS, build_model


def stage_fit(which: list[str], context: PipelineContext | None = None) -> None:
    context = resolve_context(context)
    state = context.load_state()
    models = state.get("models", {})
    for arg in which:
        if arg not in FIT_TARGETS:
            raise SystemExit(f"unknown model {arg}; expected one of {sorted(FIT_TARGETS)}")
        key = FIT_TARGETS[arg]
        t = time.time()
        m = build_model(key, state, context.seed)
        models[key] = m
        rep = m.report
        if rep is None:
            implied = m.implied_marginals()
            print(
                f"[{key}] {m.name}: {time.time() - t:.0f}s  moment-calibrated (no likelihood fit)"
            )
            print("   implied:", {k: round(float(v), 3) for k, v in implied.items()})
        else:
            print(
                f"[{key}] {m.name}: {time.time() - t:.0f}s  ll={rep.loglik:.1f}  AIC={rep.aic:.1f}  "
                f"converged={rep.converged}  warnings={rep.boundary_warnings or '-'}"
            )
        print(
            "   params:",
            {
                k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else round(float(v), 3))
                for k, v in m.params_.items()
                if not isinstance(v, np.ndarray) or v.size <= 4
            },
        )
    state["models"] = models
    context.save_state(state)
