from __future__ import annotations


import pandas as pd

from ...artifacts import write_csv
from ...data import expanding_window_folds
from ...metrics import base_marginals, quiet_run_vs_geometric
from ...evaluation import benchmark_table, evaluate_candidate
from ..config import FOLD_KWARGS, N_PATHS_EVAL, N_PATHS_FOLD
from ..context import PipelineContext, resolve_context
from ..randomness import SeedStream, generator, seed
from ..state import fitted_models


def _select_winner(results: list[dict]) -> str:
    """Return the key of the candidate passing the most gates, ties broken by D_IDC.

    `evaluate_gates` stops at the first failure, so the number of passed
    results is how far a candidate got up the sequential ladder.
    """

    def rank(res: dict) -> tuple:
        passed = sum(g.passed for g in res["gates"].results)
        return (-passed, float(res["scores"]["D_IDC"]))

    candidates = [r for r in results if r.get("role", "candidate") == "candidate"]
    if not candidates:
        raise SystemExit("no candidate models to select from (benchmarks cannot win)")
    return min(candidates, key=rank)["key"]


def stage_evaluate(context: PipelineContext | None = None) -> None:
    context = resolve_context(context)
    state = context.load_state()
    n, thr, env = state["counts"].values, state["busy_threshold"], state["envelope"]
    models = fitted_models(state)
    folds = expanding_window_folds(state["counts"], **FOLD_KWARGS)
    print(
        f"gate 7 folds: {[f.name for f in folds]} "
        f"(validation {FOLD_KWARGS['validation_days']:.0f}d each)"
    )
    results, marg_rows = (
        [],
        [dict(model="TRACE", role="-", **state["trace_marginals"], gates="-", first_fail="-")],
    )
    for name, model in models.items():
        # the single global scalar rate calibration, applied as an intercept shift
        cal = state["trace_marginals"]["mean"] / float(model.params_["mean"])
        # candidates are graded sequentially; benchmarks and ablations are scored
        # on every gate, to record where an alternative mechanism breaks
        role = getattr(model, "role", "candidate")
        res = evaluate_candidate(
            model,
            n,
            thr,
            env,
            n_paths=N_PATHS_EVAL,
            seed=seed(SeedStream.EVALUATION, base=context.seed),
            fold_n_paths=N_PATHS_FOLD,
            folds=folds,
            idc_envelope=state.get("idc_envelope"),
            stop_at_first_failure=(role == "candidate"),
            alpha=cal,
        )
        res["key"] = name
        results.append(res)
        stats = pd.DataFrame([base_marginals(p) for p in res["paths"]]).mean().to_dict()
        gr = res["gates"]
        marg_rows.append(
            dict(
                model=name,
                role=role,
                rate_cal=round(cal, 3),
                **stats,
                gates="PASS" if gr.passed else "FAIL",
                first_fail=(gr.first_failure.name if gr.first_failure else "-"),
            )
        )
        write_csv(gr.to_frame(), context.result(f"gates_{name}.csv"), index=False)
    marg = pd.DataFrame(marg_rows)
    bench = benchmark_table(results)
    write_csv(marg, context.result("marginals_by_model.csv"), index=False)
    write_csv(bench, context.result("benchmark_table.csv"), index=False)
    print(f"four-marginal reproduction (mean over {N_PATHS_EVAL} paths):")
    print(marg.round(3).to_string(index=False))
    print("\nranking / gate table:")
    print(bench.round(4).to_string(index=False))

    winner = _select_winner(results)
    print(f"\nwinner by gates passed, then D_IDC: {winner}")
    model = models[winner]
    fold_ev = next(r["fold_evidence"] for r in results if r["key"] == winner)
    if fold_ev is not None and len(fold_ev):
        write_csv(fold_ev, context.result(f"folds_{winner}.csv"), index=False)
        worst = fold_ev.loc[fold_ev["Z"].idxmax()]
        print(
            f"gate 7 worst Z {worst['Z']:.2f} ({worst['metric']} on {worst['fold']}); "
            f"per-fold worst: {fold_ev.groupby('fold')['Z'].max().round(2).to_dict()}"
        )
    cal_w = state["trace_marginals"]["mean"] / float(model.params_["mean"])
    q = quiet_run_vs_geometric(
        model.simulate(
            state["n_bins"], generator(SeedStream.BASELINE, base=context.seed), alpha=cal_w
        )
    )
    write_csv(q, context.result(f"quiet_run_{winner}.csv"), index=True)
    print("\nwinner quiet-run vs geometric:", {k: round(v, 2) for k, v in q.items()})
    state["winner"] = winner
    context.save_state(state)
