from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TypeAlias

import numpy as np
import pandas as pd
from scipy import stats

from ...artifacts import write_csv
from ...data import JOBSIZE_STATES, arrival_view, jobsize_view, load_trace
from ...jobsize import (
    ConditionalJobSize,
    EmpiricalJobSize,
    LogNormalMixture,
    ScipyJobSize,
    TAIL_QUANTILE,
    TwoPartJobSize,
    attach_activity,
    batch_frame,
    bin_grid_index,
    component_scoreboard,
    fit_jobsize,
    jobsize_table,
    offered_work_stats,
    roundtrip_error_s,
    step_time_s,
    write_jobsize_model,
)
from ..context import PipelineContext, resolve_context

_ECHO_GPUS = (1, 8, 16, 64, 256)
_N_COMPOSE = 40
_TAIL_SWEEP = (0.95, 0.98, 0.99, 0.995)
_N_SWEEP = 10

MarginalModel: TypeAlias = LogNormalMixture | ScipyJobSize
DrawRule: TypeAlias = Callable[[np.random.Generator], np.ndarray]


def _compose(
    rule: DrawRule,
    bin_of_job: np.ndarray,
    n_jobs_by_bin: np.ndarray,
    observed_sizes: np.ndarray,
    n_rep: int,
    seed: int,
) -> dict[str, float]:
    rows, ks_scores, mean_ratios = [], [], []
    for replicate in range(n_rep):
        drawn = rule(np.random.default_rng(seed + replicate))
        work = pd.Series(drawn).groupby(bin_of_job).sum().to_numpy()
        rows.append(offered_work_stats(work, n_jobs_by_bin))
        ks_scores.append(float(stats.ks_2samp(drawn, observed_sizes).statistic))
        mean_ratios.append(float(np.mean(drawn) / np.mean(observed_sizes)))
    result = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    result["ks_job_marginal"] = float(np.mean(ks_scores))
    result["mean_work_ratio"] = float(np.mean(mean_ratios))
    return result


def _load_work_units(context: PipelineContext) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    raw = load_trace(context.trace_csv)
    table = jobsize_table(jobsize_view(raw, "SFT"))
    sizes = table["job_size"].to_numpy()
    error = roundtrip_error_s(table)
    print(
        f"SFT jobs with a usable duration and GPU count: {len(table)} "
        f"(states {', '.join(JOBSIZE_STATES)})"
    )
    print(f"inversion round-trip max abs error: {error:.3e} s")
    if error > 1e-6:
        raise RuntimeError(f"work-unit inversion does not round-trip ({error:.3e} s)")

    print("\nwork units:")
    print(table["job_size"].describe().to_string())
    median = float(np.median(sizes))
    print(f"\nthe median job of {median:,.0f} units, replayed at f = 1 GHz:")
    for gpu_count in _ECHO_GPUS:
        seconds = median * float(step_time_s(gpu_count, 1.0))
        print(f"  n = {gpu_count:4d}  ->  {seconds:9.0f} s  ({seconds / 60:7.1f} min)")
    return raw, table, sizes


def _fit_marginal(sizes: np.ndarray, context: PipelineContext) -> MarginalModel:
    board = fit_jobsize(sizes, seed=context.seed)
    print("\ncandidate laws, best AIC first:")
    print(board.drop(columns="params").round(4).to_string(index=False))
    board.assign(
        params=board["params"].map(
            lambda params: json.dumps(list(params) if isinstance(params, tuple) else params)
        )
    ).pipe(write_csv, context.result("jobsize_candidates.csv"), index=False)

    best = board.iloc[0]
    model: MarginalModel
    if best["kind"] == "mixture_lognormal":
        model = LogNormalMixture.from_dict(best["params"]).freeze()
        assert model.report is not None
        print(
            f"\nfrozen law: {best['distribution']}  "
            f"AIC {model.report.aic:,.1f}  BIC {model.report.bic:,.1f}"
        )
        for weight, location, scale in zip(model.weights, model.mu, model.sigma):
            print(
                f"  w {weight:.4f}  mu {location:8.4f}  sigma {scale:.4f}  "
                f"(median {np.exp(location):,.0f} units)"
            )
    else:
        model = ScipyJobSize(name=str(best["distribution"]), params=best["params"])
        print(f"\nfrozen law: {best['distribution']} {model.params}")
    write_jobsize_model(model, context.result("jobsize_model.json"))
    return model


def _fit_structure(
    raw: pd.DataFrame,
    table: pd.DataFrame,
    sizes: np.ndarray,
    marginal: MarginalModel,
    context: PipelineContext,
) -> tuple[pd.DataFrame, pd.DataFrame, ConditionalJobSize]:
    jobs = attach_activity(table, arrival_view(raw, "SFT"))
    batches = batch_frame(jobs)
    log_sizes = np.log(jobs["job_size"].to_numpy())
    conditional = ConditionalJobSize.fit(jobs, seed=context.seed, with_latent=False).freeze()
    log_variance = float(np.var(log_sizes, ddof=1))
    icc = 1.0 - float(conditional.weights @ conditional.sigma_w**2) / log_variance
    log_activity = np.log(batches["activity"].to_numpy(dtype=float))
    r_squared = float(np.corrcoef(log_activity, batches["z_bar"].to_numpy())[0, 1] ** 2)
    print(
        f"\nbatches {len(batches):,} over {jobs['bin'].nunique():,} occupied bins; "
        f"{(batches['cardinality'] > 1).mean():.3f} hold more than one job"
    )
    print(
        f"within-batch sd {np.round(conditional.sigma_w, 4)} by component against "
        f"{np.sqrt(log_variance):.4f} overall  ->  ICC {icc:.4f}"
    )
    print(f"activity slope beta {conditional.beta:+.4f}  (R^2 {r_squared:.4f})")

    orders = component_scoreboard(jobs, conditional.beta, seed=context.seed)
    print("component counts on the activity-adjusted residuals:")
    print(orders.round(4).to_string(index=False))
    print(f"selected k = {conditional.n_components}")
    print(f"\nconditional law: mu {conditional.mu:.4f}  beta {conditional.beta:+.4f}")
    for weight, location, batch_sd, job_sd, within_sd in zip(
        conditional.weights,
        conditional.m,
        conditional.s_batch,
        conditional.s_job,
        conditional.sigma_w,
    ):
        print(
            f"  pi {weight:.4f}  m {location:+8.4f}  s_batch {batch_sd:.4f}  "
            f"sigma_w {within_sd:.4f}  (s_job {job_sd:.4f})"
        )
    write_jobsize_model(conditional, context.result("jobsize_conditional.json"))

    ranked = np.sort(sizes)[::-1]
    analytic_mean = (
        float(np.sum(marginal.weights * np.exp(marginal.mu + 0.5 * marginal.sigma**2)))
        if isinstance(marginal, LogNormalMixture)
        else np.nan
    )
    print(
        f"top 100 of {len(sizes):,} jobs carry "
        f"{ranked[:100].sum() / ranked.sum():.4f} of all work; the frozen marginal's "
        f"analytic mean is {analytic_mean / sizes.mean():.4f} of the sample mean"
    )
    pd.DataFrame(
        [
            {
                "n_jobs": len(jobs),
                "n_batches": len(batches),
                "n_occupied_bins": int(jobs["bin"].nunique()),
                "sd_log_size": float(np.sqrt(log_variance)),
                "sigma_w_mean": float(conditional.weights @ conditional.sigma_w),
                "icc_within_batch": icc,
                "beta": conditional.beta,
                "beta_r2": r_squared,
                "top100_work_share": float(ranked[:100].sum() / ranked.sum()),
                "marginal_mean_ratio": float(analytic_mean / sizes.mean()),
                "n_components": conditional.n_components,
                **{f"aic_k{int(row.k)}": row.AIC for row in orders.itertuples()},
                **{f"slack_k{int(row.k)}": row.tightest_slack for row in orders.itertuples()},
            }
        ]
    ).pipe(write_csv, context.result("jobsize_structure.csv"), index=False)
    return jobs, batches, conditional


def _fit_two_part(
    jobs: pd.DataFrame,
    batches: pd.DataFrame,
    sizes: np.ndarray,
    context: PipelineContext,
) -> tuple[TwoPartJobSize, np.ndarray, np.ndarray, np.ndarray]:
    model = TwoPartJobSize.fit(jobs, seed=context.seed).freeze()
    activity = batches["activity"].to_numpy(dtype=float)
    cardinality = batches["cardinality"].to_numpy(dtype=int)
    bin_index = bin_grid_index(batches["bin"])
    n_tail = int((sizes >= model.cut).sum())
    print(
        f"\ntail cut at the {TAIL_QUANTILE:.1%} quantile = {model.cut:,.0f} units: "
        f"{n_tail} jobs above it"
    )
    print(
        f"exceedances: generalized Pareto shape {model.tail_shape:+.4f}  "
        f"scale {model.tail_scale:,.0f}"
    )
    print(
        f"large jobs per bin: log lambda = {model.rate_a:+.4f} "
        f"{model.rate_b:+.4f} log N_k  -> "
        f"{model.tail_rate(2.0)[()]:.3f} at N_k = 2, "
        f"{model.tail_rate(100.0)[()]:.3f} at N_k = 100"
    )
    bulk = model.bulk
    print(f"bulk law ({model.bulk_kind}): k = {bulk.n_components}, mu {bulk.mu:.4f}")
    print(f"  component locations {np.round(bulk.m, 3)}")
    print(f"  job sd {np.round(bulk.s_job, 3)}   batch sd {np.round(bulk.s_batch, 3)}")
    print(
        f"  weight logit: a {np.round(bulk.logit_a, 3)}  "
        f"b {np.round(bulk.logit_b, 4)}   escape {bulk.escape:.4f}"
    )
    for count in (2, 20, 200):
        print(f"  weights at N_k = {count:3d}: {np.round(bulk.weights([float(count)])[0], 3)}")
    write_jobsize_model(model, context.result("jobsize_two_part.json"))
    return model, activity, cardinality, bin_index


def _composition_checks(
    jobs: pd.DataFrame,
    batches: pd.DataFrame,
    sizes: np.ndarray,
    marginal: MarginalModel,
    conditional: ConditionalJobSize,
    two_part: TwoPartJobSize,
    activity: np.ndarray,
    cardinality: np.ndarray,
    bin_index: np.ndarray,
    context: PipelineContext,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    conditional_two_part = TwoPartJobSize.fit(
        jobs,
        n_init=4,
        seed=context.seed,
        bulk_kind="conditional",
        with_latent=False,
    )
    flat = ConditionalJobSize.fit(
        jobs,
        n_components=conditional.n_components,
        seed=context.seed,
        beta=0.0,
        with_latent=False,
    )
    unshifted = ConditionalJobSize.fit(
        jobs,
        n_components=conditional.n_components,
        seed=context.seed,
        level="log_mean",
        with_latent=False,
    )
    print(
        f"\nlevel: mu {conditional.mu:.4f} matching mean work against "
        f"{unshifted.mu:.4f} at the residual mean, a factor "
        f"{np.exp(conditional.mu - unshifted.mu):.3f} on every job"
    )
    parametric_tail = replace(two_part, tail_kind="genpareto")
    grouped = jobs.groupby("bin")
    n_jobs_by_bin = grouped.size().to_numpy()
    observed = grouped["job_size"].sum().to_numpy()
    bin_of_job: np.ndarray = np.repeat(batches["bin"].to_numpy(), cardinality)
    rules: dict[str, DrawRule] = {
        "iid_per_job": lambda rng: marginal.rvs(len(jobs), rng),
        "two_part_conditional": lambda rng: conditional_two_part.rvs(
            activity, cardinality, bin_index, rng
        ),
        "batch_only": lambda rng: flat.rvs(activity, cardinality, rng),
        "conditional_mean_work": lambda rng: conditional.rvs(activity, cardinality, rng),
        "conditional_log_mean": lambda rng: unshifted.rvs(activity, cardinality, rng),
        "two_part": lambda rng: two_part.rvs(activity, cardinality, bin_index, rng),
        "two_part_genpareto": lambda rng: parametric_tail.rvs(
            activity, cardinality, bin_index, rng
        ),
    }
    rows = [
        {
            "rule": "observed",
            **offered_work_stats(observed, n_jobs_by_bin),
            "ks_job_marginal": 0.0,
            "mean_work_ratio": 1.0,
        }
    ]
    rows.extend(
        {
            "rule": name,
            **_compose(
                rule,
                bin_of_job,
                n_jobs_by_bin,
                sizes,
                _N_COMPOSE,
                context.seed,
            ),
        }
        for name, rule in rules.items()
    )
    checks = pd.DataFrame(rows)
    print(f"\nwork per occupied bin, {_N_COMPOSE} replicates on the trace's own batches:")
    print(checks.round(4).to_string(index=False))
    write_csv(checks, context.result("jobsize_conditional_checks.csv"), index=False)
    return checks, bin_of_job, n_jobs_by_bin


def _tail_threshold_sweep(
    jobs: pd.DataFrame,
    sizes: np.ndarray,
    activity: np.ndarray,
    cardinality: np.ndarray,
    bin_index: np.ndarray,
    bin_of_job: np.ndarray,
    n_jobs_by_bin: np.ndarray,
    context: PipelineContext,
) -> pd.DataFrame:
    rows = []
    for quantile in _TAIL_SWEEP:
        model = TwoPartJobSize.fit(jobs, quantile=quantile, n_init=4, seed=context.seed)

        def draw(rng: np.random.Generator) -> np.ndarray:
            return model.rvs(activity, cardinality, bin_index, rng)

        rows.append(
            {
                "quantile": quantile,
                "cut": model.cut,
                "n_tail": int((sizes >= model.cut).sum()),
                "bulk_k": model.bulk.n_components,
                "gpd_shape": model.tail_shape,
                "rate_a": model.rate_a,
                "rate_b": model.rate_b,
                **_compose(
                    draw,
                    bin_of_job,
                    n_jobs_by_bin,
                    sizes,
                    _N_SWEEP,
                    context.seed,
                ),
            }
        )
    sweep = pd.DataFrame(rows)
    print(f"\ntail threshold sweep, {_N_SWEEP} replicates (reported, not selected on):")
    print(sweep.round(4).to_string(index=False))
    write_csv(sweep, context.result("jobsize_tail_sweep.csv"), index=False)
    return sweep


def stage_jobsize(context: PipelineContext | None = None) -> None:
    """Fit and export the marginal, structured, and two-part size laws."""
    started = time.time()
    context = resolve_context(context)
    context.results.mkdir(parents=True, exist_ok=True)
    raw, table, sizes = _load_work_units(context)
    marginal = _fit_marginal(sizes, context)
    jobs, batches, conditional = _fit_structure(raw, table, sizes, marginal, context)
    two_part, activity, cardinality, bin_index = _fit_two_part(jobs, batches, sizes, context)
    _, bin_of_job, n_jobs_by_bin = _composition_checks(
        jobs,
        batches,
        sizes,
        marginal,
        conditional,
        two_part,
        activity,
        cardinality,
        bin_index,
        context,
    )
    _tail_threshold_sweep(
        jobs,
        sizes,
        activity,
        cardinality,
        bin_index,
        bin_of_job,
        n_jobs_by_bin,
        context,
    )
    pool = EmpiricalJobSize.from_table(table)
    pool.to_npz(context.result("jobsize_pool.npz"))
    print(
        "\nwrote jobsize_candidates.csv, jobsize_model.json, "
        "jobsize_conditional.json, jobsize_two_part.json, jobsize_structure.csv, "
        "jobsize_conditional_checks.csv, jobsize_tail_sweep.csv and a "
        f"{len(pool.sizes)}-job empirical pool to {context.results}"
    )
    print(f"jobsize stage done in {time.time() - started:.0f}s")
