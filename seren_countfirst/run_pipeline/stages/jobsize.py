"""Invert absorbed-work durations into work units and freeze a law for them."""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from ... import (
    ConditionalJobSize,
    EmpiricalJobSize,
    LogNormalMixture,
    ScipyJobSize,
    TAIL_QUANTILE,
    TwoPartJobSize,
    arrival_view,
    attach_activity,
    batch_frame,
    bin_grid_index,
    component_scoreboard,
    fit_jobsize,
    jobsize_table,
    jobsize_view,
    load_trace,
    offered_work_stats,
    roundtrip_error_s,
    step_time_s,
    write_jobsize_model,
)
from ...data import JOBSIZE_STATES
from ...paths import RESULTS, TRACE_CSV
from ..config import SEED

# GPU counts the size summary is echoed at: a work unit is invariant, the
# duration it implies is not
_ECHO_GPUS = (1, 8, 16, 64, 256)

# replicates of the composition check; each redraws every size against the
# trace's own batch structure, so only the size model varies
_N_COMPOSE = 40

# thresholds the tail split is reported at; the frozen one is TAIL_QUANTILE and
# is not chosen from this sweep
_TAIL_SWEEP = (0.95, 0.98, 0.99, 0.995)
_N_SWEEP = 10


def _compose(rule, bin_of_job, n_jobs_by_bin, observed_sizes, n_rep: int, seed: int) -> dict:
    """Average the offered-work statistics over `n_rep` redraws of every size."""
    rows, ks, ratio = [], [], []
    for r in range(n_rep):
        drawn = rule(np.random.default_rng(seed + r))
        work = pd.Series(drawn).groupby(bin_of_job).sum().to_numpy()
        rows.append(offered_work_stats(work, n_jobs_by_bin))
        ks.append(float(stats.ks_2samp(drawn, observed_sizes).statistic))
        ratio.append(float(np.mean(drawn) / np.mean(observed_sizes)))
    out = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
    out["ks_job_marginal"] = float(np.mean(ks))
    out["mean_work_ratio"] = float(np.mean(ratio))
    return out


def stage_jobsize() -> None:
    """Write the work units, the candidate scoreboard and the frozen samplers.

    Reads the trace directly rather than the pipeline state: job sizes are a
    separate view of it (the states whose runtime the cluster absorbed) and
    nothing here feeds the count models.
    """
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = load_trace(TRACE_CSV)
    table = jobsize_table(jobsize_view(raw, "SFT"))
    sizes = table["job_size"].to_numpy()

    err = roundtrip_error_s(table)
    print(f"SFT jobs with a usable duration and GPU count: {len(table)} "
          f"(states {', '.join(JOBSIZE_STATES)})")
    print(f"inversion round-trip max abs error: {err:.3e} s")
    # the inversion is an identity: a large error means the latency coefficients
    # no longer match the ones the durations were read at
    if err > 1e-6:
        raise RuntimeError(f"work-unit inversion does not round-trip ({err:.3e} s)")

    print("\nwork units:")
    print(table["job_size"].describe().to_string())
    median = float(np.median(sizes))
    print(f"\nthe median job of {median:,.0f} units, replayed at f = 1 GHz:")
    for n in _ECHO_GPUS:
        secs = median * float(step_time_s(n, 1.0))
        print(f"  n = {n:4d}  ->  {secs:9.0f} s  ({secs / 60:7.1f} min)")

    board = fit_jobsize(sizes, seed=SEED)
    print("\ncandidate laws, best AIC first:")
    print(board.drop(columns="params").round(4).to_string(index=False))
    # params serialised as json so a figure can rebuild any row's law directly
    board.assign(params=board["params"].map(
        lambda p: json.dumps(list(p) if isinstance(p, tuple) else p)
    )).to_csv(RESULTS / "jobsize_candidates.csv", index=False)

    best = board.iloc[0]
    if best["kind"] == "mixture_lognormal":
        k = int(best["distribution"].rsplit("_", 1)[1])
        model = LogNormalMixture.fit(sizes, n_components=k, seed=SEED).freeze()
        print(f"\nfrozen law: {best['distribution']}  "
              f"AIC {model.report.aic:,.1f}  BIC {model.report.bic:,.1f}")
        for w, mu, sg in zip(model.weights, model.mu, model.sigma):
            print(f"  w {w:.4f}  mu {mu:8.4f}  sigma {sg:.4f}  "
                  f"(median {np.exp(mu):,.0f} units)")
    else:
        model = ScipyJobSize(name=str(best["distribution"]), params=best["params"])
        print(f"\nfrozen law: {best['distribution']} {model.params}")
    write_jobsize_model(model, RESULTS / "jobsize_model.json")

    # --- the structure the marginal law ignores -----------------------------
    jobs = attach_activity(table, arrival_view(raw, "SFT"))
    batches = batch_frame(jobs)
    z = np.log(jobs["job_size"].to_numpy())
    # the ladder row: the design before the composition law, without the bin level
    conditional = ConditionalJobSize.fit(jobs, seed=SEED, with_latent=False).freeze()
    var_z = float(np.var(z, ddof=1))
    icc = 1.0 - float(conditional.weights @ conditional.sigma_w ** 2) / var_z
    x = np.log(batches["activity"].to_numpy(dtype=float))
    r2 = float(np.corrcoef(x, batches["z_bar"].to_numpy())[0, 1] ** 2)
    print(f"\nbatches {len(batches):,} over {jobs['bin'].nunique():,} occupied bins; "
          f"{(batches['cardinality'] > 1).mean():.3f} hold more than one job")
    print(f"within-batch sd {np.round(conditional.sigma_w, 4)} by component against "
          f"{np.sqrt(var_z):.4f} overall  ->  ICC {icc:.4f}")
    print(f"activity slope beta {conditional.beta:+.4f}  (R^2 {r2:.4f})")

    orders = component_scoreboard(jobs, conditional.beta, seed=SEED)
    print("component counts on the activity-adjusted residuals:")
    print(orders.round(4).to_string(index=False))
    print(f"selected k = {conditional.n_components}")

    print(f"\nconditional law: mu {conditional.mu:.4f}  beta {conditional.beta:+.4f}")
    for w, m, sb, sj, sw in zip(conditional.weights, conditional.m,
                                conditional.s_batch, conditional.s_job,
                                conditional.sigma_w):
        print(f"  pi {w:.4f}  m {m:+8.4f}  s_batch {sb:.4f}  sigma_w {sw:.4f}  "
              f"(s_job {sj:.4f})")
    write_jobsize_model(conditional, RESULTS / "jobsize_conditional.json")
    # a likelihood fit need not reproduce the sample mean: the top jobs carry it
    ranked = np.sort(sizes)[::-1]
    analytic_mean = (float(np.sum(model.weights * np.exp(model.mu + 0.5 * model.sigma ** 2)))
                     if isinstance(model, LogNormalMixture) else np.nan)
    print(f"top 100 of {len(sizes):,} jobs carry "
          f"{ranked[:100].sum() / ranked.sum():.4f} of all work; the frozen marginal's "
          f"analytic mean is {analytic_mean / sizes.mean():.4f} of the sample mean")
    pd.DataFrame([{
        "n_jobs": len(jobs), "n_batches": len(batches),
        "n_occupied_bins": int(jobs["bin"].nunique()),
        "sd_log_size": float(np.sqrt(var_z)),
        "sigma_w_mean": float(conditional.weights @ conditional.sigma_w),
        "icc_within_batch": icc, "beta": conditional.beta, "beta_r2": r2,
        "top100_work_share": float(ranked[:100].sum() / ranked.sum()),
        "marginal_mean_ratio": float(analytic_mean / sizes.mean()),
        "n_components": conditional.n_components,
        **{f"aic_k{int(r.k)}": r.AIC for r in orders.itertuples()},
        **{f"slack_k{int(r.k)}": r.tightest_slack for r in orders.itertuples()},
    }]).to_csv(RESULTS / "jobsize_structure.csv", index=False)

    # --- the tail: placed per bin, not per job ------------------------------
    two_part = TwoPartJobSize.fit(jobs, seed=SEED).freeze()
    activity = batches["activity"].to_numpy(dtype=float)
    card = batches["cardinality"].to_numpy(dtype=int)
    bin_index = bin_grid_index(batches["bin"])
    n_above = int((sizes >= two_part.cut).sum())
    print(f"\ntail cut at the {TAIL_QUANTILE:.1%} quantile = {two_part.cut:,.0f} units: "
          f"{n_above} jobs above it")
    print(f"exceedances: generalized Pareto shape {two_part.tail_shape:+.4f}  "
          f"scale {two_part.tail_scale:,.0f}")
    print(f"large jobs per bin: log lambda = {two_part.rate_a:+.4f} "
          f"{two_part.rate_b:+.4f} log N_k  -> "
          f"{two_part.tail_rate(2.0)[()]:.3f} at N_k = 2, "
          f"{two_part.tail_rate(100.0)[()]:.3f} at N_k = 100")
    bulk = two_part.bulk
    print(f"bulk law ({two_part.bulk_kind}): k = {bulk.n_components}, "
          f"mu {bulk.mu:.4f}")
    print(f"  component locations {np.round(bulk.m, 3)}")
    print(f"  job sd {np.round(bulk.s_job, 3)}   batch sd {np.round(bulk.s_batch, 3)}")
    print(f"  weight logit: a {np.round(bulk.logit_a, 3)}  b {np.round(bulk.logit_b, 4)}"
          f"   escape {bulk.escape:.4f}")
    for n in (2, 20, 200):
        print(f"  weights at N_k = {n:3d}: {np.round(bulk.weights([float(n)])[0], 3)}")
    write_jobsize_model(two_part, RESULTS / "jobsize_two_part.json")

    # --- does the composed work per bin look like the trace's? --------------
    # the conditional bulk, kept as a comparison row
    conditional_two_part = TwoPartJobSize.fit(jobs, n_init=4, seed=SEED,
                                              bulk_kind="conditional", with_latent=False)
    flat = ConditionalJobSize.fit(jobs, n_components=conditional.n_components, seed=SEED,
                                  beta=0.0, with_latent=False)
    unshifted = ConditionalJobSize.fit(jobs, n_components=conditional.n_components,
                                       seed=SEED, level="log_mean", with_latent=False)
    print(f"\nlevel: mu {conditional.mu:.4f} matching mean work against "
          f"{unshifted.mu:.4f} at the residual mean, a factor "
          f"{np.exp(conditional.mu - unshifted.mu):.3f} on every job")
    parametric_tail = replace(two_part, tail_kind="genpareto")
    by_bin = jobs.groupby("bin")
    n_jobs_by_bin = by_bin.size().to_numpy()
    observed = by_bin["job_size"].sum().to_numpy()
    bin_of_job = np.repeat(batches["bin"].to_numpy(), card)
    rules = {
        "iid_per_job": lambda rng: model.rvs(len(jobs), rng),
        "two_part_conditional": lambda rng: conditional_two_part.rvs(activity, card,
                                                                     bin_index, rng),
        "batch_only": lambda rng: flat.rvs(activity, card, rng),
        "conditional_mean_work": lambda rng: conditional.rvs(activity, card, rng),
        "conditional_log_mean": lambda rng: unshifted.rvs(activity, card, rng),
        "two_part": lambda rng: two_part.rvs(activity, card, bin_index, rng),
        "two_part_genpareto": lambda rng: parametric_tail.rvs(activity, card, bin_index,
                                                              rng),
    }
    rows = [{"rule": "observed", **offered_work_stats(observed, n_jobs_by_bin),
             "ks_job_marginal": 0.0, "mean_work_ratio": 1.0}]
    rows += [{"rule": name, **_compose(rule, bin_of_job, n_jobs_by_bin, sizes,
                                       _N_COMPOSE, SEED)}
             for name, rule in rules.items()]
    checks = pd.DataFrame(rows)
    print(f"\nwork per occupied bin, {_N_COMPOSE} replicates on the trace's own batches:")
    print(checks.round(4).to_string(index=False))
    checks.to_csv(RESULTS / "jobsize_conditional_checks.csv", index=False)

    sweep = []
    for q in _TAIL_SWEEP:
        part = TwoPartJobSize.fit(jobs, quantile=q, n_init=4, seed=SEED)
        sweep.append({
            "quantile": q, "cut": part.cut, "n_tail": int((sizes >= part.cut).sum()),
            "bulk_k": part.bulk.n_components, "gpd_shape": part.tail_shape,
            "rate_a": part.rate_a, "rate_b": part.rate_b,
            **_compose(lambda rng, p=part: p.rvs(activity, card, bin_index, rng),
                       bin_of_job, n_jobs_by_bin, sizes, _N_SWEEP, SEED),
        })
    sweep = pd.DataFrame(sweep)
    print(f"\ntail threshold sweep, {_N_SWEEP} replicates (reported, not selected on):")
    print(sweep.round(4).to_string(index=False))
    sweep.to_csv(RESULTS / "jobsize_tail_sweep.csv", index=False)

    pool = EmpiricalJobSize.from_table(table)
    pool.to_npz(RESULTS / "jobsize_pool.npz")
    print(f"\nwrote jobsize_candidates.csv, jobsize_model.json, "
          f"jobsize_conditional.json, jobsize_two_part.json, jobsize_structure.csv, "
          f"jobsize_conditional_checks.csv, jobsize_tail_sweep.csv and a "
          f"{len(pool.sizes)}-job empirical pool to {RESULTS}")
    print(f"jobsize stage done in {time.time() - t0:.0f}s")
