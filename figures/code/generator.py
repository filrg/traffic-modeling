from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from _common import plt

N_PATHS_LOAD = 600
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
# alpha = 1 is the trace's own load level, so it gets a tick of its own
XTICKS = (0, 1, 4, 8, 12, 16)
Z95 = 1.959964

# the work sweep composes sizes onto reconstructed events, so it runs its own grid
WORK_ALPHAS = (0.5, 1.0, 4.0, 8.0, 12.0, 16.0)
N_PATHS_WORK = 40
WORK_UNIT = 1e9

N_CANDIDATES = 200
# horizons in days; the trace supplies 138
RUN_HORIZON_D = (30, 90, 365, 1095)

# load factors drawn as runs, all on one latent path so the comparison is of load
# alone. alpha = 1 is omitted: `arrivals.series` carries it on this same realisation.
RUN_ALPHAS = (0.5, 2.0, 8.0, 16.0)
N_PATHS_RUN_ENS = 120

# the combined panel runs its own smaller sweep
N_PATHS_COMBINED_LOAD = 120
N_PATHS_COMBINED_HORIZON = 400
COMBINED_ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
COMBINED_HORIZON_D = (1, 7, 30, 90, 180, 365, 1095)
BAND_FILL = "#dce9f9"

_CACHE: dict = {}


def _load_sweep() -> tuple:
    """Simulate the paired load sweep once; both load views read it.

    Paths are paired by common random numbers: path i is seeded identically at
    every alpha, so it draws the same latent. Seeding one stream per alpha does
    not pair them -- the count draws consume a variable number of values and
    desynchronise it after the first path.
    """
    if "load" in _CACHE:
        return _CACHE["load"]
    from seren_countfirst.metrics import idc
    from seren_countfirst.models.gated import _second_moment_ratio

    st = C.state()
    model, cal = C.winner(st)
    n_bins = int(st["n_bins"])
    p = model.params_
    m, kappa = float(p["mean"]), float(p["kappa"])
    # A = 1 + mR/kappa and B = m(R-1) of the analytic affine law
    R = _second_moment_ratio(p["a"], p["b"], p["c"], p["sigma_fast"])
    A, B = 1.0 + m * R / kappa, m * (R - 1.0)

    mean_size = float(np.mean(st["batches_sizes"]))
    lam1 = float(st["trace_marginals"]["mean"])
    boot = np.random.default_rng(C.SEED)

    rows = []
    for a in ALPHAS:
        paths = [
            model.simulate(n_bins, np.random.default_rng(C.SEED + 6 + i), alpha=cal * a)
            for i in range(N_PATHS_LOAD)
        ]
        jobs = np.array([p_.sum() for p_ in paths]) * mean_size / 1e3
        idle = np.array([(p_ == 0).mean() for p_ in paths]) * 100
        vals = np.array([idc(p_) for p_ in paths])
        # path totals are right-skewed, so this normal-approx interval is only
        # trustworthy where a percentile bootstrap of the mean agrees with it
        draws = boot.integers(0, N_PATHS_LOAD, size=(2000, N_PATHS_LOAD))
        means = jobs[draws].mean(axis=1)
        rows.append(
            {
                "alpha": a,
                "jobs_k": float(jobs.mean()),
                "jobs_k_sd": float(jobs.std(ddof=1)),
                "jobs_k_se": float(jobs.std(ddof=1) / np.sqrt(N_PATHS_LOAD)),
                "path_cv": float(jobs.std(ddof=1) / jobs.mean()),
                "boot_lo_k": float(np.quantile(means, 0.025)),
                "boot_hi_k": float(np.quantile(means, 0.975)),
                "idle_pct": float(idle.mean()),
                "idle_pct_se": float(idle.std(ddof=1) / np.sqrt(N_PATHS_LOAD)),
                # the same rate with no burstiness
                "poisson_idle_pct": float(np.exp(-a * lam1)) * 100,
                "idc": float(vals.mean()),
                "idc_median": float(np.median(vals)),
                "idc_se": float(vals.std(ddof=1) / np.sqrt(N_PATHS_LOAD)),
                "idc_analytic": A + B * a,
            }
        )
    _CACHE["load"] = (pd.DataFrame(rows), A, B)
    return _CACHE["load"]


def _load_axes(ax, ylabel: str) -> None:
    ax.set_xlabel("Load factor")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, ALPHAS[-1] * 1.06)
    ax.set_ylim(0, None)
    ax.set_xticks(list(XTICKS))
    C.style_axes(ax)
    C.frame_legend(
        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=1.0,
            fontsize=8,
            handlelength=2.4,
            labelspacing=0.35,
            borderpad=0.5,
        )
    )


def load_arrivals() -> None:
    """Arrivals scale linearly with the load factor.

    Linearity is exact by construction, so this checks the implementation and
    the count-to-job reconstruction rather than a statistical claim.
    """
    df, _, _ = _load_sweep()

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    a = df["alpha"].to_numpy()
    grid = np.linspace(0, ALPHAS[-1] * 1.06, 100)
    per_unit = float(df.loc[df["alpha"] == 1.0, "jobs_k"].iloc[0])

    # the reference is drawn wider and beneath the measured line
    ax.plot(
        grid,
        grid * per_unit,
        color=C.MUTED,
        linewidth=1.9,
        linestyle=(0, (2.4, 2.4)),
        label="Linear reference",
        zorder=2,
    )
    ax.errorbar(
        a,
        df["jobs_k"].to_numpy(),
        yerr=Z95 * df["jobs_k_se"].to_numpy(),
        color=C.BLUE,
        linewidth=1.0,
        marker="o",
        markersize=4.0,
        elinewidth=0.9,
        capsize=2.6,
        capthick=0.9,
        label="GLI-NB generated, 95% CI",
        zorder=3,
    )
    _load_axes(ax, "Jobs (thousands)")
    fig.tight_layout(pad=0.4)
    C.save(fig, "fig_load")

    cols = [
        "alpha",
        "jobs_k",
        "jobs_k_sd",
        "jobs_k_se",
        "path_cv",
        "boot_lo_k",
        "boot_hi_k",
        "idle_pct",
        "idle_pct_se",
        "poisson_idle_pct",
    ]
    C.write_csv(df[cols], "fig_load", index=False)
    out = df[cols].copy()
    out["linear_ref_k"] = out["alpha"] * per_unit
    out["deviation_%"] = 100 * (out["jobs_k"] / out["linear_ref_k"] - 1)
    out["ci95_%"] = 100 * Z95 * out["jobs_k_se"] / out["jobs_k"]
    show = [
        "alpha",
        "jobs_k",
        "linear_ref_k",
        "deviation_%",
        "ci95_%",
        "boot_lo_k",
        "boot_hi_k",
        "path_cv",
        "idle_pct",
    ]
    print(out[show].round(3).to_string(index=False))


def _work_sweep() -> pd.DataFrame:
    """Compose work at each load factor; alpha is arrival-side only.

    The size law is already conditional on `N_k`, so the load factor reaches
    offered work through the counts alone and `assign_work` takes no alpha.
    """
    if "work" in _CACHE:
        return _CACHE["work"]
    from seren_countfirst.data import ActivityScheme
    from seren_countfirst.jobsize import assign_work, read_two_part_jobsize
    from seren_countfirst.reconstruct import (
        make_iid_mark_sampler,
        PatternTemplates,
        reconstruct_events,
    )

    st = C.state()
    model, cal = C.winner(st)
    counts = st["counts"]
    bin_s = float(counts.bin_seconds)
    sizes_model = read_two_part_jobsize(
        C.RESULTS / "jobsize_two_part.json", C.RESULTS / "jobsize_pool.npz"
    )
    scheme = st["scheme"]
    if isinstance(scheme, dict):
        scheme = ActivityScheme(
            edges=tuple(scheme["edges"]),
            labels=tuple(scheme["labels"]),
            busy_threshold=scheme["busy_threshold"],
            quiet_threshold=scheme["quiet_threshold"],
            source=scheme["source"],
        )
    times_s = (
        (pd.DatetimeIndex(st["batch_times"]) - counts.index[0])
        .total_seconds()
        .to_numpy(dtype=float)
    )
    templates = PatternTemplates.from_trace(
        times_s, bin_s, counts.values, classes=scheme.classify(counts.values)
    )
    marks = make_iid_mark_sampler(np.asarray(st["batches_sizes"], dtype=int))

    rows = []
    for a in WORK_ALPHAS:
        for i in range(N_PATHS_WORK):
            syn = model.simulate(len(counts), np.random.default_rng(C.SEED + 4 + i), alpha=cal * a)
            # sub-second placement throughout: a path that does not fit on whole
            # seconds must not be dropped, which would bias the totals quiet
            res = reconstruct_events(
                syn,
                bin_s,
                np.random.default_rng(C.SEED + 7000 + i),
                marks,
                offsets=templates,
                on_overflow="subsecond",
                alpha=a,
            )
            # no alpha here: the counts already carry it
            work = assign_work(
                res.batch_times_s,
                res.batch_sizes,
                syn,
                bin_s,
                sizes_model,
                np.random.default_rng(C.SEED + 7000 + i),
            )
            big = work >= sizes_model.cut
            rows.append(
                {
                    "alpha": a,
                    "work": float(work.sum()) / WORK_UNIT,
                    "large_job_frac": float(big.mean()),
                    "large_work_share": float(work[big].sum() / work.sum()),
                }
            )
    df = (
        pd.DataFrame(rows)
        .groupby("alpha")
        .agg(
            work=("work", "mean"),
            sd=("work", lambda x: x.std(ddof=1)),
            se=("work", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            large_job_frac=("large_job_frac", "mean"),
            large_work_share=("large_work_share", "mean"),
        )
        .reset_index()
    )
    _CACHE["work"] = df
    return df


def load_work() -> None:
    """Offered work against the load factor, the companion to `load-arrivals`.

    The knob that moves arrivals linearly does not move work linearly, because
    the size law reads bin activity and activity is what the knob changes.
    """
    df = _work_sweep().sort_values("alpha")

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    grid = np.linspace(0, WORK_ALPHAS[-1] * 1.06, 100)
    unit = float(df.loc[df["alpha"] == 1.0, "work"].iloc[0])
    ax.plot(
        grid,
        grid * unit,
        color=C.MUTED,
        linewidth=1.9,
        linestyle=(0, (2.4, 2.4)),
        label="Linear reference",
        zorder=2,
    )
    ax.errorbar(
        df["alpha"].to_numpy(),
        df["work"].to_numpy(),
        yerr=Z95 * df["se"].to_numpy(),
        color=C.BLUE,
        linewidth=1.0,
        marker="o",
        markersize=4.0,
        elinewidth=0.9,
        capsize=2.6,
        capthick=0.9,
        label="GLI-NB generated, 95% CI",
        zorder=3,
    )
    _load_axes(ax, "Offered work (billion units)")
    fig.tight_layout(pad=0.4)
    C.save(fig, "fig_load_work")

    out = df.copy()
    out["linear_ref"] = out["alpha"] * unit
    out["vs_linear"] = out["work"] / out["linear_ref"]
    C.write_csv(out, "fig_load_work", index=False)
    print(f"offered work in billions of units, {N_PATHS_WORK} paths per load factor:")
    print(out.round(4).to_string(index=False))
    print(
        "\nthe shortfall against linear is a local reading over this alpha range, "
        "not a scaling law: work is bracketed by two lines through the origin and "
        "tends to linear, so no exponent fitted here may be quoted as one"
    )
    print(
        f"large jobs carry {out['large_work_share'].iloc[0]:.3f} of the work at "
        f"alpha = {out['alpha'].iloc[0]}, {out['large_work_share'].iloc[-1]:.3f} at "
        f"alpha = {out['alpha'].iloc[-1]}, because the per-bin tail rate is flat in "
        "N_k. That is most of the shortfall, and it is an open claim: the flat rate "
        "was measured within the trace at one load"
    )


def load_idc() -> None:
    """Dispersion is affine in the load factor, not proportional to it."""
    df, A, B = _load_sweep()

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    a = df["alpha"].to_numpy()
    grid = np.linspace(0, ALPHAS[-1] * 1.06, 100)
    # attenuation is load-invariant, so rescaling by it leaves the shape in alpha
    r = float(df.loc[df["alpha"] == 1.0, "idc"].iloc[0]) / (A + B)

    ax.plot(
        grid,
        r * (A + B * grid),
        color=C.MUTED,
        linewidth=1.9,
        linestyle=(0, (2.4, 2.4)),
        label="Affine reference",
        zorder=2,
    )
    ax.errorbar(
        a,
        df["idc"].to_numpy(),
        yerr=Z95 * df["idc_se"].to_numpy(),
        color=C.BLUE,
        linewidth=1.0,
        marker="o",
        markersize=4.0,
        elinewidth=0.9,
        capsize=2.6,
        capthick=0.9,
        label="GLI-NB generated, 95% CI",
        zorder=3,
    )
    _load_axes(ax, "Index of dispersion (IDC)")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout(pad=0.4)
    C.save(fig, "fig_load_idc")

    cols = ["alpha", "idc", "idc_median", "idc_se", "idc_analytic"]
    C.write_csv(df[cols], "fig_load_idc", index=False)
    unit = float(df.loc[df["alpha"] == 1.0, "idc"].iloc[0])
    out = df[cols].copy()
    out["affine_ref"] = r * (A + B * out["alpha"])
    out["dev_vs_affine_%"] = 100 * (out["idc"] / out["affine_ref"] - 1)
    # against a proportional extrapolation from unit load
    out["proportional_ref"] = out["alpha"] * unit
    out["dev_vs_proportional_%"] = 100 * (out["idc"] / out["proportional_ref"] - 1)
    out["ci95_%"] = 100 * Z95 * out["idc_se"] / out["idc"]
    print(f"analytic law: IDC = {A:.3f} + {B:.3f} alpha")
    print(f"finite-length attenuation at alpha = 1: {r:.3f}\n")
    show = [
        "alpha",
        "idc",
        "idc_median",
        "idc_analytic",
        "affine_ref",
        "dev_vs_affine_%",
        "dev_vs_proportional_%",
        "ci95_%",
    ]
    print(out[show].round(3).to_string(index=False))


def horizon_runs() -> None:
    """Synthetic runs at four horizons, none of which the trace can supply."""
    st = C.state()
    model, cal = C.winner(st)
    nominal = float(st["trace_marginals"]["mean"])

    runs = {}
    for days in RUN_HORIZON_D:
        rng = np.random.default_rng(C.SEED + 5)
        cand = [model.simulate(days * C.BINS_PER_DAY, rng, alpha=cal) for _ in range(N_CANDIDATES)]
        runs[days] = cand[int(np.argmin([abs(c.mean() - nominal) for c in cand]))]

    C.house_style()
    fig, axes = plt.subplots(len(RUN_HORIZON_D), 1, figsize=(7.2, 5.4), sharey=True)
    top = max(r.max() for r in runs.values()) * C.BINS_PER_H * 1.06

    for ax, days in zip(axes, RUN_HORIZON_D):
        data = runs[days]
        t = np.arange(len(data)) / C.BINS_PER_DAY
        # thinner strokes as the run gets longer
        lw = 0.45 if days <= 90 else (0.3 if days <= 365 else 0.18)
        ax.plot(t, data * C.BINS_PER_H, color=C.BLUE, linewidth=lw, zorder=3)
        ax.set_xlim(0, days)
        ax.set_ylim(0, top)
        C.style_axes(ax)
        # the duration labels the panel instead of a legend
        ax.text(
            -0.075,
            1.04,
            f"{days} days",
            transform=ax.transAxes,
            color=C.INK,
            fontsize=10,
            va="bottom",
            ha="left",
        )

    axes[-1].set_xlabel("Time (days)")
    fig.supylabel("Arrival rate (batches/h)", color=C.INK, fontsize=9, x=0.005)
    fig.tight_layout(pad=0.4, h_pad=1.0)
    C.save(fig, "fig_horizon_runs")

    rows = [
        {
            "days": d,
            "bins": len(v),
            "mean_batches_per_h": v.mean() * C.BINS_PER_H,
            "peak_batches_per_h": v.max() * C.BINS_PER_H,
            "zero_bin_fraction": float((v == 0).mean()),
        }
        for d, v in runs.items()
    ]
    C.write_csv(pd.DataFrame(rows), "fig_horizon_runs", index=False)
    print(f"\nnominal {nominal * C.BINS_PER_H:.2f} batches/h")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def load_runs() -> None:
    """The run `arrivals.series` displays, redrawn at four load factors.

    The same realisation rather than another draw, and each panel is scaled by
    its own load, so a proportional response would draw four identical panels.
    """
    from seren_countfirst.metrics import idc

    st = C.state()
    model, cal = C.winner(st)
    counts = st["counts"].values
    n_bins = len(counts)

    # `redraw` restores the stream that produced the displayed run, so the latent
    # path is shared across load factors rather than merely resampled at one seed
    pool, states = C.display_pool(model, cal, n_bins)
    pick = C.representative(pool, float(counts.mean()))
    runs = {a: C.redraw(model, states[pick], n_bins, cal * a) for a in RUN_ALPHAS}

    C.house_style()
    fig, axes = plt.subplots(len(RUN_ALPHAS), 1, figsize=(7.2, 5.4), sharex=True)
    # a common per-unit-load ceiling, so every panel spans the same scaled range
    top = max(r.max() / a for a, r in runs.items()) * C.BINS_PER_H * 1.06

    rows = []
    for ax, a in zip(axes, RUN_ALPHAS):
        v = runs[a]
        t = np.arange(len(v)) / C.BINS_PER_DAY
        ax.plot(t, v * C.BINS_PER_H, color=C.BLUE, linewidth=0.45, zorder=3)
        ax.set_xlim(0, len(v) / C.BINS_PER_DAY)
        ax.set_ylim(0, top * a)
        C.style_axes(ax)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.text(
            -0.085,
            1.04,
            f"α = {a:g}",
            transform=ax.transAxes,
            color=C.INK,
            fontsize=10,
            va="bottom",
            ha="left",
        )
        # annotated from the ensemble below, never from the single drawn path
        rows.append(
            {
                "alpha": a,
                "drawn_mean_batches_per_h": v.mean() * C.BINS_PER_H,
                "drawn_peak_batches_per_h": v.max() * C.BINS_PER_H,
                "drawn_zero_bin_fraction": float((v == 0).mean()),
            }
        )

    axes[-1].set_xlabel("Time (days)")
    fig.supylabel("Arrival rate (batches/h)", color=C.INK, fontsize=9, x=0.005)
    fig.tight_layout(pad=0.4, h_pad=1.0)
    C.save(fig, "fig_load_runs")

    ens = []
    for a in RUN_ALPHAS:
        # one stream per path index, so the load factors are compared on the same
        # paths; a stream per alpha pairs only its first path and lets the rest
        # carry a spurious trend across the sweep
        ep = [
            model.simulate(n_bins, np.random.default_rng(C.SEED + 2000 + i), alpha=cal * a)
            for i in range(N_PATHS_RUN_ENS)
        ]
        ens.append(
            {
                "alpha": a,
                "idc": float(np.median([idc(p) for p in ep])),
                "cv2": float(np.median([p.var(ddof=1) / p.mean() ** 2 for p in ep])),
                "zero_bin_fraction": float(np.median([(p == 0).mean() for p in ep])),
                "mean_occupied": float(np.median([p[p > 0].mean() for p in ep])),
            }
        )
    out = pd.DataFrame(rows).merge(pd.DataFrame(ens), on="alpha")
    C.write_csv(out, "fig_load_runs", index=False)

    ref = pool[pick]
    print(f"displaying run {pick} of {len(pool)}, the run arrivals.series shows")
    print(
        "  redraw reproduces it at alpha = 1: "
        f"{np.array_equal(C.redraw(model, states[pick], n_bins, cal), ref)}"
    )
    print(
        "  correlation of each panel with it: "
        + ", ".join(f"{a:g}: {np.corrcoef(runs[a], ref)[0, 1]:.4f}" for a in RUN_ALPHAS)
    )
    print(f"\nensemble over {N_PATHS_RUN_ENS} paths per load factor:")
    print(out.round(4).to_string(index=False))
    lo, hi = out.iloc[0], out.iloc[-1]
    print(
        f"\nacross a {hi['alpha'] / lo['alpha']:.0f}x load range: IDC "
        f"{lo['idc']:,.0f} -> {hi['idc']:,.0f}, but CV^2 only {lo['cv2']:.1f} -> {hi['cv2']:.1f}. "
        f"Superposing independent streams would divide CV^2 by {hi['alpha'] / lo['alpha']:.0f}."
    )
    print(
        f"empty bins {lo['zero_bin_fraction']:.3f} -> {hi['zero_bin_fraction']:.3f}; "
        f"mean count over occupied bins {lo['mean_occupied']:.2f} -> {hi['mean_occupied']:.2f} "
        f"({hi['mean_occupied'] / lo['mean_occupied']:.1f}x against the "
        f"{hi['alpha'] / lo['alpha']:.0f}x load)"
    )


def combined() -> None:
    """Two panels: dispersion against load, and realised rate against run length."""
    from seren_countfirst.metrics import idc

    st = C.state()
    model, cal = C.winner(st)
    n_bins = int(st["n_bins"])

    load = []
    for a in COMBINED_ALPHAS:
        rng = np.random.default_rng(C.SEED + 2)
        vals = [
            idc(model.simulate(n_bins, rng, alpha=cal * a)) for _ in range(N_PATHS_COMBINED_LOAD)
        ]
        load.append({"alpha": a, "idc": float(np.median(vals))})
    load = pd.DataFrame(load)

    horizon = []
    for d in COMBINED_HORIZON_D:
        rng = np.random.default_rng(C.SEED + 3)
        rates = (
            np.array(
                [
                    model.simulate(d * C.BINS_PER_DAY, rng, alpha=cal).mean()
                    for _ in range(N_PATHS_COMBINED_HORIZON)
                ]
            )
            * C.BINS_PER_H
        )
        horizon.append(
            {
                "days": d,
                "median": float(np.median(rates)),
                "q025": float(np.quantile(rates, 0.025)),
                "q975": float(np.quantile(rates, 0.975)),
            }
        )
    horizon = pd.DataFrame(horizon)
    nominal = float(model.params_["mean"]) * cal * C.BINS_PER_H

    C.house_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # (a) the load knob; the reference is anchored on the measured alpha = 1
    a = load["alpha"].to_numpy()
    idc_1 = float(load.loc[load["alpha"] == 1.0, "idc"].iloc[0])
    grid = np.linspace(0, COMBINED_ALPHAS[-1] * 1.06, 100)
    ax1.plot(
        grid,
        1 + grid * (idc_1 - 1),
        color=C.MUTED,
        linewidth=1.3,
        linestyle=(0, (2.2, 2.2)),
        label="Linear in load factor",
        zorder=2,
    )
    ax1.plot(
        a,
        load["idc"].to_numpy(),
        color=C.BLUE,
        linewidth=0,
        marker="o",
        markersize=5.0,
        label="Measured",
        zorder=3,
    )
    ax1.set_xlim(0, COMBINED_ALPHAS[-1] * 1.06)
    ax1.set_ylim(0, None)
    ax1.set_xticks([0, 4, 8, 12, 16])
    ax1.set_xlabel("Load factor")
    ax1.set_ylabel("Index of dispersion (IDC)")
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # (b) horizons
    x = np.arange(len(COMBINED_HORIZON_D), dtype=float)
    ax2.fill_between(
        x,
        horizon["q025"].to_numpy(),
        horizon["q975"].to_numpy(),
        color=BAND_FILL,
        linewidth=0,
        label="95% of runs",
        zorder=1,
    )
    ax2.axhline(
        nominal,
        color=C.MUTED,
        linewidth=1.3,
        linestyle=(0, (2.2, 2.2)),
        label="Nominal rate",
        zorder=2,
    )
    ax2.plot(
        x,
        horizon["median"].to_numpy(),
        color=C.BLUE,
        linewidth=1.7,
        marker="o",
        markersize=4.2,
        label="Median run",
        zorder=3,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in COMBINED_HORIZON_D])
    ax2.set_xlim(x[0] - 0.25, x[-1] + 0.25)
    ax2.set_ylim(0, None)
    ax2.set_xlabel("Run length (days)")
    ax2.set_ylabel("Arrival rate (batches/h)")

    for ax, tag in ((ax1, "(a)"), (ax2, "(b)")):
        C.style_axes(ax)
        ax.text(
            -0.17,
            1.02,
            tag,
            transform=ax.transAxes,
            color=C.INK,
            fontsize=10,
            va="bottom",
            ha="left",
        )
        C.frame_legend(
            ax.legend(
                loc="upper left",
                frameon=True,
                framealpha=1.0,
                fontsize=8,
                handlelength=2.4,
                labelspacing=0.35,
                borderpad=0.5,
            )
        )

    fig.tight_layout(pad=0.4, w_pad=2.2)
    C.save(fig, "fig_load_horizon")

    C.write_csv(load, "fig_load_horizon_panel_a", index=False)
    C.write_csv(horizon, "fig_load_horizon_panel_b", index=False)
    load["linear_reference"] = 1 + load["alpha"] * (idc_1 - 1)
    print(f"\nnominal rate {nominal:.2f} batches/h")
    print("\n(a) load knob:\n", load.round(2).to_string(index=False))
    print("\n(b) horizons:\n", horizon.round(3).to_string(index=False))


INPUTS = ("state", "counts", "models", "batch_times", "jobsize_two_part", "jobsize_pool")

VIEWS = {
    "load-arrivals": load_arrivals,
    "load-work": load_work,
    "load-idc": load_idc,
    "horizon-runs": horizon_runs,
    "load-runs": load_runs,
    "combined": combined,
}

if __name__ == "__main__":
    C.run(
        VIEWS,
        "load-arrivals",
        "Generator properties: the load knob and behaviour across horizons.",
        INPUTS,
    )
