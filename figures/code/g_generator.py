"""G1/G2 generator properties: the load knob and behaviour across horizons.

Neither is a fidelity claim. The trace holds one load level and one duration, so
it cannot validate either; both are statements about how the generator behaves
when its knobs move.

`load-arrivals` and `load-idc` are drawn from the *same* runs (same per-path
seeds), so the two panels sit side by side as one experiment: the knob delivers
volume linearly, and it makes the workload relatively spikier as it does so.
`combined` is the compact two-panel version for the paper, at its own path counts.
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd

import _common as C
from _common import plt

N_PATHS_LOAD = 600
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
# alpha = 1 is the trace's own load level, so it gets a tick of its own
XTICKS = (0, 1, 4, 8, 12, 16)
Z95 = 1.959964

N_CANDIDATES = 200
# one unit for the axis: days. The trace supplies 138; the last two are beyond it.
RUN_HORIZON_D = (30, 90, 365, 1095)

# the combined panel runs its own smaller sweep
N_PATHS_COMBINED_LOAD = 120
N_PATHS_COMBINED_HORIZON = 400
COMBINED_ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
COMBINED_HORIZON_D = (1, 7, 30, 90, 180, 365, 1095)
BAND_FILL = "#dce9f9"

_CACHE: dict = {}


def _load_sweep() -> tuple:
    """Simulate the paired load sweep once; both load views read it.

    Paths are paired across load factors by common random numbers: path i uses
    the same seed at every alpha, so it draws the same latent path. That makes
    the error bars common-mode and the departure from linearity <0.1%. Resetting
    one stream per alpha does not pair them, because `rng.gamma` and
    `rng.poisson` consume a variable number of draws and desynchronise the
    stream after the first path.
    """
    if "load" in _CACHE:
        return _CACHE["load"]
    from seren_countfirst import idc
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
        paths = [model.simulate(n_bins, np.random.default_rng(C.SEED + 6 + i), alpha=cal * a)
                 for i in range(N_PATHS_LOAD)]
        jobs = np.array([p_.sum() for p_ in paths]) * mean_size / 1e3
        idle = np.array([(p_ == 0).mean() for p_ in paths]) * 100
        vals = np.array([idc(p_) for p_ in paths])
        # path totals are right-skewed, so check the normal-approx interval
        # against a percentile bootstrap of the mean before trusting the bar
        draws = boot.integers(0, N_PATHS_LOAD, size=(2000, N_PATHS_LOAD))
        means = jobs[draws].mean(axis=1)
        rows.append({
            "alpha": a,
            "jobs_k": float(jobs.mean()),
            "jobs_k_sd": float(jobs.std(ddof=1)),
            "jobs_k_se": float(jobs.std(ddof=1) / np.sqrt(N_PATHS_LOAD)),
            "path_cv": float(jobs.std(ddof=1) / jobs.mean()),
            "boot_lo_k": float(np.quantile(means, 0.025)),
            "boot_hi_k": float(np.quantile(means, 0.975)),
            "idle_pct": float(idle.mean()),
            "idle_pct_se": float(idle.std(ddof=1) / np.sqrt(N_PATHS_LOAD)),
            # a workload of the same rate with no burstiness at all
            "poisson_idle_pct": float(np.exp(-a * lam1)) * 100,
            "idc": float(vals.mean()),
            "idc_median": float(np.median(vals)),
            "idc_se": float(vals.std(ddof=1) / np.sqrt(N_PATHS_LOAD)),
            "idc_analytic": A + B * a,
        })
    _CACHE["load"] = (pd.DataFrame(rows), A, B)
    return _CACHE["load"]


def _load_axes(ax, ylabel: str) -> None:
    ax.set_xlabel("Load factor")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, ALPHAS[-1] * 1.06)
    ax.set_ylim(0, None)
    ax.set_xticks(list(XTICKS))
    C.style_axes(ax)
    C.frame_legend(ax.legend(loc="upper left", frameon=True, framealpha=1.0, fontsize=8,
                             handlelength=2.4, labelspacing=0.35, borderpad=0.5))


def load_arrivals() -> None:
    """Arrivals scale linearly with the load factor.

    Linearity is exact by construction — `_eta_from_states` puts log(alpha*mean)
    in the intensity — so this checks the implementation and the count-to-job
    reconstruction, not a statistical claim. The bar is wide (~6%) because a
    single run is: per-path CV is 0.76 at this horizon, since the gate latent's
    8.2-day timescale leaves only ~17 independent stretches in 138 days.
    """
    df, _, _ = _load_sweep()

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    a = df["alpha"].to_numpy()
    grid = np.linspace(0, ALPHAS[-1] * 1.06, 100)
    per_unit = float(df.loc[df["alpha"] == 1.0, "jobs_k"].iloc[0])

    # the reference is drawn wider and beneath, so it stays visible under the
    # measured line rather than being hidden by it
    ax.plot(grid, grid * per_unit, color=C.MUTED, linewidth=1.9,
            linestyle=(0, (2.4, 2.4)), label="Linear reference", zorder=2)
    ax.errorbar(a, df["jobs_k"].to_numpy(), yerr=Z95 * df["jobs_k_se"].to_numpy(),
                color=C.BLUE, linewidth=1.0, marker="o", markersize=4.0,
                elinewidth=0.9, capsize=2.6, capthick=0.9,
                label="GLI-NB generated, 95% CI", zorder=3)
    _load_axes(ax, "Jobs (thousands)")
    fig.tight_layout(pad=0.4)
    C.save(fig, "fig_load")

    cols = ["alpha", "jobs_k", "jobs_k_sd", "jobs_k_se", "path_cv", "boot_lo_k",
            "boot_hi_k", "idle_pct", "idle_pct_se", "poisson_idle_pct"]
    C.write_csv(df[cols], "fig_load", index=False)
    out = df[cols].copy()
    out["linear_ref_k"] = out["alpha"] * per_unit
    out["deviation_%"] = 100 * (out["jobs_k"] / out["linear_ref_k"] - 1)
    out["ci95_%"] = 100 * Z95 * out["jobs_k_se"] / out["jobs_k"]
    show = ["alpha", "jobs_k", "linear_ref_k", "deviation_%", "ci95_%",
            "boot_lo_k", "boot_hi_k", "path_cv", "idle_pct"]
    print(out[show].round(3).to_string(index=False))


def load_idc() -> None:
    """Dispersion is affine in the load factor, not proportional to it."""
    df, A, B = _load_sweep()

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    a = df["alpha"].to_numpy()
    grid = np.linspace(0, ALPHAS[-1] * 1.06, 100)
    # attenuation measured at unit load; it is load-invariant, so rescaling the
    # analytic law by it leaves the shape in alpha untouched
    r = float(df.loc[df["alpha"] == 1.0, "idc"].iloc[0]) / (A + B)

    ax.plot(grid, r * (A + B * grid), color=C.MUTED, linewidth=1.9,
            linestyle=(0, (2.4, 2.4)), label="Affine reference", zorder=2)
    ax.errorbar(a, df["idc"].to_numpy(), yerr=Z95 * df["idc_se"].to_numpy(),
                color=C.BLUE, linewidth=1.0, marker="o", markersize=4.0,
                elinewidth=0.9, capsize=2.6, capthick=0.9,
                label="GLI-NB generated, 95% CI", zorder=3)
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
    # comparison against a proportional extrapolation from unit load, which the
    # affine intercept makes overstate dispersion
    out["proportional_ref"] = out["alpha"] * unit
    out["dev_vs_proportional_%"] = 100 * (out["idc"] / out["proportional_ref"] - 1)
    out["ci95_%"] = 100 * Z95 * out["idc_se"] / out["idc"]
    print(f"analytic law: IDC = {A:.3f} + {B:.3f} alpha")
    print(f"finite-length attenuation at alpha = 1: {r:.3f}\n")
    show = ["alpha", "idc", "idc_median", "idc_analytic", "affine_ref",
            "dev_vs_affine_%", "dev_vs_proportional_%", "ci95_%"]
    print(out[show].round(3).to_string(index=False))


def horizon_runs() -> None:
    """Synthetic runs at four horizons, none of which the trace can supply."""
    st = C.state()
    model, cal = C.winner(st)
    nominal = float(st["trace_marginals"]["mean"])

    runs = {}
    for days in RUN_HORIZON_D:
        rng = np.random.default_rng(C.SEED + 5)
        cand = [model.simulate(days * C.BINS_PER_DAY, rng, alpha=cal)
                for _ in range(N_CANDIDATES)]
        runs[days] = cand[int(np.argmin([abs(c.mean() - nominal) for c in cand]))]

    C.house_style()
    fig, axes = plt.subplots(len(RUN_HORIZON_D), 1, figsize=(7.2, 5.4), sharey=True)
    top = max(r.max() for r in runs.values()) * C.BINS_PER_H * 1.06

    for ax, days, tag in zip(axes, RUN_HORIZON_D, ["(a)", "(b)", "(c)", "(d)"]):
        data = runs[days]
        t = np.arange(len(data)) / C.BINS_PER_DAY
        # thinner strokes as the run gets longer, so density stays readable
        lw = 0.45 if days <= 90 else (0.3 if days <= 365 else 0.18)
        ax.plot(t, data * C.BINS_PER_H, color=C.BLUE, linewidth=lw, zorder=3)
        ax.set_xlim(0, days)
        ax.set_ylim(0, top)
        C.style_axes(ax)
        # the duration rides with the panel tag, replacing a one-entry legend
        ax.text(-0.075, 1.04, f"{tag}  {days} days", transform=ax.transAxes,
                color=C.INK, fontsize=10, va="bottom", ha="left")

    axes[-1].set_xlabel("Time (days)")
    fig.supylabel("Arrival rate (batches/h)", color=C.INK, fontsize=9, x=0.005)
    fig.tight_layout(pad=0.4, h_pad=1.0)
    C.save(fig, "fig_horizon_runs")

    rows = [{"days": d, "bins": len(v),
             "mean_batches_per_h": v.mean() * C.BINS_PER_H,
             "peak_batches_per_h": v.max() * C.BINS_PER_H,
             "zero_bin_fraction": float((v == 0).mean())} for d, v in runs.items()]
    C.write_csv(pd.DataFrame(rows), "fig_horizon_runs", index=False)
    print(f"\nnominal {nominal * C.BINS_PER_H:.2f} batches/h")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def combined() -> None:
    """Two panels: dispersion against load, and realised rate against run length."""
    from seren_countfirst import idc

    st = C.state()
    model, cal = C.winner(st)
    n_bins = int(st["n_bins"])

    load = []
    for a in COMBINED_ALPHAS:
        rng = np.random.default_rng(C.SEED + 2)
        vals = [idc(model.simulate(n_bins, rng, alpha=cal * a))
                for _ in range(N_PATHS_COMBINED_LOAD)]
        load.append({"alpha": a, "idc": float(np.median(vals))})
    load = pd.DataFrame(load)

    horizon = []
    for d in COMBINED_HORIZON_D:
        rng = np.random.default_rng(C.SEED + 3)
        rates = np.array([model.simulate(d * C.BINS_PER_DAY, rng, alpha=cal).mean()
                          for _ in range(N_PATHS_COMBINED_HORIZON)]) * C.BINS_PER_H
        horizon.append({"days": d, "median": float(np.median(rates)),
                        "q025": float(np.quantile(rates, 0.025)),
                        "q975": float(np.quantile(rates, 0.975))})
    horizon = pd.DataFrame(horizon)
    nominal = float(model.params_["mean"]) * cal * C.BINS_PER_H

    C.house_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # (a) the load knob: dispersion is linear in alpha, so a linear axis makes
    # the law a straight line; the reference is anchored on the measured alpha = 1
    a = load["alpha"].to_numpy()
    idc_1 = float(load.loc[load["alpha"] == 1.0, "idc"].iloc[0])
    grid = np.linspace(0, COMBINED_ALPHAS[-1] * 1.06, 100)
    ax1.plot(grid, 1 + grid * (idc_1 - 1), color=C.MUTED, linewidth=1.3,
             linestyle=(0, (2.2, 2.2)), label="Linear in load factor", zorder=2)
    ax1.plot(a, load["idc"].to_numpy(), color=C.BLUE, linewidth=0, marker="o",
             markersize=5.0, label="Measured", zorder=3)
    ax1.set_xlim(0, COMBINED_ALPHAS[-1] * 1.06)
    ax1.set_ylim(0, None)
    ax1.set_xticks([0, 4, 8, 12, 16])
    ax1.set_xlabel("Load factor")
    ax1.set_ylabel("Index of dispersion (IDC)")
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # (b) horizons
    x = np.arange(len(COMBINED_HORIZON_D), dtype=float)
    ax2.fill_between(x, horizon["q025"].to_numpy(), horizon["q975"].to_numpy(),
                     color=BAND_FILL, linewidth=0, label="95% of runs", zorder=1)
    ax2.axhline(nominal, color=C.MUTED, linewidth=1.3, linestyle=(0, (2.2, 2.2)),
                label="Nominal rate", zorder=2)
    ax2.plot(x, horizon["median"].to_numpy(), color=C.BLUE, linewidth=1.7,
             marker="o", markersize=4.2, label="Median run", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in COMBINED_HORIZON_D])
    ax2.set_xlim(x[0] - 0.25, x[-1] + 0.25)
    ax2.set_ylim(0, None)
    ax2.set_xlabel("Run length (days)")
    ax2.set_ylabel("Arrival rate (batches/h)")

    for ax, tag in ((ax1, "(a)"), (ax2, "(b)")):
        C.style_axes(ax)
        ax.text(-0.17, 1.02, tag, transform=ax.transAxes, color=C.INK,
                fontsize=10, va="bottom", ha="left")
        C.frame_legend(ax.legend(loc="upper left", frameon=True, framealpha=1.0,
                                 fontsize=8, handlelength=2.4, labelspacing=0.35,
                                 borderpad=0.5))

    fig.tight_layout(pad=0.4, w_pad=2.2)
    C.save(fig, "fig_load_horizon")

    C.write_csv(load, "fig_load_horizon_panel_a", index=False)
    C.write_csv(horizon, "fig_load_horizon_panel_b", index=False)
    load["linear_reference"] = 1 + load["alpha"] * (idc_1 - 1)
    print(f"\nnominal rate {nominal:.2f} batches/h")
    print("\n(a) load knob:\n", load.round(2).to_string(index=False))
    print("\n(b) horizons:\n", horizon.round(3).to_string(index=False))


VIEWS = {
    "load-arrivals": load_arrivals,
    "load-idc": load_idc,
    "horizon-runs": horizon_runs,
    "combined": combined,
}

if __name__ == "__main__":
    C.run(VIEWS, "load-arrivals", __doc__)
