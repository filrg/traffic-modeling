from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from _common import plt


# --- the fitted law ---------------------------------------------------


# the empirical series takes the ink slot the trace holds elsewhere
LAWS = [
    ("mix_lognormal_3", "Lognormal mix (k=3)", "#2a78d6"),
    ("lognormal", "Lognormal", "#eb6834"),
    ("burr12", "Burr XII", "#1baf7a"),
]
# display window and binning, fixed so the panels compare
DISPLAY_Q = 99.0
N_BINS = 70
N_GRID_PDF = 1000
# the CDF is read off a log-spaced grid, which is exact for a step function
N_GRID_CDF = 600

_LAW_CACHE: dict = {}


def _pool_and_board() -> tuple:
    """Return (sizes, scoreboard), loaded once."""
    if not _LAW_CACHE:
        from seren_countfirst.jobsize import EmpiricalJobSize

        _LAW_CACHE["sizes"] = EmpiricalJobSize.from_npz(C.RESULTS / "jobsize_pool.npz").sizes
        _LAW_CACHE["board"] = pd.read_csv(C.RESULTS / "jobsize_candidates.csv").set_index(
            "distribution"
        )
    return _LAW_CACHE["sizes"], _LAW_CACHE["board"]


def _law(row: pd.Series, name: str):
    """Rebuild a scored candidate law from the json the jobsize stage persisted."""
    from seren_countfirst.jobsize import LogNormalMixture, ScipyJobSize

    blob = json.loads(row["params"])
    if row["kind"] == "mixture_lognormal":
        return LogNormalMixture.from_dict(blob)
    return ScipyJobSize(name=name, params=tuple(blob))


def _size_axes(ax, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("Job size (work units)")
    ax.set_ylabel(ylabel)
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.6, color=C.AXIS)


def pdf() -> None:
    """Density per decade, empirical histogram against the fitted laws."""
    sizes, board = _pool_and_board()
    shown = sizes[sizes <= np.percentile(sizes, DISPLAY_Q)]
    grid = np.geomspace(shown.min(), shown.max(), N_GRID_PDF)
    curves = {"job_size": grid}
    for key, label, _ in LAWS:
        # f_X(x) -> density of log10 X, the quantity a log axis can be read on
        curves[label] = _law(board.loc[key], key).pdf(grid) * grid * np.log(10.0)
    table = pd.DataFrame(curves).set_index("job_size")

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    edges = np.geomspace(shown.min(), shown.max(), N_BINS + 1)
    counts, _ = np.histogram(shown, bins=edges)
    # per decade, so bar area is probability
    heights = counts / len(shown) / np.diff(np.log10(edges))
    ax.bar(
        edges[:-1],
        heights,
        width=np.diff(edges),
        align="edge",
        color=C.INK,
        alpha=0.16,
        edgecolor=C.INK,
        linewidth=0.25,
        zorder=2,
    )
    for _, label, color in LAWS:
        ax.plot(
            grid,
            table[label].to_numpy(),
            color=color,
            linewidth=1.3,
            linestyle=C.MODEL_DASH,
            zorder=3,
        )

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, None)
    _size_axes(ax, "Share of jobs per decade")
    handles = [Patch(facecolor=(0.04, 0.04, 0.04, 0.16), edgecolor=C.INK, linewidth=0.25)]
    handles += [Line2D([], [], color=c, linewidth=1.3, linestyle=C.MODEL_DASH) for _, _, c in LAWS]
    C.frame_legend(
        ax.legend(
            handles,
            ["Empirical"] + [label for _, label, _ in LAWS],
            loc="upper left",
            frameon=True,
            framealpha=1.0,
            fontsize=6.5,
            handlelength=2.6,
            labelspacing=0.28,
            borderpad=0.45,
        )
    )
    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_jobsize_pdf")

    C.write_csv(table, "fig_jobsize_pdf")
    print(
        f"\n{len(sizes)} completed jobs; the panel shows the {len(shown)} at or below "
        f"the {DISPLAY_Q:.0f}th percentile ({np.percentile(sizes, DISPLAY_Q):,.0f} units)"
    )
    print(
        f"the {len(sizes) - len(shown)} jobs above it reach {sizes.max():,.0f} units, "
        f"against a median of {np.median(sizes):,.0f}"
    )


def cdf() -> None:
    """Cumulative distribution over the full range, empirical against the laws."""
    sizes, board = _pool_and_board()
    ordered = np.sort(sizes)
    grid = np.geomspace(ordered[0], ordered[-1], N_GRID_CDF)
    # the empirical CDF on the same grid
    curves = {
        "job_size": grid,
        "Empirical": np.searchsorted(ordered, grid, side="right") / len(ordered),
    }
    for key, label, _ in LAWS:
        curves[label] = _law(board.loc[key], key).cdf(grid)
    table = pd.DataFrame(curves).set_index("job_size")

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.plot(grid, table["Empirical"].to_numpy(), color=C.INK, linewidth=1.5, zorder=4)
    for _, label, color in LAWS:
        ax.plot(
            grid,
            table[label].to_numpy(),
            color=color,
            linewidth=1.3,
            linestyle=C.MODEL_DASH,
            zorder=3,
        )

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _size_axes(ax, "Fraction of jobs at or below")
    handles = [Line2D([], [], color=C.INK, linewidth=1.5)]
    handles += [Line2D([], [], color=c, linewidth=1.3, linestyle=C.MODEL_DASH) for _, _, c in LAWS]
    C.frame_legend(
        ax.legend(
            handles,
            ["Empirical"] + [label for _, label, _ in LAWS],
            loc="upper left",
            frameon=True,
            framealpha=1.0,
            fontsize=6.5,
            handlelength=2.6,
            labelspacing=0.28,
            borderpad=0.45,
        )
    )
    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_jobsize_cdf")

    C.write_csv(table, "fig_jobsize_cdf")
    emp = table["Empirical"].to_numpy()
    print("largest vertical gap from the empirical CDF (the KS statistic's grid read):")
    for _, label, _ in LAWS:
        print(f"  {label:22s} {np.max(np.abs(table[label].to_numpy() - emp)):.4f}")


# --- generated draws --------------------------------------------------


# redraws of every size against the trace's own batches, so only the size model varies
N_REP = 200
# the sizes are read as a step function on a log grid, which is exact
N_GRID = 700
BAND_FILL = "#dce9f9"

# replicates for the composed view; each redraws every size
N_WORK_REP = 40
WORK_STRATA = 8
# the naive baseline takes the alternative-model slot it holds elsewhere
IID_COLOR = "#eb6834"

_DRAWS_CACHE: dict = {}
_STRUCT: dict = {}


def _structure() -> tuple:
    """Return the frozen law, the trace's jobs, and the batch arrays it draws on."""
    if not _STRUCT:
        from seren_countfirst.data import arrival_view, jobsize_view, load_trace
        from seren_countfirst.jobsize import (
            attach_activity,
            batch_frame,
            bin_grid_index,
            jobsize_table,
            read_two_part_jobsize,
        )
        from seren_countfirst.paths import TRACE_CSV

        model = read_two_part_jobsize(
            C.RESULTS / "jobsize_two_part.json", C.RESULTS / "jobsize_pool.npz"
        )
        raw = load_trace(TRACE_CSV)
        jobs = attach_activity(jobsize_table(jobsize_view(raw, "SFT")), arrival_view(raw, "SFT"))
        batches = batch_frame(jobs)
        card = batches["cardinality"].to_numpy(dtype=int)
        _STRUCT["value"] = (
            model,
            jobs,
            batches["activity"].to_numpy(dtype=float),
            card,
            # the real grid position: the tail is placed per bin,
            # so a compacted index would misplace it
            bin_grid_index(batches["bin"]),
            np.repeat(batches["bin"].to_numpy(), card),
        )
    return _STRUCT["value"]


def _draws() -> tuple:
    """Return the frozen law, the trace's sizes, and `N_REP` draws for the same jobs."""
    model, jobs, activity, card, index, _ = _structure()
    drawn = [
        model.rvs(activity, card, index, np.random.default_rng(C.SEED + r)) for r in range(N_REP)
    ]
    return model, jobs["job_size"].to_numpy(dtype=float), len(card), drawn


def _ccdf(sample: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return the share of `sample` strictly above each point of `grid`."""
    ordered = np.sort(np.asarray(sample, dtype=float))
    return 1.0 - np.searchsorted(ordered, grid, side="right") / len(ordered)


def _visible(y: np.ndarray, floor: float) -> np.ndarray:
    """Blank what a log axis cannot show, so the curve ends instead of diving."""
    return np.where(y > floor, y, np.nan)


def _curves() -> tuple:
    """Return the shared grid, the trace's CCDF on it, and one CCDF per redraw."""
    if not _DRAWS_CACHE:
        model, sizes, n_batches, drawn = _draws()
        grid = np.geomspace(
            min(sizes.min(), min(d.min() for d in drawn)),
            max(sizes.max(), max(d.max() for d in drawn)),
            N_GRID,
        )
        _DRAWS_CACHE["value"] = (
            model,
            sizes,
            n_batches,
            drawn,
            grid,
            _ccdf(sizes, grid),
            np.array([_ccdf(d, grid) for d in drawn]),
        )
    return _DRAWS_CACHE["value"]


def _report(model, sizes, n_batches, drawn, grid, trace, band) -> None:
    """Print what the panel is worth: the fit, its spread, and what is replayed."""
    ks = np.array([stats.ks_2samp(d, sizes).statistic for d in drawn])
    mean_ratio = np.array([d.mean() / sizes.mean() for d in drawn])
    med_ratio = np.array([np.median(d) / np.median(sizes) for d in drawn])
    q025, q975 = np.quantile(band, [0.025, 0.975], axis=0)
    inside = float(np.mean((trace >= q025) & (trace <= q975)))
    n_above = np.array([int((d >= model.cut).sum()) for d in drawn])
    print(
        f"\n{len(sizes):,} jobs in {n_batches:,} batches, redrawn {N_REP} times on "
        f"the trace's own batch structure"
    )
    print(f"KS against the trace   {ks.mean():.4f}  ({ks.min():.4f} to {ks.max():.4f})")
    print(
        f"mean work ratio        {mean_ratio.mean():.4f}  "
        f"({mean_ratio.min():.4f} to {mean_ratio.max():.4f})"
    )
    print(
        f"median ratio           {med_ratio.mean():.4f}  "
        f"({med_ratio.min():.4f} to {med_ratio.max():.4f})"
    )
    print(f"the trace's curve lies inside the band over {inside:.1%} of the grid")
    print(
        f"\ncut S* {model.cut:,.0f} units: {int((sizes >= model.cut).sum())} jobs above "
        f"it in the trace against {n_above.mean():.0f} drawn. Above the cut the model "
        f"resamples those exceedances, so that stretch is replayed, not fitted"
    )
    print(
        "in-sample: this window is the one the law was fitted on. Held-out transfer "
        "is in results/jobsize_folds_summary.csv"
    )


def ccdf() -> None:
    """Survival function over the full range, trace against the generator's draws."""
    model, sizes, n_batches, drawn, grid, trace, paths = _curves()
    median = np.median(paths, axis=0)
    q025, q975 = np.quantile(paths, [0.025, 0.975], axis=0)
    table = pd.DataFrame(
        {
            "job_size": grid,
            "trace": trace,
            "model_median": median,
            "model_q025": q025,
            "model_q975": q975,
        }
    ).set_index("job_size")

    # one job in the pool: the smallest share either sample can report
    floor = 0.5 / len(sizes)
    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.fill_between(
        grid,
        np.maximum(q025, floor),
        np.maximum(q975, floor),
        color=BAND_FILL,
        linewidth=0,
        zorder=1,
    )
    ax.axvline(model.cut, color=C.MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    ax.annotate(
        "$S^\\star$",
        (model.cut, 0.32),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6.5,
        color=C.INK_2,
    )
    # the model is drawn over the trace here, against the usual order
    ax.plot(grid, _visible(trace, floor), color=C.INK, linewidth=1.5, zorder=3)
    ax.plot(
        grid, _visible(median, floor), color=C.BLUE, linewidth=1.3, linestyle=C.MODEL_DASH, zorder=4
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(floor, 1.4)
    ax.set_xlabel("Job size (work units)")
    ax.set_ylabel("Fraction of jobs above")
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_tick_params(which="minor", length=1.5, width=0.6, color=C.AXIS)
    handles = [
        Line2D([], [], color=C.INK, linewidth=1.5),
        Line2D([], [], color=C.BLUE, linewidth=1.3, linestyle=C.MODEL_DASH),
        Patch(facecolor=BAND_FILL, edgecolor="none"),
    ]
    C.frame_legend(
        ax.legend(
            handles,
            ["Trace", "Generated", "95% of draws"],
            loc="lower left",
            frameon=True,
            framealpha=1.0,
            fontsize=6.5,
            handlelength=2.6,
            labelspacing=0.28,
            borderpad=0.45,
        )
    )
    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_jobsize_ccdf")

    C.write_csv(table, "fig_jobsize_ccdf")
    _report(model, sizes, n_batches, drawn, grid, trace, paths)


def draws_cdf() -> None:
    """Cumulative distribution, trace against the generator's draws."""
    model, sizes, n_batches, drawn, grid, trace, paths = _curves()
    # the same curves read from below
    seen, band = 1.0 - trace, 1.0 - paths
    median = np.median(band, axis=0)
    q025, q975 = np.quantile(band, [0.025, 0.975], axis=0)
    table = pd.DataFrame(
        {
            "job_size": grid,
            "trace": seen,
            "model_median": median,
            "model_q025": q025,
            "model_q975": q975,
        }
    ).set_index("job_size")

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.fill_between(grid, q025, q975, color=BAND_FILL, linewidth=0, zorder=1)
    ax.axvline(model.cut, color=C.MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    ax.annotate(
        "$S^\\star$",
        (model.cut, 0.5),
        xytext=(-3, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=6.5,
        color=C.INK_2,
    )
    ax.plot(grid, seen, color=C.INK, linewidth=1.5, zorder=3)
    ax.plot(grid, median, color=C.BLUE, linewidth=1.3, linestyle=C.MODEL_DASH, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Job size (work units)")
    ax.set_ylabel("Fraction of jobs at or below")
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.6, color=C.AXIS)
    handles = [
        Line2D([], [], color=C.INK, linewidth=1.5),
        Line2D([], [], color=C.BLUE, linewidth=1.3, linestyle=C.MODEL_DASH),
        Patch(facecolor=BAND_FILL, edgecolor="none"),
    ]
    C.frame_legend(
        ax.legend(
            handles,
            ["Trace", "Generated", "95% of draws"],
            loc="upper left",
            frameon=True,
            framealpha=1.0,
            fontsize=6.5,
            handlelength=2.6,
            labelspacing=0.28,
            borderpad=0.45,
        )
    )
    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_jobsize_fit_cdf")

    C.write_csv(table, "fig_jobsize_fit_cdf")
    _report(model, sizes, n_batches, drawn, grid, trace, paths)
    widest = int(np.argmax(np.abs(median - seen)))
    print(
        f"\nthe CDF's largest vertical gap is {np.abs(median - seen)[widest]:.4f} at "
        f"{grid[widest]:,.0f} units. Everything above S* is compressed into the top "
        f"{float(1.0 - seen[np.searchsorted(grid, model.cut)]):.3f} of this axis, "
        f"which is what the ccdf view exists to show"
    )


def work_per_bin() -> None:
    """Work offered per bin against activity: trace, an iid marginal, the two-part law."""
    from seren_countfirst.jobsize import read_jobsize_model

    model, jobs, activity, card, index, bin_of_job = _structure()
    marginal = read_jobsize_model(C.RESULTS / "jobsize_model.json")
    observed = jobs.groupby("bin")["job_size"].sum()
    bins = observed.index
    act = jobs.groupby("bin")["activity"].first().reindex(bins).to_numpy(dtype=float)

    def compose(rule) -> np.ndarray:
        """Sum each replicate's sizes into the bins the trace put those jobs in."""
        return np.array(
            [
                pd.Series(rule(np.random.default_rng(C.SEED + r)))
                .groupby(bin_of_job)
                .sum()
                .reindex(bins)
                .to_numpy()
                for r in range(N_WORK_REP)
            ]
        )

    arms = [
        ("Two-part model", C.BLUE, compose(lambda rng: model.rvs(activity, card, index, rng))),
        ("iid marginal", IID_COLOR, compose(lambda rng: marginal.rvs(len(jobs), rng))),
    ]

    # bins are the unit here
    edges = np.unique(np.quantile(act, np.linspace(0.0, 1.0, WORK_STRATA + 1)))
    strata = np.clip(np.searchsorted(edges, act, side="right") - 1, 0, len(edges) - 2)
    n_strata = len(edges) - 1
    centres = np.array([np.median(act[strata == b]) for b in range(n_strata)])
    seen = np.array([np.median(observed.to_numpy()[strata == b]) for b in range(n_strata)])
    curves = {
        label: np.array([np.mean([np.median(x[strata == b]) for x in w]) for b in range(n_strata)])
        for label, _, w in arms
    }
    table = pd.DataFrame({"N_k": centres, "trace": seen, **curves}).set_index("N_k")

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    for label, color, _ in arms:
        ax.plot(
            centres,
            curves[label],
            color=color,
            linewidth=1.3,
            linestyle=C.MODEL_DASH,
            marker="o",
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
        )
    ax.plot(
        centres,
        seen,
        color=C.INK,
        linewidth=1.5,
        marker="o",
        markersize=3.2,
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(centres[0] / 1.5, centres[-1] * 1.5)
    ax.set_xlabel("Batch count in bin, $N_k$")
    ax.set_ylabel("Median work per bin (work units)")
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_tick_params(which="minor", length=1.5, width=0.6, color=C.AXIS)
    C.stroke_legend(
        ax,
        [("Trace", C.INK, 1.5, "-")]
        + [(label, color, 1.3, C.MODEL_DASH) for label, color, _ in arms],
        loc="upper left",
    )
    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_jobsize_work_per_bin")

    C.write_csv(table, "fig_jobsize_work_per_bin")
    total = observed.sum()
    print(
        f"\nmedian work per bin over {n_strata} strata equal in bins "
        f"({len(observed):,} occupied bins, {total:.4g} work units total):"
    )
    show = table.copy()
    for label, _, _ in arms:
        show[f"{label} / trace"] = table[label] / table["trace"]
    print(show.round(2).to_string())
    print("\nspan from the quietest stratum to the busiest:")
    for name, v in [("trace", seen)] + [(label, curves[label]) for label, _, _ in arms]:
        print(f"  {name:20s} {v[0]:>12,.0f} -> {v[-1]:>12,.0f}   factor {v[-1] / v[0]:6.1f}")
    print("\ntotal work delivered, as a share of the trace's:")
    for label, _, w in arms:
        print(f"  {label:20s} {w.sum(axis=1).mean() / total:.4f}")
    print(
        "\nthe trace's own span is the target; a law drawn per job cannot help "
        "tracking the job count, which is what the iid arm's span shows"
    )


INPUTS = (
    "jobsize_pool",
    "jobsize_candidates",
    "state",
    "counts",
    "jobsize_two_part",
    "jobsize_model",
)

VIEWS = {
    "pdf": pdf,
    "cdf": cdf,
    "ccdf": ccdf,
    "draws-cdf": draws_cdf,
    "work-per-bin": work_per_bin,
}

if __name__ == "__main__":
    C.run(
        VIEWS,
        "pdf",
        "Job sizes: the trace against the fitted laws, and against the generator's draws.",
        INPUTS,
    )
