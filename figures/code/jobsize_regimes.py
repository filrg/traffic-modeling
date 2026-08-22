from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from _common import plt

# validated categorical slots, fixed order and never cycled; the exceedances
# take the ink slot the trace holds elsewhere
COMPONENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
N_GRID = 400
VIOLIN_HALF_WIDTH = 0.38
# a violin is cut where its density falls below this share of its own peak
TRIM_FRACTION = 0.02

# activity strata, shared by `weights` and `level` so the two panels read across
ACTIVITY_BINS = 8
# the fourth validated hue, outside the component slots: `level` and `tail-rate`
# use it for the series that must not read as a component
ALT_COLOR = "#4a3aa7"
MARKER_SIZE = 3.4
# activity is carried past the trace's busiest bin; the panel marks where it stops
LEVEL_MAX = 3000.0
LEVEL_GRID = 22
# singleton batches per activity point, so the sample statistic is the law's own
LEVEL_DRAWS = 200_000
SHARE_GRID, SHARE_DRAWS = 30, 200_000
# components share an x, so their markers are nudged apart by this factor
SHARE_DODGE = 1.13
# enough resamples that the interval's 2.5% edge is not a handful of draws
N_BOOT = 2000


def _bulk():
    """Return the frozen two-part law, its bulk, the exceedances and the pool size."""
    from seren_countfirst.jobsize import EmpiricalJobSize, read_two_part_jobsize

    model = read_two_part_jobsize(
        C.RESULTS / "jobsize_two_part.json", C.RESULTS / "jobsize_pool.npz"
    )
    n_pool = len(EmpiricalJobSize.from_npz(C.RESULTS / "jobsize_pool.npz").sizes)
    return model, model.bulk, np.asarray(model.tail_pool, dtype=float), n_pool


def _colors(k: int) -> list:
    """Return one hue per component, refusing to cycle the categorical slots."""
    if k > len(COMPONENT_COLORS):
        raise SystemExit(
            f"{k} components exceeds the {len(COMPONENT_COLORS)} validated hues. "
            "Cycling them would make two components share an identity; draw small "
            "multiples instead, or fold the smallest components into one panel."
        )
    return COMPONENT_COLORS[:k]


def _component_sigma(bulk) -> np.ndarray:
    """Return each component's marginal log-work sd: batch and job levels combined."""
    return np.sqrt(bulk.s_batch**2 + bulk.s_job**2)


def _mean_weights(bulk) -> np.ndarray:
    """Return component shares averaged over the trace's bin-activity distribution."""
    counts = C.state()["counts"].values
    active = counts[counts > 0].astype(float)
    # weighted by batches, which is the unit the component is drawn for
    return (bulk.weights(active) * active[:, None]).sum(axis=0) / active.sum()


def _trace_jobs():
    """Return every sized job with its bin and that bin's batch count."""
    from seren_countfirst.jobsize import JOBSIZE_STATES, attach_activity, jobsize_table
    from seren_countfirst.run_pipeline.context import default_context
    from traffic_modeling.datasets import DatasetCapability

    dataset = default_context().load_dataset(
        DatasetCapability.ARRIVAL_EVENTS,
        DatasetCapability.JOB_STATES,
        DatasetCapability.EXECUTION_DURATIONS,
        DatasetCapability.GPU_COUNTS,
    )
    return attach_activity(jobsize_table(dataset.jobs_in_states(JOBSIZE_STATES)), dataset.jobs)


def _activity_bins(activity: np.ndarray) -> tuple:
    """Return each job's activity stratum and the stratum edges, equal-count."""
    edges = np.unique(np.quantile(activity, np.linspace(0.0, 1.0, ACTIVITY_BINS + 1)))
    strata = np.clip(np.searchsorted(edges, activity, side="right") - 1, 0, len(edges) - 2)
    return strata, edges


def _day_blocks(
    bins, strata: np.ndarray, values: np.ndarray, n_strata: int, statistic=np.median
) -> tuple:
    """Return each stratum's `statistic` at the 2.5th and 97.5th percentile of day resamples.

    Resamples whole days, not rows, because sizes and large-job arrivals cluster
    inside a day.
    """
    day = pd.factorize(pd.DatetimeIndex(bins).floor("D"))[0]
    by_day = [np.nonzero(day == d)[0] for d in range(int(day.max()) + 1)]
    rng = np.random.default_rng(C.SEED)
    boot = np.full((N_BOOT, n_strata), np.nan)
    for r in range(N_BOOT):
        take = np.concatenate([by_day[d] for d in rng.integers(0, len(by_day), len(by_day))])
        drawn_strata, drawn_values = strata[take], values[take]
        for b in range(n_strata):
            sel = drawn_values[drawn_strata == b]
            if len(sel):
                boot[r, b] = statistic(sel)
    return np.nanquantile(boot, [0.025, 0.975], axis=0)


class _TriDot(HandlerBase):
    """Legend handle drawing one marker split into every component's colour.

    One entry stands for all the measured series, so the handle carries every hue.
    """

    def __init__(self, colors, radius: float = 3.0):
        super().__init__()
        self.colors, self.radius = colors, radius

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        centre = (0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent)
        step = 360.0 / len(self.colors)
        wedges = [
            Wedge(
                centre,
                self.radius,
                90 + i * step,
                90 + (i + 1) * step,
                facecolor=c,
                edgecolor="white",
                linewidth=0.5,
            )
            for i, c in enumerate(self.colors)
        ]
        for w in wedges:
            w.set_transform(trans)
        return wedges


def _measured(ax, x, y, lo, hi, color, ci: bool, clip: bool = True, zorder: int = 4):
    """Draw the measured points, carrying their interval when `ci` asks for it."""
    shared = {
        "markersize": MARKER_SIZE,
        "color": color,
        "markeredgecolor": "white",
        "markeredgewidth": 0.5,
        "clip_on": clip,
        "zorder": zorder,
    }
    if not ci:
        return ax.plot(x, y, linestyle="none", marker="o", **shared)[0]
    return ax.errorbar(
        x,
        y,
        yerr=[np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)],
        fmt="o",
        elinewidth=0.8,
        capsize=1.8,
        capthick=0.8,
        **shared,
    )


def _band_rule(bulk, activity: np.ndarray, log_cut: float) -> tuple:
    """Return (classify, thresholds): the component whose fitted density is highest.

    One partition of the size axis, applied to every job whatever its bin. The
    rule is activity-free, so the shares it produces measure the trend rather
    than restate it.
    """
    mu_c, sd_c = bulk.mu + bulk.m, _component_sigma(bulk)
    prior = np.log(bulk.weights(activity).mean(axis=0))

    def classify(z: np.ndarray) -> np.ndarray:
        return (prior + stats.norm.logpdf(np.asarray(z, dtype=float)[:, None], mu_c, sd_c)).argmax(
            axis=1
        )

    scan = np.linspace(float(mu_c.min() - 6.0 * sd_c.max()), log_cut, 20_000)
    label = scan_label = classify(scan)
    edges = np.nonzero(np.diff(label))[0]
    # each component must win exactly one stretch, in order, for the panel to
    # read as size bands
    if list(scan_label[np.r_[0, edges + 1]]) != list(range(len(mu_c))):
        raise SystemExit(
            "the components do not partition the size axis into one band each, so "
            "the shares cannot be described as size bands. Draw the responsibilities "
            "instead, or report the rule as an argmax rather than as thresholds."
        )
    return classify, np.exp(scan[edges])


def _pct_labels(share: np.ndarray) -> list:
    """Format shares as whole percents that still sum to 100, by largest remainder."""
    scaled = np.asarray(share, dtype=float) * 100.0
    whole = np.floor(scaled).astype(int)
    short = int(round(scaled.sum())) - int(whole.sum())
    for i in np.argsort(whole - scaled)[:short]:
        whole[i] += 1
    return [f"({v}%)" for v in whole]


def _violin(ax, x: float, grid: np.ndarray, density: np.ndarray, color: str) -> None:
    """Draw one mirrored density, trimmed where it is too thin to read."""
    keep = density > TRIM_FRACTION * density.max()
    if not keep.any():
        return
    y, d = grid[keep], density[keep]
    half = VIOLIN_HALF_WIDTH * d / density.max()
    ax.fill_betweenx(
        y, x - half, x + half, facecolor=color, alpha=0.22, edgecolor=color, linewidth=0.9, zorder=2
    )


def components() -> None:
    """Fitted component laws and the observed exceedances, as violins."""
    model, bulk, tail, n_pool = _bulk()
    k = bulk.n_components
    colors = _colors(k)
    mu_c = bulk.mu + bulk.m
    sd_c = _component_sigma(bulk)
    # the component weights are shares of the bulk, the tail is a share of every
    # job; deflating the first by the second puts all four on one denominator so
    # the column reads as a partition
    tail_share = len(tail) / n_pool
    share = np.append(_mean_weights(bulk) * (1.0 - tail_share), tail_share)
    hues = colors + [ALT_COLOR]
    log_cut = float(np.log10(model.cut))
    ln10 = np.log(10.0)

    lo = float(np.log10(np.exp((mu_c - 3.2 * sd_c).min())))
    hi = float(np.log10(tail.max()))
    grid = np.linspace(lo, hi, N_GRID)

    # bulk truncated at the cut, exceedances starting there: neither density
    # crosses to the other's side
    below, above = grid <= log_cut, grid >= log_cut
    curves = {"log10_work_units": grid}
    for c in range(k):
        mass = stats.norm.cdf(log_cut * ln10, mu_c[c], sd_c[c])
        d = stats.norm.pdf(grid * ln10, mu_c[c], sd_c[c]) * ln10 / max(mass, 1e-12)
        curves[f"component_{c + 1}"] = np.where(below, d, 0.0)
    kde = stats.gaussian_kde(np.log10(tail))(grid)
    curves["exceedances"] = np.where(above, kde, 0.0)
    table = pd.DataFrame(curves).set_index("log10_work_units")

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.axhline(log_cut, color=C.MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.annotate(
        "$S^\\star$",
        (k + 0.62, log_cut),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
        color=C.INK_2,
    )
    for i, name in enumerate(table.columns):
        _violin(ax, i, grid, table[name].to_numpy(), hues[i])

    medians = list(np.exp(mu_c)) + [float(np.median(tail))]
    for i, med in enumerate(medians):
        y = float(np.log10(med))
        ax.plot(
            [i - VIOLIN_HALF_WIDTH * 0.55, i + VIOLIN_HALF_WIDTH * 0.55],
            [y, y],
            color=hues[i],
            linewidth=1.6,
            solid_capstyle="butt",
            zorder=4,
        )

    labels = _pct_labels(share)
    ax.set_xticks(range(k + 1))
    ax.set_xticklabels([f"C{c + 1}\n{labels[c]}" for c in range(k)] + [f"Tail\n{labels[k]}"])
    ax.set_xlim(-0.7, k + 0.7)
    ax.set_ylim(lo, hi + 0.15)
    ax.set_ylabel("Job size (work units)")
    ticks = np.arange(np.ceil(lo), np.floor(hi) + 1, 2 if hi - lo > 6 else 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"$10^{{{int(t)}}}$" for t in ticks])
    C.style_axes(ax, grid_axis="y", labelsize=7.5, ticklen=2.5)
    # the fill is the violin's own, so the swatch maps onto the shape it names
    C.frame_legend(
        ax.legend(
            [Patch(facecolor=h, alpha=0.22, edgecolor=h, linewidth=0.9) for h in hues],
            [f"Component {c + 1}" for c in range(k)] + ["Tail"],
            loc="upper left",
            frameon=True,
            framealpha=1.0,
            fontsize=6.5,
            handlelength=1.6,
            handleheight=1.1,
            labelspacing=0.28,
            borderpad=0.45,
        )
    )
    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_jobsize_regimes")

    C.write_csv(table, "fig_jobsize_regimes")
    print(f"k = {k} bulk components; medians {', '.join(f'{v:,.0f}' for v in np.exp(mu_c))} units")
    print(
        f"shares of every job, the four partitioning the pool: "
        f"{', '.join(f'{s:.4f}' for s in share)}  (sum {share.sum():.4f})"
    )
    print(
        f"the component weights before deflating by the tail's share: "
        f"{', '.join(f'{s:.3f}' for s in _mean_weights(bulk))}"
    )
    print(f"cut {model.cut:,.0f}; {len(tail):,} exceedances, median {np.median(tail):,.0f} units")


def weights() -> None:
    """Share of bulk jobs in each component's size band, against bin activity."""
    model, bulk, _, _ = _bulk()
    k = bulk.n_components
    colors = _colors(k)
    log_cut = float(np.log(model.cut))

    jobs = _trace_jobs()
    sizes = jobs["job_size"].to_numpy(dtype=float)
    keep = sizes < model.cut
    act = jobs["activity"].to_numpy(dtype=float)[keep]
    classify, thresholds = _band_rule(bulk, act, log_cut)
    label = classify(np.log(sizes[keep]))

    strata, _ = _activity_bins(act)
    n_strata = int(strata.max()) + 1
    centres = np.array([np.median(act[strata == b]) for b in range(n_strata)])
    seen = np.array([[np.mean(label[strata == b] == c) for c in range(k)] for b in range(n_strata)])
    # a share is the mean of an indicator, so the same day-block resample serves
    bounds = [
        _day_blocks(jobs.loc[keep, "bin"], strata, (label == c).astype(float), n_strata, np.mean)
        for c in range(k)
    ]
    lo = np.column_stack([b[0] for b in bounds])
    hi = np.column_stack([b[1] for b in bounds])

    # the model's share of the same bands, classified by the same rule
    grid = np.unique(np.round(np.geomspace(max(act.min(), 1.0), act.max(), SHARE_GRID)))
    rng = np.random.default_rng(C.SEED)
    card = np.ones(SHARE_DRAWS, dtype=int)
    drawn = np.array(
        [
            np.bincount(
                classify(np.log(bulk.rvs(np.full(SHARE_DRAWS, n), card, rng, 1.0, log_cut))),
                minlength=k,
            )
            / SHARE_DRAWS
            for n in grid
        ]
    )
    table = pd.DataFrame(
        drawn, index=pd.Index(grid, name="N_k"), columns=[f"component_{c + 1}" for c in range(k)]
    )

    # intervals off unless asked; `tail-rate` keeps them on
    ci = C.show_ci(False)
    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    for c in range(k):
        ax.plot(grid, drawn[:, c], color=colors[c], linewidth=1.3, linestyle=C.MODEL_DASH, zorder=3)
        # nudged apart only when the intervals are drawn; bare markers stay at
        # the stratum's own activity
        at = centres * SHARE_DODGE ** (c - (k - 1) / 2) if ci else centres
        _measured(ax, at, seen[:, c], lo[:, c], hi[:, c], colors[c], ci)

    ax.set_xscale("log")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Batch count in bin, $N_k$")
    ax.set_ylabel("Bulk component share, $\\pi_c(N_k)$")
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.6, color=C.AXIS)
    # colour carries the component; one split marker stands for every measured series
    tri = Line2D([], [], linestyle="none")
    handles = [Line2D([], [], color=c, linewidth=1.3, linestyle=C.MODEL_DASH) for c in colors] + [
        tri
    ]
    C.frame_legend(
        ax.legend(
            handles,
            [f"Component {c + 1}" for c in range(k)] + ["Trace, 95% CI" if ci else "Trace"],
            handler_map={tri: _TriDot(colors)},
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
    C.save(fig, "fig_jobsize_weights")

    C.write_csv(table, "fig_jobsize_weights")
    cols = [f"component_{c + 1}" for c in range(k)]
    obs = pd.concat(
        [
            pd.DataFrame(
                a,
                index=pd.Index(np.round(centres, 1), name="N_k"),
                columns=[f"{n}_{c}" for c in cols],
            )
            for n, a in (("share", seen), ("lo", lo), ("hi", hi))
        ],
        axis=1,
    )
    C.write_csv(obs, "fig_jobsize_weights_measured")
    at = np.column_stack([np.interp(np.log(centres), np.log(grid), drawn[:, c]) for c in range(k)])
    inside = (at >= lo) & (at <= hi)
    print(
        f"{int(keep.sum()):,} bulk jobs classified by size alone into {k} bands: "
        f"under {thresholds[0]:,.0f}, to {thresholds[1]:,.0f}, then to the cut"
    )
    print(
        "the rule is activity-free, so the shares below measure the trend rather "
        "than restate the logit that generated it"
    )
    print("\nshare per band by activity stratum, trace against model:")
    for b in range(n_strata):
        cells = "  ".join(
            f"C{c + 1} {seen[b, c]:.3f} [{lo[b, c]:.3f},{hi[b, c]:.3f}] vs {at[b, c]:.3f}"
            for c in range(k)
        )
        print(f"  N_k {centres[b]:6.1f}  {cells}")
    print(
        f"\nmodel inside the trace interval in {int(inside.sum())} of {inside.size} "
        f"cells; per component {[int(inside[:, c].sum()) for c in range(k)]} of "
        f"{n_strata}"
    )
    # both counts are printed, so a widening interval can be read against them
    days = pd.DatetimeIndex(jobs.loc[keep, "bin"]).floor("D")
    print("\nwhy the intervals widen to the right (markers are nudged apart on x):")
    print(
        pd.DataFrame(
            {
                "N_k": np.round(centres, 1),
                "jobs": [int((strata == b).sum()) for b in range(n_strata)],
                "days": [days[strata == b].nunique() for b in range(n_strata)],
                "widest_CI": np.round((hi - lo).max(axis=1), 3),
            }
        ).to_string(index=False)
    )
    print(
        f"trace  C1 {seen[0, 0]:.3f} -> {seen[-1, 0]:.3f}   "
        f"C{k} {seen[0, k - 1]:.3f} -> {seen[-1, k - 1]:.3f}"
    )
    print(
        f"model  C1 {at[0, 0]:.3f} -> {at[-1, 0]:.3f}   "
        f"C{k} {at[0, k - 1]:.3f} -> {at[-1, k - 1]:.3f}"
    )


def level() -> None:
    """Median job size against bin activity: activity on the weights, or on the level."""
    from seren_countfirst.jobsize import read_conditional_jobsize

    model, bulk, _, _ = _bulk()
    # the alternative design: one location moved by beta log N_k
    on_level = read_conditional_jobsize(C.RESULTS / "jobsize_conditional.json")
    jobs = _trace_jobs()
    act = jobs["activity"].to_numpy(dtype=float)
    sizes = jobs["job_size"].to_numpy(dtype=float)

    grid = np.unique(np.round(np.geomspace(1.0, LEVEL_MAX, LEVEL_GRID)))
    rng = np.random.default_rng(C.SEED)
    card = np.ones(LEVEL_DRAWS, dtype=int)
    log_cut = float(np.log(model.cut))
    by_weights = np.array(
        [np.median(bulk.rvs(np.full(LEVEL_DRAWS, n), card, rng, 1.0, log_cut)) for n in grid]
    )
    by_level = np.array([np.median(on_level.rvs(np.full(LEVEL_DRAWS, n), card, rng)) for n in grid])
    table = pd.DataFrame({"N_k": grid, "on_weights": by_weights, "on_level": by_level}).set_index(
        "N_k"
    )

    strata, edges = _activity_bins(act)
    n_strata = len(edges) - 1
    centres = np.array([np.median(act[strata == b]) for b in range(n_strata)])
    observed = np.array([np.median(sizes[strata == b]) for b in range(n_strata)])
    lo, hi = _day_blocks(jobs["bin"], strata, sizes, n_strata)

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    # stepped over each stratum's span, never interpolated between centres
    ax.fill_between(
        edges,
        np.append(lo, lo[-1]),
        np.append(hi, hi[-1]),
        step="post",
        color=C.BAND,
        linewidth=0,
        zorder=1,
    )
    ax.axvline(act.max(), color=C.MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    top = max(hi.max(), by_level.max()) * 1.5
    ax.annotate(
        "busiest bin",
        (act.max(), top * 0.62),
        xytext=(-3, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=6.5,
        color=C.INK_2,
    )
    ax.plot(grid, by_level, color=ALT_COLOR, linewidth=1.3, linestyle=C.MODEL_DASH, zorder=3)
    ax.plot(
        grid, by_weights, color=COMPONENT_COLORS[0], linewidth=1.3, linestyle=C.MODEL_DASH, zorder=4
    )
    ax.plot(
        centres,
        observed,
        linestyle="none",
        marker="o",
        markersize=3.4,
        color=C.INK,
        markeredgecolor="white",
        markeredgewidth=0.5,
        zorder=5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(min(lo.min(), by_level.min()) / 1.5, top)
    ax.set_xlabel("Batch count in bin, $N_k$")
    ax.set_ylabel("Median job size (work units)")
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_tick_params(which="minor", length=1.5, width=0.6, color=C.AXIS)
    handles = [
        Line2D(
            [],
            [],
            color=C.INK,
            linestyle="none",
            marker="o",
            markersize=3.4,
            markeredgecolor="white",
            markeredgewidth=0.5,
        ),
        Patch(facecolor=C.BAND, edgecolor="none"),
        Line2D([], [], color=COMPONENT_COLORS[0], linewidth=1.3, linestyle=C.MODEL_DASH),
        Line2D([], [], color=ALT_COLOR, linewidth=1.3, linestyle=C.MODEL_DASH),
    ]
    C.frame_legend(
        ax.legend(
            handles,
            ["Trace", "95% CI", "Activity on weights", "Activity on level"],
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
    C.save(fig, "fig_jobsize_level")

    C.write_csv(table, "fig_jobsize_level")
    seen = pd.DataFrame(
        {"N_k": centres.round(1), "median": observed, "block_lo": lo, "block_hi": hi}
    ).set_index("N_k")
    C.write_csv(seen, "fig_jobsize_level_measured")
    at_weights = np.interp(np.log(centres), np.log(grid), by_weights)
    at_level = np.interp(np.log(centres), np.log(grid), by_level)
    inside = [int(((c >= lo) & (c <= hi)).sum()) for c in (at_weights, at_level)]
    print(
        f"median job size falls {by_weights[0]:,.0f} -> {by_weights[-1]:,.0f} on the "
        f"weights (factor {by_weights[0] / by_weights[-1]:.1f}) against "
        f"{by_level[0]:,.0f} -> {by_level[-1]:,.0f} on the level "
        f"(factor {by_level[0] / by_level[-1]:.1f}) over N_k 1 to {grid[-1]:.0f}"
    )
    print(
        f"the two cross near N_k = "
        f"{np.exp(np.interp(0.0, np.log(by_level / by_weights)[::-1], np.log(grid)[::-1])):.0f}"
        f"; ratio {by_level[0] / by_weights[0]:.2f} at N_k = 1 and "
        f"{by_level[-1] / by_weights[-1]:.2f} at N_k = {grid[-1]:.0f}"
    )
    print(
        f"\ntrace medians over {n_strata} equal-count activity strata, "
        f"{N_BOOT} day-block resamples:"
    )
    print(seen.round(0).to_string())
    print(f"widest block interval spans a factor of {(hi / lo).max():.1f}")
    print(
        f"inside it: weights in {inside[0]} of {n_strata} strata, "
        f"level in {inside[1]}. Within the range the trace covers the two designs "
        f"are not separated by this evidence; they separate past it, where only "
        f"one is bounded"
    )


def tail_rate() -> None:
    """Jobs above the cut per bin against bin activity, measured against the fitted rate."""
    model, _, _, _ = _bulk()
    jobs = _trace_jobs()
    above = jobs["job_size"].to_numpy(dtype=float) >= model.cut
    # the rate is per bin, so bins are the unit: counted, stratified and weighted
    per_bin = (
        jobs.assign(_above=above)
        .groupby("bin")
        .agg(k=("_above", "sum"), n_jobs=("_above", "size"), activity=("activity", "first"))
    )
    act = per_bin["activity"].to_numpy(dtype=float)
    k = per_bin["k"].to_numpy(dtype=float)

    strata, edges = _activity_bins(act)
    n_strata = len(edges) - 1
    centres = np.array([np.median(act[strata == b]) for b in range(n_strata)])
    observed = np.array([k[strata == b].mean() for b in range(n_strata)])
    lo, hi = _day_blocks(per_bin.index, strata, k, n_strata, np.mean)

    grid = np.geomspace(max(act.min(), 1.0), act.max(), N_GRID)
    fitted = model.tail_rate(grid)
    table = pd.DataFrame({"N_k": grid, "fitted_rate": fitted}).set_index("N_k")

    # one series, so the intervals fit and stay on by default
    ci = C.show_ci(True)
    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.plot(grid, fitted, color=ALT_COLOR, linewidth=1.3, linestyle=C.MODEL_DASH, zorder=3)
    # unclipped: the lowest stratum sits exactly on the axis minimum
    seen = _measured(ax, centres, observed, lo, hi, ALT_COLOR, ci, clip=False)

    ax.set_xscale("log")
    # the same limits `weights` sets, so the two panels align
    ax.set_xlim(grid[0], grid[-1])
    # a zero baseline, so a cropped axis cannot magnify the scatter into a trend
    ax.set_ylim(0.0, max(hi.max() if ci else observed.max(), fitted.max()) * 1.18)
    ax.set_xlabel("Batch count in bin, $N_k$")
    ax.set_ylabel("Tail jobs per bin ($\\nu(N_k)$)")
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.6, color=C.AXIS)
    C.frame_legend(
        ax.legend(
            [seen, Line2D([], [], color=ALT_COLOR, linewidth=1.3, linestyle=C.MODEL_DASH)],
            ["Trace, 95% CI" if ci else "Trace", "Tail fitted rate"],
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
    C.save(fig, "fig_jobsize_tail_rate")

    C.write_csv(table, "fig_jobsize_tail_rate")
    n_bins_in = np.array([int((strata == b).sum()) for b in range(n_strata)])
    at = model.tail_rate(centres)
    obs = pd.DataFrame(
        {
            "N_k": centres.round(1),
            "bins": n_bins_in,
            "above_cut": [int(k[strata == b].sum()) for b in range(n_strata)],
            "observed": observed,
            "block_lo": lo,
            "block_hi": hi,
            "fitted": at,
        }
    ).set_index("N_k")
    C.write_csv(obs, "fig_jobsize_tail_rate_measured")
    print(
        f"{len(per_bin):,} occupied bins hold {int(k.sum())} jobs above "
        f"S* = {model.cut:,.0f}, {k.mean():.4f} per bin"
    )
    print(
        f"fitted log lambda = {model.rate_a:+.4f} {model.rate_b:+.5f} log N_k, so the "
        f"rate moves by a factor of {fitted.max() / fitted.min():.3f} over the whole "
        f"range of N_k"
    )
    print(
        f"\nper stratum ({n_strata} equal in bins; activity is integer, so the low end "
        f"cannot be split finer):"
    )
    print(obs.round(4).to_string())
    print(
        f"observed spread is a factor of {observed.max() / observed.min():.2f}; the "
        f"fitted rate is inside the block interval in "
        f"{int(((at >= lo) & (at <= hi)).sum())} of {n_strata} strata"
    )
    per_job = np.array(
        [0.01 * per_bin["n_jobs"].to_numpy()[strata == b].mean() for b in range(n_strata)]
    )
    print(
        f"placing the top 1% per job instead would run {per_job.min():.4f} to "
        f"{per_job.max():.4f} across the same strata, a factor of "
        f"{per_job.max() / per_job.min():.0f}. That is what the per-bin rate avoids, "
        f"and it is not drawn on the panel"
    )


INPUTS = ("state", "counts", "jobsize_two_part", "jobsize_pool", "jobsize_conditional")

VIEWS = {"components": components, "weights": weights, "level": level, "tail-rate": tail_rate}

if __name__ == "__main__":
    C.run(
        VIEWS,
        "components",
        "Job-size regimes: the fitted components, and how activity moves between them.",
        INPUTS,
    )
