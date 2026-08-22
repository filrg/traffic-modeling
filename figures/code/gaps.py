from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from _common import plt


# --- the gap side -----------------------------------------------------


N_PATHS = 60
VOL_TOL = 0.20
# a gap CV needs a few gaps; below this the window is not scored at all
MIN_ARRIVALS = 5

# window lengths in minutes, evenly spaced on the axis
WINDOWS_MIN = (5, 15, 60, 360, 1440, 2880, 10080)
WINDOW_TICKS = ("0.08", "0.25", "1", "6", "24", "48", "168")

# blocks of consecutive gaps for the interval dispersion index
IDI_K = (1, 2, 5, 20, 100, 500, 2000)

RENEWAL_FIT = "Gamma renewal"
RENEWAL_EMP = "Renewal, trace gaps"

_CACHE: dict = {}


def _trace_events(st) -> tuple:
    """Return trace batch times in seconds from the first bin edge, sizes and classes."""
    from seren_countfirst.data import ActivityScheme

    counts = st["counts"]
    times_s = (pd.DatetimeIndex(st["batch_times"]) - counts.index[0]).total_seconds().to_numpy(float)
    scheme = st["scheme"]
    if isinstance(scheme, dict):
        scheme = ActivityScheme(edges=tuple(scheme["edges"]), labels=tuple(scheme["labels"]),
                                busy_threshold=scheme["busy_threshold"],
                                quiet_threshold=scheme["quiet_threshold"], source=scheme["source"])
    return times_s, np.asarray(st["batches_sizes"], int), scheme


def ensemble() -> dict:
    """Trace events, volume-matched winner reconstructions and the two renewal arms.

    Every view here reads an event stream rather than counts, so the ensemble
    carries reconstructed timestamps. Runs are held at the trace's volume,
    without which a gap statistic reads per-path volume instead of shape.
    """
    from seren_countfirst.reconstruct import (
        make_iid_mark_sampler, OffsetTemplates, PatternTemplates, reconstruct_events)

    if _CACHE:
        return _CACHE
    st = C.state()
    counts = st["counts"]
    x, bin_s = np.asarray(counts.values, float), float(counts.bin_seconds)
    horizon = len(x) * bin_s
    times_s, sizes, scheme = _trace_events(st)
    gaps = np.diff(np.sort(times_s))

    # both placement layers on the same count paths and seed, so the pair differs
    # only in placement
    placements = {
        "syn": PatternTemplates.from_trace(times_s, bin_s, x, classes=scheme.classify(x)),
        "syn_iid": OffsetTemplates.from_trace(times_s, bin_s, x, classes=scheme.classify(x)),
    }
    sampler = make_iid_mark_sampler(sizes)
    model, cal = C.winner(st)
    rng = np.random.default_rng(C.SEED + 3)
    paths = [model.simulate(len(x), rng, alpha=cal) for _ in range(N_PATHS)]

    syn, syn_iid = [], []
    for i, p in enumerate(paths):
        if abs(p.mean() / x.mean() - 1.0) > VOL_TOL:
            continue
        try:
            got = [reconstruct_events(p, bin_s, np.random.default_rng(C.SEED + 5000 + i),
                                      sampler, offsets=law, classes=scheme.classify(p))
                   for law in placements.values()]
        except ValueError:
            # an overflowing bin is dropped, never merged, and drops both arms so
            # the two stay paired
            continue
        syn.append(got[0].batch_times_s)
        syn_iid.append(got[1].batch_times_s)

    shape, _, scale = stats.gamma.fit(gaps[gaps > 0], floc=0)
    fitted, empirical = [], []
    for rep in range(10):
        r = np.random.default_rng(C.SEED + 900 + rep)
        for pool, draw in ((fitted, stats.gamma.rvs(shape, scale=scale, size=int(len(gaps) * 1.5),
                                                    random_state=r)),
                           (empirical, r.choice(gaps, size=int(len(gaps) * 1.4), replace=True))):
            t = np.cumsum(draw)
            pool.append(t[t < horizon])

    _CACHE.update(trace=times_s, gaps=gaps, horizon=horizon, syn=syn, syn_iid=syn_iid,
                  fitted=fitted, empirical=empirical, gamma=(shape, scale),
                  counts=x, bin_s=bin_s)
    print(f"ensemble: {len(syn)} of {N_PATHS} winner paths inside +-{VOL_TOL:.0%} of trace volume")
    return _CACHE


def _cv_curve(t: np.ndarray, horizon: float) -> tuple:
    """Median within-window gap CV per window length, with the scored-window share."""
    t = np.sort(np.asarray(t, float))
    cv, cover = {}, {}
    for wmin in WINDOWS_MIN:
        w = wmin * 60.0
        wid = (t // w).astype(int)
        starts = np.flatnonzero(np.r_[True, np.diff(wid) != 0])
        vals = []
        for a, b in zip(starts, np.r_[starts[1:], len(t)]):
            if b - a >= MIN_ARRIVALS:
                d = np.diff(t[a:b])
                if d.mean() > 0:
                    vals.append(d.std(ddof=1) / d.mean())
        cv[wmin] = float(np.median(vals)) if vals else np.nan
        cover[wmin] = len(vals) / int(np.ceil(horizon / w))
    return pd.Series(cv), pd.Series(cover)


def cv_window() -> None:
    """Within-window gap CV against the window it is measured in."""
    e = ensemble()
    rows = {"Trace": _cv_curve(e["trace"], e["horizon"])[0]}
    for label, pool in (("GLI-NB", e["syn"]), (RENEWAL_FIT, e["fitted"]),
                        (RENEWAL_EMP, e["empirical"])):
        rows[label] = pd.DataFrame([_cv_curve(t, e["horizon"])[0] for t in pool]).median()
    table = pd.DataFrame(rows)

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    x = np.arange(len(WINDOWS_MIN), dtype=float)
    entries = [("Trace", C.INK, 1.5, "-")]

    ax.plot(x, table["Trace"].to_numpy(), color=C.INK, linewidth=1.5,
            marker="o", markersize=3.2, zorder=4)
    # the renewal arms go to the csv twin, not the panel; `idi` is the view that
    # has to show them
    color = C.SERIES[0][2]
    ax.plot(x, table["GLI-NB"].to_numpy(), color=color, linewidth=1.3,
            linestyle=C.MODEL_DASH, marker="o", markersize=3.0, zorder=3)
    entries.append(("GLI-NB", color, 1.3, C.MODEL_DASH))
    # a Poisson process reads 1 at every window
    ax.axhline(1.0, color=C.MUTED, linewidth=0.8, linestyle=(0, (1.5, 1.5)), zorder=2)
    entries.append(("Poisson", C.MUTED, 0.8, (0, (1.5, 1.5))))

    ax.set_xticks(x)
    ax.set_xticklabels(WINDOW_TICKS)
    ax.set_xlim(x[0] - 0.25, x[-1] + 0.25)
    ax.set_ylim(0, 7.4)
    ax.set_xlabel("Measurement window (h)")
    ax.set_ylabel("Inter-arrival CV")
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    C.stroke_legend(ax, entries, loc="upper left")

    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_cv_window")

    table.index = list(WINDOW_TICKS)
    table.index.name = "window_h"
    table["trace_window_coverage"] = _cv_curve(e["trace"], e["horizon"])[1].to_numpy()
    C.write_csv(table, "fig_cv_window")
    print("\ntable view (the figure's twin):")
    print(table.round(3).to_string())
    g = e["gaps"]
    print(f"\nglobal gap CV over the whole trace: {g.std(ddof=1) / g.mean():.2f}")


def _idi(gaps: np.ndarray) -> pd.Series:
    """Interval dispersion index over blocks of k consecutive gaps."""
    out = {}
    for k in IDI_K:
        nb = len(gaps) // k
        if nb < 8:
            out[k] = np.nan
            continue
        s = gaps[: nb * k].reshape(nb, k).sum(axis=1)
        out[k] = float(k * s.var(ddof=1) / s.mean() ** 2)
    return pd.Series(out)


def idi() -> None:
    """Interval dispersion against block length: flat is the iid-gap signature."""
    e = ensemble()
    rows = {"Trace": _idi(e["gaps"])}
    for label, pool in (("GLI-NB", e["syn"]), (RENEWAL_FIT, e["fitted"]),
                        (RENEWAL_EMP, e["empirical"])):
        rows[label] = pd.DataFrame([_idi(np.diff(np.sort(t))) for t in pool]).median()
    table = pd.DataFrame(rows)

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    x = np.arange(len(IDI_K), dtype=float)
    entries = [("Trace", C.INK, 1.5, "-")]

    ax.plot(x, table["Trace"].to_numpy(), color=C.INK, linewidth=1.5,
            marker="o", markersize=3.2, zorder=4)
    for (_, _, color), label in zip(C.SERIES, ("GLI-NB", RENEWAL_FIT, RENEWAL_EMP)):
        ax.plot(x, table[label].to_numpy(), color=color, linewidth=1.3,
                linestyle=C.MODEL_DASH, marker="o", markersize=3.0, zorder=3)
        entries.append((label, color, 1.3, C.MODEL_DASH))

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in IDI_K])
    ax.set_xlim(x[0] - 0.25, x[-1] + 0.25)
    ax.set_yscale("log")
    ax.set_ylim(2.5, 1.2e4)
    ax.set_xlabel("Consecutive inter-arrivals per block")
    ax.set_ylabel("Interval dispersion (IDI)")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    C.stroke_legend(ax, entries, loc="upper left")

    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_idi")

    table.index = list(IDI_K)
    table.index.name = "block_k"
    C.write_csv(table, "fig_idi")
    print("\ntable view (the figure's twin):")
    print(table.round(1).to_string())


def idc_ceiling() -> None:
    """Count dispersion against the ceiling a finite-variance renewal is held under."""
    from seren_countfirst.data import SCALES
    from seren_countfirst.metrics import idc_curve

    e = ensemble()
    x, bin_s = e["counts"], e["bin_s"]
    n = len(x)

    def counts_of(t):
        idx = np.clip((np.asarray(t) // bin_s).astype(int), 0, n - 1)
        return np.bincount(idx, minlength=n).astype(float)

    rows = {"Trace": idc_curve(x)}
    for label, pool in ((RENEWAL_FIT, e["fitted"]), (RENEWAL_EMP, e["empirical"])):
        rows[label] = pd.DataFrame([idc_curve(counts_of(t)) for t in pool]).median()
    table = pd.DataFrame(rows)
    g = e["gaps"]
    ceiling = float((g.std(ddof=1) / g.mean()) ** 2)

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    xs = np.arange(len(SCALES), dtype=float)
    entries = [("Trace", C.INK, 1.5, "-")]

    ax.plot(xs, table["Trace"].to_numpy(), color=C.INK, linewidth=1.5,
            marker="o", markersize=3.2, zorder=4)
    # slots 2 and 3; the winner's blue is reserved
    for (_, _, color), label in zip(C.SERIES[1:], (RENEWAL_FIT, RENEWAL_EMP)):
        ax.plot(xs, table[label].to_numpy(), color=color, linewidth=1.3,
                linestyle=C.MODEL_DASH, marker="o", markersize=3.0, zorder=3)
        entries.append((label, color, 1.3, C.MODEL_DASH))
    # every renewal process with finite gap variance converges to CV^2 from below
    ax.axhline(ceiling, color=C.MUTED, linewidth=0.9, linestyle=(0, (1.5, 1.5)), zorder=2)
    entries.append(("Renewal ceiling", C.MUTED, 0.9, (0, (1.5, 1.5))))

    ax.set_xticks(xs)
    ax.set_xticklabels(["0.25", "1", "6", "24", "48", "96", "168"])
    ax.set_xlim(xs[0] - 0.25, xs[-1] + 0.25)
    ax.set_yscale("log")
    ax.set_ylim(2, 9000)
    ax.set_xlabel("Aggregation window (h)")
    ax.set_ylabel("Index of dispersion (IDC)")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    C.stroke_legend(ax, entries, loc="upper left")

    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_idc_ceiling")

    table.index.name = "scale"
    table["renewal_ceiling"] = ceiling
    C.write_csv(table, "fig_idc_ceiling")
    print("\ntable view (the figure's twin):")
    print(table.round(1).to_string())
    print(f"\nrenewal ceiling CV^2 = {ceiling:.1f}")


def iat_ccdf() -> None:
    """Inter-arrival survival, trace against the generator's reconstructed gaps."""
    e = ensemble()
    g = e["gaps"]
    grid = np.unique(np.round(np.geomspace(1.0, g.max(), 220)))
    shape, scale = e["gamma"]

    def surv(sample):
        s = np.sort(np.asarray(sample, float))
        return 1.0 - np.searchsorted(s, grid, side="right") / len(s)

    laws = {
        "Exponential": 1.0 - stats.expon.cdf(grid, scale=g.mean()),
        "Gamma": 1.0 - stats.gamma.cdf(grid, shape, scale=scale),
        "Lognormal": None,
    }
    ln = stats.lognorm.fit(g[g > 0], floc=0)
    laws["Lognormal"] = 1.0 - stats.lognorm.cdf(grid, *ln)
    model = np.median([surv(np.diff(np.sort(t))) for t in e["syn"]], axis=0)

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    entries = [("Trace", C.INK, 1.5, "-")]
    ax.plot(grid, surv(g), color=C.INK, linewidth=1.5, zorder=5)
    # the fitted laws are computed and written to the csv twin, but not drawn:
    # the KS table below carries the comparison
    color = C.SERIES[0][2]
    ax.plot(grid, model, color=color, linewidth=1.2, linestyle=C.MODEL_DASH, zorder=4)
    entries.append(("GLI-NB", color, 1.2, C.MODEL_DASH))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1.2e6)
    ax.set_ylim(1e-5, 1.4)
    ax.set_xlabel("Inter-arrival time (s)")
    ax.set_ylabel("P(gap > t)")
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    C.stroke_legend(ax, entries, loc="lower left")

    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_iat_ccdf")

    out = pd.DataFrame({"t_s": grid, "trace": surv(g), "GLI_NB": model,
                        "gamma": laws["Gamma"], "lognormal": laws["Lognormal"],
                        "exponential": laws["Exponential"]})
    C.write_csv(out, "fig_iat_ccdf", index=False)

    def ks(a, b):
        a, b = np.sort(a), np.sort(b)
        v = np.concatenate([a, b])
        return float(np.max(np.abs(np.searchsorted(a, v, "right") / len(a)
                                   - np.searchsorted(b, v, "right") / len(b))))

    print("\nKS against the trace's gaps:")
    for name, dist, args in (("Exponential", stats.expon, (0, g.mean())),
                             ("Gamma", stats.gamma, (shape, 0, scale)),
                             ("Lognormal", stats.lognorm, ln)):
        print(f"  {name:12s} {stats.kstest(g[g > 0], dist.cdf, args=args).statistic:.4f}")
    print(f"  {'GLI-NB':12s} {np.median([ks(np.diff(np.sort(t)), g) for t in e['syn']]):.4f}")


def _placement_curves(e: dict) -> tuple:
    """Grid and survival curves for the trace, both placements and the best fitted law."""
    g = e["gaps"]
    grid = np.unique(np.round(np.geomspace(1.0, g.max(), 220)))

    def surv(sample):
        s = np.sort(np.asarray(sample, float))
        return 1.0 - np.searchsorted(s, grid, side="right") / len(s)

    ln = stats.lognorm.fit(g[g > 0], floc=0)
    tab = pd.DataFrame({
        "Trace": surv(g),
        "GLI-NB": np.median([surv(np.diff(np.sort(t))) for t in e["syn"]], axis=0),
        "GLI-NB, iid offsets": np.median([surv(np.diff(np.sort(t))) for t in e["syn_iid"]], axis=0),
        "Lognormal": 1.0 - stats.lognorm.cdf(grid, *ln),
    }, index=grid)
    tab.index.name = "t_s"
    return grid, tab


def _placement_ks(e: dict) -> None:
    """Print the gap-marginal KS of each arm against the trace."""
    g = e["gaps"]
    for label, pool in (("GLI-NB", e["syn"]), ("GLI-NB, iid offsets", e["syn_iid"])):
        d = np.median([stats.ks_2samp(np.diff(np.sort(t)), g).statistic for t in pool])
        print(f"  {label:22s} {d:.4f}")
    ln = stats.lognorm.fit(g[g > 0], floc=0)
    print(f"  {'Lognormal':22s} {stats.kstest(g[g > 0], stats.lognorm.cdf, args=ln).statistic:.4f}")


def _placement_panel(e: dict, log_y: bool, name: str, ylabel: str) -> None:
    """Draw the placement comparison as survival or as its complement."""
    grid, tab = _placement_curves(e)
    labels = ("GLI-NB", "GLI-NB, iid offsets", "Lognormal")

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    entries = [("Trace", C.INK, 1.5, "-")]
    conv = (lambda y: y) if log_y else (lambda y: 1.0 - y)
    ax.plot(grid, conv(tab["Trace"].to_numpy()), color=C.INK, linewidth=1.5, zorder=5)
    for (_, _, color), label in zip(C.SERIES, labels):
        ax.plot(grid, conv(tab[label].to_numpy()), color=color, linewidth=1.2,
                linestyle=C.MODEL_DASH, zorder=4)
        entries.append((label, color, 1.2, C.MODEL_DASH))

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
        ax.set_xlim(1, 1.2e6)
        ax.set_ylim(1e-5, 1.4)
        loc = "lower left"
    else:
        # the body, where the placements differ
        ax.set_xlim(1, 1e5)
        ax.set_ylim(0.0, 1.02)
        loc = "lower right"
    ax.set_xlabel("Inter-arrival time (s)")
    ax.set_ylabel(ylabel)
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    C.stroke_legend(ax, entries, loc=loc)

    fig.tight_layout(pad=0.3)
    C.save(fig, name)
    C.write_csv(tab if log_y else 1.0 - tab, name)
    print("\nKS against the trace's gaps:")
    _placement_ks(e)


def ccdf_placement() -> None:
    """Inter-arrival survival, the shipped placement against the one it replaced."""
    _placement_panel(ensemble(), True, "fig_iat_ccdf_placement", "P(gap > t)")


def gap_cdf() -> None:
    """Inter-arrival CDF, where the body of the distribution actually is."""
    _placement_panel(ensemble(), False, "fig_gap_cdf", "P(gap <= t)")


# --- placement inside a bin -------------------------------------------


N_REPS = 6
# occupancy buckets, doubling, ending at the trace's busiest bin
BUCKETS = ((2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255), (256, 530))
TICKS = ("2", "4", "8", "16", "32", "64", "128", "256+")


def _events(st) -> tuple:
    """Trace batch times in seconds from the first bin edge, with sizes and classes."""
    from seren_countfirst.data import ActivityScheme

    counts = st["counts"]
    t = (pd.DatetimeIndex(st["batch_times"]) - counts.index[0]).total_seconds().to_numpy(float)
    scheme = st["scheme"]
    if isinstance(scheme, dict):
        scheme = ActivityScheme(edges=tuple(scheme["edges"]), labels=tuple(scheme["labels"]),
                                busy_threshold=scheme["busy_threshold"],
                                quiet_threshold=scheme["quiet_threshold"], source=scheme["source"])
    return np.sort(t), np.asarray(st["batches_sizes"], int), scheme


def _bucket(occ: np.ndarray, value: np.ndarray) -> pd.Series:
    """Median of `value` over the bins in each occupancy bucket."""
    out = {}
    for lo, hi in BUCKETS:
        m = (occ >= lo) & (occ <= hi) & np.isfinite(value)
        out[f"{lo}-{hi}"] = float(np.median(value[m])) if m.any() else np.nan
    return pd.Series(out)


def _occupancy(t: np.ndarray, bin_s: float) -> tuple:
    """Per-bin occupancy and the median within-bin gap over the mean spacing."""
    t = np.sort(np.asarray(t, float))
    b = (t // bin_s).astype(int)
    starts = np.flatnonzero(np.r_[True, np.diff(b) != 0])
    stops = np.r_[starts[1:], len(t)]
    occ = stops - starts
    ratio = np.full(len(occ), np.nan)
    for i, (a, z) in enumerate(zip(starts, stops)):
        if z - a >= 2:
            ratio[i] = np.median(np.diff(t[a:z])) / (bin_s / (z - a + 1))
    return occ, ratio


def _clustering(t: np.ndarray, bin_s: float) -> pd.Series:
    """Median within-bin gap over the mean spacing, per occupancy bucket."""
    return _bucket(*_occupancy(t, bin_s))


def _independent(occ: np.ndarray) -> pd.Series:
    """Return `(1 - 2**(-1/n))(n + 1)`, median over mean for gaps that are Beta(1, n).

    The reference under independent placement is a curve in n, never a constant.
    """
    n = np.asarray(occ, dtype=float)
    return _bucket(occ, np.where(n >= 2, (1.0 - 2.0 ** (-1.0 / np.maximum(n, 1))) * (n + 1.0),
                                 np.nan))


def clustering() -> None:
    """Within-bin clustering against bin occupancy, trace against two placements."""
    from seren_countfirst.reconstruct import (
        make_iid_mark_sampler, OffsetTemplates, PatternTemplates, reconstruct_events)

    st = C.state()
    counts = st["counts"]
    x, bin_s = np.asarray(counts.values, float), float(counts.bin_seconds)
    times, sizes, scheme = _events(st)
    classes = scheme.classify(x)
    sampler = make_iid_mark_sampler(sizes)

    # placed on the trace's own counts, so every arm sees the same bins
    laws = {
        "Pattern resampling": PatternTemplates.from_trace(times, bin_s, x, classes=classes),
        "Independent offsets": OffsetTemplates.from_trace(times, bin_s, x, classes=classes),
    }
    occ, _ = _occupancy(times, bin_s)
    rows = {"Trace": _clustering(times, bin_s), "Uniform, exact": _independent(occ)}
    for label, law in laws.items():
        reps = [_clustering(reconstruct_events(x, bin_s, np.random.default_rng(C.SEED + 100 + r),
                                               sampler, offsets=law,
                                               classes=classes).batch_times_s, bin_s)
                for r in range(N_REPS)]
        rows[label] = pd.DataFrame(reps).median()
    table = pd.DataFrame(rows)

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    xs = np.arange(len(BUCKETS), dtype=float)
    entries = [("Trace", C.INK, 1.5, "-")]

    # the closed form for independent placement; the panel is the distance below it
    ax.plot(xs, table["Uniform, exact"].to_numpy(), color=C.MUTED, linewidth=0.9,
            linestyle=(0, (1.5, 1.5)), zorder=2)
    ax.plot(xs, table["Trace"].to_numpy(), color=C.INK, linewidth=1.5,
            marker="o", markersize=3.2, zorder=5)
    for (_, _, color), label in zip(C.SERIES, laws):
        ax.plot(xs, table[label].to_numpy(), color=color, linewidth=1.3,
                linestyle=C.MODEL_DASH, marker="o", markersize=3.0, zorder=4)
        entries.append((label, color, 1.3, C.MODEL_DASH))
    entries.append(("Uniform, exact", C.MUTED, 0.9, (0, (1.5, 1.5))))

    ax.set_xticks(xs)
    ax.set_xticklabels(TICKS)
    ax.set_xlim(xs[0] - 0.25, xs[-1] + 0.25)
    ax.set_ylim(0.25, 1.05)
    ax.set_xlabel("Batches in the bin")
    ax.set_ylabel("Median gap / mean spacing")
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    C.stroke_legend(ax, entries, loc="upper right")

    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_placement_clustering")

    table.index.name = "occupancy"
    C.write_csv(table, "fig_placement_clustering")
    print("\ntable view (the figure's twin):")
    print(table.round(3).to_string())


INPUTS = ("state", "counts", "models", "batch_times")

VIEWS = {
    "cv-window": cv_window,
    "idi": idi,
    "idc-ceiling": idc_ceiling,
    "iat-ccdf": iat_ccdf,
    "ccdf-placement": ccdf_placement,
    "gap-cdf": gap_cdf,
    "clustering": clustering,
}

if __name__ == "__main__":
    C.run(VIEWS, "cv-window",
          "The gap side: inter-arrival statistics, and where in a bin the batches land.",
          INPUTS)
