from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from _common import plt


# --- occupancy --------------------------------------------------------


N_CANDIDATES = C.DISPLAY_CANDIDATES
# `excursion` and `generator.load-runs` select from this same pool, so all three
# show one run of the winner


def _collect() -> tuple:
    st = C.state()
    counts = st["counts"].values
    model, cal = C.winner(st)
    from seren_countfirst.metrics import base_marginals

    tm = base_marginals(counts)
    keys = ("mean", "IDC_15m", "zero_frac", "acf_1h_lag1")
    # one rng per series in `series_paths`, so drawing the winner alone
    # reproduces its pool there exactly and the run indices agree
    runs, _ = C.display_pool(model, cal, len(counts), N_CANDIDATES)
    pick = C.representative(runs, float(counts.mean()))
    # reported, never selected on
    dist = np.array([sum(abs(np.log(max(base_marginals(r)[k], 1e-9) / tm[k]))
                         for k in keys) for r in runs])
    order = np.argsort(dist)
    report = pd.DataFrame({
        "run": ["selected", "closest on stats", "median", "furthest"],
        "stat_distance": [dist[pick], dist[order[0]],
                          dist[order[len(order) // 2]], dist[order[-1]]],
    })
    return counts, runs[pick], float(counts.mean()), dist[pick], report, pick


def series() -> None:
    """The full real series beside a synthetic run of equal length."""
    trace_win, syn_win, target, picked_d, report, pick = _collect()
    from seren_countfirst.metrics import base_marginals

    C.house_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 3.8), sharex=True, sharey=True)
    t = np.arange(len(trace_win)) / C.BINS_PER_DAY
    span = len(trace_win) / C.BINS_PER_DAY
    top = max(trace_win.max(), syn_win.max()) * C.BINS_PER_H * 1.06

    for ax, data, color, label, tag in (
        (ax1, trace_win, C.INK, "Trace", "(a)"),
        (ax2, syn_win, C.BLUE, "GLI-NB", "(b)"),
    ):
        ax.plot(t, data * C.BINS_PER_H, color=color, linewidth=0.45, zorder=3)
        ax.set_ylim(0, top)
        ax.set_xlim(0, span)
        C.style_axes(ax)
        ax.text(-0.075, 1.04, tag, transform=ax.transAxes, color=C.INK,
                fontsize=10, va="bottom", ha="left")
        C.frame_legend(ax.legend([plt.Line2D([], [], color=color, linewidth=1.4)], [label],
                                 loc="upper right", frameon=True, framealpha=1.0,
                                 fontsize=8, handlelength=1.8, borderpad=0.45))

    ax2.set_xticks(np.arange(0, span + 1, 20))
    ax2.set_xlabel("Time (days)")
    fig.supylabel("Arrival rate (batches/h)", color=C.INK, fontsize=9, x=0.005)
    fig.tight_layout(pad=0.4, h_pad=1.0)
    C.save(fig, "fig_series")

    C.write_csv(pd.DataFrame({"day": np.arange(len(trace_win)) / C.BINS_PER_DAY,
                              "trace_batches_per_h": trace_win * C.BINS_PER_H,
                              "model_batches_per_h": syn_win * C.BINS_PER_H}),
                "fig_series", index=False)
    print(f"\nselection: run {pick} of {N_CANDIDATES}, stat distance {picked_d:.4f}")
    print(report.round(4).to_string(index=False))
    print(f"\ntrace overall mean {target * C.BINS_PER_H:.2f} batches/h")
    for name, w in (("trace", trace_win), ("model run", syn_win)):
        bm = base_marginals(w)
        print(f"  {name:10} mean {bm['mean'] * C.BINS_PER_H:6.2f}  IDC {bm['IDC_15m']:7.2f}  "
              f"zero {bm['zero_frac']:.3f}  acf_1h {bm['acf_1h_lag1']:.3f}  "
              f"peak {bm['max_bin'] * C.BINS_PER_H:7.0f}")


def _stack(rows: list, name: str) -> None:
    """Draw one series per row on a shared scale, and write the csv twin."""
    from seren_countfirst.metrics import base_marginals

    C.house_style()
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 5.4), sharex=True, sharey=True)
    span = len(rows[0][1]) / C.BINS_PER_DAY
    t = np.arange(len(rows[0][1])) / C.BINS_PER_DAY
    # one shared scale across all five rows
    top = max(r.max() for _, r, _ in rows) * C.BINS_PER_H * 1.06

    for ax, (label, data, color) in zip(axes, rows):
        ax.plot(t, data * C.BINS_PER_H, color=color, linewidth=0.4, zorder=3)
        ax.set_ylim(0, top)
        ax.set_xlim(0, span)
        ax.set_yticks([0, 1000, 2000])
        ax.annotate(label, (0.008, 0.9), xycoords="axes fraction", ha="left", va="top",
                    fontsize=8, color=color, zorder=4)
        C.style_axes(ax, grid_axis="y", labelsize=7)

    axes[-1].set_xticks(np.arange(0, span + 1, 20))
    axes[-1].set_xlabel("Time (days)")
    fig.supylabel("Arrival rate (batches/h)", color=C.INK, fontsize=9, x=0.005)
    fig.tight_layout(pad=0.4, h_pad=0.45)
    C.save(fig, name)

    C.write_csv(pd.DataFrame({"day": t, **{label: data * C.BINS_PER_H
                                           for label, data, _ in rows}}), name, index=False)
    print("\ntable view (the figure's twin):")
    for label, data, _ in rows:
        bm = base_marginals(data)
        print(f"  {label:11s} mean {bm['mean'] * C.BINS_PER_H:6.2f}  IDC {bm['IDC_15m']:7.2f}  "
              f"zero {bm['zero_frac']:.3f}  acf_1h {bm['acf_1h_lag1']:.3f}  "
              f"peak {bm['max_bin'] * C.BINS_PER_H:7.0f}")


def _stacked_rows(best_baselines: bool) -> list:
    """Return (label, run, colour) per series, from the shared display ensemble."""
    st = C.state()
    counts = st["counts"].values
    tm = float(counts.mean())
    pools = C.series_paths(st, N_CANDIDATES, seed_offset=C.DISPLAY_OFFSET)
    winner = C.SERIES[0][1]

    rows = [("Trace", counts, C.INK)]
    for _, label, color in C.SERIES:
        runs = pools[label]
        if best_baselines and label != winner:
            pick, how = C.closest_to_trace(runs, counts, tm), "best at volume"
        else:
            pick, how = C.representative(runs, tm), "typical"
        rows.append((label, runs[pick], color))
        print(f"{label:11s} run {pick} of {N_CANDIDATES} ({how})")
    return rows


def series_all() -> None:
    """The real series and one typical run of every model, on one shared scale."""
    _stack(_stacked_rows(best_baselines=False), "fig_series_all")


def series_all_best() -> None:
    """The same stack with each baseline at its most trace-like run instead.

    The winner keeps its typical run, so the asymmetry runs against it.
    """
    _stack(_stacked_rows(best_baselines=True), "fig_series_all_best")


# --- burstiness across scales -----------------------------------------


IDC_N_PATHS = 200
# the seven scales are a chosen set, not a continuum, so they are spaced evenly.
# Hours throughout: days would put three decimals on the short end.
TICKS = {"15m": "0.25", "1h": "1", "6h": "6", "1d": "24",
         "2d": "48", "4d": "96", "7d": "168"}


def idc_scale() -> None:
    """Index of dispersion against aggregation scale, trace against the field."""
    from seren_countfirst.data import SCALES
    from seren_countfirst.metrics import idc_curve

    st = C.state()
    trace = idc_curve(st["counts"].values)
    paths = C.series_paths(st, IDC_N_PATHS)
    curves = {label: pd.DataFrame([idc_curve(p) for p in runs]).median()
              for label, runs in paths.items()}

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    x = np.arange(len(SCALES), dtype=float)

    ax.plot(x, trace.to_numpy(), color=C.INK, linewidth=1.5, linestyle="-",
            marker="o", markersize=3.2, label="Trace", zorder=4)
    for _, label, color in C.SERIES:
        ax.plot(x, curves[label].to_numpy(), color=color, linewidth=1.3,
                linestyle=C.MODEL_DASH, marker="o", markersize=3.0, label=label, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([TICKS[s] for s in SCALES])
    ax.set_xlim(x[0] - 0.25, x[-1] + 0.25)
    # log y: the field spans three decades, and a linear axis collapses it
    ax.set_yscale("log")
    ax.set_ylim(5, 7000)
    ax.set_xlabel("Aggregation window (h)")
    ax.set_ylabel("Index of dispersion (IDC)")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    C.model_legend(ax, loc="upper left")

    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_idc_scale")

    tbl = pd.DataFrame(curves)
    tbl.insert(0, "trace", trace)
    C.write_csv(tbl, "fig_idc_scale")
    print("\ntable view (the figure's twin):")
    print(tbl.round(1).to_string())


# --- memory -----------------------------------------------------------


ACF_N_PATHS = 60
# chosen lags, evenly spaced; one unit for the axis (hours)
LAGS = (1, 2, 4, 8, 24, 48, 96, 192, 384)


def acf() -> None:
    """Count autocorrelation against lag, trace against the models."""
    from seren_countfirst.metrics import acf as acf_of

    st = C.state()
    rows = {"Trace": acf_of(st["counts"].values, LAGS)}
    for label, runs in C.series_paths(st, ACF_N_PATHS).items():
        rows[label] = pd.DataFrame([acf_of(p, LAGS) for p in runs]).median()
    table = pd.DataFrame(rows)

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    x = np.arange(len(LAGS), dtype=float)

    ax.axhline(0.0, color=C.AXIS, linewidth=0.7, zorder=1)
    ax.plot(x, table["Trace"].to_numpy(), color=C.INK, linewidth=1.5,
            marker="o", markersize=3.2, label="Trace", zorder=4)
    for _, label, color in C.SERIES:
        ax.plot(x, table[label].to_numpy(), color=color, linewidth=1.3,
                linestyle=C.MODEL_DASH, marker="o", markersize=3.0, label=label, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lag * C.BIN_H:g}" for lag in LAGS])
    ax.set_xlim(x[0] - 0.25, x[-1] + 0.25)
    ax.set_ylim(-0.06, 0.78)
    ax.set_xlabel("Lag (h)")
    ax.set_ylabel("Count autocorrelation")
    # nine lag labels, so the tick text comes down with the panel
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    C.model_legend(ax, loc="upper right")

    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_acf")

    table.index = [f"{lag * C.BIN_H:g}" for lag in LAGS]
    table.index.name = "lag_h"
    C.write_csv(table, "fig_acf")
    print("\ntable view (the figure's twin):")
    print(table.round(3).to_string())


INPUTS = ("state", "counts", "models")

VIEWS = {
    "series": series,
    "series-all": series_all,
    "series-all-best": series_all_best,
    "idc-scale": idc_scale,
    "acf": acf,
}

if __name__ == "__main__":
    C.run(VIEWS, "series",
          "Arrival fidelity: the series, its dispersion across scales, and its memory.",
          INPUTS)
