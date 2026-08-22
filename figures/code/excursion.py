from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
from _common import plt

N_PATHS = C.DISPLAY_CANDIDATES
DAYS_PER_WEEK = 7.0

# rounded, so `REPORT_Q` indexes the grid without comparing floats for equality
Q = np.round(np.linspace(0.01, 0.99, 99), 2)
REPORT_Q = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
# quiet end, typical day, busy day; the bar spans TYP..BUSY and LOW is the whisker
Q_LOW, Q_TYP, Q_BUSY = 0.10, 0.50, 0.90

# disjoint bands over the daily range, split at the frozen busy threshold
DAY_EDGES = [0, 25, 100, 200, np.inf]
DAY_LABELS = ["<25", "25-100", "100-200", "200+"]
BIN_EDGES = [25, 50, 100, 200, np.inf]
BIN_LABELS = ["25-50", "50-100", "100-200", "200+"]
# the band whose bars carry value labels
LABELLED_BAND = "200+"

_CACHE: dict = {}


def _ensemble() -> tuple:
    """Return (trace counts, {label: N_PATHS paths}), simulated once and reused.

    Drawn at `DISPLAY_OFFSET`, so the winner's run is the one `arrivals.series` shows.
    """
    if not _CACHE:
        st = C.state()
        _CACHE["counts"] = st["counts"].values
        _CACHE["paths"] = C.series_paths(st, N_PATHS, seed_offset=C.DISPLAY_OFFSET)
    return _CACHE["counts"], _CACHE["paths"]


def _band_counts(values: np.ndarray, edges) -> np.ndarray:
    """Return how many of `values` fall in each disjoint band."""
    return np.array([((values >= lo) & (values < hi)).sum()
                     for lo, hi in zip(edges[:-1], edges[1:])], dtype=float)


def _days_per_week(counts: np.ndarray) -> np.ndarray:
    peaks = C.daily_peaks(counts)
    return _band_counts(peaks, DAY_EDGES) / len(peaks) * DAYS_PER_WEEK


def _bins_per_week(counts: np.ndarray) -> np.ndarray:
    x = np.asarray(counts, dtype=float)
    weeks = len(x) * C.BIN_H / 24.0 / 7.0
    return _band_counts(x, BIN_EDGES) / weeks


def _bar_panel(df: pd.DataFrame, labels, xlabel: str, ylabel: str, name: str,
               figsize: tuple, ylim: tuple, yticks=None, leg_fontsize: float = 8,
               leg_handlelength: float = 1.6, annot_fontsize: float = 6.5) -> None:
    """Draw one grouped-bar panel; the two band views differ only in their data."""
    C.house_style()
    fig, ax = plt.subplots(figsize=figsize)
    entries = [("Trace", C.INK)] + [(label, color) for _, label, color in C.SERIES]
    x = np.arange(len(labels), dtype=float)
    # gap between adjacent bars, in points
    width = 0.84 / len(entries)
    for i, (label, color) in enumerate(entries):
        offset = (i - (len(entries) - 1) / 2) * width
        bars = ax.bar(x + offset, df[label].to_numpy(), width * 0.88, color=color,
                      label=label, zorder=3)
        if LABELLED_BAND in df.index:
            j = list(df.index).index(LABELLED_BAND)
            v = float(df[label].iloc[j])
            ax.annotate(f"{v:.2f}" if v else "0",
                        (bars[j].get_x() + bars[j].get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 2.5), ha="center",
                        fontsize=annot_fontsize, color=C.INK_2, rotation=90, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    C.style_axes(ax, grid_axis="y")
    C.frame_legend(ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in entries],
                             loc="upper right", frameon=True, framealpha=1.0,
                             fontsize=leg_fontsize, handlelength=leg_handlelength,
                             labelspacing=0.3 if leg_fontsize < 8 else 0.35,
                             borderpad=0.45 if leg_fontsize < 8 else 0.5))
    fig.tight_layout(pad=0.3)
    C.save(fig, name)


def daily_peak() -> None:
    """Quantile curve of the busiest quarter-hour of each day, across days."""
    counts, paths = _ensemble()
    curves = {"Trace": np.quantile(C.daily_peaks(counts), Q)}
    for label, runs in paths.items():
        per_path = np.array([np.quantile(C.daily_peaks(p), Q) for p in runs])
        curves[label] = np.median(per_path, axis=0)
    df = pd.DataFrame(curves, index=Q)

    C.house_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.plot(Q, df["Trace"].to_numpy(), color=C.INK, linewidth=1.8, label="Trace", zorder=4)
    for _, label, color in C.SERIES:
        ax.plot(Q, df[label].to_numpy(), color=color, linewidth=1.6,
                linestyle=C.MODEL_DASH, label=label, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 300)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of days, quietest to busiest")
    ax.set_ylabel("Batches in the day's busiest 15 minutes")
    C.style_axes(ax)
    C.model_legend(ax, loc="upper left", trace_lw=1.8, model_lw=1.6,
                   fontsize=8, handlelength=3.2, labelspacing=0.35, borderpad=0.5)
    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_daily_peak")

    report = df.loc[REPORT_Q]
    report.index = [f"{q:.0%} of days" for q in REPORT_Q]
    report.index.name = "quantile"
    C.write_csv(report, "fig_daily_peak")
    print("\ntable view (the figure's twin) - batches in the busiest 15 min:")
    print(report.round(1).to_string())


def daily_peak_range() -> None:
    """Range bars from the median day to the 90th-percentile day."""
    counts, paths = _ensemble()
    qs = [Q_LOW, Q_TYP, Q_BUSY]
    rows = {"Trace": np.quantile(C.daily_peaks(counts), qs)}
    for label, runs in paths.items():
        # median across paths of the per-path quantile
        rows[label] = np.median(np.array([np.quantile(C.daily_peaks(p), qs) for p in runs]),
                                axis=0)
    df = pd.DataFrame(rows, index=["quiet_day_p10", "typical_day_p50", "busy_day_p90"]).T

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    rows_ord = [("Trace", C.INK)] + [(label, color) for _, label, color in C.SERIES]
    ys = list(range(len(rows_ord) - 1, -1, -1))  # trace on top, reading order

    # the trace's two values repeated on every row
    for q in ("typical_day_p50", "busy_day_p90"):
        ax.axvline(float(df.loc["Trace", q]), color=C.AXIS, linewidth=0.7,
                   linestyle=(0, (1, 2.2)), zorder=1)

    for (label, color), y in zip(rows_ord, ys):
        lo, typ, busy = (float(df.loc[label, c]) for c in df.columns)
        ax.plot([lo, typ], [y, y], color=color, linewidth=1.0, alpha=0.45,
                solid_capstyle="butt", zorder=2)
        ax.plot([lo], [y], marker="|", markersize=4.5, color=color, alpha=0.55, zorder=2)
        ax.plot([typ, busy], [y, y], color=color, linewidth=2.0,
                solid_capstyle="round", zorder=3)
        ax.plot([typ, busy], [y, y], marker="o", markersize=5.5, color=color,
                markeredgecolor="white", markeredgewidth=0.8, linestyle="none", zorder=4)
        for x in (typ, busy):
            ax.annotate(f"{x:.0f}", (x, y), textcoords="offset points", xytext=(0, -8.5),
                        ha="center", va="center", fontsize=6.5, color=C.INK_2, zorder=4)
        # names sit above their bars rather than on ticks, to save panel width
        ax.annotate(label, (2.0, y + 0.34), ha="left", va="center",
                    fontsize=7, color=color, zorder=4)

    # labelled once, over the guide lines
    ax.annotate("typical day", (float(df.loc["Trace", "typical_day_p50"]), ys[0] + 0.88),
                ha="left", va="center", fontsize=6.5, color=C.MUTED)
    ax.annotate("busy day", (float(df.loc["Trace", "busy_day_p90"]), ys[0] + 0.88),
                ha="right", va="center", fontsize=6.5, color=C.MUTED)

    ax.set_xlim(0, 200)
    ax.set_ylim(-0.55, len(rows_ord) + 0.15)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([])
    ax.set_xlabel("Batches in the day's busiest 15 minutes")
    C.style_axes(ax, grid_axis="x", hide_left=True)
    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_daily_peak_range")

    df.index.name = "series"
    C.write_csv(df, "fig_daily_peak_range")
    print("\ntable view (the figure's twin) - batches in the day's busiest 15 min:")
    print(df.round(1).to_string())


def daily_peak_bands() -> None:
    """Days per week whose busiest bin falls in each amplitude band."""
    counts, paths = _ensemble()
    rows = {"Trace": _days_per_week(counts)}
    for label, runs in paths.items():
        rows[label] = np.median(np.array([_days_per_week(p) for p in runs]), axis=0)
    df = pd.DataFrame(rows, index=DAY_LABELS)

    _bar_panel(df, DAY_LABELS, "Batches in the day's busiest 15 minutes", "Days per week",
               "fig_daily_peak_bands", figsize=(3.5, 2.9), ylim=(0, DAYS_PER_WEEK),
               yticks=[0, 1, 2, 3, 4, 5, 6, 7], leg_fontsize=7, leg_handlelength=1.4,
               annot_fontsize=6)

    df.index.name = "peak_batches_in_day"
    C.write_csv(df, "fig_daily_peak_bands")
    print("\ntable view (the figure's twin) - days per week:")
    print(df.round(2).to_string())
    print("\nbands are disjoint; the trace sums to 7 exactly and a model column,"
          "\nbeing a per-band median across paths, lands near it:")
    print(df.sum().round(2).to_string())


def daily_peak_strips() -> None:
    """One strip per series: the busiest quarter-hour of every day, day by day."""
    counts, paths = _ensemble()
    strips = {"Trace": C.daily_peaks(counts)}
    for label, runs in paths.items():
        pick = C.representative(runs, float(counts.mean()))
        strips[label] = C.daily_peaks(runs[pick])
        print(f"{label:11s} strip: run {pick} of {N_PATHS}")
    df = pd.DataFrame(strips, index=pd.RangeIndex(1, len(strips["Trace"]) + 1, name="day"))

    C.house_style()
    rows = [("Trace", C.INK)] + [(label, color) for _, label, color in C.SERIES]
    day = df.index.to_numpy(dtype=float)
    ymax = float(np.ceil(df.to_numpy().max() / 50.0) * 50.0)
    print(f"shared y scale 0-{ymax:.0f} (tallest day {df.to_numpy().max():.0f})")

    fig, axes = plt.subplots(len(rows), 1, figsize=(3.5, 3.2), sharex=True, sharey=True)
    for ax, (label, color) in zip(axes, rows):
        ax.fill_between(day, df[label].to_numpy(), 0.0, color=color,
                        linewidth=0.0, alpha=0.9, zorder=3)
        ax.annotate(label, (0.015, 0.93), xycoords="axes fraction", ha="left", va="top",
                    fontsize=6.5, color=color, zorder=4)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, ymax / 2])
        C.style_axes(ax, grid_axis="y", labelsize=6.5, ticklen=2.5)

    axes[-1].set_xlim(0, day[-1])
    axes[-1].set_xlabel("Time (days)")
    axes[-1].xaxis.label.set_color(C.INK)
    # the label goes on the middle strip, so `tight_layout` accounts for it
    axes[len(rows) // 2].set_ylabel("Batches in the day's busiest 15 min", fontsize=8)
    axes[len(rows) // 2].yaxis.label.set_color(C.INK)

    fig.tight_layout(pad=0.3, h_pad=0.35)
    C.save(fig, "fig_daily_peak_strips")

    C.write_csv(df, "fig_daily_peak_strips")
    summary = pd.DataFrame({
        "quiet days (peak < 25)": (df < 25).sum(),
        "burst days (peak > 100)": (df > 100).sum(),
        "tallest day": df.max(),
    })
    print("\ntable view (the figure's twin) - days out of", len(df), "in the drawn strip:")
    print(summary.round(0).to_string())


def burst_rate() -> None:
    """How often a burst of each amplitude happens, in bins per week."""
    counts, paths = _ensemble()
    rows = {"Trace": _bins_per_week(counts)}
    for label, runs in paths.items():
        rows[label] = np.median(np.array([_bins_per_week(p) for p in runs]), axis=0)
    df = pd.DataFrame(rows, index=BIN_LABELS)

    _bar_panel(df, BIN_LABELS, "Batches in the bin", "Bins per week",
               "fig_burst_rate", figsize=(5.6, 3.5), ylim=(0, 17.5))

    df.index.name = "batches_in_bin"
    C.write_csv(df, "fig_burst_rate")
    print("\ntable view (the figure's twin) - bins per week:")
    print(df.round(2).to_string())


INPUTS = ("state", "counts", "models")

VIEWS = {
    "daily-peak": daily_peak,
    "daily-peak-range": daily_peak_range,
    "daily-peak-bands": daily_peak_bands,
    "daily-peak-strips": daily_peak_strips,
    "burst-rate": burst_rate,
}

if __name__ == "__main__":
    C.run(VIEWS, "daily-peak",
          "Excursion structure: how big the bursts get, and how often they happen.",
          INPUTS)
