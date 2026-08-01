"""Job-size law: the empirical work units against the fitted candidates.

Two views of one fit. `pdf` is truncated at the 99th percentile for legibility
and reads density **per decade**, so bar area is probability on a log axis; `cdf`
is drawn over the whole seven-decade range and is where the fits visibly
separate. Neither is complete alone, which is why they are a pair.

Reads `results/` — run `python -m seren_countfirst.run_pipeline jobsize` first.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import _common as C
from _common import plt

# the empirical series takes the ink slot the trace holds elsewhere
LAWS = [
    ("mix_lognormal_3", "Lognormal mix (k=3)", "#2a78d6"),
    ("lognormal", "Lognormal", "#eb6834"),
    ("burr12", "Burr XII", "#1baf7a"),
]
# the source notebook's display window and binning, kept so the panels compare
DISPLAY_Q = 99.0
N_BINS = 70
N_GRID_PDF = 1000
# 39e3 steps exceed what a 3.5 in panel resolves, so the CDF is read off a
# log-spaced grid, which is exact for a step function
N_GRID_CDF = 600

_CACHE: dict = {}


def _pool_and_board() -> tuple:
    """Return (sizes, scoreboard), loaded once."""
    if not _CACHE:
        from seren_countfirst import EmpiricalJobSize

        _CACHE["sizes"] = EmpiricalJobSize.from_npz(C.RESULTS / "jobsize_pool.npz").sizes
        _CACHE["board"] = pd.read_csv(
            C.RESULTS / "jobsize_candidates.csv").set_index("distribution")
    return _CACHE["sizes"], _CACHE["board"]


def _law(row: pd.Series, name: str):
    """Rebuild a scored candidate law from the json the jobsize stage persisted."""
    from seren_countfirst import LogNormalMixture, ScipyJobSize

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
    ax.bar(edges[:-1], heights, width=np.diff(edges), align="edge", color=C.INK,
           alpha=0.16, edgecolor=C.INK, linewidth=0.25, zorder=2)
    for _, label, color in LAWS:
        ax.plot(grid, table[label].to_numpy(), color=color, linewidth=1.3,
                linestyle=C.MODEL_DASH, zorder=3)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, None)
    _size_axes(ax, "Share of jobs per decade")
    handles = [Patch(facecolor=(0.04, 0.04, 0.04, 0.16), edgecolor=C.INK, linewidth=0.25)]
    handles += [Line2D([], [], color=c, linewidth=1.3, linestyle=C.MODEL_DASH)
                for _, _, c in LAWS]
    C.frame_legend(ax.legend(handles, ["Empirical"] + [label for _, label, _ in LAWS],
                             loc="upper left", frameon=True, framealpha=1.0, fontsize=6.5,
                             handlelength=2.6, labelspacing=0.28, borderpad=0.45))
    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_jobsize_pdf")

    C.write_csv(table, "fig_jobsize_pdf")
    print(f"\n{len(sizes)} completed jobs; the panel shows the {len(shown)} at or below "
          f"the {DISPLAY_Q:.0f}th percentile ({np.percentile(sizes, DISPLAY_Q):,.0f} units)")
    print(f"the {len(sizes) - len(shown)} jobs above it reach {sizes.max():,.0f} units, "
          f"against a median of {np.median(sizes):,.0f}")


def cdf() -> None:
    """Cumulative distribution over the full range, empirical against the laws."""
    sizes, board = _pool_and_board()
    ordered = np.sort(sizes)
    grid = np.geomspace(ordered[0], ordered[-1], N_GRID_CDF)
    # the empirical CDF on the same grid: the share of jobs at or below
    curves = {"job_size": grid,
              "Empirical": np.searchsorted(ordered, grid, side="right") / len(ordered)}
    for key, label, _ in LAWS:
        curves[label] = _law(board.loc[key], key).cdf(grid)
    table = pd.DataFrame(curves).set_index("job_size")

    C.house_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.plot(grid, table["Empirical"].to_numpy(), color=C.INK, linewidth=1.5, zorder=4)
    for _, label, color in LAWS:
        ax.plot(grid, table[label].to_numpy(), color=color, linewidth=1.3,
                linestyle=C.MODEL_DASH, zorder=3)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _size_axes(ax, "Fraction of jobs at or below")
    handles = [Line2D([], [], color=C.INK, linewidth=1.5)]
    handles += [Line2D([], [], color=c, linewidth=1.3, linestyle=C.MODEL_DASH)
                for _, _, c in LAWS]
    C.frame_legend(ax.legend(handles, ["Empirical"] + [label for _, label, _ in LAWS],
                             loc="upper left", frameon=True, framealpha=1.0, fontsize=6.5,
                             handlelength=2.6, labelspacing=0.28, borderpad=0.45))
    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_jobsize_cdf")

    C.write_csv(table, "fig_jobsize_cdf")
    emp = table["Empirical"].to_numpy()
    print("largest vertical gap from the empirical CDF (the KS statistic's grid read):")
    for _, label, _ in LAWS:
        print(f"  {label:22s} {np.max(np.abs(table[label].to_numpy() - emp)):.4f}")


VIEWS = {"pdf": pdf, "cdf": cdf}

if __name__ == "__main__":
    C.run(VIEWS, "pdf", __doc__)
