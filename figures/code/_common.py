"""House style, state loading and the view-selection CLI shared by every figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Mapping

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seren_countfirst.paths import RESULTS, STATE_DIR  # noqa: E402,F401
from seren_countfirst.tools.migrate_state import (  # noqa: E402
    load_portable_state,
    rebuild_model,
)

SEED = 20260729
BIN_H = 0.25
BINS_PER_H = 4
BINS_PER_DAY = 96

# this file lives in figures/code; outputs are sorted into pdf/ png/ csv/
OUT = Path(__file__).resolve().parents[1]

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#e8e7e2"
BLUE = "#2a78d6"

# black solid is the measurement, coloured dash is a model. Hue alone carries
# identity: the four slots clear all-pairs CVD dE 9.2 (target 8) and
# normal-vision 16.3 (floor 15) against white paper, so no dash relief is owed.
MODEL_DASH = (0, (5, 1.6))

# categorical slots 1-3 plus violet. Violet is slot 7, not slot 4: as a fourth,
# yellow / magenta / red fail the normal-vision floor of 15 (13.7 / 12.9 / 7.1)
# and green fails CVD at 3.2.
SERIES = [
    ("gated_mm4", "GLI-NB", "#2a78d6"),
    ("gated_nogate", "LI-NB", "#eb6834"),
    ("onoff", "ON/OFF", "#1baf7a"),
    ("ingarch", "NB-INGARCH", "#4a3aa7"),
]


def house_style() -> None:
    """Apply the shared rcParams. Type 42 embeds TrueType rather than Type 3."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 9,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes(ax, grid_axis: str = "both", labelsize: float | None = None,
               ticklen: float = 3.0, hide_left: bool = False) -> None:
    """Apply the shared grid, spine and tick treatment to one axis."""
    ax.grid(True, axis=grid_axis, which="major", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    dead = ("top", "right", "left") if hide_left else ("top", "right")
    for side in dead:
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        if side in dead:
            continue
        ax.spines[side].set_color(AXIS)
    kw = {"colors": MUTED, "labelcolor": INK_2, "length": ticklen, "width": 0.7}
    if labelsize is not None:
        kw["labelsize"] = labelsize
    ax.tick_params(**kw)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def frame_legend(leg):
    """Recolour a legend frame and its text to the house palette."""
    leg.get_frame().set_edgecolor(AXIS)
    leg.get_frame().set_linewidth(0.6)
    for t in leg.get_texts():
        t.set_color(INK_2)
    return leg


def stroke_legend(ax, entries, loc, fontsize: float = 6.5, labelspacing: float = 0.28,
                  handlelength: float = 2.6, borderpad: float = 0.45):
    """Draw a legend whose handles are bare strokes matching each curve's style.

    Handles are proxies, not the artists: a 4pt marker centred in the default
    19pt handle covers most of a 6.6pt dash period, rendering every entry as a
    solid stroke. The handle is lengthened to fit two full dash periods.
    """
    handles = [Line2D([], [], color=c, linewidth=lw, linestyle=ls) for _, c, lw, ls in entries]
    return frame_legend(ax.legend(
        handles, [label for label, _, _, _ in entries], loc=loc, frameon=True,
        framealpha=1.0, fontsize=fontsize, handlelength=handlelength,
        labelspacing=labelspacing, borderpad=borderpad))


def model_legend(ax, loc, trace_lw: float = 1.5, model_lw: float = 1.3, **kw):
    """Draw the standard trace-plus-four-models stroke legend."""
    return stroke_legend(ax, [("Trace", INK, trace_lw, "-")]
                         + [(label, color, model_lw, MODEL_DASH) for _, label, color in SERIES],
                         loc=loc, **kw)


def save(fig, name: str) -> None:
    """Write `name`.pdf and `name`.png into the figure tree."""
    for ext in ("pdf", "png"):
        (OUT / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / ext / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "pdf" / f"{name}.pdf", "and .png")


def write_csv(df: pd.DataFrame, name: str, index: bool = True) -> None:
    """Write the figure's csv twin."""
    (OUT / "csv").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "csv" / f"{name}.csv", index=index)


def state() -> dict:
    """Load the portable pipeline state."""
    return load_portable_state(STATE_DIR)


def calibrated(st: dict, key: str) -> tuple:
    """Return (model, alpha) for one state key, with the global rate calibration."""
    m = rebuild_model(st["models_json"][key])
    return m, st["trace_marginals"]["mean"] / float(m.params_["mean"])


def winner(st: dict) -> tuple:
    """Return (model, alpha) for the frozen winner."""
    return calibrated(st, st["winner"])


def series_paths(st: dict, n_paths: int, seed_offset: int = 1) -> dict:
    """Simulate `n_paths` at trace length for each model in `SERIES`.

    One RNG per series, drawn sequentially, so every view built on this ensemble
    sees the same paths.
    """
    n_bins = len(st["counts"].values)
    out = {}
    for key, label, _ in SERIES:
        model, cal = calibrated(st, key)
        rng = np.random.default_rng(SEED + seed_offset)
        out[label] = [model.simulate(n_bins, rng, alpha=cal) for _ in range(n_paths)]
    return out


def daily_peaks(counts: np.ndarray) -> np.ndarray:
    """Return the largest bin in each whole day; a trailing partial day is dropped."""
    x = np.asarray(counts, dtype=float)
    n_days = len(x) // BINS_PER_DAY
    return x[: n_days * BINS_PER_DAY].reshape(n_days, BINS_PER_DAY).max(axis=1)


def run(views: Mapping[str, Callable[[], None]], default: str, description: str) -> None:
    """Parse `--style` / `--all` / `--list` and run the selected views."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--style", choices=list(views), default=default,
                    help=f"which view to render (default: {default})")
    ap.add_argument("--all", action="store_true", help="render every view")
    ap.add_argument("--list", action="store_true", help="list the views and exit")
    args = ap.parse_args()

    if args.list:
        for name, fn in views.items():
            mark = " (default)" if name == default else ""
            print(f"{name}{mark}\n    {(fn.__doc__ or '').splitlines()[0]}")
        return

    chosen = list(views) if args.all else [args.style]
    for i, name in enumerate(chosen):
        if i:
            print()
        print(f"=== {name} ===")
        views[name]()
