from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seren_countfirst.artifacts import write_csv as atomic_write_csv
from seren_countfirst.ensemble import (
    BINS_PER_DAY,
    DISPLAY_CANDIDATES,
    DISPLAY_OFFSET,
    PEAK_Q,
    SEED,
    calibrated,
    closest_to_trace,
    daily_peaks,
    display_pool,
    redraw,
    representative,
    state,
    winner,
)
from seren_countfirst.run_pipeline.context import PortableStateStore, default_context

_RUNTIME = default_context()
RESULTS = _RUNTIME.results
if not isinstance(_RUNTIME.state_store, PortableStateStore):
    raise TypeError("figure generation requires a portable filesystem state store")
STATE_DIR = _RUNTIME.state_store.directory

__all__ = [
    "BINS_PER_DAY",
    "DISPLAY_CANDIDATES",
    "DISPLAY_OFFSET",
    "PEAK_Q",
    "SEED",
    "calibrated",
    "closest_to_trace",
    "daily_peaks",
    "display_pool",
    "redraw",
    "representative",
    "state",
    "winner",
]

# set by `run` from --error-bars / --no-error-bars; None leaves each view on its
# own default. The csv twin always holds the bounds whether or not they are drawn.
SHOW_CI: bool | None = None
BIN_H = 0.25
BINS_PER_H = 4

# this file lives in figures/code; outputs are sorted into pdf/ png/ csv/
OUT = Path(__file__).resolve().parents[1]

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#e8e7e2"
BLUE = "#2a78d6"

# black solid is the measurement, coloured dash is a model; hue alone carries identity
MODEL_DASH = (0, (5, 1.6))

# revalidate the palette for colour-vision and print contrast before adding a series
SERIES = [
    ("gated_mm4", "GLI-NB", "#2a78d6"),
    ("gated_nogate", "LI-NB", "#eb6834"),
    ("onoff_long", "ON/OFF", "#1baf7a"),
    ("ingarch_long", "NB-INGARCH", "#4a3aa7"),
]


# artifact key -> (path, the pipeline stage that writes it). The only place a
# figure input path is written down; a module names keys, never paths.
ARTIFACTS = {
    "state": (STATE_DIR / "meta.json", "data"),
    "counts": (STATE_DIR / "counts.npy", "data"),
    "models": (STATE_DIR / "models.json", "fit"),
    "batch_times": (STATE_DIR / "batch_times.csv", "data"),
    "jobsize_model": (RESULTS / "jobsize_model.json", "jobsize"),
    "jobsize_pool": (RESULTS / "jobsize_pool.npz", "jobsize"),
    "jobsize_candidates": (RESULTS / "jobsize_candidates.csv", "jobsize"),
    "jobsize_conditional": (RESULTS / "jobsize_conditional.json", "jobsize"),
    "jobsize_two_part": (RESULTS / "jobsize_two_part.json", "jobsize"),
}


def check_inputs(keys) -> bool:
    """Print ok / MISSING for each artifact, naming the stage that writes it."""
    ok = True
    for key in keys:
        path, stage = ARTIFACTS[key]
        here = path.exists()
        ok &= here
        mark = "ok     " if here else "MISSING"
        note = "" if here else f"   <- run `run_pipeline {stage}`"
        print(f"  {mark} {key:20s} {path}{note}")
    return ok


def show_ci(default: bool) -> bool:
    """Whether to draw a measurement's interval; the flag overrides the view's default."""
    return default if SHOW_CI is None else SHOW_CI


def house_style() -> None:
    """Apply the shared rcParams. Type 42 embeds TrueType rather than Type 3."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
            "font.size": 9,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(
    ax,
    grid_axis: str = "both",
    labelsize: float | None = None,
    ticklen: float = 3.0,
    hide_left: bool = False,
) -> None:
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


def stroke_legend(
    ax,
    entries,
    loc,
    fontsize: float = 6.5,
    labelspacing: float = 0.28,
    handlelength: float = 2.6,
    borderpad: float = 0.45,
):
    """Draw a legend whose handles are bare strokes matching each curve's style.

    The handles are proxies rather than the plotted artists, and are lengthened
    to hold two full dash periods.
    """
    handles = [Line2D([], [], color=c, linewidth=lw, linestyle=ls) for _, c, lw, ls in entries]
    return frame_legend(
        ax.legend(
            handles,
            [label for label, _, _, _ in entries],
            loc=loc,
            frameon=True,
            framealpha=1.0,
            fontsize=fontsize,
            handlelength=handlelength,
            labelspacing=labelspacing,
            borderpad=borderpad,
        )
    )


def model_legend(ax, loc, trace_lw: float = 1.5, model_lw: float = 1.3, **kw):
    """Draw the standard trace-plus-four-models stroke legend."""
    return stroke_legend(
        ax,
        [("Trace", INK, trace_lw, "-")]
        + [(label, color, model_lw, MODEL_DASH) for _, label, color in SERIES],
        loc=loc,
        **kw,
    )


def save(fig, name: str) -> None:
    """Write `name`.pdf and `name`.png into the figure tree."""
    for ext in ("pdf", "png"):
        (OUT / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / ext / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "pdf" / f"{name}.pdf", "and .png")


def write_csv(df: pd.DataFrame, name: str, index: bool = True) -> None:
    """Write the figure's csv twin."""
    (OUT / "csv").mkdir(parents=True, exist_ok=True)
    atomic_write_csv(df, OUT / "csv" / f"{name}.csv", index=index)


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


def run(
    views: Mapping[str, Callable[[], None]],
    default: str,
    description: str,
    inputs: Sequence[str] = (),
) -> None:
    """Parse `--style` / `--all` / `--list` / `--check` and run the selected views."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--style",
        choices=list(views),
        default=default,
        help=f"which view to render (default: {default})",
    )
    ap.add_argument("--all", action="store_true", help="render every view")
    ap.add_argument("--list", action="store_true", help="list the views and exit")
    ap.add_argument(
        "--check", action="store_true", help="report whether this module's inputs exist, and exit"
    )
    ap.add_argument(
        "--error-bars",
        dest="error_bars",
        action="store_true",
        default=None,
        help="draw the measurement's confidence intervals in every view "
        "that has them, overriding the view's own default",
    )
    ap.add_argument(
        "--no-error-bars",
        dest="error_bars",
        action="store_false",
        help="hide them in every view, likewise",
    )
    args = ap.parse_args()

    global SHOW_CI
    SHOW_CI = args.error_bars

    if args.check:
        raise SystemExit(0 if check_inputs(inputs) else 1)

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
