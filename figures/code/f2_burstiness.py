"""F2 burstiness across scales: dispersion against aggregation window."""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd

import _common as C
from _common import plt

N_PATHS = 200
# the seven scales are a chosen set, not a continuum, so they are spaced evenly.
# Hours throughout: days would put three decimals on the short end.
TICKS = {"15m": "0.25", "1h": "1", "6h": "6", "1d": "24",
         "2d": "48", "4d": "96", "7d": "168"}


def idc_scale() -> None:
    """Index of dispersion against aggregation scale, trace against the field."""
    from seren_countfirst import SCALES, idc_curve

    st = C.state()
    trace = idc_curve(st["counts"].values)
    paths = C.series_paths(st, N_PATHS)
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
    # log y: the field spans 7.7 to 4423, and on a linear axis the three
    # alternatives collapse at the bottom, hiding that ON/OFF matches the trace
    # at 15m (73.1 vs 73.2) and only then plateaus
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


VIEWS = {"idc-scale": idc_scale}

if __name__ == "__main__":
    C.run(VIEWS, "idc-scale", __doc__)
