"""F3 memory: how far back the count series predicts itself."""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C
from _common import plt

N_PATHS = 60
# chosen lags, evenly spaced; one unit for the axis (hours)
LAGS = (1, 2, 4, 8, 24, 48, 96, 192, 384)


def acf() -> None:
    """Count autocorrelation against lag, trace against the models."""
    from seren_countfirst import acf as acf_of

    st = C.state()
    rows = {"Trace": acf_of(st["counts"].values, LAGS)}
    for label, runs in C.series_paths(st, N_PATHS).items():
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
    # nine lag labels in 3.5 in: the tick text has to come down with the panel
    C.style_axes(ax, labelsize=7.5, ticklen=2.5)
    C.model_legend(ax, loc="upper right")

    fig.tight_layout(pad=0.3)
    C.save(fig, "fig_acf")

    table.index = [f"{lag * C.BIN_H:g}" for lag in LAGS]
    table.index.name = "lag_h"
    C.write_csv(table, "fig_acf")
    print("\ntable view (the figure's twin):")
    print(table.round(3).to_string())


VIEWS = {"acf": acf}

if __name__ == "__main__":
    C.run(VIEWS, "acf", __doc__)
