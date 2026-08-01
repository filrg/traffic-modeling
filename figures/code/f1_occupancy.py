"""F1 occupancy: the arrival series itself, trace against a synthetic run."""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C
from _common import plt

N_CANDIDATES = 200
# "level" picks on mean rate, "stats" on all four base-bin facts, "typical" the
# median of those. Selection never looks at when a busy regime falls: the model
# is stationary and carries no calendar, so regime placement is arbitrary and
# picking on it costs real accuracy.
SELECT = "level"


def _collect() -> tuple:
    st = C.state()
    counts = st["counts"].values
    model, cal = C.winner(st)
    from seren_countfirst import base_marginals

    tm = base_marginals(counts)
    keys = ("mean", "IDC_15m", "zero_frac", "acf_1h_lag1")
    rng = np.random.default_rng(C.SEED + 4)
    runs = [model.simulate(len(counts), rng, alpha=cal) for _ in range(N_CANDIDATES)]
    dist = np.array([sum(abs(np.log(max(base_marginals(r)[k], 1e-9) / tm[k]))
                         for k in keys) for r in runs])
    order = np.argsort(dist)
    if SELECT == "level":
        pick = int(np.argmin([abs(r.mean() - float(counts.mean())) for r in runs]))
    elif SELECT == "stats":
        pick = int(order[0])
    else:
        pick = int(order[len(order) // 2])
    report = pd.DataFrame({
        "run": ["selected", "closest on stats", "median", "furthest"],
        "stat_distance": [dist[pick], dist[order[0]],
                          dist[order[len(order) // 2]], dist[order[-1]]],
    })
    return counts, runs[pick], float(counts.mean()), dist[pick], report


def series() -> None:
    """The full real series beside a synthetic run of equal length."""
    trace_win, syn_win, target, picked_d, report = _collect()
    from seren_countfirst import base_marginals

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
    print(f"\nselection: {SELECT} of {N_CANDIDATES}, distance {picked_d:.4f}")
    print(report.round(4).to_string(index=False))
    print(f"\ntrace overall mean {target * C.BINS_PER_H:.2f} batches/h")
    for name, w in (("trace", trace_win), ("model run", syn_win)):
        bm = base_marginals(w)
        print(f"  {name:10} mean {bm['mean'] * C.BINS_PER_H:6.2f}  IDC {bm['IDC_15m']:7.2f}  "
              f"zero {bm['zero_frac']:.3f}  acf_1h {bm['acf_1h_lag1']:.3f}  "
              f"peak {bm['max_bin'] * C.BINS_PER_H:7.0f}")


VIEWS = {"series": series}

if __name__ == "__main__":
    C.run(VIEWS, "series", __doc__)
