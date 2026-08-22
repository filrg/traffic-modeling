# seren_countfirst

A count-first arrival generator for LLM fine-tuning workloads, plus a separate
job-size pipeline. The two halves share the trace and nothing else; they meet
only in the `workload` stage, through `jobsize.folds.assign_work`.

Rationale lives in `docs/`, not here: `design-history.md` (why, as a decision
log), `HANDOFF.md` (verified numbers), `design-notes.md` (model and module
rules).

## The arrival half

| module | holds |
|---|---|
| `data.py` | trace views, the regular count grid, contiguous and rolling-origin splits |
| `metrics.py` | the standard metric set and the discrepancy definitions, all on base-resolution counts |
| `bootstrap.py` | block-bootstrap baselines and the trace uncertainty envelopes candidates are graded against |
| `models/` | the ladder: rungs 2-4b, plus the benchmark rows and the ablation it is read against |
| `evaluation/` | the eight sequential structural gates, rolling-origin folds, horizon and recovery studies |
| `scaling.py` | load scaling by intensity multiplication, superposition as a benchmark, the final scalar rate calibration |
| `reconstruct.py` | counts to batch timestamps to sizes to job arrivals, with collision accounting |

## The size half

`jobsize/` inverts absorbed-work durations into work units and fits a law to
them. It has its own README.

## Shared

| module | holds |
|---|---|
| `paths.py` | filesystem locations, resolved relative to the repository and env-overridable |
| `state.py` | the resumable pipeline state as npy + csv + json, and model rebuilding; never pickle |
| `artifacts.py` | atomic writers for pipeline and figure artifacts |
| `ensemble.py` | the display-ensemble rules: which simulated run is representative, and how one is redrawn at another load |
| `run_pipeline/` | the staged CLI, stage registry, named seed streams and injectable path/state context |
| `experiments/` | standalone studies that are not part of any stage |
| `tests/` | harness invariants, numerical pins and a smoke suite |

`ensemble.py` sits in the package rather than in `figures/` because the rules it
carries are defended in the paper and are pinned by tests; `figures/code/_common.py`
re-exports it so a figure calls one name.
