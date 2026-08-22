# seren_countfirst

A trace-calibrated **arrival generator** for LLM fine-tuning workloads.

It produces synthetic job-arrival streams for scheduling research: any horizon,
any load factor, repeatable from a seed, and statistically faithful to a real
cluster trace.

## Reference data

The generator is calibrated against the SFT subset of a public GPU-cluster
trace: 66,971 jobs → 49,747 exact-timestamp batches over 138.2 days, binned to
13,272 fifteen-minute intervals.

## Getting the trace

`trace_seren.csv` is not in the repository: it is 99 MB and belongs to
[AcmeTrace](https://github.com/InternLM/AcmeTrace), whose Seren cluster trace it
comes from. Put it at the repository root, or point `SEREN_TRACE_CSV` at it. The
`data` stage verifies it on load against

```
sha256  747854c73ce8b3263b764aa9ff8d866d89903705d31fcd0e0520c33dd14af9a1
```

and stops if it does not match, so a different snapshot cannot be mistaken for
this one.

## Environment

`requirements.lock` pins the exact versions the published `results/` and
`figures/` were produced under; `pyproject.toml` carries the floors. The pipeline
is a pure function of `SEED = 20260729`, so a rebuild in that environment is
reproducible:

```bash
python reproduce.py --quick            # every stage except the three long ones, ~20 min
python reproduce.py --verify           # full rebuild, then diff it against results/
```

`docs/refactor-todo.md` records the one place the rebuild is known to differ.

## Quickstart

```bash
# ingest the trace, verify the audit facts, build the count series
python -m seren_countfirst.run_pipeline data

# calibrate the model ladder
python -m seren_countfirst.run_pipeline fit r2 r3 r4
python -m seren_countfirst.run_pipeline fit mm4

# score candidates, sweep load, sweep horizon, reconstruct events, export
python -m seren_countfirst.run_pipeline evaluate scaling horizon reconstruct baselines export
```

Every stage is resumable: state persists as npy, csv and json, never pickle.
Stages after `fit` may be listed together; `fit` takes rung names, so it runs on
its own.

Generating a workload directly:

```python
import numpy as np
from seren_countfirst.state import load_portable_state, rebuild_model
from seren_countfirst.paths import STATE_DIR

state = load_portable_state(STATE_DIR)
model = rebuild_model(state["models_json"][state["winner"]])

rng = np.random.default_rng(0)
counts = model.simulate(n_bins=96 * 30, rng=rng, alpha=1.0)   # 30 days at nominal load
```

`alpha` is the load factor. Event-level output — timestamps, batch sizes and
job arrivals — comes from the `reconstruct` stage.

## Paths and configuration

Paths resolve relative to the repository and are environment-overridable:

| variable | default | holds |
|---|---|---|
| `SEREN_TRACE_CSV` | `./trace_seren.csv` | the raw trace |
| `SEREN_RESULTS` | `./results` | exported tables and the model card |
| `SEREN_STATE_DIR` | `./results/state_portable_data/state_portable` | resumable pipeline state |

To run without touching an existing results directory:

```bash
SEREN_RESULTS=$PWD/results_scratch python -m seren_countfirst.run_pipeline evaluate
```

## Repository map

| path | contents |
|---|---|
| `seren_countfirst/data.py` | trace ingestion, batching, the count series, contiguous and rolling-origin splits |
| `seren_countfirst/metrics.py` | dispersion across scales, autocorrelation, episode and run-length statistics, microstructure |
| `seren_countfirst/models/` | the model ladder: iid negative binomial, latent AR, regime-switch benchmark, gated latent intensity |
| `seren_countfirst/evaluation/` | the sequential structural gates, rolling-origin folds, horizon and recovery studies |
| `seren_countfirst/bootstrap.py` | block resamplers and the envelopes candidates are graded against |
| `seren_countfirst/scaling.py` | load scaling, superposition, the final scalar rate calibration |
| `seren_countfirst/reconstruct.py` | counts to batch timestamps to sizes to job arrivals, with collision accounting |
| `seren_countfirst/jobsize/` | the separate size pipeline: the work-unit inversion and the laws fitted to it |
| `seren_countfirst/state.py` | the typed, resumable pipeline state as npy, csv and json; never pickle |
| `seren_countfirst/artifacts.py` | atomic CSV, JSON and NPY writers used by the pipeline |
| `seren_countfirst/ensemble.py` | which simulated run is representative, and how one is redrawn at another load |
| `seren_countfirst/run_pipeline/` | the staged CLI with a central registry and injectable runtime context |
| `seren_countfirst/experiments/` | standalone feasibility studies |
| `seren_countfirst/tests/` | harness invariants and numerical regression pins |
| `figures/code/` | the manuscript figures; `make_all.py` renders every one |

Each package folder carries its own README; `seren_countfirst/README.md` is the
module map.

## Figures

```bash
python figures/code/make_all.py            # every figure and its csv twin
python figures/code/make_all.py --check    # which inputs are missing, and the stage that writes them
python figures/code/excursion.py --list    # the views one module offers
```

## Tests

```bash
python -m pytest -q                      # everything
python -m pytest -q -m "not slow"        # the fast lane, for use between edits
```

Four kinds: harness invariants that encode the methodology as executable rules,
numerical pins that fix simulation checksums and the metric vector, a smoke suite
that fits, simulates and freezes every model on a short synthetic series, and
layout invariants that keep the house style and the seeding discipline from
drifting back. A failing pin means a numerical result moved — treat it as the
specification working, not as a test to update.

Design rationale is in `docs/`; `docs/README.md` is the map.
