# seren_countfirst

A trace-calibrated **arrival generator** for LLM fine-tuning workloads.

It produces synthetic job-arrival streams for scheduling research: any horizon,
any load factor, repeatable from a seed, and statistically faithful to a real
cluster trace.

## Reference data

The generator is calibrated against the SFT subset of a public GPU-cluster
trace: 66,971 jobs → 49,747 exact-timestamp batches over 138.2 days, binned to
13,272 fifteen-minute intervals.

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

Every stage is resumable: state persists between invocations as npy, csv and
json, never pickle. Stages after `fit` may be listed together and run in
sequence; `fit` takes rung names, so it runs on its own.

Generating a workload directly:

```python
import numpy as np
from seren_countfirst.tools.migrate_state import load_portable_state, rebuild_model
from seren_countfirst.paths import STATE_DIR

state = load_portable_state(STATE_DIR)
model = rebuild_model(state["models_json"][state["winner"]])

rng = np.random.default_rng(0)
counts = model.simulate(n_bins=96 * 30, rng=rng, alpha=1.0)   # 30 days at nominal load
```

`alpha` is the load factor. Event-level output — timestamps, batch sizes and
job arrivals — comes from the `reconstruct` stage.

## Paths and configuration

Paths resolve relative to the repository and are environment-overridable, so
the package runs unchanged in a sandbox, in CI and on a workstation:

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
| `seren_countfirst/reconstruct.py` | counts → batch timestamps → sizes → job arrivals, with collision accounting |
| `seren_countfirst/run_pipeline/` | the staged command-line driver |
| `seren_countfirst/experiments/` | standalone feasibility studies |
| `seren_countfirst/tests/` | harness invariants and numerical regression pins |

## Tests

```bash
python -m pytest seren_countfirst/tests -q
```

Three kinds: harness invariants that encode the methodology as executable
rules, numerical pins that fix simulation checksums and the metric vector, and
a smoke suite that fits, simulates and freezes every model on a short synthetic
series. A failing pin means a numerical result moved — treat it as the
specification working, not as a test to update.
