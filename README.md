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
python reproduce.py --quick            # skip four long/dependent stages, ~20 min
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

# inspect everything required for a target without executing it
python -m seren_countfirst.run_pipeline --plan workload
python -m seren_countfirst.run_pipeline --plan --with-deps evaluate

# skip only an exact invocation whose full provenance still verifies
python -m seren_countfirst.run_pipeline --resume evaluate scaling

# resolve and safely execute omitted dependencies from the Seren recipe
python -m seren_countfirst.run_pipeline --with-deps evaluate
python -m seren_countfirst.run_pipeline --resume --with-deps workload
```

Stages can be invoked independently against portable npy/csv/json state. The
stage-level dependency planner is inspection-only: omitted prerequisites are
assumed to already exist in that state. When several stages are listed,
requested dependencies must appear first. `fit` takes model names, so it runs
on its own. `--resume fit mm4` matches the arguments as well as the stage name.
`--with-deps` expands the stage DAG through the concrete full-reproduction
recipe, prints the invocation plan, and processes shared dependencies once.
Dependency invocations use verified resume automatically; requested targets run
normally unless `--resume` is also present. Invalid or missing recipe commands
fail before any stage runs.

Executed invocations append a validated `run_manifest.json` under the selected
results directory. It records the immutable run identity, resolved configuration,
status, timing, runtime versions, source revision, and every result/state file
read or rewritten by that invocation with its size and SHA-256. Manifest schema
v3 records whether dependency input lineage is complete. `--resume` skips only
when RunSpec/config, exact invocation, clean Git revision, runtime versions, raw
dataset, dependency inputs and outputs all verify; otherwise it prints the miss
reason and runs normally. Manifests v1/v2 remain readable but cannot authorize a
skip. `--plan` remains a read-only, stage-level view; `--with-deps` is the opt-in
execution mode that resolves abstract stages such as `fit` into exact model
arguments. `reproduce.py` writes its outer timing summary separately as
`reproduction_report.json`.

Generating a workload directly:

```python
import numpy as np
from seren_countfirst.run_pipeline.context import default_context
from seren_countfirst.state import rebuild_model

state = default_context().load_state()
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
| `SEREN_RESULTS` | `./results` | exported tables, the model card and run manifest |
| `SEREN_STATE_DIR` | `./results/state_portable_data/state_portable` | resumable pipeline state |
| `SEREN_RUNS_DIR` | unset | opt-in root for RunSpec-isolated results and state |

To run without touching an existing results directory:

```bash
SEREN_RESULTS=$PWD/results_scratch python -m seren_countfirst.run_pipeline evaluate
```

For multiple datasets, algorithms or configurations, use the namespaced mode:

```bash
SEREN_RUNS_DIR=$PWD/runs python -m seren_countfirst.run_pipeline data
```

The actual directory is `SEREN_RUNS_DIR/<dataset identity>/<model identity>/<run-id>/`;
the default study currently resolves to
`runs/internlm/acmetrace-seren/unversioned/sft/full/count-first/gli-nb/gated-mm4/<run-id>/`.
Portable state lives below that same directory. `SEREN_RUNS_DIR` is intentionally
mutually exclusive with `SEREN_RESULTS` and `SEREN_STATE_DIR`. With none of them
set, the published `results/` layout remains unchanged. No state is copied or
silently loaded from legacy `results/` when namespaced mode is selected.

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
| `seren_countfirst/run_pipeline/` | staged orchestration, registries, runtime context, output tracking and manifest persistence |
| `seren_countfirst/experiments/` | standalone feasibility studies |
| `seren_countfirst/tests/` | harness invariants and numerical regression pins |
| `traffic_modeling/core/` | dataset/model/run identities and typed execution-manifest records |
| `traffic_modeling/datasets/` | canonical trace contracts, capabilities and provider adapters |
| `figures/code/` | the manuscript figures; `make_all.py` renders every one |
| `tools/` | local quality ratchet and isolated pipeline profiler |

Each package folder carries its own README; `seren_countfirst/README.md` is the
module map.

## Figures

```bash
python figures/code/make_all.py            # every figure and its csv twin
python figures/code/make_all.py --check    # which inputs are missing, and the stage that writes them
python figures/code/excursion.py --list    # the views one module offers
```

## Quality

```bash
python tools/quality.py fast       # lint + typed scope + fast tests + wheel smoke
python tools/quality.py source     # tracked source and package checks; no internal tests
python tools/quality.py full       # include slow scientific tests
python tools/quality.py package    # isolated wheel build/install/import smoke only
```

The Mypy scope is an explicit ratchet in `pyproject.toml`: clean modules may be
added, but a failing module is never removed merely to make the command green.
The package smoke builds from a temporary copy, installs outside the checkout,
checks both import roots and the `seren-pipeline` entry point, and rejects wheels
that contain internal tests. `source` is therefore usable without `docs/` or
`seren_countfirst/tests/`; `fast` is the normal local checkpoint.

The internal suite has four kinds: harness invariants that encode the methodology,
numerical pins that fix simulation checksums and the metric vector, a smoke suite
that fits, simulates and freezes every model on a short synthetic series, and
layout invariants that keep the house style and the seeding discipline from
drifting back. A failing pin means a numerical result moved — treat it as the
specification working, not as a test to update.

Profile a real stage only in an isolated run namespace:

```bash
python tools/profile_pipeline.py \
  --runs-dir runs/profiles \
  --output runs/profiles/data.prof \
  data
```

The profiler refuses the repository root and published `results/`, writes a
standard `cProfile` file and prints cumulative hotspots. Keep profiling separate
from the quality gate: heavy stages are evidence runs, not per-edit checks.

Design rationale is in `docs/`; `docs/README.md` is the map.
