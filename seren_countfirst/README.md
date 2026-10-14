# seren_countfirst

The Seren study: the GLI-NB arrival generator, the job-size model, and the
pipeline that calibrates, evaluates and combines them. Arrivals and sizes share
the trace but nothing else; they meet only in the `workload` stage.

## Modules

| module | contents |
|---|---|
| `models/` | count models: IID and latent-AR negative binomial, gated latent (GLI-NB) |
| `candidates/` | the GLI-NB variants that can be selected as the generator |
| `baselines/` | ON/OFF, NB-INGARCH and regime-switch comparison models |
| `evaluation/` | model scoring, horizon and parameter-recovery studies |
| `reconstruct.py` | batch counts to timestamps to job arrivals |
| `scaling.py` | load scaling and the global rate calibration |
| `jobsize/` | durations to work units, and the size laws ([README](jobsize/README.md)) |
| `ensemble.py` | representative-run selection for figures |
| `state.py` | pipeline state as npy, csv and json |
| `run_pipeline/` | stages, registries, run manifests and configuration |
| `experiments/` | standalone feasibility studies |

## Pipeline

```bash
python -m seren_countfirst.run_pipeline <stage> [<stage> ...]
```

| stage | does |
|---|---|
| `data` | reads the trace, verifies its audit facts, builds the 15-minute count series |
| `fit <model> ...` | fits the named models |
| `evaluate` | scores every model on eight sequential gates and selects the generator |
| `scaling`, `certify` | load-scaling sweep; gate scores over ten seeds |
| `horizon` | statistics by horizon length |
| `reconstruct` | counts to batch timestamps to job arrivals |
| `baselines` | bootstrap reference for the gates |
| `export` | the model card |
| `jobsize`, `jobsize_folds` | the size laws, and their rolling-origin validation |
| `workload` | arrivals with sizes, across load factors |

`fit` takes model names and runs alone. `mm4` is GLI-NB, the published generator;
`nogate` is its ablation LI-NB; `onoff_long` and `ingarch_long` are the ON/OFF and
NB-INGARCH baselines of the paper. Only GLI-NB variants can be selected.

`--plan <stage>` prints the dependency order without running it, `--with-deps`
also runs the dependencies, and `--resume` skips invocations whose recorded inputs
and outputs still verify. Every invocation is recorded in
`results/run_manifest.json`. `SEREN_TRACE_CSV`, `SEREN_RESULTS` and
`SEREN_STATE_DIR` relocate the trace, results and state; `SEREN_RUNS_DIR`
isolates each run under its own identity.
