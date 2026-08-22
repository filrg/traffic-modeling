# seren_countfirst

The Seren count-first study: a trace-calibrated arrival generator, a separate
job-size model, and the pipeline that combines them into offered workloads.
Generic experiment identities and dataset contracts live in `traffic_modeling/`;
this package owns the Seren-specific scientific implementation.

```text
canonical Seren trace
    +-- arrival counts -> candidate/baseline models -> shared evaluation
    +-- job durations  -> work units -> job-size laws
                                      |
                         workload <---+
```

The arrival and size models share the canonical trace but remain independent.
They meet only in the `workload` stage through `jobsize.folds.assign_work`.

## Package map

| location | responsibility |
|---|---|
| `candidates/` | candidate-family definitions; `gli_nb.py` owns typed GLI-NB variants, scientific knobs and builders |
| `models/` | count-model implementations and shared mathematical primitives: NB2, AR(1), IID, latent-AR and gated latent models |
| `baselines/` | comparison algorithms, their immutable specifications and fit/calibration builders |
| `evaluation/` | the shared metric, fold and eight-gate evaluation harness used by every model role |
| `jobsize/` | duration-to-work inversion, size laws, dependence structure and rolling-origin validation |
| `run_pipeline/` | CLI orchestration, stage/model registries, run context, seed streams and study configuration |
| `experiments/` | standalone diagnostic studies that are not production pipeline stages |

The root modules provide reusable domain or infrastructure functions:

| module | responsibility |
|---|---|
| `data.py` | count grids, batch tables, activity schemes and time-aware splits from a canonical dataset view |
| `metrics.py` | IDC, ACF, episodes, tails, wavelets and discrepancy scores |
| `bootstrap.py` | trace-resampling references and uncertainty envelopes; these are not model baselines |
| `scaling.py` | load scaling, superposition comparisons and scalar rate calibration |
| `reconstruct.py` | count-bin paths to event timestamps and collision diagnostics |
| `state.py` | portable JSON/CSV/NPY state and model reconstruction; never pickle |
| `artifacts.py` | atomic CSV, JSON and NPY writers |
| `ensemble.py` | representative-run selection and reproducible redraws for figures |
| `paths.py` | environment-overridable compatibility paths |
| `provenance.py` | interpreter, dependency and platform metadata |

## Model roles

The three roles are separate by construction:

- candidates are GLI-NB ladder entries eligible for winner selection;
- baselines are external comparison mechanisms and cannot win;
- ablations isolate one candidate mechanism and cannot win.

All three use `evaluation.evaluate_model`, so dataset views, folds, seeds,
metrics and gate tolerances stay comparable. Serialized `role` values remain for
artifact compatibility; registry membership owns selection policy.

Each GLI-NB variant is one immutable record under `candidates/gli_nb.py`. That
record binds its model identity, implementation, calibration route, fit budget
and optional burst texture. `run_pipeline/candidate_registry.py` only composes
candidate families; a future algorithm can be added as another family module
without extending a string-based `if/else` chain.

## Pipeline structure

`python -m seren_countfirst.run_pipeline <stage>` dispatches through the typed
stage registry. The main flow is:

```text
data -> fit -> evaluate -> scaling/horizon/reconstruct -> export
  +-------> jobsize -> jobsize_folds --------------------> workload
```

`run_pipeline/stages/baselines.py` is the historical bootstrap-reference stage;
algorithm baselines live under `baselines/` and are fitted through the regular
`fit` stage.

Scientific calculations stay in domain modules. Stage modules load inputs, call
those calculations and write artifacts through an injected `PipelineContext`.

## Job size

`jobsize/README.md` documents the work-unit definition and the frozen marginal,
composition and two-part laws. Arrival generation remains size-agnostic.
