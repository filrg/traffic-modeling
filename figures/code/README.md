# Figure code

Figure modules are grouped by scientific objective, not manuscript panel number.
Every view writes a PDF, PNG, and CSV twin while preserving the published
filenames.

## Modules

| module | purpose | views | default |
|---|---|---|---|
| `arrivals.py` | occupancy, burstiness, and memory | `series`, `series-all`, `series-all-best`, `idc-scale`, `acf` | `series` |
| `excursion.py` | daily peaks and burst excursions | `daily-peak`, `daily-peak-range`, `daily-peak-bands`, `daily-peak-strips`, `burst-rate` | `daily-peak` |
| `gaps.py` | gaps and within-bin placement | `cv-window`, `idi`, `idc-ceiling`, `iat-ccdf`, `ccdf-placement`, `gap-cdf`, `clustering` | `cv-window` |
| `generator.py` | load and horizon behaviour | `load-arrivals`, `load-work`, `load-idc`, `load-runs`, `horizon-runs`, `combined` | `load-arrivals` |
| `jobsize.py` | size-law and workload fidelity | `pdf`, `cdf`, `ccdf`, `draws-cdf`, `work-per-bin` | `pdf` |
| `jobsize_regimes.py` | size components and activity | `components`, `weights`, `tail-rate`, `level` | `components` |
| `studies.py` | supporting tables, no figures | `rate-error`, `horizon`, `load-scaling`, `latent-frontier` | `rate-error` |

## Running

```bash
python figures/code/make_all.py
python figures/code/make_all.py --check
python figures/code/make_all.py --module arrivals

python figures/code/excursion.py
python figures/code/excursion.py --style burst-rate
python figures/code/excursion.py --all
python figures/code/excursion.py --list
```

Measurement intervals can be overridden with `--error-bars` or
`--no-error-bars`. Model-spread bands are part of the view and are not affected
by those flags.

## Inputs and outputs

`_common.ARTIFACTS` is the single map from an input key to its path and producing
pipeline stage. Figure modules declare `INPUTS`; they do not duplicate paths.

Outputs are written under:

- `figures/pdf/`
- `figures/png/`
- `figures/csv/`

`make_all.py --check` reports missing inputs without rendering.

## Reproducibility

- Shared views reuse the same simulation pool.
- Single-run panels select through `seren_countfirst.ensemble.representative`.
- Load comparisons redraw the same latent path through
  `seren_countfirst.ensemble.redraw`.
- Seeds come from the pipeline seed; a figure must not introduce an unseeded RNG.
- Changing pool size, seed offsets, or selection metrics is a numerical change,
  not a layout refactor.

## House style

`_common.py` owns the palette, rcParams, legends, artifact lookup, and shared
CLI. Keep view-specific calculations in the module that owns the view.

- Measurements: solid black.
- Models: coloured dashed lines.
- Model labels: GLI-NB, LI-NB, ON/OFF, and NB-INGARCH.
- A fifth categorical colour requires accessibility and print-contrast checks.
- Use one module per objective and one function per view.
- Keep empirical measurements and manuscript argument out of source comments.

The two system-overview diagrams live in `figures/` as standalone SVG source
scripts because they are manuscript assets rather than data-driven views.
