# Figure code

One module per objective. Each view writes a PDF, a PNG and a CSV of its data to
`figures/pdf/`, `figures/png/` and `figures/csv/`.

| module | views (the first is the default) |
|---|---|
| `arrivals.py` | `series`, `series-all`, `series-all-best`, `idc-scale`, `acf` |
| `excursion.py` | `daily-peak`, `daily-peak-range`, `daily-peak-bands`, `daily-peak-strips`, `burst-rate` |
| `gaps.py` | `cv-window`, `idi`, `idc-ceiling`, `iat-ccdf`, `ccdf-placement`, `gap-cdf`, `clustering` |
| `generator.py` | `load-arrivals`, `load-work`, `load-idc`, `load-runs`, `horizon-runs`, `combined` |
| `jobsize.py` | `pdf`, `cdf`, `ccdf`, `draws-cdf`, `work-per-bin` |
| `jobsize_regimes.py` | `components`, `weights`, `tail-rate`, `level` |
| `studies.py` | tables only: `rate-error`, `horizon`, `load-scaling`, `latent-frontier` |

```bash
python figures/code/make_all.py                    # every view
python figures/code/make_all.py --check            # missing inputs, and the stage that writes each
python figures/code/arrivals.py --style acf        # one view; --all, --list, --error-bars
```

Measurements are solid black and models coloured dashed lines, labelled GLI-NB,
LI-NB, ON/OFF and NB-INGARCH; ON/OFF and NB-INGARCH are the `onoff_long` and
`ingarch_long` calibrations. Single-run panels use
`seren_countfirst.ensemble.representative`, and every random draw is seeded from
the pipeline seed. `_common.py` holds the palette, the input paths and the shared
command line.
