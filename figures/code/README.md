# Figure code

One module per **objective**, not per plot. The grouping follows the metric
families in `docs/results.md`: what a figure is *for* decides where it lives, and
several ways of drawing the same claim are views inside one module rather than
separate files.

Outputs go to `figures/pdf/`, `figures/png/` and `figures/csv/`. Every view ships
a csv twin. Filenames are unchanged from the one-file-per-plot layout, so any
`\includegraphics` still resolves.

## Modules

| module | family | views | default |
|---|---|---|---|
| `f1_occupancy.py` | F1 occupancy | `series` | `series` |
| `f2_burstiness.py` | F2 burstiness across scales | `idc-scale` | `idc-scale` |
| `f3_memory.py` | F3 memory | `acf` | `acf` |
| `f4_excursion.py` | F4 excursion structure | `daily-peak`, `daily-peak-range`, `daily-peak-bands`, `daily-peak-strips`, `burst-rate` | `daily-peak` |
| `g_generator.py` | G1 load, G2 horizon | `load-arrivals`, `load-idc`, `horizon-runs`, `combined` | `load-arrivals` |
| `jobsize_law.py` | job sizes (separate pipeline) | `pdf`, `cdf` | `pdf` |
| `analysis.py` | supporting studies, no figures | `rate-error`, `horizon`, `load-scaling`, `latent-frontier` | `rate-error` |

`_common.py` holds the palette, the house rcParams, state loading and the CLI.
It is not runnable.

## Running

```bash
cd figures/code

python f4_excursion.py                        # the default view
python f4_excursion.py --style burst-rate     # one named view
python f4_excursion.py --all                  # every view in the module
python f4_excursion.py --list                 # what the module offers
```

Everything reads the frozen pipeline state, so run the pipeline first:

```bash
python -m seren_countfirst.run_pipeline data
python -m seren_countfirst.run_pipeline fit mm4 onoff ingarch nogate
python -m seren_countfirst.run_pipeline evaluate     # sets state["winner"]
python -m seren_countfirst.run_pipeline jobsize      # only for jobsize_law.py
```

To regenerate every figure:

```bash
for m in f1_occupancy f2_burstiness f3_memory f4_excursion g_generator jobsize_law; do
    python $m.py --all
done
```

## Why some modules have several views

Views are *alternative visualisations of the same claim*.

**`f4_excursion`** uses several daily-peak views to show both quiet-day coverage and
busy-day magnitude. Quantiles give the full distribution, range bars show extremes,
band bars show frequencies, and strips show the raw sequence. `burst-rate` examines
the same behaviour per bin.

**`g_generator`** combines two parts of one load experiment: `load-arrivals` shows 
linear volume scaling, while `load-idc` shows increasing burstiness. `combined` is the compact paper version.

**`jobsize_law`** includes `pdf` and `cdf`: the PDF is truncated at the 99th percentile for readability, while the CDF covers the full seven-decade range.

## Shared ensembles

Views in the same module reuse simulations instead of repeating them. `f4_excursion` shares 60 paths per model, 
`g_generator` shares one 600-path load sweep, and `jobsize_law` loads its pool and scoreboard once.

## House style

`_common.py` defines the palette, rcParams, axis styling, and legend helpers. 
Figure conventions are documented in `docs/results.md` §4b and should be changed there first.

Measurements use solid black; models use coloured dashed lines. The four-colour palette is validated 
for colour-vision accessibility and print contrast. Revalidate it before adding another series.

Series are labelled by model name—GLI-NB, LI-NB, ON/OFF, and NB-INGARCH—rather than “proposed model.” The caption identifies the authors’ model.
