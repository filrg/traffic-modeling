# jobsize

Durations to work units, and the laws fitted to them. Separate from the arrival
generator: the two share the trace and nothing else, and the arrival generator
stays size-agnostic.

A size is **the work the cluster absorbed**, never seconds. A duration is what a
job took on the GPU count the scheduler happened to give it, so it is inverted
through the simulator's own A100 latency model at the trace's operating point.
That inversion is an identity, not an estimate, and the stage raises if it fails
to round-trip.

| module | holds |
|---|---|
| `work.py` | the A100 inversion, the activity attachment, the batch and bin index helpers |
| `laws.py` | the marginal laws: lognormal mixture, the scipy candidates, the empirical pool, and their json readers |
| `latent.py` | `SizeLatent`, a two-scale stationary bin level, and the within-batch variance it is calibrated against |
| `composition.py` | `CompositionJobSize` — the frozen bulk law: the component belongs to the job, and activity moves the weights, not the level |
| `conditional.py` | `ConditionalJobSize` — the superseded design where activity moves the level, kept as the recorded alternative |
| `two_part.py` | `TwoPartJobSize` — bulk drawn per job, the top 1% placed per bin at a rate free of the job count |
| `folds.py` | rolling-origin windows, per-window scoring, and `assign_work`, the one function the arrival half calls |

## Three laws are frozen, not one

- `results/jobsize_model.json` — the marginal: a 3-component lognormal mixture.
- `results/jobsize_conditional.json` — adds the batch level and the activity slope.
- `results/jobsize_two_part.json` — the bulk/tail split that the workload stage uses.

Always say which fit a number came from: dropping the top 1% moves the largest
component's median from 47,475 (marginal fit, all jobs) to about 120,000 (bulk
fit, sub-threshold jobs).

`k = 3` is a declared modelling choice, not a selection. AIC and BIC keep
improving through k = 6, and held-out likelihood cannot separate any k from any
other. `docs/design-notes.md` 3c carries the measurements.

`SizeLatent` is real and is off by default: it cannot be added to the composition
bulk additively, because the variance budget is already full. The principled
version drives the weights.
