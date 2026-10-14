# jobsize

Job durations to work units, and the size laws fitted to them. The arrival
generator does not depend on this package; the two meet only in `assign_work`.

A size is the work the cluster absorbed, over completed, failed and cancelled
jobs: the job's duration inverted through an A100 latency model at the trace's
operating point, so it does not depend on the GPU count the job ran on.

| module | contents |
|---|---|
| `work.py` | the A100 inversion and the bin activity of each job |
| `laws.py` | marginal laws: lognormal mixture, scipy families, the empirical pool |
| `composition.py` | `CompositionJobSize`: a mixture whose weights move with bin activity |
| `two_part.py` | `TwoPartJobSize`: the bulk drawn per job, the top 1% placed per bin |
| `conditional.py` | `ConditionalJobSize`: the alternative where activity moves the level |
| `latent.py` | `SizeLatent`: a bin-level latent, off by default |
| `folds.py` | rolling-origin validation and `assign_work` |

The `jobsize` stage writes three laws to `results/`: the marginal three-component
lognormal mixture (`jobsize_model.json`), the conditional law
(`jobsize_conditional.json`) and the two-part law the workload uses
(`jobsize_two_part.json`). Three components is a modelling choice; neither AIC,
BIC nor held-out likelihood selects the number.
