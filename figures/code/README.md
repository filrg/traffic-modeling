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
| `arrivals.py` | arrival fidelity: occupancy, burstiness across scales, memory | `series`, `series-all`, `series-all-best`, `idc-scale`, `acf` | `series` |
| `excursion.py` | excursion structure | `daily-peak`, `daily-peak-range`, `daily-peak-bands`, `daily-peak-strips`, `burst-rate` | `daily-peak` |
| `gaps.py` | the gap side and the placement layer | `cv-window`, `idi`, `idc-ceiling`, `iat-ccdf`, `ccdf-placement`, `gap-cdf`, `clustering` | `cv-window` |
| `generator.py` | load and horizon | `load-arrivals`, `load-work`, `load-idc`, `load-runs`, `horizon-runs`, `combined` | `load-arrivals` |
| `jobsize.py` | job sizes (separate pipeline): the fitted law, and what the generator draws | `pdf`, `cdf`, `ccdf`, `draws-cdf`, `work-per-bin` | `pdf` |
| `jobsize_regimes.py` | job-size regimes and their activity dependence | `components`, `weights`, `tail-rate`, `level` | `components` |
| `studies.py` | supporting studies, no figures | `rate-error`, `horizon`, `load-scaling`, `latent-frontier` | `rate-error` |

`_common.py` holds the palette, the house rcParams, the artifact manifest and the
shared CLI. It is not runnable. It no longer holds the ensemble rules: which run
is `representative`, how one is `redraw`n at another load, and the rate
calibration live in `seren_countfirst/ensemble.py`, because they are defended in
the paper and are pinned by tests. `_common` re-exports them, so a view still
calls one name.

Within a module, a `# ---` banner opens each family it absorbed, and views keep
their names: `jobsize`'s `cdf` is the fitted law's, `draws-cdf` the generator's.

`make_all.py` renders every module. **Modules are named for what they show, never
for a panel number** — panel numbers move between drafts, and the output
filenames, which the manuscript cites, do not change with them.

The two system-overview diagrams are hand-built SVG rather than matplotlib and sit
one level up, in `figures/`. They were left as two self-contained scripts on
purpose: only four of their nine shared helpers are actually identical, so folding
them into a shared library would move about 25 lines and put a manuscript asset at
risk for it.

## Running

Modules run from anywhere: each prepends its own directory to `sys.path`.

```bash
python figures/code/excursion.py                        # the default view
python figures/code/excursion.py --style burst-rate     # one named view
python figures/code/excursion.py --all                  # every view in the module
python figures/code/excursion.py --list                 # what the module offers

python figures/code/jobsize_regimes.py --style weights --error-bars      # add the intervals
python figures/code/jobsize_regimes.py --style tail-rate --no-error-bars # drop them
```

Each view that carries a **measurement's** interval sets its own default and the
two flags override in either direction, through `_common.show_ci`. `tail-rate`
draws them (one series of seven points; without them a flat line through
scattered dots asserts more than it shows) and `weights` does not (three series
of eight, whose intervals overlap into a smear). The csv twin always holds the
bounds whether or not they are drawn.

The flags do not touch `level`, whose band is the panel's whole argument, nor the
model-spread bands in `jobsize`'s draw views and in `generator`, which are a different
object: those hold 95% of an ensemble rather than bounding an estimate.

Everything reads the frozen pipeline state, so run the pipeline first:

```bash
python -m seren_countfirst.run_pipeline data
python -m seren_countfirst.run_pipeline fit mm4 onoff ingarch nogate
python -m seren_countfirst.run_pipeline evaluate     # sets state["winner"]
python -m seren_countfirst.run_pipeline jobsize      # only for jobsize.py
```

To regenerate every figure, and to find out what is missing first:

```bash
python figures/code/make_all.py --check   # ok / MISSING per input, and the stage that writes it
python figures/code/make_all.py           # every view of every module, plus its csv twin
python figures/code/make_all.py --module generator
```

`--check` is on every module too. Each declares the artifacts it needs as keys in
`INPUTS`; `_common.ARTIFACTS` maps a key to its path and its producing stage, and
is the only place a figure input path is written down.

## Why some modules have several views

Views are *alternative visualisations of the same claim*.

The daily-peak claim has four views because no single treatment carries both ends
at once: the quantile curve is complete but must be read off a curve, the range
bars put both ends at a glance, the band bars give frequencies a reader can quote,
and the strips show the raw alternation with no statistic in between. `burst-rate`
is the same question asked per bin rather than per day.

**`excursion`** uses several daily-peak views to show both quiet-day coverage and
busy-day magnitude. Quantiles give the full distribution, range bars show extremes,
band bars show frequencies, and strips show the raw sequence. `burst-rate` examines
the same behaviour per bin.

**Neither of `generator`'s claims is a fidelity claim.** The trace holds one load
level and one duration, so it cannot validate either; both are statements about
how the generator behaves when its knobs move. `load-arrivals` and `load-idc` are
drawn from the *same* runs at the same per-path seeds, so the two panels sit side
by side as one experiment: the knob delivers volume linearly, and it makes the
workload relatively spikier as it does so. `combined` is the compact two-panel
version for the paper, at its own path counts.

**`generator`** combines two parts of one load experiment: `load-arrivals` shows 
linear volume scaling, while `load-idc` shows increasing burstiness. `combined` is the compact paper version.

`load-work` is the panel `load-arrivals` is meant to sit beside, and it is the
one place the generator's headline knob stops behaving. Arrivals scale linearly
because `_eta_from_states` puts `log(alpha*mean)` in the intensity; **work does
not**, because the size law reads bin activity and activity is what the knob
changes. At α = 16 offered work is **0.25×** what a linear reading implies.

**α is arrival-side only, and `assign_work` is called without one** — settled by
the author on 2026-08-08. The size law is already conditional on `N_k`, so the
knob reaches offered work through the counts alone; the superposition reading,
which divides a bin's occupancy among α notional streams before the size model
sees it, is a second knob with no support in the trace and is dropped. Earlier
sweeps in `results/workload_by_alpha.csv` carry a `reading` column from before
that decision; the `intensity` rows are the live ones.

**`load-runs` is drawn to stand beside `arrivals`'s `series`, and shows that
figure's realisation rather than a second draw.** `series` carries α = 1, so this one
omits it and extends the same run to 0.5 / 2 / 8 / 16 — the reader recognises the
activity pattern across both figures and reads load alone. The ladder is a
regular ×4 that stops at 16 because 16 is the ceiling of every α sweep here; the
closing ×2 step is that ceiling, not a choice. Measured on the drawn run, the
panel-to-panel change in zero fraction is 0.093 / 0.114 / 0.061, against
0.093 / 0.057 / 0.118 for a 0.5-2-4-16 ladder, whose flat step would fall in the
middle of the figure instead of at its end.

Every panel is scaled by its own α, so a purely proportional response would draw
four identical panels: what is left to see is the departure. Empty bins fall
0.822 → 0.554 while peak per unit load is flat from α = 2 up (1712 / 1745 / 1752
batches/h), which is the run-level picture of load filling empty bins rather than
making occupied ones proportionally busier.

**Do not fit an exponent here and call it a scaling law.** Every term in the
per-job mean is a bounded convex combination, so offered work stays bracketed by
two lines through the origin and tends to linear as the mixture weights
concentrate. The shortfall is a transient, and any slope read off this α range is
local to it.

Most of the shortfall is the tail: `large_work_share` falls 0.76 → 0.42 from
α = 1 to 16 and `large_job_frac` 0.016 → 0.003, because `ν(N_k)` is flat in
`N_k`, so a bin sixteen times busier gets essentially the same number of huge
jobs. **That is an open claim, not a result** — the flat rate was measured within
the trace at one load, and extrapolating it to a sixteen-fold cluster-wide rise
is something the trace cannot settle. Either defend it in the text or let the
tail rate scale.

This view is far heavier than the count-only ones — it reconstructs events and
composes sizes at every load factor — so `--all` on this module is minutes, not
seconds.

**Linear in the horizon is a different question, and there the answer is yes.**
The count model is stationary and the size model reads bin activity, whose
distribution does not depend on where in the run a bin sits, so expected work is
`T` times expected work per bin by construction. Measured over 30 / 90 / 138 /
365 days at unit load it comes out at an exponent of 0.975, consistent but not
tight: per-path CV is 1.01 at 30 days and still 0.24 at 365, so a single short
run says very little. The knob and the clock are not symmetric — the clock
changes how many bins you draw, the knob changes what is inside them, and sizes
read that.

**`gaps`** answers one question four ways: what the classical inter-arrival
statistics say about this trace, and what they cannot say. Every view is a
statistic of an **event stream**, so the module reconstructs timestamps and holds
its runs at the trace's volume — a gap statistic reads the rate directly, and the
winner's per-path volume spans an order of magnitude, so an unmatched ensemble
would report the volume spread as a gap result.

Its two renewal arms are the point of the module and are not interchangeable.
`Gamma renewal` is the law the serving-trace literature actually fits, and it
fails on the marginal as well as the dynamics. `Renewal, trace gaps` resamples
the trace's own gaps iid — the best an *iid renewal* model could ever do, with
zero estimation error on the marginal — so it isolates what is lost by assuming
independence alone. It does not bound gap-first modelling in general: a
dependent-gap law (ACD, a Gaussian-copula gap sequence) is still gap-first and is
not held under the ceiling. What these arms refute is the independence
assumption, not the choice to model gaps.

`cv-window` is the only view drawn on a measurement window rather than a process
property: the same trace reads CV 1.03 at five minutes and 6.13 at a week, so a
single reported CV is a statement about the window. **It draws two series, trace
and GLI-NB, and neither renewal arm.** The fitted one is flat at 2.15 and the
empirical reaches 12.8, which would take three quarters of the axis; both stay in
the csv twin, and `idi` is where a renewal arm has to be seen, flatness being the
whole claim there. Do not read the rising trace curve as a renewal rejection — a
heavy-tailed renewal rises here too. The rejection is `idi`.

`idi` and `idc-ceiling` are the same statement from the two sides. A
finite-variance renewal process is pinned to `CV²` at every block length and
converges to `CV²` at every aggregation scale, so one draws a flat line and the
other a ceiling. The trace clears both by a factor of about ten. Both qualifiers
are load-bearing: infinite gap variance escapes the ceiling, and dependence
escapes the flat line.

`iat-ccdf` **also draws two series, trace and GLI-NB.** The fitted laws are still
computed, written to the csv twin and scored in the printed KS table — the panel
shows the fit and the numbers carry the comparison. Drawn, they lose the tail
entirely by 10⁴ s while the generator holds it to 10⁶ s, which cost the panel
three curves for a claim two numbers make: **KS 0.055 against a fitted
lognormal's 0.129**, so the generator beats the best fitted law by 2.4× on a
marginal it never fitted. It read 0.205 while the placement layer drew offsets
independently; the 0.055 is on `PatternTemplates`, and since every view in this
module reconstructs, all of them moved when the placement default changed. The
manuscript quotes neither number: as a two-series panel it claims that the design
reproduces inter-arrival behavior, not that it beats a fitted law, and it carries
the mechanism instead — the placement rule sets the short gaps and the count
model's runs of empty bins set the long ones, which `gap-cdf` measures. Quote the
KS pair only if the stronger claim is made.

`ccdf-placement` and `gap-cdf` are **diagnostics, not manuscript panels**: the
same four series drawn as survival and as its complement, with the placement the
generator shipped against the independent offsets it replaced, paired on the same
count paths and the same seeds. They exist to show that the axis choice decides
what is visible. On the log-log survival the two placements lie on top of each
other, because both reproduce the tail and the tail is what that axis resolves.
On the linear CDF they separate by 13 points at one second. Above about 100 s the
two are indistinguishable (0.863 against 0.865) — those are across-bin gaps and
belong to the count layer, so the panel also draws the boundary between the two
layers. Prefer `iat-ccdf` for the paper; it is the two-series panel and carries
the fitted-law comparison as KS numbers, where this pair only re-cuts the same
ensemble. Both still draw four series, so they are the place to look when the
question is what the placement layer changed.

**`clustering`** carries the reason `iat-ccdf` moved, and is the only view here
drawn against an exact reference rather than a baseline. n points dropped
independently into a bin leave gaps that are Beta(1, n) on it, so the median over
the mean spacing is `(1 − 2^(−1/n))(n + 1)` — 0.879 at n = 2, ln 2 from about
n = 50 up, and **never 1**. `Independent offsets` lands on that curve at every
bucket from n = 8 (0.742 / 0.711 / 0.703 / 0.710 against 0.733 / 0.713 / 0.704 /
0.698), which is what establishes that the layer it replaced was doing
independent placement rather than merely resembling it. The trace runs 0.35-0.53
across the occupancies that carry its batches, and `Pattern resampling` tracks it
to within 0.05.

Two features of that panel are the grid rather than the workload. The rise above
n ≈ 64 is saturation — a bin holding 530 batches in 900 one-second slots is 59%
full and has no room left to cluster in — and in the `256+` bucket all four
series collapse onto one value because it holds two bins and the grid dictates
the placement. Both are kept rather than trimmed: the axis stops where the trace
stops.

Runs are on the trace's own counts, so the occupancy axis carries the same bin
membership in every series and the panel isolates placement from the count model.

**`jobsize`'s law views** are `pdf` and `cdf`: the PDF is truncated at the 99th percentile for readability, while the CDF covers the full seven-decade range.

**`jobsize_regimes`** pairs the two halves of the bulk model. `components` draws
each fitted component's marginal law as a violin, truncated at the cut because
that is how the bulk is drawn, with the observed exceedances above it — so the
two-part split is visible rather than described.

`weights` draws how the bulk's composition moves with activity, trace against
model. Every bulk job is sorted into the size band where one component's fitted
density is highest — under 19,480 units, then to 89,089, then to the cut — using
a prior fixed at the activity-averaged weights. That makes the rule **one
partition of the size axis applied to every job whatever its bin**, so the trend
the shares show is measured rather than assumed. The model side is the same count
on its own draws at each activity, so both sides are one statistic, as in
`tail-rate`. When the intervals are drawn the markers are nudged apart on x by
`SHARE_DODGE`, without which the three components' intervals overlap into a
smear; bare markers sit at the stratum's own activity.

**Its y-axis says $\pi_c(N_k)$, and that symbol has to be defined as the share of
jobs falling in component $c$'s band, not as the mixing weight.** The two are not
the same and are not close: at $N_k = 1$ the fitted logit reads 0.258 / 0.177 /
0.565 while the realised band shares are 0.262 / 0.376 / 0.361, because component
3 is wide (log-sd 1.01) and spills into component 2's band. Largest gap over the
range is 0.20. Plotting the logit instead would leave the trace side with no
model-free counterpart, which is the whole reason the panel was rebuilt — so the
quantity is right and the definition has to follow it.

It does not fit everywhere, and that is worth keeping visible: the model is
inside the trace's interval in 19 of 24 cells, but per component 8 / 7 / **4** of
8. Component 3 is the miss — the model holds too many large bulk jobs in busy
bins (0.085 against a measured 0.029 at $N_k = 187$) and too few in quiet ones
(0.239 against 0.342 at $N_k = 6$). Component 2's measured share rises 0.24 to
0.43 while the model's stays flat near 0.385.

The intervals widen sharply to the right, and not because those strata hold fewer
jobs — they are equal-count. The top stratum's 8,223 jobs come from **12 days**
against 113 for the quietest, so busy bins are concentrated in time and a day
resample moves them a long way. The same concentration is what `level` runs into.

An earlier version of this panel drew the fitted logit itself against shares from
the model's own **posterior** assignment. That assignment read the bin's
activity, so the points partly reproduced the curve by construction. Do not go
back to it.

`tail-rate` is the companion `weights` is drawn beside: one panel per half of the
two-part law's activity dependence, both measured against fitted, both on $N_k$.
The bulk's mix moves with activity; the tail's rate does not, and that is the
point. The fitted rate spans a factor of 1.07 over the whole range of $N_k$
against an observed spread of 1.78, and sits inside the day-block interval in all
7 strata. At the 90% interval this was drawn with first it was 6 of 7, missing at
$N_k = 23$; the interval widened, the evidence did not. What makes flat worth a
panel is the alternative it rules out:
placing the top 1% per job instead would run 0.010 to 1.041 across the same
strata, a factor of 101. That number is printed, not drawn — the panel is
observed against fitted and nothing else.

Its strata are equal in **bins**, not in jobs as `weights` uses, because the rate
is per bin; that is why its points sit left of the other panel's on a shared
axis. Most bins are quiet, most jobs are in busy ones. Activity is a small
integer at the low end, so `ACTIVITY_BINS` collapses from 8 strata to 7 here and
cannot be split finer.

It draws in `ALT_COLOR`, the fourth validated hue, rather than a component slot:
placed beside `weights` a blue series here would read as component 1. Points and
curve share that colour because they are two views of one quantity, where in
`weights` colour has to carry the component instead and the mark type separates
the sides.

`level` is a fourth view and is **not** part of that pair — it answers whether
activity should have entered the level instead. It carries the trace's binned
medians with a day-block interval spanning a factor of 2.7 to 33, and both
designs sit inside all eight: within the range the trace covers, this evidence
does not separate them. They separate past it, which is why the panel is read to
`LEVEL_MAX` and marks where the trace stops. Do not present it as a fit
comparison — the argument is that only one of the two designs is bounded.

Day blocks, in both: sizes and large-job arrivals cluster inside a day, and a
per-row resample would report either statistic as far more precise than it is.
Intervals are 95%, and `N_BOOT` is sized so the 2.5% edge is an order statistic
well inside the resamples drawn rather than one of the few most extreme. They are
labelled "95% CI" rather than following `generator`'s "95% of runs" or
`jobsize`'s "95% of draws" — those bands really do hold 95% of an
ensemble's members, whereas these are intervals on a statistic, and "95% of day
blocks" would read as a proportion of the days. The caption carries the method.

The `components` and `weights` panels report *different* quantities and should
not be read across: `components` labels each violin with the batch-weighted
**prior** weight averaged over the trace's activity, while `weights` plots the
**realised** share of jobs landing in each size band, which differs because the
components overlap — a job drawn from one can land in another's band.

`components`' four labels are deflated onto one denominator so they partition the
pool and sum to 100: the mixture weights are shares of the *bulk* (0.496 / 0.307
/ 0.197) and the tail is a share of *every* job, so the printed labels read
0.4912 / 0.3036 / 0.1952 / 0.0100. `_pct_labels` rounds by largest remainder,
because rounding each independently can land on 99 or 101 and the panel's claim
is that these four are exhaustive. Its tail takes `ALT_COLOR`, matching
`tail-rate`, and its components the same slots `weights` uses, so a reader moving
between the three panels keeps one colour key.
The violin density itself is verified against what `rvs` generates —
KS 0.002-0.004 per component — so the marginal truncation drawn there is a
faithful stand-in for the per-batch renormalised truncation the model applies.

**`jobsize`'s draw views** — `ccdf` and `draws-cdf` — are the goodness-of-fit
half of the module and are not more law candidates: the law views compare the
trace against *candidate analytic marginals* and are where the family is chosen,
while these compare it against what the *frozen two-part generator draws*, on the
trace's own batch structure. The two are the same 200 redraws read from opposite
ends, and the choice
between them is not cosmetic. `ccdf` is on log-log because that is the only way
the tail appears at all: on `cdf`, every job above $S^\star$ — the 640 that carry
about 40% of all work — is compressed into the **top 0.010** of the axis. `cdf`
resolves the body instead, where the largest vertical gap is 0.0246 at 2,602
units. Neither shows both, and `cdf` shows two curves lying on top of each other,
so it demonstrates agreement without demonstrating anything else. Two things the
`ccdf` panel is drawn to keep visible: the model
is plotted over the trace rather than under it, since the two coincide closely
enough that the usual order would hide one; and $S^\star$ is marked, because
above it the model resamples the observed exceedances instead of fitting them,
so that stretch matches by construction. The window is also the one the law was
fitted on — `results/jobsize_folds_summary.csv` holds the held-out arm.

`work-per-bin` is the view that earns the size model its keep, and neither
distribution view can stand in for it. **On the marginal, the whole two-part
construction buys almost nothing** — KS 0.026 against 0.028 for sizes drawn iid
from the frozen marginal mixture — so a reviewer looking at `cdf` or `ccdf` is
entitled to ask why the machinery exists. It exists because the generator feeds a
scheduler with work arriving over time, not with a histogram, and drawn per job
the sizes cannot help tracking the job count:

| median work per bin | quietest stratum | busiest | span | total work |
|---|---|---|---|---|
| trace | 214,959 | 3,413,432 | **15.9×** | 1.000 |
| two-part | 91,157 | 4,831,176 | 53.0× | 0.998 |
| iid marginal | 22,712 | 8,927,869 | **393×** | **0.746** |

The iid arm is 0.11× the trace in the quietest bins and 2.62× in the busiest, and
delivers a quarter less work overall, because a likelihood fit need not reproduce
the sample mean — `marginal_mean_ratio` is 0.745, and the two-part law solves its
level against composed work instead. The two-part arm is not exact either: 0.42×
at the low end and 1.42× at the high one. Caption it as a large win over iid with
a residual miss at both extremes, not as a match.

Strata are equal in **bins**, as in `tail-rate`, because work per bin is per bin.
No intervals are drawn: the effect is a factor of 25 larger than the day-block
uncertainty on the neighbouring panels, and three lines with bars would be
unreadable.

Three job-size distribution files now sit one word apart, and they are not
interchangeable. Check which one an `\includegraphics` means:

| file | module | trace against |
|---|---|---|
| `fig_jobsize_cdf` | `jobsize` `cdf` | candidate **analytic laws** — where the family is chosen |
| `fig_jobsize_ccdf` | `jobsize` `ccdf` | the **generator's draws**, log-log, tail visible |
| `fig_jobsize_fit_cdf` | `jobsize` `draws-cdf` | the same draws, linear-y, body visible |

`components` and `weights` read the component count from the frozen model, so
refitting at a different k regenerates them unchanged. Past four components the module raises
rather than cycling the categorical hues, since a repeated hue would give two
components the same identity — draw small multiples instead.

**`studies`** emits tables and csv, no figures. It backs the method section's
claims about load, horizon and feasibility, and sits here rather than in
`seren_countfirst/experiments/` because it shares the model-loading and the view
CLI with the figures; `load-scaling` reuses `rate-error`'s autocorrelation
machinery.

## Shared ensembles

Views in the same module reuse simulations instead of repeating them. `excursion` shares 200 paths per model, 
`generator` shares one 600-path load sweep, and `jobsize`'s law views load the pool and scoreboard once.

**`arrivals`'s `series`, `excursion` and `generator`'s `load-runs` share one
ensemble across modules**, at `_common.DISPLAY_OFFSET` and
`_common.DISPLAY_CANDIDATES`, and every figure showing a single run picks it
through `_common.representative`: closest to that model's *own* median daily-peak
quantiles and hourly autocorrelation, re-ranked among runs at the trace's volume
where the typical one does not carry it. Model-internal by design, so one rule
serves all four series. The pool size is shared rather than repeated per module
because the rule ranks against the pool's own median, so a module drawing a
different number of candidates would silently display a different run.

`load-runs` reaches other load factors through `_common.redraw`, which restores
the stream state that produced the displayed run. `simulate` draws the latent
before the counts and α reaches only the intensity and κ, so the latent path is
identical across load factors and the counts alone respond. **Re-seeding per α
does not pair anything past the first path** — the count draws consume a variable
number of values — which is the same trap `_load_sweep` documents.

Settled in review 2026-08-09, each clause paying for a failure found there: the
two modules drew different runs under different rules; ranking on mean rate alone
picked a run holding 93% of its volume in one quarter; peak quantiles alone are
blind to memory and drew `ingarch` at the lowest autocorrelation of its 200 runs;
and `gated_nogate`'s typical run carries 0.42 of its own ensemble mean, which put
shape and volume in one panel with no way to separate them. **Do not switch the
reference to the trace for every series** — it finds a `gated_nogate` path with
the trace's 88 quiet days by running 46% hot at a zero fraction of 0.454 against
0.775. `closest_to_trace` exists for `series-all-best` only, where it handicaps
the winner and never flatters it.

`series-all` is the paper figure. `series-all-best` is the same stack with each
baseline at its best run at volume, kept as the robustness check behind the claim
that no baseline is rescued by a lucky draw.

## Numbers behind the choices

Kept here rather than in the source, where a measurement goes stale silently and
a reader cannot act on it.

**Palette.** The four model slots clear all-pairs CVD ΔE 9.2 against a target of
8, and normal-vision 16.3 against a floor of 15, on white paper — which is why
hue alone carries identity and no dash relief is owed. Violet is categorical slot
7, not slot 4: as a fourth, yellow / magenta / red fail the normal-vision floor
at 13.7 / 12.9 / 7.1, and green fails CVD at 3.2. Revalidate before adding a
fifth series.

**Run selection.** `representative` ranks on daily-peak quantiles *and* the hourly
lag-one autocorrelation. Ranking on mean rate alone, which `series` did before,
buys the level by packing volume into one stretch: it selected a run holding 93%
of its batches in the first quarter, with twice the trace's count of empty days.

**`load-arrivals` error bar.** Wide (~6%) because a single run is: per-path CV is
0.76 at this horizon, since the gate latent's 8.2-day timescale leaves only ~17
independent stretches in 138 days. Path totals are right-skewed, so the drawn
normal-approximation interval is only trustworthy where a percentile bootstrap of
the mean agrees with it.

**`idc-scale` log axis.** The field spans 7.7 to 4423. On a linear axis the three
alternatives collapse at the bottom, hiding that ON/OFF matches the trace at 15 min
(73.1 against 73.2) and only then plateaus.

**`gaps`.** The renewal arms are in the csv twin, not the `cv-window` panel:
the fitted one is flat at 2.15 and the empirical reaches 12.8, three quarters of
the axis, and neither carries that panel's claim. `idi` is where a renewal arm has
to be seen, flatness being the whole statement there. In `iat-ccdf` the three
fitted laws are computed and scored but not drawn — drawn, they lose the tail by
1e4 s and cost three curves for a claim two numbers make. The `ccdf-placement`
inset shows the body, which carries 94% of the gaps.

**Resample and grid sizes.** `excursion` has 138 whole days, so the finest
resolvable quantile is about 1/138 and `REPORT_Q` stops at 0.99.
`jobsize`'s draw views use 200 redraws, which is what the band's 2.5% edge needs
to stop moving. `jobsize_regimes` uses 2000 day-resamples, so the 2.5% edge is the
50th order statistic rather than a handful of extreme draws, and nudges component
markers by 0.05 decades. `jobsize` reads the law CDF off a log-spaced grid because
39e3 steps exceed what a 3.5 in panel resolves.

**`clustering` reference.** `(1 - 2**(-1/n))(n + 1)` is 0.879 at n = 2 and reaches
ln 2 from about n = 50 up — a curve in n, and never 1.

## House style

`_common.py` defines the palette, rcParams, axis styling, and legend helpers. 
Figure conventions are documented in `docs/results.md` §4b and should be changed there first.

Measurements use solid black; models use coloured dashed lines. The four-colour palette is validated 
for colour-vision accessibility and print contrast. Revalidate it before adding another series.

Series are labelled by model name—GLI-NB, LI-NB, ON/OFF, and NB-INGARCH—rather than “proposed model.” The caption identifies the authors’ model.
