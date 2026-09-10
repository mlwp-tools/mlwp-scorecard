# mlwp-scorecards — a Python package for weather forecasting scorecards

## Context

`/Users/B280936/git-repos/mlwp/mlwp-scorecards/` holds one file: `scorecards-47r1ENS.html`,
a 7.4 MB ECMWF-generated scorecard page kept as a design reference. There is no package, no
git repo, no source.

The goal is a package that generates scorecards of that kind for machine-learning weather
prediction — an ML model against an NWP baseline, or two ML models head-to-head. It joins
three sibling packages under `/Users/B280936/git-repos/mlwp/` (`mlwp-data-loaders`,
`mlwp-data-specs`, `mxalign`; GitHub org `mlwp-tools`) and follows their conventions.

### What a scorecard is for

When a forecasting system changes — a new IFS cycle, a new ML model, a retrained
checkpoint — the question is rarely "is it good?" but "is it better than what we already
have, and where is it worse?" The evidence is thousands of numbers: every variable,
level, spatial region, metric and lead time. Nobody reads thousands of line plots.

A scorecard compresses that evidence onto one page that can be scanned in a minute. Its
job is to make three things immediately visible:

- **whether the change is a net improvement**, across every physical variable,
  spatial region and lead time at once;
- **where it regresses** — the handful of cells that got worse. This is usually the
  operative finding: a model that improves the global mean while degrading tropical
  precipitation at day 10 is not shippable;
- **whether either of those is real**, or is within sampling noise.

It is a decision artefact — an operational centre uses one to decide whether to promote a
cycle — and that sets the bar for the rendering. Direction must be unambiguous. Magnitude
must stay visible rather than saturating away at the extremes. A cell with no data must
look different from a cell with no difference. Significance must be distinguishable from
size — see *Significance is not size* below. Each of those constrains a design choice
here, and each is a way the reference implementation can be improved on rather than
copied.

### What a scorecard looks like

The simplest useful card: one grouping on each axis. One truth source (the analysis), no
spatial aggregation — a single global area — and four surface variables, so there are no
pressure levels either. Rows are **variable**, columns are **metric**, and each cell is a
short row of boxes, one per forecast lead time.

```
┌──────────╥─────────┬─────────┬─────────┐
│ variable ║   rmse  │   crps  │  spread │   ← one column grouping
├──────────╫─────────┼─────────┼─────────┤
│ 2t       ║ ▁▂▃▄▅▆█ │ ▁▂▄▅▆▇█ │ ░░▒▒▓▓▓ │
│ msl      ║ ▁▂▃▄▅▆▇ │ ▂▃▄▅▆▇█ │ ░▒▒▓▓▓▓ │
│ 10ff     ║ ▁▁▂▃▄▅▆ │ ▁▂▃▄▅▆▇ │ ░░▒▒▓▓▓ │
│ tp       ║ ▂▃▄▅▆▇█ │ ▁▃▄▅▆▇█ │ ░▒▓▓▓▓▓ │
└──────────╨─────────┴─────────┴─────────┘
     ↑          ↑
     │          └─ one cell: 7 lead times, T+24 → T+168, earliest on the left
  one row grouping

  rows    = ["variable"]
  columns = ["metric"]
```

Reading it:

- **One box is one lead time.** Left to right within a cell is the forecast growing older.
- **Colour is the difference** between the two prediction sources at that lead time — hue
  for direction, intensity for size.
- **A row is one thing being forecast**, a **column one way of measuring agreement.**
  `spread` is shaded differently because more spread is neither better nor worse — see
  *Significance is not size*.

Everything else is the same picture with more groupings nested onto the two axes. Adding
`level` inside `variable` and `truth_source` outside it on the rows, and `spatial_region`
outside `metric` on the columns, gives the shape of the ECMWF reference card (two regions
shown; it has ten):

```
┌─────────┬─────┬──────╥───────┬───────┬───────╥───────┬───────┬───────┐
│         │     │      ║         n.hem         ║         s.hem         │   ┐ depth 0: "spatial_region"
├─────────┼─────┼──────╫───────┬───────┬───────╫───────┬───────┬───────┤   ├ two nested column groupings
│  truth  │ var │level ║  rmse │  crps │  sprd ║  rmse │  crps │  sprd │   ┘ depth 1: "metric"
├─────────┼─────┼──────╫───────┼───────┼───────╫───────┼───────┼───────┤
│ analysis│  z  │   50 ║▁▃▅▆█▇▇│▂▄▅▇███│░░▒▒▓▓▓║▂▃▄▅▆▇█│▁▂▄▅▆▇█│░▒▒▓▓▓▓│
│         │     │  500 ║▁▂▄▅▇██│▁▃▄▆███│░░▒▓▓▓▓║▁▂▃▄▅▆▇│▂▃▄▅▇██│░░▒▒▓▓▓│
│         │     │  850 ║▂▃▄▅▆▇█│▁▂▃▅▆▇█│░▒▒▓▓▓▓║▁▁▂▃▄▅▆│▁▂▃▄▅▆▇│░░▒▓▓▓▓│
│         │ msl │   ── ║▁▂▃▄▅▆▇│▂▃▄▆▇██│░░▒▒▓▓▓║▂▂▃▄▅▆▇│▁▃▄▅▆▇█│░▒▓▓▓▓▓│
│ obs     │  z  │   50 ║▁▂▃▄▅▆▇│▁▂▄▅▆▇█│░░▒▒▓▓▓║       │       │       │
└─────────┴─────┴──────╨───────┴───────┴───────╨───────┴───────┴───────┘
     ↑       ↑     ↑                                               ↑
     └───────┴─────┘                                               └─ an empty crossing: no data at all
  three nested row groupings

  rows    = ["truth_source", "variable", "level"]
  columns = ["spatial_region", "metric"]
```

At full size that is 45 rows x 30 columns x 15 lead times — 20,250 boxes on one page.
Which grouping goes where is declared by the caller; nothing in the layout engine privileges
any of them, and `lead_time` can sit on the rows or columns just as readily as inside the
cell.

### The three data sources

Three, not two. Each is the same kind of thing — **a source of values for physical
variables at specific points in space and time** — and they differ only in how the
comparison uses them. Throughout, a **spatial region** is one named area the scores were
averaged over (`n.hem`, `europe`, `tropics`); it is always the `spatial_region`
coordinate, never anything else.

| Source | How the comparison uses it | In the reference |
|---|---|---|
| **truth source** | taken as the believed-true value; both predictions are scored against it | `an` (analyses) or `ob` (observations) — a *dimension*, both shown on one card |
| **control** | scored against that truth source; the baseline | IFS cycle 47r0 |
| **experiment** | scored against the *same* truth source; the one under test | IFS cycle 47r1 |

Because they are one kind of thing, how each source is used is a property of the
*comparison*, not of the data — so it is named at call time rather than baked into the
dataset, and one file can produce many cards.

A scorecard is therefore **not** a picture of one model against truth. It is a picture of
**two prediction sources, each already scored against a common truth source**, with the
*difference between their scores* driving the colour. "Better" on the page means better
than the other prediction source, not better than truth.

The reference confirms this in its own metadata — `reftypes=['an','ob']`,
`expvers=(cntrl:[...], exper:[...])` — and its legend reads "red = the experiment (esuite) is
worse than the control". Its structure, and the parts of it not worth copying, are recorded
in the appendix.

### Decisions taken

1. **Input is pre-computed verification statistics** — no scoring, no statistics, in this
   package.
2. **Two outputs**: self-contained interactive HTML, and static matplotlib PNG/SVG/PDF.
3. **Axes fully generic** — the caller names which coordinates nest on rows and on columns.
4. **CVD-safe palette is the default**; the exact ECMWF palette ships as an opt-in preset.
5. **Vendor a small local `ValidationReport`** — no `mlwp-data-specs` dependency.
6. **Walking skeleton first**: scaffolding + core + HTML backend end-to-end. Matplotlib
   backend, drill-down charts, pagination and the ECMWF preset come in pass 2.
7. **Significance is supplied, not derived** (see *Significance* below).

---

## What the input dataset contains

Not physical variables — **verification statistics**, one step downstream of a per-case,
per-gridpoint error field, after two collapses have already happened.

One element is:

> a summary statistic describing how well one **prediction source** agreed with one **truth
> source**, for one physical variable, aggregated over a spatial region and over a set of
> forecast cases, at one forecast lead time.

`ds.z.sel(truth_source="analysis", prediction_source="GraphCast", metric="rmse", level=500,
spatial_region="n.hem", lead_time="3 days", stat="mean")` → `41.7` — over all forecast cases in the
study, GraphCast's 3-day forecast of 500 hPa geopotential height had an RMSE of 41.7 m against
the analysis, averaged over the northern hemisphere.

| Quantity in the raw comparison | Fate in this dataset |
|---|---|
| latitude, longitude | **collapsed** (1) → `spatial_region` names the area averaged over |
| forecast case / initialisation date | **collapsed** (2) → an ancillary count variable records how many |
| the pointwise error itself | **collapsed** → `metric` names how it was reduced |
| forecast lead time | kept → `lead_time` |
| physical variable | kept → the data variable name, plus `level` |
| which forecast system | kept → `prediction_source` |
| what it was scored against | kept → `truth_source` |
| sampling uncertainty over cases | `stat` ∈ {mean, lower, upper} at `confidence` |

No gridpoints and no dates remain. It is `mxalign`'s verification output after both collapses
— the table you would otherwise stare at as a wall of line plots.

### The two collapses, and which one carries the uncertainty

They are different in kind, and only the second is a *sample*:

1. **Over space, within a single forecast case.** The spatial region's gridpoints reduce
   to one number per case, area-weighted by cos(latitude); the reduction is metric-specific
   (RMSE takes the root of the area-weighted mean squared error, CRPS the area-weighted mean
   of the pointwise CRPS). **No sampling uncertainty attaches here** — it is a deterministic
   reduction of that case's error field.
2. **Over forecast cases.** The initialisation dates reduce to a mean, which is `stat="mean"`.
   *This* is the sample: N weather situations drawn from the population of possible ones,
   whose mean score estimates the expected score with a standard error. The ancillary count
   variable holds N; `lower`/`upper` are its sampling uncertainty.

Confirmed against the reference: its `popul` falls by exactly 2 per 24-hour lead step in
13,447 of 14,154 transitions, with twice-daily case dates — the signature of cases dropping
off as the verification time runs past the end of the study period.

**The count needs the full indexing.** Measured on the reference, it varies with every
dimension: variable (248 of 396 groups differ), level (93/411), spatial region (54/135), metric
(31/337) and truth source (432/579). It cannot be factored into a smaller array, which is why
it is a sibling variable rather than a coordinate or a scalar.

**The package is agnostic to the interval method.** It requires only `lower ≤ mean ≤ upper` at
the stated `confidence`. What matters is that the resampling is over *cases*, and that for the
difference it is a **paired** resample.

### What `n` and `stat` mean here

`n` counts the forecast cases that survive into the mean, so it falls by **2 per 24-hour
step** with twice-daily runs — cases near the end of the study have no truth to verify
against. That reproduces the reference's 417→389 pattern exactly.

`lower`/`upper` are percentiles of the **bootstrap distribution of the mean**, not of the
data. Full code with array shapes is in *Computing significance* below.

2 m temperature against observations at T+48, in kelvin:

```
stat                 mean   lower   upper
prediction_source
IFS-HRES           2.2204  2.0805  2.3765
GraphCast          2.0806  1.9439  2.2252
```

### Why the difference interval cannot be reconstructed

Those two intervals overlap heavily, yet the paired difference is significant at every lead:

```
 lead  diff mean         paired 95% CI      "independent" 95% CI
   24    -0.0794   [-0.1025, -0.0555]     [-0.1838, +0.0250]
   48    -0.1375   [-0.1740, -0.1002]     [-0.3405, +0.0654]
   72    -0.1544   [-0.2198, -0.0924]     [-0.4766, +0.1678]
   96    -0.3935   [-0.5238, -0.2781]     [-0.8941, +0.1070]

mean CI width: paired 0.1235, independent 0.5650  ->  4.6x wider
paired excludes zero at every lead: True      independent: False
```

The only difference is that **one** resample index array is applied to the already-differenced
per-case values, so every replicate compares the two sources on the same weather. That has to
happen in the scoring step: by the time the summary dataset exists, the per-case values are
gone. Hence the *Open assumption* below.

**Caveat to document, not enforce:** for RMSE the two collapses do not commute —
`mean_over_cases(sqrt(mean_over_space(e²)))` ≠ `sqrt(mean_over_cases_and_space(e²))`. The
package performs neither, but `lower`/`upper` must be consistent with whichever produced
`mean`.

### Schema

**One score variable per physical variable, plus an optional ancillary count.
Everything else is a coordinate.**

```
<xarray.Dataset>
Dimensions:  (truth_source: 2, prediction_source: 2, level: 7,
              spatial_region: 10, metric: 3, lead_time: 15, stat: 3)
Coordinates:
  * truth_source       (truth_source)      <U12   'observations' 'analysis'
  * prediction_source  (prediction_source) <U16   'IFS-HRES' 'GraphCast'
  * level              (level)             f8     50.0 100.0 250.0 500.0 850.0 ...
  * spatial_region     (spatial_region)    <U9    'n.hem' 's.hem' 'tropics' 'europe' ...
  * metric             (metric)            <U6    'rmse' 'crps' 'spread'
  * lead_time          (lead_time)         m8[ns] 1 days 2 days ... 15 days
  * stat               (stat)              <U5    'mean' 'lower' 'upper'
    confidence                             f8     0.95
Data variables:
    z                    (truth_source, prediction_source, level, spatial_region, metric, lead_time, stat) f8
    t                    (truth_source, prediction_source, level, spatial_region, metric, lead_time, stat) f8
    msl                  (truth_source, prediction_source,        spatial_region, metric, lead_time, stat) f8
    z_number_of_cases    (truth_source, prediction_source, level, spatial_region, metric, lead_time) i8
    msl_number_of_cases  (truth_source, prediction_source,        spatial_region, metric, lead_time) i8

>>> ds.z.attrs
{'standard_name': 'geopotential_height', 'long_name': 'Geopotential height',
 'units': 'm', 'ancillary_variables': 'z_number_of_cases'}
```

- **`stat` has exactly one meaning**: which view of the estimated score — the point estimate
  and its two interval edges. It never carries anything of a different kind or unit.
- **`truth_source` and `prediction_source` are disjoint lists.** A source is one or the other;
  the cross product of all sources against all sources is mostly meaningless and not stored.
- **A variable omits dimensions that don't apply to it.** `msl` has no `level`. This is the
  whole reason each physical variable gets its own data variable — raggedness then needs no
  sentinel value.
- **`units`, `standard_name`, `long_name`** live on the data variable, where CF puts them.
  rmse, mae, crps and spread all inherit the physical variable's units. A dimensionless
  metric such as ACC needs `units` as a coordinate indexed by `metric` — supported, not
  the default.
- **The case count is a sibling variable**, associated through CF's `ancillary_variables`
  attribute, with a `_number_of_cases` / `_n` suffix as a fallback if the attribute is absent.
  It keeps its integer dtype and carries no units. Entirely optional — it only enriches the
  tooltip ("414 cases"), which is worth having, since a cell computed from 40 cases deserves
  less trust than one from 417.
- **`stat="lower"`/`"upper"`** are the confidence-interval bounds on the mean, at
  `confidence`. Also optional. `confidence` may be promoted to a dimension
  (`[0.68, 0.95, 0.997]`) for multi-level significance, in which case `mean` must not vary
  along it.

Accepted equivalents, normalised on ingest: a flat cube that already has a `variable`
dimension, a tidy `pandas.DataFrame`, or a sequence of records.

Size is negligible: `2 × 2 × 7 × 10 × 3 × 15 × 4` ≈ 25k points per physical variable, a
few MB for a full card. Density is not a concern.

### Normalisation on ingest

`ingest.prepare(ds, spec)` turns the CF form into a uniform cube so the layout engine treats
every coordinate identically:

1. Partition the data variables into **scores** and **ancillaries**, by following each score's
   `ancillary_variables` attribute (falling back to a `_number_of_cases` / `_n` suffix).
2. For each score variable, `expand_dims` any *optional* layout coordinate it lacks with a
   single `NaN` element — so `msl` gains `level = [nan]`, producing **one** row, not seven.
3. `xr.concat` along a new `variable` dimension with `join="outer"`; do the same for the
   counts.
4. Hoist `units`, `standard_name`, `long_name` from each score's `.attrs` into non-dimension
   coordinates indexed by `variable`.

Result: `score(variable, truth_source, prediction_source, level, spatial_region, metric,
stat)` and an aligned `n(...)` without `stat`. `layout.resolve` then flattens once to a frame
for the group-by.

### Missing vs NaN

| State | Meaning | Render |
|---|---|---|
| all-NaN along `lead_time` for a (row-key, col-key) | cell missing | blank grey tile, excluded from `Layout.cells` |
| NaN at some lead times only | step has no datum | neutral tick inside an otherwise normal cell |

---

## Significance is not size

Two independent facts about a cell, and all four combinations occur:

| | significant | not significant |
|---|---|---|
| **large difference** | a real improvement — act on it | large but noisy, or too few cases |
| **small difference** | real, but may not matter operationally | nothing |

If one visual channel carried both, a dark cell would be ambiguous — a reader could not
separate a large-but-noisy regression from a small-but-certain one. So they get **separate
channels**: fill encodes magnitude, border encodes significance. That part of the reference
design is right and is kept.

---

## Computing significance

All of this happens **upstream** of this package — it consumes `stat="lower"`/`"upper"` and
only asks that `lower <= mean <= upper` at the stated `confidence`. It is documented here
because the numbers are meaningless without knowing how they were produced, and because
getting them wrong is easy (see *Two ways this goes wrong*).

### The definition

From the paired per-case differences `d_i = score_exp(case i) - score_ctl(case i)`:

- **Null hypothesis:** `E[d] = 0` — the two prediction sources have the same expected score.
- **Bootstrap:** resample the case set with replacement B times; take the mean of `d` in each
  resample; the confidence interval is the appropriate percentile pair of that distribution.
- **Significant at 95%** ⟺ the 95% interval excludes zero.

For the reference's three-level scheme, compute intervals at 68%, 95% and 99.7% (the 1σ, 2σ
and 3σ points of a normal) and report the tightest level whose interval excludes zero, signed
by direction:

| `siglev` | meaning |
|---|---|
| ±3 | interval excludes zero at 99.7% |
| ±2 | excludes zero at 95% |
| ±1 | excludes zero at 68% |
| 0 | interval includes zero even at 68% |

A paired t-test on `d_i`, or a Wilcoxon signed-rank test, are alternatives. The bootstrap is
assumption-light and is what the reference uses.

### Worked example, with array shapes

Runnable, and the output below is what it actually produces. Every array is annotated with
what its axes mean.

```python
import numpy as np

rng = np.random.default_rng(0)
N_CASE, N_LEAD, N_POINT = 400, 4, 500      # 400 twice-daily runs, 4 lead times, 500 stations
leads = np.array([24, 48, 72, 96])         # hours

# ── what you start with ────────────────────────────────────────────────────────
lat   # (point,)                     latitude of each verification point
obs   # (case, lead_time, point)     truth, NaN where unavailable
fcst  # (source, case, lead_time, point)   source axis is [control, experiment]

# A case initialised at t verifies at t + lead_time. Near the end of the study there
# is no truth to verify against, so those entries are NaN and drop out of `n`.
valid = np.arange(N_CASE)[:, None] * 12 + leads[None, :]     # (case, lead_time), hours
obs   = np.where(valid[:, :, None] <= (N_CASE - 1) * 12, obs, np.nan)

# ── collapse 1: over space, within each case ───────────────────────────────────
# Area weights. For a lat/lon grid this is cos(latitude); for stations, equal or
# by representativity. Weighted so that a dense region does not dominate.
w = np.cos(np.deg2rad(lat)); w /= w.sum()                    # (point,) sums to 1

se = np.nansum(w * (fcst - obs) ** 2, axis=-1)               # (source, case, lead_time)
allnan = np.isnan(obs).all(-1)                               # (case, lead_time)
se = np.where(allnan[None, :, :], np.nan, se)                # a case with no truth is NaN,
                                                             #   not a spuriously perfect 0
score_per_case = np.sqrt(se)                                 # (source, case, lead_time)
# ^ ONE NUMBER PER CASE. This is the unit that gets resampled below.

# ── collapse 2: over cases ─────────────────────────────────────────────────────
# The PAIRED difference: same case, both sources. Pairing is what makes the interval
# tight enough to be useful — see "Why the difference interval cannot be reconstructed".
d = score_per_case[1] - score_per_case[0]                    # (case, lead_time)
n = np.sum(~np.isnan(d), axis=0)                             # (lead_time,) -> [398 396 394 392]

mean = np.nanmean(d, axis=0)                                 # (lead_time,) the point estimate

# ── the bootstrap ──────────────────────────────────────────────────────────────
# Moving-block, NOT iid: consecutive cases share a weather system. L is in cases,
# so L = 24 twice-daily cases = 12 days. See "Two ways this goes wrong".
N_BOOT, L = 2000, 24
n_block = int(np.ceil(N_CASE / L))
starts  = rng.integers(0, N_CASE - L + 1, (N_BOOT, n_block))         # (boot, block)
idx     = (starts[:, :, None] + np.arange(L)).reshape(N_BOOT, -1)[:, :N_CASE]
                                                                      # (boot, case)
# ONE index array, applied to the already-differenced d -> the pairing is preserved
# inside every replicate. Resampling the two sources independently would not do this.
boot = np.nanmean(d[idx], axis=1)                            # (boot, lead_time)
#      ^ the sampling distribution of the mean difference

# ── stat = mean / lower / upper ────────────────────────────────────────────────
lower = np.percentile(boot,  2.5, axis=0)                    # (lead_time,)
upper = np.percentile(boot, 97.5, axis=0)                    # (lead_time,)

# ── siglev: tightest level whose interval excludes zero, signed by direction ───
siglev = np.zeros(N_LEAD, dtype=np.int8)                     # (lead_time,)
for k, conf in enumerate((0.68, 0.95, 0.997), start=1):      # 1σ, 2σ, 3σ
    a = (1 - conf) / 2 * 100
    lo = np.percentile(boot, a,       axis=0)                # (lead_time,)
    hi = np.percentile(boot, 100 - a, axis=0)                # (lead_time,)
    siglev = np.where((lo > 0) | (hi < 0), k, siglev)        # later levels overwrite earlier

# Polarity turns "which direction" into "better or worse". For an error-like metric
# a NEGATIVE difference means the experiment scored lower, i.e. better -> positive siglev.
siglev = np.where(mean < 0, siglev, -siglev)                 # NEGATIVE_IS_BETTER
```

Output:

```
 lead      ctl      exp      diff                 95% CI  siglev     n
   24   1.0710   1.0087   -0.0623   [-0.0691, -0.0564]       3   398
   48   2.1782   2.0542   -0.1240   [-0.1331, -0.1138]       3   396
   72   3.1656   2.9786   -0.1870   [-0.2030, -0.1707]       3   394
   96   4.3290   4.0581   -0.2709   [-0.2979, -0.2451]       3   392
```

`n` falls by 2 per 24-hour step, reproducing the reference's signature. `siglev = 3` with a
negative difference means the experiment is better, significant at 99.7%.

Two shape facts worth internalising:

- `score_per_case` is `(source, case, lead_time)` — space is **gone**. Everything after this
  point resamples along `case` only.
- `boot` is `(boot, lead_time)` — one sampling distribution per lead time, but built from a
  **single** `idx` of shape `(boot, case)`. Reusing that one index array across lead times is
  what keeps a row of boxes coherent rather than independently noisy.

### What counts as one sample

**One forecast case — one initialisation time.** By the time significance is computed, space
has already been collapsed (collapse 1), so a case contributes a *single scalar* per
(variable, level, spatial_region, metric, lead_time). The resampling population is the set of
initialisation times, of size `n`. That is exactly what the reference's `popul` counts, and
why it falls by 2 per 24 hours with twice-daily runs.

What is **not** a sample:

- **Gridpoints.** Already collapsed, and heavily spatially correlated — a 500 hPa geopotential
  error field has a correlation length of order 1000 km, so 10^5 gridpoints carry perhaps 10^2
  independent pieces of information. Resampling them yields absurdly tight intervals.
- **(initialisation, lead time) pairs.** The same run at T+24 and T+48 shares its initial
  condition. Resample *initialisations* and carry all their lead times along — which is also
  what preserves the visual coherence along a row of boxes.
- **Valid times.** A different slicing with the same dependence problem.

If the metric is a non-linear function of the per-case values — pooled RMSE rather than the
mean of per-case RMSEs — the resampling unit is still the case; the statistic is simply
recomputed from the resampled set within each replicate.

### Two ways this goes wrong

Both apply to the reference card as much as to this one, and both are worth stating in the
rendered legend rather than leaving implicit:

1. **Forecast cases are autocorrelated**, and this is severe. Runs 12 hours apart share the
   same weather system, so they are not independent draws. Measured on synthetic data with
   AR(1) dependence (phi = 0.75, ~2-day decorrelation), 400 cases, and a true difference of
   exactly zero — so every "significant" result is a false positive:

   ```
    block length L   = days   false positives   CI width
                 1        0             43.8%     0.0192   <- naive iid bootstrap
                 2        1             33.0%     0.0255
                 4        2             22.5%     0.0330
                 8        4             15.5%     0.0396
                16        8             12.5%     0.0443
                24       12              8.2%     0.0459   <- best
                32       16              9.8%     0.0463
                48       24             12.8%     0.0439
                64       32             14.5%     0.0436

    nominal false-positive rate: 5.0%
   ```

   A naive per-case bootstrap declares significance **44% of the time when there is no effect
   at all**. A moving-block bootstrap reduces that to 8%, and *no* block length reaches
   nominal: too short and blocks do not span the decorrelation time, too long and there are
   too few blocks to resample. Effective sample size here is about 57 of 400.

   The fix — a moving-block bootstrap with the block sized to the synoptic timescale, or
   thinning to independent cases — belongs in the scoring step. This package cannot detect
   the problem, because by the time it sees the data the per-case values are gone. The
   practical consequence is that borderline significance on a scorecard should be treated as
   suggestive rather than decisive.

2. **A card is thousands of simultaneous tests.** At 45 x 30 x 15 it is 20,250 of them; at 95%
   confidence roughly 1,000 cells will read as significant by chance alone. So an isolated
   significant cell carries little weight, while a coherent block of them across neighbouring
   levels or lead times carries a lot. Neither the reference nor this package corrects for
   multiplicity; the honest response is to say so on the page, and to let the eye use spatial
   coherence — which is precisely what a dense tabular layout is good for.

---

## Significance: supplied, not derived

The card's **value** is `experiment - control`. Its **interval cannot be derived**
from the two sources' individual intervals: both are scored on the same forecast
cases, so their errors are strongly paired and the paired interval is far tighter
than any combination of the marginals. Treating them as independent overstates the
interval and marks genuine improvements as insignificant.

So the paired statistics are an **input**, supplied as an ancillary variable
indexed by an ordered pair of prediction sources:

```
    <var>_difference(truth_source, control_source, experiment_source,
                     ..., metric, lead_time, stat)
```

Sparse by construction — only the pairs actually computed are filled, and the
diagonal is left NaN. It is found through CF's `ancillary_variables`, and
classified by its *dimensions* rather than its name: the variable indexed by an
ordered pair of sources is the difference, anything else attached to a score is a
case count.

A cell is **significant when that interval excludes zero**, and `Step.significant`
drives the border. Without the input nothing is marked and the card shows magnitude
only, which `ingest` reports as a warning rather than leaving to be noticed.

Producing it needs one line in the scoring step and is easy to get wrong:

```python
idx = rng.integers(0, n_case, (n_boot, n_case))   # ONE resample...
d   = per_case_experiment - per_case_control      # ...applied to the paired
boot = d[idx].mean(axis=1)                        #    difference, not to each
```

Measured on the HARMONIE/AIFS card, the paired interval is 1.0x to 3.2x tighter
than the naive independent one — so the naive version would have hidden real
results.

**Two channels, independently legible.** Fill carries magnitude, border carries
significance, and the border must stay readable on every fill. Making it a darkened
fill fails at the saturated end — dark-on-dark falls to 1.35:1 — so the
significance channel goes blank exactly where the differences are largest. The
border therefore flips: dark on light fills, light on dark ones, never below 2.5:1,
which `test_borders_stay_visible_against_their_own_fill` enforces.

**A high fraction of significant cells is a warning, not a result.** With few or
autocorrelated cases a naive bootstrap marks almost everything; the HARMONIE card
comes out 92% significant from five initialisations spanning 24 hours, which is
over-claiming. `Layout` adds a note to the card when fewer than 30 cases back a
cell, and another when over 75% of boxes are marked, rather than letting the
borders speak for themselves.

## Layout vocabulary

The picture is in *What a scorecard looks like* above; this names its parts.

`Layout` is the whole resolved thing — the container, not a part. (It was called `Grid`;
renamed because "grid" already means lat/lon gridpoints in meteorology.)

```python
layout = build_layout(ds, spec)

layout.rows            # (Line, ...)  — 45, in display order
layout.columns         # (Line, ...)  — 30 leaf columns
layout.row_headers     # 3 tuples of HeaderCell, one per nesting depth
layout.column_headers  # 2 tuples of HeaderCell
layout.lead_times      # (24.0, 48.0, ..., 360.0)
layout.stats           # LayoutStats(n_rows=45, n_cols=30, n_cells_present=1011, ...)
```

### Indexing

**By label — the public interface.** Cells are addressed by the coordinate values that define
them, following xarray's `sel`/`isel` split:

```python
# keyword form: self-describing, order-independent
layout.sel(truth_source="analysis", variable="z", level=50, spatial_region="n.hem", metric="rmse")

# tuple form: terse, in nesting order — (row key, column key)
layout[("analysis", "z", 50), ("n.hem", "rmse")]

# either returns Cell | None; None means a blank tile
```

Rows and columns are selectable the same way:

```python
layout.row(truth_source="analysis", variable="z", level=50)   # -> Line
layout.column(spatial_region="n.hem", metric="rmse")          # -> Line
```

**By position — for renderers only.** A renderer walks the card in document order, because
position on the page *is* its output:

```python
layout.isel(row=0, col=0)          # -> Cell | None
layout.iter_cells()                # -> Iterator[tuple[int, int, Cell]], populated only
```

`Layout.cells` is keyed by `(row_key, col_key)` label tuples, so `sel` is a direct lookup and
`isel` is one hop through `rows[r].key`. Keying by labels rather than integers also means the
mapping survives a reordering of the rows.

The `Layout` holds everything needed to draw the card and nothing about *how* to draw it —
which is what lets the HTML and matplotlib backends share it.

Every name below is a **frozen dataclass or a `StrEnum`**. None is a matplotlib type —
`matplotlib.axes.Axes` appears only inside `render/static/backend.py`, where the entire card is
drawn into a single `Axes`; the layout types are plain data and never import matplotlib.

**Spec types — you construct these** (`spec.py`):

| Name | Python type | Example |
|---|---|---|
| `ScorecardSpec` | frozen dataclass | `ScorecardSpec(rows=["truth_source","variable","level"], columns=["spatial_region","metric"])` |
| `Dimension` | frozen dataclass | `Dimension("level", sort="numeric", optional=True)` |
| `Category` | frozen dataclass | `Category("n.hem", label="N. Hemisphere", tooltip="lat 20 to 90", tint="#eee")` |
| `MetricSpec` | frozen dataclass, subclasses `Category` | `MetricSpec("crps", polarity=Polarity.NEGATIVE_IS_BETTER)` |
| `Polarity` | `StrEnum` | `Polarity.ACTIVITY` |

**Layout types — the package builds these, renderers read them** (`model.py`):

| Name | Python type | Example |
|---|---|---|
| `Layout` | frozen dataclass | what `build_layout(ds, spec)` returns — the whole card |
| `Line` | frozen dataclass | `layout.row(truth_source="analysis", variable="z", level=850)` |
| `HeaderCell` | frozen dataclass | `HeaderCell(dim="spatial_region", key="n.hem", ..., span=3)` |
| `Cell` | frozen dataclass | `layout[("analysis","z",850), ("n.hem","rmse")]` |
| `Step` | frozen dataclass | `layout[...].steps[7]` → the T+192 box |
| `CellDetail` | frozen dataclass | `layout[...].detail` → control/experiment series |
| `LayoutStats` | frozen dataclass | `layout.stats.n_cells_present` |

**Words, not types**:

| Name | What it actually is |
|---|---|
| nesting depth | `int` — an index into the `rows` / `columns` list; 0 is outermost |
| header level | `tuple[HeaderCell, ...]` — one element of `layout.row_headers` |
| scorecard | the output `.html` / `.png`, not a class |

---

## Declaring the layout

`rows` and `columns` take **lists of coordinate names**. Reordering the nesting is reordering
the list. Any element may instead be a `Dimension` object, needed only for tooltips, tints or
explicit category ordering.

```python
import xarray as xr
from mlwp_scorecards import Dimension, MetricSpec, Polarity, make_scorecard

ds = xr.open_dataset("verification_summary.nc")

make_scorecard(
    ds,
    truth_source=["observations", "analysis"],   # subsets; must appear in rows or columns
    control="IFS-HRES",                          # collapsed by differencing
    experiment="GraphCast",
    rows=["truth_source", "variable", Dimension("level", sort="numeric", optional=True)],
    columns=["spatial_region", "metric"],
    cell="lead_time",
    metrics={
        "rmse":   MetricSpec("rmse",   polarity=Polarity.NEGATIVE_IS_BETTER),
        "crps":   MetricSpec("crps",   polarity=Polarity.NEGATIVE_IS_BETTER),
        "spread": MetricSpec("spread", polarity=Polarity.ACTIVITY, family="activity"),
    },
    title="GraphCast vs IFS HRES",
    output=["scorecard.html", "scorecard.png"],
)
```

The same file yields another card by naming a different pair — which is why which source
is truth, control or experiment is an argument rather than baked into the data.

**How each coordinate is consumed.** Every coordinate is placeable on the card *except* two,
which the rendering consumes:

- `prediction_source` — **collapsed by differencing**: `experiment − control`.
- `stat` — becomes **colour and (later) border**: `mean` drives the fill; `lower`/`upper` feed
  the drill-down error bars and, once paired statistics exist, significance.

Validated with a clear error rather than a downstream `KeyError`: every name in
`rows + columns + [cell]` must be a coordinate of the prepared cube; every coordinate must
appear exactly once across those three, `prediction_source`, and `stat`; `control` and
`experiment` must be members of `prediction_source`; passing a list to `truth_source` without
naming it in `rows` or `columns` is an error, since there would be nowhere to show both.

---

## Architecture

```
xr.Dataset ─► ingest.prepare ─► uniform cube ─► layout.resolve ─► Layout ─┬─► render.html   ─► .html
                                                     ▲                    └─► render.static ─► .png/.svg/.pdf
                                              ScorecardSpec
```

`Layout` is the sole renderer contract. Renderers import only `model`, `colours` and
`geometry` — never `xarray`, never the raw spec.

**Where colour is decided.** The layout engine computes a signed integer **level**
(`-14..14`) per `Step` via `Scaling`, and stores `level`, `significant` and `family` on it.
Renderers map `(family, level, significant) → Swatch` through the shared `ColourScheme`.
Semantics live in one place; the HTML backend can still emit ~120 short CSS rules instead of
15,165 inline styles (this is what makes the file 8× smaller), and matplotlib can build one
`(N,4)` RGBA array. Colours are never inferred from a metric's name.

---

## Files to create

```
mlwp-scorecards/
├── pyproject.toml  .python-version  .flake8  .pre-commit-config.yaml  .gitignore
├── README.md  CHANGELOG.md  AGENTS.md
├── .github/workflows/{ci,pre-commit}.yml
├── src/mlwp_scorecards/
│   ├── __init__.py        version + public re-exports
│   ├── api.py             make_scorecard / build_layout / render / validate_input / load_spec
│   ├── spec.py            Category, MetricSpec, Dimension, ScorecardSpec, YAML I/O
│   ├── model.py           HeaderCell, Line, Step, Cell, CellDetail, LayoutStats, Layout
│   ├── ingest.py          CF Dataset -> uniform cube -> frame; vendored ValidationReport
│   ├── adapters.py        from_dataframe, from_records, read_dataset (nc/zarr)
│   ├── layout.py          resolve(cube, spec) -> Layout
│   ├── colours.py         Swatch, Ramp, Family, ColourScheme, Scaling, SCHEMES
│   ├── symbols.py         significance level -> unicode glyph / matplotlib marker
│   ├── presets/           ecmwf.py — the reference spec, tints, tooltips, lat/lon boxes
│   ├── render/
│   │   ├── __init__.py    Renderer protocol, registry, lazy backend import
│   │   ├── geometry.py    Geometry, span_cells() — shared, dependency-free
│   │   ├── payload.py     compact drill-down JSON + gzip/base64
│   │   ├── html.py        self-contained interactive page + drill-down
│   │   └── static.py      matplotlib backend
│   └── cli.py             mlwp.make_scorecard
└── tests/                 conftest.py, data/, baseline/, test_*.py
```

---

## Key types

```python
class Polarity(StrEnum):
    NEGATIVE_IS_BETTER = "negative_is_better"   # rmse, crps, mae
    POSITIVE_IS_BETTER = "positive_is_better"   # acc, csi, skill scores
    ACTIVITY           = "activity"             # spread: more vs less, no valence
    NEUTRAL            = "neutral"

@dataclass(frozen=True, slots=True)
class Category:
    key: str
    label: str | None = None       # defaults to key
    tooltip: str | None = None     # "NHem Extratropics (lat 20.0 to 90.0, ...)"
    tint: str | None = None        # header-cell background

@dataclass(frozen=True, slots=True)
class MetricSpec(Category):
    polarity: Polarity             # REQUIRED, no silent default
    family: str = "error"          # key into ColourScheme.families
    saturate: float | None = None  # per-metric ramp saturation override

@dataclass(frozen=True, slots=True)
class Dimension:
    name: str                                # a coordinate of the prepared cube
    order: tuple[str, ...] | None = None     # GLOBAL ordering, filtered by presence
    categories: Mapping[str, Category] = ...
    sort: Literal["given", "alpha", "numeric", "appearance"] = "given"
    optional: bool = False                   # may be absent for some variables
    drop_unused: bool = True

@dataclass(frozen=True, slots=True)
class Step:                    # one lead_time inside a Cell -> one drawn box
    lead_time: float
    value: float | None        # experiment - control, in the variable's units
    relative: float | None     # value / |control|, drives the colour ramp
    control: float | None; experiment: float | None
    err_hi: float | None; err_lo: float | None
    n: int | None
    level: int                 # signed ramp bucket, computed by Scaling
    family: str                # "error" | "activity"
    significant: bool          # paired interval excludes zero
    tooltip: str               # "T+24 8.81% better (414 cases)"

Key = tuple[str | float, ...]      # coordinate values in nesting order

@dataclass(frozen=True, slots=True)
class Line:                    # one row, or one leaf column
    index: int                 # position on the card
    key: Key                   # ("analysis", "z", 50)
    headers: tuple[HeaderCell, ...]   # one per nesting depth, never ragged
    slug: str                  # stable, DOM-safe

@dataclass(frozen=True, slots=True)
class Cell:                    # one row Line x one column Line
    row_key: Key; col_key: Key
    row: int; col: int         # positions, for renderers
    cell_id: str               # unique, DOM-safe
    metric: MetricSpec
    units: str | None          # physical units, from the CF attrs
    steps: tuple[Step, ...]    # aligned 1:1 with Layout.lead_times
    detail: CellDetail | None  # control/experiment series for drill-down

@dataclass(frozen=True, slots=True)
class Layout:
    spec: ScorecardSpec
    rows: tuple[Line, ...]; columns: tuple[Line, ...]
    row_headers: tuple[tuple[HeaderCell, ...], ...]     # [depth][block]
    column_headers: tuple[tuple[HeaderCell, ...], ...]
    lead_times: tuple[float, ...]
    cells: Mapping[tuple[Key, Key], Cell]               # keyed by labels, not positions
    stats: LayoutStats

    # label access
    def sel(self, **coords: str | float) -> Cell | None: ...
    def __getitem__(self, keys: tuple[Key, Key]) -> Cell | None: ...
    def row(self, **coords: str | float) -> Line: ...
    def column(self, **coords: str | float) -> Line: ...
    # positional access, for renderers
    def isel(self, *, row: int, col: int) -> Cell | None: ...
    def iter_cells(self) -> Iterator[tuple[int, int, Cell]]: ...
```

**Invariants renderers may rely on** (each gets a test):

1. `rows`/`columns` are in final display order; `rows[i].index == i`.
2. Every `Line.headers` has length `row_depth`/`col_depth` — **never ragged**. A
   not-applicable coordinate is `HeaderCell(is_na=True, label="", span=1)`. All raggedness is
   absorbed here so neither renderer branches on it.
3. `row_headers[d]` / `column_headers[d]` are contiguous, non-overlapping, cover `[0, n)`.
4. A lookup returning `None` ⟺ blank tile. `sel`, `__getitem__` and `isel` agree:
   `layout.isel(row=r, col=c) is layout[rows[r].key, columns[c].key]`. A key that names no
   row or column at all raises `KeyError` — distinct from a valid crossing with no data.
5. `len(cell.steps) == len(layout.lead_times)`, aligned by index.
6. `Cell.cell_id` and `Line.slug` are unique and DOM-safe; **cell identity in the HTML is an
   integer index** (`data-i`), enumerated once, never derived from labels.
7. `Layout` is immutable and picklable.

---

## Layout engine (`layout.py`)

`resolve(cube, spec, *, strict=False) -> Layout`:

1. Resolve `rows` / `columns` / `cell` against the cube's coordinates; error on an unknown or
   unassigned one.
2. Select the truth subset; select `control` and `experiment` along `prediction_source` and
   difference them, dropping that coordinate.
3. Resolve rows and columns independently. **Ordering rule: one global per-dimension `order`,
   filtered by presence within each parent branch.** Verified against the reference — a single
   13-element variable list reproduces both the `an` and `ob` row sequences exactly, with no
   per-branch ordering.
4. Drop categories with no data (`drop_unused`), so unused `(variable, level)` combinations
   never become rows. The `NaN` level introduced by ingest yields a single row with an `is_na`
   `HeaderCell`, keeping header tuples rectangular.
5. Assign `start`/`span` by post-order accumulation of leaf counts.
6. Group by row + column keys; reindex each group onto `Layout.lead_times`. A group that is
   all-NaN is a missing cell, not a `Cell` with NaN steps.
7. Compute `relative` and `level` per `Step` via `Scaling`; freeze.

**Label repetition: model the span, render as blank by default.** `HeaderCell` always carries
`span` and `is_group_start`; `label_repeat` selects the rendering. The HTML backend defaults
to blank continuation cells — not for fidelity, but because `rowspan` in the body breaks the
moment a row inside the span is hidden by the filter checkboxes, and conflicts with
`position: sticky`. The matplotlib backend uses the span unconditionally (no toggling in a
static image), drawing one centred label per block. Two renderers, one `if`.

`FixedScaling` (absolute breakpoints on `|relative|`) is the default rather than quantile
scaling: quantile scaling makes two scorecards from different experiments non-comparable,
which defeats the purpose. `QuantileScaling` exists but stamps a warning into the legend.

---

## HTML backend

**jinja2 for the page shell, a hand-rolled Python loop for the 15,165-box body.** The shell is
~300 lines of conditional markup and needs `ChoiceLoader` template overrides (DMI vs ECMWF
branding will be asked for). Jinja's per-node overhead over 227k emissions is seconds, and the
boxes must be emitted with no inter-element whitespace, so the cell loop is a Python function
returning `Markup`.

Departures from the reference, each deliberate:

| Reference | Here | Why |
|---|---|---|
| 2 spans/step (box + hidden triangle) | 1 `<i>`/step, mode switched in CSS | halves node count |
| ~95 B inline style per span | level class + CSS custom properties | 27 unique values → ~120 rules, not 15k styles |
| `title=` on every span, twice | `title=` on `<td>`; per-step tip built on hover | ~600 kB saved |
| duplicate `id` on ~45 `<td>`s | `data-col="nhem\|rmse"` | ids must be unique |
| label-concatenated cell id | integer `data-i` | breaks on `_`, `.`, unicode |
| `<td>` row labels | `<th scope="row">` | screen readers |
| `setVisibility` writes 15k inline styles | one injected CSS rule in `<style id="sc-dyn">` | hundreds of ms → one reflow |
| Plotly 2.7 MB over `http://`, unpinned | ~140 lines of hand-rolled SVG charts | offline, no CDN, prints as vector |
| GTM + full ECMWF site template | removed | ~200 kB, and no third-party tracking |

Clicking a cell opens a drill-down with two charts: the difference over lead time, and the
two sources' own values with their confidence intervals. It uses a native `<dialog>` — focus
trap, Esc and backdrop come free — rather than rewriting the cell's `innerHTML`, which in the
reference loses the mode state and leaks the old DOM. The payload is rounded to 4 significant
figures, gzipped with `mtime=0`, base64'd into a `<script>`, and inflated lazily on first
click via `DecompressionStream` (which works on `file://`).

**The difference has no confidence band**, and the dialog says so rather than leaving the
absence to be noticed: that needs a paired resample, which is not yet an input. Overlapping
bands on the second chart therefore do *not* mean the difference is insignificant — the
marginal intervals are much wider than the paired one (see *Why the difference interval
cannot be reconstructed*).

Measured on a 45x30x15 card: **1.64 MB against the reference's 7.4 MB**, of which the
drill-down payload is 0.46 MB against the reference's 3.2 MB — 6.9x smaller for the same
data, from rounding plus gzip. `detail=False` drops it to 1.17 MB.

---

## Static backend (pass 2)

**One `Axes` with manual patches, not `GridSpec`.** GridSpec cannot express ragged
colspan/rowspan header merges without hand-computing the geometry anyway; 1,350 Axes is
pathological for both figure creation and PDF size. A single Axes with `set_ylim(H, 0)` gives a
top-left-origin point space identical to the CSS box model, so `geometry.py` is genuinely
shared, and all 20,250 boxes become one `PatchCollection` drawn in a single pass.

Sizing: a 45×30×15 card is ~3,330 pt = 1.17 m wide. **Never shrink the font below 6.5 pt** —
grow the figure. `max_width_mm` triggers pagination at depth-0 column-group boundaries (never
mid-group), repeating the row labels on each page; PDF gets one `PdfPages`, PNG/SVG get
`card-01`, `card-02`, … `dpi="auto"` snaps boxes to integer device pixels to avoid uneven
half-pixel edges.

---

## Colour and accessibility

- **`cvd` scheme is the default**: RdBu endpoints (`#2166ac` / `#b2182b`) for the error family,
  BrBG (`#8c510a` / `#01665e`) for activity. The reference's purple↔green activity pair is
  rejected — under deuteranopia both arms land in the same muddy yellow-grey band and are
  indistinguishable at 5×9 px.
- **`ecmwf` scheme reproduces the reference verbatim**, opt-in, for visual continuity. Its ramp
  is recoverable exactly: 15 fill steps, bin width 25/14 = 1.7857 %, saturating at 25 %, each
  fill paired with one saturated border.
- Ramps are **committed hex tables**, generated offline by a `dev/` script using
  `colorspacious` — a dev-only dependency, never imported at runtime, so output is
  byte-reproducible.
- **A diverging ramp is luminance-symmetric, so sign is unrecoverable in greyscale for any hue
  pair.** No palette fixes this, which is why the significance-glyph mode (pass 2) is planned
  as the **print default** rather than a fallback. Tests *assert* the greyscale collapse,
  documenting why.
- The reference's `triangleDict` is replaced: it maps `-1` and `+1` to the same glyph, and maps
  `0` (weakest signal) to a full block (heaviest glyph). Use a monotone triangle family.
- The legend is generated from the same `ColourScheme` that colours the cells, with a test
  asserting the hexes match — the reference's legend says "red" while its cells are orange.
- The legend also states the two significance caveats from *Significance is not size*: that
  cases are autocorrelated, and that a card is thousands of simultaneous tests, so isolated
  significant cells mean little and coherent blocks mean a lot. `LayoutStats` carries the
  test count so the wording can quote it.

---

## Public API

```python
make_scorecard(data, *, truth_source=None, control, experiment,
               rows=..., columns=..., cell="lead_time", metrics=None,
               output, formats=None, spec=None, strict=False,
               return_validation_report=False, **renderer_kwargs) -> list[Path]
build_layout(data, spec, **kw) -> Layout
render(layout, output, *, format=None, **kwargs) -> Path
validate_input(data, spec=None, **kw) -> ValidationReport
load_spec(path) -> ScorecardSpec
```

The keyword form builds a `ScorecardSpec` for you; `spec=` accepts a prebuilt one or a YAML
path for reuse across cards. Re-exported from `__init__.py` alongside `ScorecardSpec`,
`Dimension`, `Category`, `MetricSpec`, `Polarity`, `Layout`, `ColourScheme`, `SCHEMES`.

CLI `mlwp.make_scorecard`, argparse + `@logger.catch`, mirroring `mlwp_data_loaders/cli.py` in
style; exit 1 when `report.has_fails()`.

---

## pyproject

Match the siblings: setuptools>=69 + setuptools-scm>=8 + wheel, `dynamic = ["version"]`,
`package-dir = {"" = "src"}`, `requires-python = ">=3.12"`, `[tool.isort] profile = "black"`,
`.flake8` with `max-line-length = 88` / `ignore = E203,E501,W503`, the sibling
`.pre-commit-config.yaml` verbatim, `[dependency-groups] dev`, and the `uv run --all-extras
pytest` CI workflow.

```toml
dependencies = ["xarray>=2024.1", "pandas>=2.2", "numpy>=1.26",
                "jinja2>=3.1", "pyyaml>=6.0", "loguru>=0.7"]

[project.optional-dependencies]
static  = ["matplotlib>=3.8,<4"]
netcdf  = ["netcdf4>=1.6", "h5netcdf>=1.3"]
zarr    = ["zarr>2,<3"]
test    = ["pytest>=8", "pytest-mpl>=0.17", "syrupy>=4", "hypothesis>=6.100",
           "matplotlib~=3.10.0", "netcdf4>=1.6"]

[project.scripts]
"mlwp.make_scorecard" = "mlwp_scorecards.cli:main"
```

`xarray` is core (it brings `pandas` and `numpy`, so the frame path is free). File-format
backends are extras, matching `mlwp-data-loaders`' split. `matplotlib` is an extra so an
HTML-only install stays light; `render/__init__.py` imports backends lazily and raises
`ImportError("install mlwp-scorecards[static]")`. No `plotly`, no `lxml`/`beautifulsoup4`
(stdlib `html.parser` suffices for tests), no `mlwp-data-specs`.

---

## Build order

**Pass 1 — walking skeleton (this task)**

1. Write this plan to `PLAN.md` in the repo root, so it is versioned alongside the code and
   reviewable in the same place. Repo scaffolding: `git init`, pyproject, `.flake8`,
   pre-commit, CI, README, AGENTS.md.
2. `spec.py` + `model.py` + vendored `ValidationReport` in `ingest.py`.
3. `ingest.py` — CF-to-cube normalisation, validation; `adapters.from_dataframe`,
   `from_records`, `read_dataset`.
4. `colours.py` (`cvd` + `ecmwf` schemes, `FixedScaling`) and `symbols.py`.
5. `layout.py` — truth subset, control/experiment differencing, coordinate resolution,
   ordering, raggedness, spans, cell assembly.
6. `render/geometry.py` (`span_cells`) — pure, unblocks both backends.
7. `render/html/` — markup and CSS first, then JS column toggling.
8. `api.py`, `cli.py`, tests throughout.

**Pass 2** — pagination for the static backend, `presets/ecmwf.py`, the
significance-glyph mode for print and greyscale, accessibility pass, `compact_cells` gradient
path if real cards justify it.

---

## Verification

**Determinism is a prerequisite for every snapshot test.** No `datetime.now()` in the renderer;
`json.dumps(separators=(",",":"), allow_nan=False)`; `gzip.compress(mtime=0)`; float formatting
through one `%.{p}g` helper; no `set` iteration in output paths; respect `SOURCE_DATE_EPOCH`.
`test_determinism.py` renders twice in-process and once in a `subprocess` with
`PYTHONHASHSEED=1` and asserts all three are byte-identical.

- `test_ingest.py` — the CF-to-cube step: `msl` (no `level` dim) yields **one** row not seven;
  `units`/`standard_name` reach `Cell.units`; the count is located via `ancillary_variables`
  and, separately, via the suffix fallback, and keeps its integer dtype through to
  `Step.n`; a dataset with no counts at all renders with countless tooltips rather than
  failing; an all-NaN `(row, col)` slice becomes a missing cell while a partially NaN slice
  becomes a `Cell` with neutral steps; the CF form, the flat cube and the equivalent tidy
  frame produce byte-identical HTML.
- `test_spec.py` — coordinate-list resolution: an unknown name errors clearly; a coordinate
  assigned to neither rows, columns, `cell`, `prediction_source` nor `stat` errors clearly; a
  `control` not in `prediction_source` errors clearly; a list `truth_source` not named in
  `rows`/`columns` errors clearly; reordering `rows` reorders the nesting; a bare string and an
  equivalent `Dimension` give the same layout.
- `test_layout.py` is the load-bearing one: the global-order-plus-presence rule reproduces the
  exact ECMWF `an` and `ob` row sequences; `msl` yields `is_na` at depth 2 with
  `len(headers) == 3`; `sum(h.span) == n_rows` at every depth; `experiment − control` is
  differenced in the right direction (a deliberately-worse experiment must colour red);
  degenerate cases (depth-1 nesting, single row, empty data) raise clearly rather than
  `IndexError`.
- `test_indexing.py` — `sel`, `__getitem__` and `isel` agree for every populated crossing;
  keyword and tuple forms agree; a valid-but-empty crossing returns `None` while an unknown
  coordinate value raises `KeyError`; `iter_cells()` yields exactly `stats.n_cells_present`
  entries; reordering `rows` in the spec leaves every label lookup returning the same `Cell`.
- `test_colours.py` — the exact ECMWF hex at each bin midpoint under the `ecmwf` scheme, plus
  contrast (≥4.5:1 fg-on-fill), CVD separability under Machado matrices (ΔE > 12 for each
  family's two arms), and the greyscale collapse.
- HTML: one `syrupy` snapshot over a small fixture (4×3×5, including a missing cell and a
  variable with no level), plus ~15 structural tests using stdlib `html.parser` — every
  `data-i` resolves, all `id`s unique, `colspan` matches visible children, and
  `test_no_external_urls` guards against reintroducing the GTM / `http://` Plotly defects.
- Hypothesis property test feeding labels containing `<script>`, `"`, `&`, `_`, `|` and emoji
  through a full render, asserting the parsed DOM structure is unchanged.
- matplotlib (pass 2): `pytest-mpl` baselines on the **small** fixture only, with `locked_rc`
  forcing `pdf.fonttype: 42`, `svg.fonttype: "none"`, `font.family: DejaVu Sans`. The large
  card asserts behaviour (completes < 20 s, correct page count), never pixels.
- `test_renderer_contract.py` runs both backends over one `Layout` and asserts identical cell
  counts, colour multisets and missing-cell positions — the guard against renderer drift.
- Markers: `slow`, `mpl`, `golden`. CI fast job runs `-m "not slow and not mpl"`.
- End-to-end smoke: `uv run mlwp.make_scorecard tests/data/tiny.nc --control IFS-HRES
  --experiment GraphCast -o /tmp/card.html`, then open it and check the spatial-region checkboxes and a
  hover tooltip.

---

## Proposed schema changes (not yet implemented)

Two changes to the input schema, recorded before acting on them.

### 1. Fold the metric into the variable name: `{variable}.{metric}`

Today `metric` is a coordinate and `units` is an attribute on the score variable,
which forces **one unit per variable across all metrics**. That is already wrong:
RMSE, MAE and spread inherit the field's units, but ACC, CRPSS and FSS are
dimensionless, and a threshold-based score may carry different units again. The
current escape hatch — `units` as a coordinate indexed by `metric` — is noted
elsewhere in this document as "supported, not the default", which is a sign the
shape is wrong rather than a solution.

Naming variables `z.rmse`, `z.fss`, `2t.acc` puts `units`, `standard_name` and
`long_name` back where CF expects them, one set per (variable, metric).

It also fixes something the current shape handles badly: **metric coverage is
ragged**. CRPS needs an ensemble, FSS needs a threshold and really only suits
precipitation, ACC needs a climatology. A `metric` coordinate forces a full cross
product and fills the gaps with NaN; separate variables simply omit what does not
apply — the same argument that made one-variable-per-field right in the first
place.

`ingest.prepare` already builds the `variable` dimension by stacking data-variable
names, so this extends that to split each name into two dimensions rather than one.
`METRIC_POLARITY` is unaffected: it is keyed on the metric name either way.

Three things to settle first:

- **The separator.** `.` is clean but breaks xarray's attribute access —
  `ds["z.rmse"]` works, `ds.z.rmse` does not, and the latter reads as nested
  attribute access which it is not. `__` avoids that at some cost in looks. Worth
  checking against real variable names: the ECMWF reference card carries `10ff@sea`
  and `2t`, neither containing a dot, so `.` looks safe for our data — but the
  splitter must fail loudly on a name it cannot split, not guess.
- **Round-tripping.** A dataset that has been through `prepare` and back out should
  reproduce the same names.
- **Backwards compatibility.** Accept both forms, or migrate? The existing
  HARMONIE/AIFS pipeline writes the coordinate form.

### 2. Rename `stat`, and say what a "case" is

`stat` is vague and its members are not all the same kind of thing, which is the
real problem: `mean` is a reduction over cases, but `lower` and `upper` are
percentiles of the *bootstrap distribution of that mean*. They are not reduction
operations at all.

So `cases_reduction_op` would be accurate for one member out of three and
misleading for the other two. What the dimension actually answers is "which number
about the aggregate", which suggests **`estimate`** — `ds["z.rmse"].sel(estimate="lower")`
reads correctly for all three members.

If the central reduction ever needs to vary (median rather than mean, or a
trimmed mean), that is a *different* axis and should be a different coordinate, not
extra members of this one.

**On "cases".** For the current design a case is always one forecast
initialisation — that is established in *What counts as one sample*, and being
explicit guards against the mistake of resampling gridpoints or station points
instead. So `number_of_forecasts` is more informative than the present
`number_of_cases`, and much more so than `number_of_samples`, which invites the
question "samples of what?" — exactly the question this package should not leave
open.

The cost is that the name becomes a lie if the resampling unit ever changes: a
moving-block bootstrap resamples *blocks* of forecasts, not forecasts, and that is
already the recommended fix for autocorrelation. Suggested resolution: name it
`number_of_forecasts` and carry a `resampling_unit` attribute alongside, so the
statistics can say what they actually resampled without the variable name having to
carry it.

Also worth correcting while there: the count currently declares
`standard_name = "number_of_observations"`. These are forecasts, not observations.

## Appendix: the ECMWF reference card, structurally

Reference material. `scorecards-47r1ENS.html` in the repo root is the ECMWF-generated
card this design was reverse-engineered from; this records what it does, and what it
does that should not be copied.

A single HTML `<table>` — no charting library. Plotly appears only in a click-through popup.

- **Rows**: 3 nested label columns — truth source (`an`/`ob`) → variable (`z`, `t`, `ff`,
  `msl`, `2t`, `tp`, `swh`…) → pressure level (50/100/250/500/850, absent for surface
  variables). 45 rows. Each column has its own background tint; the level tint darkens
  with pressure.
- **Columns**: 2 nested header rows — spatial region (10, each `colspan=3`, with lat/lon
  tooltips) → metric (`rmsef`, `crps`, `spread`). 30 leaf columns.
- **Each cell is a mini time series**: 15 forecast lead times (T+24…T+360) drawn as 15 small
  boxes in a row inside the cell. 1011 of the 1350 (row, col) combinations are populated;
  the rest are blank grey `#f9f9f9` tiles.
- **Two renderings per box**, both emitted and toggled by radio buttons: a filled box (fill =
  magnitude on a ~15-step ramp, border = significant at 95%) and a Unicode significance
  triangle. Four hue families: blue better / red worse, purple more-active / green less-active.
- Below the table, a 3.2 MB `SubPagesData` JSON keyed by cell id drives a click-through
  Plotly drill-down.

Defects the package should **not** reproduce: ~95 bytes of inline style on every one of 30,330
spans (only 27 distinct values exist); the same `id` reused on ~45 different `<td>`s (invalid
HTML); cell identity string-concatenated from labels (breaks on any label containing `_`);
colour family inferred from `metric.substr(0,3)=="sda"`; Plotly loaded over plain `http://`
from an unpinned CDN; embedded Google Tag Manager; `displayAsColouredBoxes` is a stub
returning `"bbb"`.
