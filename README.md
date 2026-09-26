# mlwp-scorecards

Weather forecasting scorecards for machine-learning weather prediction.

When a forecasting system changes, the question is rarely "is it good?" but "is it
better than what we already have, and **where is it worse**?" The evidence runs to
thousands of numbers — every variable, level, spatial region, metric and lead time —
and nobody reads thousands of line plots. A scorecard puts them on one page you can scan
in a minute, so a net improvement, an isolated regression, and the difference between
a real effect and sampling noise are all visible at once.

It compares **one or more forecast sources with a baseline source, all already
scored against a common truth source**, and colours the *paired difference between
their scores*. "Better" means better than the baseline, not better than truth.

This package does no scoring: metrics are computed upstream, where the fields
are. It takes one score per forecast case and performs the collapse over those
cases itself — the mean, its bootstrap interval, and the *paired* difference that
decides significance. That last one is why it wants per-case input: pairing has
to happen before the averaging, and cannot be recovered afterwards.

## Install

```bash
uv add mlwp-scorecards            # HTML output
uv add "mlwp-scorecards[static]"  # + matplotlib PNG/SVG/PDF
```

## Use

```python
import xarray as xr
from mlwp_scorecards import make_scorecard

ds = xr.open_dataset("verification_summary.nc")

make_scorecard(
    ds,
    predictions_from=["GraphCast"],
    relative_to="IFS-HRES",
    html_path="scorecard.html",
    image_path="scorecard.png",
    title="GraphCast vs IFS HRES",
)
```

Rows and columns are inferred from the dataset, or named explicitly:

```python
make_scorecard(
    ds, html_path="scorecard.html",
    predictions_from=["GraphCast"], relative_to="IFS-HRES",
    rows=["truth_source", "variable", "level"],
    columns=["spatial_region", "metric"],
    cell="lead_time",
)
```

The same file yields another card by naming different sources, so which source is
truth, baseline or forecast is an argument rather than something baked into the data.

### Several forecast sources against one baseline

Name several sources in `predictions_from` and `forecast_source` becomes a layout
axis — outermost on the rows unless you place it — with one block of rows per
source, each compared with the `relative_to` baseline:

```python
make_scorecard(
    ds, html_path="scorecard.html",
    predictions_from=["GraphCast", "AIFS", "Aurora"], relative_to="IFS-HRES",
    rows=["forecast_source", "variable", "level"],
    columns=["metric"],
    truth_source="analysis",
)
```

`...` stands for every source not otherwise named, in the order of the
`forecast_source` coordinate, and never includes the baseline. So
`predictions_from=["GraphCast", ...]` is GraphCast first and then all the rest,
and leaving `predictions_from` out is the same as `[...]`: every source but the
baseline.

Every row is exactly the two-source card for that source: the resample is shared
by all of them, so the rows are consistent with one another. The baseline may not
also be named in `predictions_from`. `relative_to` is required for now; a card of
absolute scores with no baseline is planned.

`cases=` says which forecast cases each comparison rests on, and the card says
which was used:

- `"common"` (the default): only the cases **every** selected source and the
  baseline scored. Rows are comparable with one another, but a source that runs
  one cycle a day cuts the case count for all of them.
- `"pairwise"`: the cases each forecast source shares with the baseline. Each row
  uses as much data as it can, but the rows no longer rest on the same weather and
  should not be ranked against one another.

With one forecast source the two are the same.

There is no configuration object to build: coordinate names, source names and output
paths are all the API has.

```bash
mlwp.make_scorecard verification_summary.nc \
    --predictions-from GraphCast --relative-to IFS-HRES \
    --html-path scorecard.html --image-path scorecard.png
```

`html_path=` is the interactive page and must end in `.html`. `image_path=` is
the static figure, and takes one path or several; each one's format follows its
suffix (`.png`, `.pdf`, `.svg`, ...), so `image_path=["card.png", "card.pdf"]`
writes both. At least one of the two is required, and a suffix that contradicts
its argument is refused before anything is computed.

The HTML page is self-contained — no CDN, no analytics, no webfonts, so it works
offline and from `file://`. Hover a box for its value, use the checkboxes to filter
columns, and click a cell for a drill-down showing the difference over lead time
and both sources' own values with their confidence intervals. The charts are
generated SVG rather than a plotting library.

## Input

One variable per `{metric}.{physical_variable}` pair, holding that metric's value
**for each forecast case** (different forecast runs) — not yet averaged over them. The naming convention is
[WeatherBench-X](https://github.com/google-research/weatherbenchX)'s, and its
`Aggregator.reduce_dims` is a required argument, so leaving `init_time` out of it
gives exactly this shape. Everything else is a coordinate.

A variable omits the dimensions that do not apply to it: `msl` (mean sea-level
pressure) simply has no `level`, and a metric that needs an ensemble simply has
no variable for the fields that lack one.

```
<xarray.Dataset>
Dimensions:  (truth_source: 2, forecast_source: 2, level: 7, spatial_region: 10,
              lead_time: 15, init_time: 400)
Coordinates:
  * truth_source     (truth_source)    <U12   'observations' 'analysis'
  * forecast_source  (forecast_source) <U9    'IFS-HRES' 'GraphCast'
  * level            (level)           f8     50.0 100.0 250.0 500.0 850.0 ...
  * spatial_region   (spatial_region)  <U9    'n.hem' 's.hem' 'tropics' ...
  * lead_time        (lead_time)       m8[ns] 1 days ... 15 days
  * init_time        (init_time)       M8[ns] 2024-01-01 ... 2024-07-15
Data variables:
    rmse.z    (truth_source, forecast_source, level, spatial_region, lead_time, init_time) f8
    rmse.msl  (truth_source, forecast_source,        spatial_region, lead_time, init_time) f8
    crps.z    (truth_source, forecast_source, level, spatial_region, lead_time, init_time) f8

>>> ds["rmse.z"].attrs
{'standard_name': 'geopotential_height', 'long_name': 'Geopotential height RMSE',
 'units': 'm'}
```

One element is *the score of one forecast, from one source, against one truth
source, for one physical variable, over one spatial region, at one lead time* —
after the collapse over space, before the collapse over cases. NaN marks a case
with no verification, and the case count falls out of counting the finite ones.

### The two collapses, and which one is yours

A scorecard rests on two reductions, and they are different in kind.

**Over space, within one forecast case.** The gridpoints of a region reduce to one
number, area-weighted by cos(latitude), by a formula specific to the metric. This
is deterministic, carries no sampling uncertainty, and needs the raw fields — so
it happens upstream, and the dataset above is its output.

**Over forecast cases.** The initialisations reduce to a mean. *This* is the
sample: N weather situations drawn from the population of possible ones. **The
package does this one**, because doing it well requires the per-case numbers:

- the sources are compared on the *same* cases, so the difference is taken
  per case and one resample is shared between them. Their common error then
  cancels instead of adding — measured 1.0x to 3.1x tighter than treating them as
  independent, and the only thing that can decide significance;
- consecutive forecasts share a weather system, so the resample blocks them.

Both choices are arguments, and both are printed on the card:

```python
make_scorecard(ds, html_path="scorecard.html",
               predictions_from=["GraphCast"], relative_to="IFS-HRES",
               bootstrap="moving-block",   # or "iid"
               block_length=None,          # in CASES; derived from the cadence
               n_resamples=2000,
               confidence_levels=(0.68, 0.95, 0.997),
               seed=0)
```

`seed` is fixed rather than drawn from the OS, so two runs on one file agree.

### The smallest input that renders

Almost everything is optional. Two variables, one metric, lead time and cases:

```
<xarray.Dataset>
Dimensions:          (forecast_source: 2, lead_time: 8, init_time: 40)
Coordinates:
  * forecast_source  (forecast_source) <U9    'IFS-HRES' 'GraphCast'
  * lead_time        (lead_time)       m8[ns] 0 days 06:00:00 ... 2 days
  * init_time        (init_time)       M8[ns] 2024-01-01 ... 2024-01-20
Data variables:
    rmse.2t          (forecast_source, lead_time, init_time) f8
    rmse.msl         (forecast_source, lead_time, init_time) f8
```

```python
make_scorecard(ds, html_path="scorecard.html",
               predictions_from=["GraphCast"], relative_to="IFS-HRES")
```

`truth_source`, `level` and `spatial_region` are all absent, so the inferred
layout is `rows=["variable"]`, `columns=["metric"]` — two rows and one column.
Adding `mae.2t` and `mae.msl` would give a second column.

`init_time` may be absent too, if all you have is means. Then you get a card
coloured by magnitude with no error bars, no borders and no case counts, and the
report says so:

```
warning no 'init_time' dimension: values are read as already-collapsed means,
        with no interval and nothing marked significant
```

Set `units` on each variable either way — it reaches the drill-down axes, and
nothing else supplies it.

### Which names mean something

Four dimension names are reserved. The package consumes them, and none becomes a
row or column unless you put it there — which only `forecast_source` allows, and
only when there are several forecast sources.

| Name | Required | What it does |
|---|---|---|
| `forecast_source` | yes | The sources being compared, named at call time as `predictions_from=` and `relative_to=`. The card shows `forecast - baseline`. With one forecast source the dimension is collapsed by differencing; with several it is laid out like any other axis. |
| `init_time` | no | The forecast cases. Collapsed by the bootstrap, which is where the intervals and the significance come from. Absent, the values are read as already-collapsed means. |
| `variable`, `metric` | never | **Produced** by splitting the `{metric}.{variable}` names. Supplying either as an input dimension is an error. |

Everything else is yours. `truth_source`, `level` and `spatial_region` are
conventions, not rules — they get a sensible default position because they are
what verification datasets usually carry, and `truth_source` additionally gets a
`truth_source=` filter argument for convenience. But nothing requires them, and a
dimension the package has never heard of behaves exactly the same way:

```python
# season and threshold are not special; they are just axes
make_scorecard(
    ds, html_path="scorecard.html",
    predictions_from=["GraphCast"], relative_to="IFS-HRES",
    rows=["season", "variable"],
    columns=["threshold", "metric"],
)
```

**Every non-reserved dimension must go somewhere** — rows, columns, or `cell`.
Leaving one out is an error rather than a silent average over it, because a card
that quietly averaged over your thresholds would look entirely normal and be
wrong. If you do not want a dimension on the card, pick one value:

```python
make_scorecard(ds, select={"season": "DJF"}, ...)   # drops the dimension
make_scorecard(ds.sel(season="DJF"), ...)           # the same
```

A single value in `select=` drops the dimension; a list keeps it, subset, and so
it still needs a place on the card: `select={"season": ["DJF", "JJA"]}` with
`season` on the rows.

When `rows` and `columns` are omitted they are inferred: the conventional names
above take their usual positions, and anything left over is appended to the
columns in alphabetical order. Inference is a convenience for exploration — name
the axes explicitly for a card anyone else will read.

Other notes on the shape:

- The name splits on its **last** dot, so a metric may carry parameters of its
  own: `seeps.v1.5.tp` is the metric `seeps.v1.5` of the variable `tp`. A name
  with no dot is refused rather than guessed at.
- `units`, `standard_name` and `long_name` live on the data variable, where CF
  puts them — and because the metric is in the name there is one set per
  (metric, variable), so `rmse.2t` can be in K while `acc.2t` is dimensionless.
- That is the only input shape: a netCDF or Zarr dataset laid out as above. There
  is no adapter for tidy frames, record sequences or a cube that already carries
  a `variable` dimension — the last is refused with a message pointing back here.

## Documentation

[`PLAN.md`](PLAN.md) is the design document: what the dataset conceptually contains,
how `n` and the confidence intervals are produced upstream, the layout vocabulary,
and the rendering approach.

## Related

- [`mlwp-data-specs`](https://github.com/mlwp-tools/mlwp-data-specs) — trait-based
  dataset validation
- [`mlwp-data-loaders`](https://github.com/mlwp-tools/mlwp-data-loaders) — loading raw
  MLWP source datasets
- `mxalign` — alignment and verification, the usual upstream of this package
