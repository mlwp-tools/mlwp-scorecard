# mlwp-scorecards

Weather forecasting scorecards for machine-learning weather prediction.

When a forecasting system changes, the question is rarely "is it good?" but "is it
better than what we already have, and **where is it worse**?" The evidence runs to
thousands of numbers — every variable, level, spatial region, metric and lead time —
and nobody reads thousands of line plots. A scorecard puts them on one page you can scan
in a minute, so a net improvement, an isolated regression, and the difference between
a real effect and sampling noise are all visible at once.

It compares **two prediction sources, each already scored against a common truth
source**, and colours the *difference between their scores*. "Better" means better
than the other prediction source, not better than truth.

This package does no scoring and no statistics. It consumes pre-computed verification
statistics and renders them.

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
    ["scorecard.html", "scorecard.png"],
    control="IFS-HRES",
    experiment="GraphCast",
    title="GraphCast vs IFS HRES",
)
```

Rows and columns are inferred from the dataset, or named explicitly:

```python
make_scorecard(
    ds, "scorecard.html",
    control="IFS-HRES", experiment="GraphCast",
    rows=["truth_source", "variable", "level"],
    columns=["spatial_region", "metric"],
    cell="lead_time",
)
```

The same file yields another card by naming a different pair, so which source is
truth, control or experiment is an argument rather than something baked into the data.

There is no configuration object to build: coordinate names, source names and output
paths are all the API has.

```bash
mlwp.make_scorecard verification_summary.nc \
    --control IFS-HRES --experiment GraphCast \
    -o scorecard.html -o scorecard.png
```

Output format follows the suffix: `.html` for the interactive page, `.png`, `.pdf`
or `.svg` for the static figure.

The HTML page is self-contained — no CDN, no analytics, no webfonts, so it works
offline and from `file://`. Hover a box for its value, use the checkboxes to filter
columns, and click a cell for a drill-down showing the difference over lead time
and both sources' own values with their confidence intervals. The charts are
generated SVG rather than a plotting library.

## Input

One score variable per physical variable; everything else is a coordinate. A
variable omits the dimensions that do not apply to it, so `msl` simply has no `level`.

```
<xarray.Dataset>
Dimensions:  (truth_source: 2, prediction_source: 2, level: 7,
              spatial_region: 10, metric: 3, lead_time: 15, stat: 3)
Coordinates:
  * truth_source       (truth_source)      <U12   'observations' 'analysis'
  * prediction_source  (prediction_source) <U16   'IFS-HRES' 'GraphCast'
  * level              (level)             f8     50.0 100.0 250.0 500.0 850.0 ...
  * spatial_region     (spatial_region)    <U9    'n.hem' 's.hem' 'tropics' ...
  * metric             (metric)            <U6    'rmse' 'crps' 'spread'
  * lead_time          (lead_time)         m8[ns] 1 days ... 15 days
  * stat               (stat)              <U5    'mean' 'lower' 'upper'
    confidence                             f8     0.95
Data variables:
    z                  (truth_source, prediction_source, level, spatial_region, metric, lead_time, stat) f8
    msl                (truth_source, prediction_source,        spatial_region, metric, lead_time, stat) f8
    z_number_of_cases  (truth_source, prediction_source, level, spatial_region, metric, lead_time) i8
```

One element is *a summary statistic describing how well one prediction source agreed
with one truth source, for one physical variable, aggregated over a spatial region and
over a set of forecast cases, at one forecast lead time*.

- `stat` means one thing only: which view of the estimated score — the point estimate
  and its two interval edges.
- `lower`/`upper` are the sampling uncertainty **over forecast cases**, not over
  gridpoints.
- The case count is a sibling variable found through CF's `ancillary_variables`
  attribute (with a `_number_of_cases` / `_n` suffix fallback). Optional.
- `units`, `standard_name` and `long_name` live on the score variable, where CF puts
  them.

A tidy `pandas.DataFrame`, a sequence of records, or a flat cube that already has a
`variable` dimension are all accepted and normalised on ingest.

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
