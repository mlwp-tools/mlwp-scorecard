# Prior work: harp scorecards

[harp](https://harphub.github.io/harp/) is the R verification toolkit used across
the ACCORD / HIRLAM consortium. It has a scorecard feature, described in the
[scorecards article](https://harphub.github.io/harp/articles/scorecards.html) and
implemented in two packages:

- `harpPoint::bootstrap_verify()` — computes the scores and the bootstrap
  ([source](https://github.com/harphub/harpPoint/blob/master/R/bootstrap_verify.R))
- `harpVis::plot_scorecard()` — draws the card
  ([source](https://github.com/harphub/harpVis/blob/master/R/plot_scorecard.R))

Notes taken 2026-09-27 from the article and the `master` branch sources. The
source-level details below were read from the code, not run.

## Pipeline

harp does everything end to end, starting from raw point forecasts and
observations:

```r
fcst <- read_point_forecast(dttm = ..., fcst_model = c("AROME_Arctic_prod", "MEPS_prod"),
                            fcst_type = "det", parameter = param, file_path = ...)
fcst <- common_cases(fcst)                  # keep only cases every model has
obs  <- read_point_obs(unique_valid_dttm(fcst), param, obs_path = ...,
                       stations = unique_stations(fcst))
fcst <- join_to_fcst(fcst, obs)

result <- bootstrap_verify(fcst, det_verify, {{param}}, n = 100, pool_by = "SID")
```

It repeats this for each parameter (`T2m`, `S10m`, `T850`, …, `Td500`), combines
the results with `bind_point_verif()`, then plots them:

```r
plot_scorecard(scorecard_data, fcst_model = "AROME_Arctic_prod",
               ref_model = "MEPS_prod", scores = c("rmse", "mae", "bias"))
```

Ensemble forecasts use `ens_verify` in place of `det_verify`, and the rest is the
same.

## `bootstrap_verify()`

```r
bootstrap_verify(.fcst, verif_func, obs_col, n,
                 groupings = "lead_time", pool_by = NULL, conf = 0.95,
                 min_cases = 4, perfect_scores = perfect_score(),
                 parallel = FALSE, num_cores = NULL, show_progress = TRUE, ...)
```

- **Resampling unit.** With `pool_by = NULL`, it resamples individual rows (one
  forecast–observation pair each). With `pool_by = "<column>"`, it resamples whole
  pools with replacement. A pool can be a station (`SID`, as in the article's
  example) or a forecast date (the article's suggestion). Pooling is how harp deals
  with autocorrelation. `pool_by` can also be a data frame with a `pool` column,
  for user-defined blocks.
- **Scoring.** In each replicate it calls `verif_func` on the resampled rows,
  within each `groupings` cell (lead time by default). So the score is recomputed
  from raw pairs in every replicate. Resampling happens before scoring, not after.
- **Paired.** Each replicate's sample indices are drawn once and applied to every
  model (`purrr::map(.fcst, dplyr::slice, row_numbers)`), so both models are
  scored on the same resampled cases.
- **Model pairs.** It forms every ordered pair of models. For each replicate,
  `difference = fcst_score - ref_score`.
- **`percent_better`** is the fraction of replicates in which `fcst_model` is
  closer to the perfect score than `ref_model`:
  `abs(fcst_score - perfect) < abs(ref_score - perfect)`. Ties count as not better.
  `perfect_score()` is a table of perfect values (0 for bias, rmse, mae, …), which
  is how harp handles polarity, including the signed bias.
- **Intervals.** It reports `mean`, `median`, `lower` and `upper` for the
  reference score, the forecast score and the difference. `lower` and `upper` are
  the `(1-conf)/2` and `1-(1-conf)/2` quantiles of the replicates (2.5 % and
  97.5 % by default).
- **`min_cases`** is accepted, but its enforcement is commented out in the current
  source. In the block bootstrap it means the minimum number of blocks.
- `parallel = TRUE` runs the replicates with `parallel::mclapply`.

Output columns: `fcst_model, ref_model, lead_time, score,
{ref,fcst}_score_{mean,median,upper,lower}, difference_{mean,median,upper,lower},
percent_better`.

## `plot_scorecard()`

```r
plot_scorecard(bootstrap_data, fcst_model, ref_model, scores,
  facet_by = vars(score), num_facet_rows = 1, grid_facets = FALSE, filter_by = NULL,
  significance_breaks = c(-1.1, -0.997, -0.95, -0.68, 0.68, 0.95, 0.997, 1.1),
  colours = c("#CA0020", "#CA0020", "#CA0020", "grey70", "#0571B0", "#0571B0", "#0571B0"),
  fills   = c("#CA0020", "#F4A582", NA, "grey70", NA, "#92C5DE", "#0571B0"),
  shapes  = c(25, 25, 25, 22, 24, 24, 24),
  sizes   = c(3, 2, 1, 0.5, 1, 2, 3),
  legend_labels = "auto", num_facet_cols = length(scores), ...)
```

- **Layout.** A ggplot of points. The x axis is lead time and the y axis is the
  parameter. There is one facet per score, side by side by default. `grid_facets`
  switches to a two-variable `facet_grid`.
- **Signed confidence.** `percent_better` is mapped onto −1…1 before binning:
  values below 0.5 have 1 subtracted. So 0.98 stays 0.98 ("fcst better, 98 %"),
  and 0.02 becomes −0.98 ("ref better, 98 %").
- **Seven classes** come from `cut()` at ±0.68 / ±0.95 / ±0.997 (the 1σ, 2σ
  and 3σ levels):
  - Blue up-triangles (shape 24) mean `fcst_model` is better, and red
    down-triangles (shape 25) mean it is worse.
  - Stronger confidence gives a bigger, more filled symbol: hollow at 68 %, light
    fill at 95 %, solid at 99.7 %.
  - Below 68 % either way, the cell is a small grey square (shape 22).
- **Encoding.** Colour, fill, shape and size are all mapped to the class, and each
  has its own legend. The legends read like "fcst_model better than ref_model with
  significance > X%".
- **Model pair.** It filters to exactly `fcst_model` / `ref_model` and never
  swaps a pair stored the other way round. Since `bootstrap_verify()` stores both
  orderings, this doesn't matter in practice.
- The magnitude of the difference is not shown. The card shows only which model
  wins and how confidently.

## Compared with mlwp-scorecards

| | harp | mlwp-scorecards |
|---|---|---|
| Scoring | Computes scores from raw point pairs | None; scores arrive per case from upstream (mxalign) |
| Resampling unit | Rows, or pools (station, date, custom blocks) | Blocks of consecutive forecast cases (`block_length`, derived from the initialisation cadence). Space is already collapsed upstream, so pooling by station isn't possible |
| Order of bootstrap and scoring | Resample, then score | Score per case upstream, then resample and average |
| Pairing | Same resample for both models | Paired difference per case |
| Significance statistic | `percent_better`, the fraction of replicates where fcst wins | Interval on the paired difference; excluding 0 is equivalent to a `percent_better` threshold |
| Polarity | Distance to a `perfect_score()` table, so \|bias\| nearer 0 wins | Explicit `METRIC_POLARITY` table; `polarity_of` raises on unknown metrics; `bias` is `NEUTRAL` (no winner) |
| Cell encoding | Symbol shape, size and fill; no magnitude | Colour of the difference, with significance marked separately |
| Levels | 68 / 95 / 99.7 % in one plot | 68 / 95 / 99.7 % by default (`confidence_levels`) |
| Output | Static ggplot | Static image, and HTML with drill-down |

Ideas worth considering:

- **Distance to perfect for bias.** harp calls a model better on bias when its
  |bias| is closer to 0. We currently mark bias `NEUTRAL`. This works from the
  case-mean bias, so it would be a derived quantity rather than a new polarity.
- **User-defined blocks.** harp's `pool_by` data frame lets the user choose the
  blocks explicitly. We derive `block_length` from the cadence and don't accept
  arbitrary groupings.
- **`percent_better` as an output.** It's the one number harp users read off a
  card. Our replicates can give the same number, which would make comparison with
  harp cards direct.
