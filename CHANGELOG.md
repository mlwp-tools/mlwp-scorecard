# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial development of `mlwp-scorecards`, a package for rendering weather
forecasting scorecards from pre-computed verification statistics. A scorecard
compares one or more forecast sources with a baseline source, all scored against
a common truth source, and colours the paired difference between their scores
across nested groupings of variable, level, region, metric and forecast lead time.

Design and rationale are documented in [`PLAN.md`](PLAN.md).

### Changed

- **`control=` / `experiment=` are gone.** The baseline is `colour_relative_to=`
  (`--colour-relative-to`): it decides the colouring, and the significance, and
  nothing else. The sources compared with it are a selection like any other,
  `select=dict(forecast_source=["GraphCast", ...])`, and default to every source
  but the baseline. A baseline is often not a model at all, hence "relative to"
  rather than "control". `Step.control*` /
  `Step.experiment*` become `Step.baseline*` / `Step.forecast*`, and
  `Layout.control` / `Layout.experiment` become `Layout.baseline_source` /
  `Layout.forecast_sources`. There is no alias for the old names.
- **All selection goes through `select=`**, with one rule for every coordinate:
  a single value picks that member and drops the dimension, unless the dimension
  is named in `rows=`, `columns=` or `cell=`, where it is kept one entry long; a
  list keeps the dimension, subset **in the order given** (which is the order it
  is drawn in), with at most one `...` for every other value; a slice keeps it, as
  `ds.sel` would. In `forecast_source` a `...` never includes the baseline, and
  naming the baseline is an error. `variable` and `metric` can be selected and
  ordered the same way. The `truth_source=` argument is gone:
  `select=dict(truth_source="analysis")` replaces it. On the CLI, `--truth-source`
  becomes a repeatable `--select DIM=V1,V2` (a trailing comma makes a list of one;
  values are read as the coordinate's type). A `select=` key that is not a
  dimension is an error.
- **Outputs are named arguments.** `make_scorecard(ds, "card.html", ...)` becomes
  `make_scorecard(ds, html_path="card.html", image_path=["card.png", "card.pdf"],
  ...)`, and everything after `data` is keyword-only. At least one output is
  required, and a suffix contradicting its argument (`html_path="card.png"`) is
  refused before anything is computed. The CLI's `-o/--output` becomes
  `--html-path` and a repeatable `--image-path`; `--validate-only` needs neither.

- **The input is now one score per forecast case**, and the package performs the
  collapse over cases itself: the mean, the per-source bootstrap intervals, the
  paired difference that decides significance, and the case count. `estimate`,
  `confidence`, `control_source`, `experiment_source`, every
  `number_of_forecasts.*` and every `*.difference` variable leave the schema; the
  per-case axis is `init_time`. Six reserved dimension names become four.

  This exists because which source is the control is a call-time choice, and the
  old schema forced it into the data: a pre-computed paired difference had to say
  which ordered pair it belonged to, so the roles were named in the file and named
  again at the call. Pairing has to precede the averaging and cannot be recovered
  afterwards, so the only way to keep the roles out of the file is to give the
  package the un-averaged numbers.

  It also reverses the project's "no statistics" rule, stated in `AGENTS.md` and
  the README. No *scoring* still holds — the collapse over space needs the fields
  and stays upstream.
- `prediction_source` is renamed `forecast_source`.
- `build_layout`/`make_scorecard` gain `bootstrap`, `block_length`, `n_resamples`,
  `confidence_levels` and `seed`, with matching CLI flags. `bootstrap` defaults to
  `"moving-block"`: on AR(1) synthetic data an iid resample marks ~44% of
  truly-null cells as significant against a nominal 5%, and blocking brings that
  to ~8%. `block_length` is derived from the initialisation cadence when omitted,
  and the choice is reported rather than made silently. `seed` is fixed at 0 so
  two runs on one file agree.
- The card now states how its intervals were made — method, block length,
  resample count and seed — in its notes and on `Layout`. A significance claim
  cannot be checked without them.
- `build_layout` and `make_scorecard` no longer take `prediction_dim`,
  `estimate_dim`, `confidence_dim` or `metric_dim`, and `ingest.prepare` /
  `layout.resolve` no longer take those or `variable_dim`. The schema's
  dimension names are fixed constants in `ingest.py`. A breaking signature
  change, though nothing could have depended on it: `variable_dim` was never
  forwarded from the public API, `prediction_dim` was contradicted by hardcoded
  literals in two other modules, and `metric_dim=None` raised on every dataset.
- The CLI reads `.nc`, `.nc4`, `.cdf` and `.zarr` and refuses anything else by
  name, rather than passing unrecognised paths to `xarray` for engine sniffing.
- A dataset that already carries a `variable` or `metric` dimension is now
  refused with a message naming the schema, instead of failing later on a
  variable name with no dot in it.

### Added

- **`show_values=True`** (`--show-values`) prints each source's own score in its
  boxes, formatted by one helper (`model.format_value`). With a baseline the
  colours and significance borders are unchanged, and the baseline appears as a
  grey row of its own scores, first; `forecast_source` is then always on an axis.
- **Cards with no baseline.** `colour_relative_to=None` with `show_values=True`
  shows every source's own scores on grey, with their own intervals in the
  tooltip and drill-down, and nothing compared or marked significant. Neither
  a baseline nor values is an error naming the available sources.
- `Step.text`, `Step.has_data`, `Cell.is_baseline`, `Layout.show_values`,
  `Layout.coloured`.
- The CLI's **`--open`** opens each file written in the system's default viewer.

- **Several forecast sources against one baseline.** With more than one source
  selected, `forecast_source` becomes a layout axis, outermost on the
  rows by default. Each row is exactly the two-source card for that source,
  because one resample is shared by every pair.
- **`cases="common" | "pairwise"`**: which forecast cases each comparison rests
  on. `"common"` (the default) uses only the cases every selected source scored,
  so rows are comparable; `"pairwise"` uses what each shares with the baseline.
  A card with several sources says which it used.
- `Cell.forecast_source` names the source a cell compares with the baseline.

### Fixed

- **The HTML page dropped middle column-header levels.** Only the outermost and
  leaf header rows were written, so `columns=["forecast_source",
  "spatial_region", "metric"]` lost the region headings (the static figure had
  them). Every level is now a row, and hiding columns resizes each level's spans.

- **A string dimension only some variables have** no longer shows the text "nan"
  where it does not apply. Padding used NaN, which a fixed-width string coordinate
  (what a netCDF round trip gives) turns into `'nan'`; it now pads with None for
  non-numeric coordinates, the not-applicable marker the layout already expects.
- **Such dimensions keep their order.** Combining the variables sorted the union of
  the coordinate's values, so rows no longer followed the dataset's order or a
  `select=` list's. Each is now built in first-appearance order, with the
  not-applicable entry last.

- **The difference was not paired when a source was missing cases.** Each
  source's mean and bootstrap were taken over its own finite cases, while `n`
  counted only the shared ones, so a card could compare different weather: two
  sources identical on every shared case, with the baseline bad on the cases the
  other lacks, showed a difference of −2.0 significant at 99.7% from `n = 20`.
  Every source is now masked to the shared cases before anything is averaged.

- A paired difference stored for only one ordering of a source pair left the
  reverse card with every significance border silently missing — it rendered
  perfectly and simply claimed nothing was significant. The difference is
  antisymmetric, so the transpose is now derived (mean negated, bounds swapped
  and negated) and a writer need only fill one triangle. When neither ordering
  is present the report says so instead of staying quiet.
- The unassigned-dimension error suggested subsetting the dimension away with
  `select=`, which does not work: `select=` is applied after the check and keeps
  the dimension at length one. It now names the remedy that does, `ds.sel(...)`.
- Lead times stored as `timedelta64[ns]` — what a netCDF round trip commonly
  decodes to — were labelled `T+2.16e+13` instead of `T+6` on every box.
  `.item()` on a `timedelta64` returns a `datetime.timedelta` at every coarser
  resolution but a bare int of nanoseconds at `ns`, and an int is
  indistinguishable from a lead time already given in hours. Numpy time scalars
  are no longer unwrapped.
- A paired difference supplied for some physical variables but not others
  crashed with `IndexError` when a variable without one also lacked an optional
  dimension — the exact shape the README's example describes, where `rmse.msl`
  has no `level` but `rmse.z.difference` does. The NaN filler for the missing
  pair borrowed another variable's coordinates, leaving the difference cube
  shorter than the score cube it is read against.
- `resolve` now checks that the case counts and the paired difference share the
  scores' coordinates before reading them positionally, so a future stacking bug
  is an error naming the dimension rather than a wrong number in a cell.

### Removed

- The undocumented-but-advertised support for tidy `pandas.DataFrame` inputs,
  record sequences and pre-flattened cubes. None of it was ever implemented; the
  README claimed it. `pandas` is no longer a declared dependency, since the
  package never imported it.
