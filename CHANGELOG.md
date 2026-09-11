# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial development of `mlwp-scorecards`, a package for rendering weather
forecasting scorecards from pre-computed verification statistics. A scorecard
compares two forecast sources, each scored against a common truth source, and
colours the difference between their scores across nested groupings of variable,
level, region, metric and forecast lead time.

Design and rationale are documented in [`PLAN.md`](PLAN.md).

### Changed

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

### Fixed

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
