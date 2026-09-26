# AGENTS

Guidance for agents and contributors working in this repository.

## Project intent

- `mlwp-scorecards` renders weather forecasting scorecards from **pre-computed
  verification statistics**.
- It performs **no scoring**. Metrics are computed upstream (typically by
  `mxalign`), where the fields are, and the collapse over *space* happens there.
- It **does** perform the collapse over forecast cases: the mean, its bootstrap
  interval, the paired difference that decides significance, and the case count.
  That reverses an earlier rule ("no statistics"), deliberately — pairing the two
  sources against the same weather has to happen before the averaging, so the
  package needs one score per case and must do the averaging itself.
- A scorecard compares **two forecast sources** against a **common truth source**
  and colours the *difference* between their scores. "Better" means better than the
  other model, not better than truth.

## Interfaces

- Python API: `make_scorecard(...)`, `build_layout(...)`, `render(...)`
- CLI: `mlwp.make_scorecard`

## Common commands

- Install/sync env: `uv sync --extra test --group dev`
- Run tests: `uv run python -m pytest`
- Fast tests only: `uv run python -m pytest -m "not slow and not mpl"`
- Lint: `pre-commit run --all-files`
- Regenerate matplotlib baselines: `uv run pytest --mpl-generate-path=tests/baseline`

## Structure

- Design document: `PLAN.md` — read this first; it defines the input schema and the
  layout vocabulary.
- Public API: `src/mlwp_scorecards/api.py`, `cli.py`
- Input handling: `src/mlwp_scorecards/ingest.py`
- Layout types (the renderer contract): `src/mlwp_scorecards/model.py`
- Layout engine: `src/mlwp_scorecards/layout.py`
- Colour, scaling and metric polarity: `src/mlwp_scorecards/colours.py`
- Renderers: `src/mlwp_scorecards/render/html.py`, `render/static.py`
- Synthetic test data: `tests/synthetic.py`
- Notes on other scorecard tools: `docs/prior-work/` (`harp.md`, `ecmwf-scorecard.md`, `brightband/`)

## Development expectations

- **`Layout` is the sole renderer contract.** Modules under `render/` may import
  `model`, `colours` and `geometry` only — never `xarray`, never the raw spec. If a
  renderer needs something absent from `Layout`, extend `model.py` rather than adding
  a backend-specific code path.
- **No user-facing configuration object.** The API takes coordinate names, source
  names and output paths. Do not reintroduce a spec/config class into the public
  surface; internal dataclasses are fine.
- **One input shape, and its dimension names are constants.** The package reads a
  netCDF or Zarr dataset in the schema documented in README.md — per-case scores,
  no adapters for frames or records. The schema's dimension names are the `*_DIM` constants in
  `ingest.py`, not parameters: they were parameters once, nothing ever passed a
  non-default, and two of them silently did not work because the same names were
  also hardcoded elsewhere. `rows`, `columns` and `cell` are different in kind —
  they choose where a coordinate goes, not what it is called.
- **Never guess a metric's polarity.** `colours.METRIC_POLARITY` is an explicit
  table and `polarity_of` raises for anything absent, because guessing produces a
  confidently backwards card. The reference implementation's
  `metric.substr(0,3)=="sda"` is the bug this rule exists to prevent.
- **Not-applicable coordinates are `None` in a layout key, never NaN** -- `nan != nan`
  would break every label lookup for surface variables.
- **Cell identity in HTML is an integer index**, enumerated once. Never build an id by
  concatenating labels.
- **Rendered output must be byte-reproducible.** No `datetime.now()`, no `set`
  iteration in output paths, `gzip.compress(..., mtime=0)`, float formatting through a
  single helper. `tests/test_determinism.py` enforces this.
- Keep docstrings (numpydoc) and type annotations on new code; `from __future__ import
  annotations` at the top of every module.
- Add or update tests when changing behaviour.
