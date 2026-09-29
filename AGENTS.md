# AGENTS

Guidance for agents and contributors working in this repository.

## Project intent

- `mlwp-scorecard` renders weather forecasting scorecards from **pre-computed
  verification statistics**.
- It performs **no scoring**. Metrics are computed upstream (typically by
  `mxalign`), where the fields are, and the collapse over *space* happens there.
- It **does** perform the collapse over forecast cases: the mean, its bootstrap
  interval, the paired difference that decides significance, and the case count.
  That reverses an earlier rule ("no statistics"), deliberately — pairing the two
  sources against the same weather has to happen before the averaging, so the
  package needs one score per case and must do the averaging itself.
- A scorecard compares **one or more forecast sources** with a **baseline source**,
  all scored against a **common truth source**, and colours each one's *paired
  difference* from the baseline. "Better" means better than the baseline, not
  better than truth. The baseline is never also one of the forecast sources, and
  is never compared with itself; with `show_values=True` it appears as a grey row
  of its own scores. With no baseline (`baseline=None`) nothing is
  compared and the card is each source's own scores.

## Interfaces

- Python API: `ScoreCard(ds, ...)`, then `.to_figure()` -> matplotlib `Figure`, `.to_html()` -> `str`. The caller saves.
  `api.build_layout` and the `render/` functions are internal.
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
- How the modules fit together: `DEVELOPING.md`, *Code structure*.
- Public API: `src/mlwp_scorecard/api.py`, `cli.py`
- Input handling: `src/mlwp_scorecard/ingest.py`
- Collapse over forecast cases (means, bootstrap, paired differences):
  `src/mlwp_scorecard/aggregate.py`
- Layout types (the renderer contract): `src/mlwp_scorecard/layout/model.py`
- Layout engine (`create_layout`): `src/mlwp_scorecard/layout/engine.py`
- Metric polarity (which direction is better) and its words:
  `src/mlwp_scorecard/polarity.py`
- Difference → signed level: `src/mlwp_scorecard/layout/scaling.py`
- Palettes (level → colour): `src/mlwp_scorecard/render/colours.py`
- Renderers: `src/mlwp_scorecard/render/html.py`, `render/static.py`
- Synthetic test data: `tests/synthetic.py`
- The README's example card (WeatherBench 2 scores, and the scripts that make
  and draw them): `docs/example/`
- Notes on other scorecard tools: `docs/prior-work/README.md` (overview and comparison; links to each note)

## Development expectations

- **`Layout` is the sole renderer contract.** Modules under `render/` may import
  `mlwp_scorecard.layout` (the types), `polarity` (the direction words) and each
  other only — never `layout.engine`, never `xarray`, never the input dataset. The
  layout carries no colours; the palette is chosen when drawing. `layout/__init__.py`
  therefore re-exports the model and must never import the engine (a test checks).
  If a renderer needs something absent from `Layout`, extend `layout/model.py`
  rather than adding a backend-specific code path.
- **No user-facing configuration object.** The API takes coordinate names and
  source names. Do not reintroduce a spec/config class into the public
  surface; internal dataclasses are fine.
- **One input shape, and its dimension names are constants.** The package reads a
  netCDF or Zarr dataset in the schema documented in README.md — per-case scores,
  no adapters for frames or records. The schema's dimension names are the `*_DIM` constants in
  `ingest.py`, not parameters: they were parameters once, nothing ever passed a
  non-default, and two of them silently did not work because the same names were
  also hardcoded elsewhere. `rows`, `columns` and `cell` are different in kind —
  they choose where a coordinate goes, not what it is called.
- **Never guess a metric's polarity.** `polarity.METRIC_POLARITY` is an explicit
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
- Every module, class, function and method under `src/` -- private helpers and
  nested functions included -- has a numpydoc docstring with its Parameters and
  Returns. The `numpydoc-validation` pre-commit hook enforces it; the checks are
  configured in `pyproject.toml` under `[tool.numpydoc_validation]`. A constructor's
  parameters go on the class docstring. Keep type annotations on new code, and
  `from __future__ import annotations` at the top of every module.
- Add or update tests when changing behaviour.
