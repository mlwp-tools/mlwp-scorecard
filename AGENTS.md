# AGENTS

Guidance for agents and contributors working in this repository.

## Project intent

- `mlwp-scorecards` renders weather forecasting scorecards from **pre-computed
  verification statistics**.
- It performs **no scoring and no statistics**. Metrics, confidence intervals and
  case counts are computed upstream (typically by `mxalign`) and consumed here.
- A scorecard compares **two prediction sources** against a **common truth source**
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
- Input handling: `src/mlwp_scorecards/ingest.py`, `adapters.py`
- Layout declaration: `src/mlwp_scorecards/spec.py`
- Layout types (the renderer contract): `src/mlwp_scorecards/model.py`
- Layout engine: `src/mlwp_scorecards/layout.py`
- Colour and symbols: `src/mlwp_scorecards/colours.py`, `symbols.py`
- Renderers: `src/mlwp_scorecards/render/html/`, `render/static/`

## Development expectations

- **`Layout` is the sole renderer contract.** Modules under `render/` may import
  `model`, `colours` and `geometry` only — never `xarray`, never the raw spec. If a
  renderer needs something absent from `Layout`, extend `model.py` rather than adding
  a backend-specific code path.
- **Never infer semantics from a name.** A metric's polarity and colour family are
  declared in the spec, never derived from its string. The reference implementation's
  `metric.substr(0,3)=="sda"` is the bug this rule exists to prevent.
- **Cell identity in HTML is an integer index**, enumerated once. Never build an id by
  concatenating labels.
- **Rendered output must be byte-reproducible.** No `datetime.now()`, no `set`
  iteration in output paths, `gzip.compress(..., mtime=0)`, float formatting through a
  single helper. `tests/test_determinism.py` enforces this.
- Keep docstrings (numpydoc) and type annotations on new code; `from __future__ import
  annotations` at the top of every module.
- Add or update tests when changing behaviour.
