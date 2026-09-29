# Developing

## Environment

This project uses `uv` for dependency management and local commands.

```bash
uv sync --extra test --group dev
```

For a one-off command, `uv run ...` works without activating the environment.

## Code structure

A card is made in four steps, each handing the next something simpler than it
was given:

```
xr.Dataset ──(1) ingest.prepare──▶ per-case scores + units
           ──(2) aggregate.aggregate──▶ Aggregated (collapsed over cases)
           ──(3) layout.engine.create_layout──▶ layout.model.Layout
           ──(4) render.html / render.static──▶ page (str) / figure
```

```
src/mlwp_scorecards/
├── __init__.py        public exports: ScoreCard, SCHEMES, DEFAULT_ROWS/COLUMNS
├── api.py             ScoreCard — the public object; build_layout runs (1)–(3)
│                      (dataset → Layout; internal, used by the layout tests)
├── cli.py             the mlwp.make_scorecard command, a thin wrapper around ScoreCard
├── ingest.py      (1) check the dataset's shape; flatten {metric}.{variable} into one
│                      array of per-case scores (metric, variable, …, lead_time,
│                      init_time) plus units; the schema's *_DIM names
├── aggregate.py   (2) collapse over forecast cases: means, bootstrap intervals, and
│                      the paired difference from the baseline (what decides significance)
├── polarity.py        what a metric means: which direction is better (an explicit
│                      table; unknown metrics raise), its family, the words for it
├── layout/            the card's layout
│   ├── __init__.py    re-exports the model types — never the engine (see below)
│   ├── engine.py  (3) create_layout(): stateless; difference against the baseline,
│   │                  then place rows, columns, cells, each box's family and level,
│   │                  significance, tooltips, notes (aggregated numbers → Layout)
│   ├── scaling.py     a relative difference → a signed level, within ±LEVELS
│   └── model.py       Layout, Cell, Step, Line, …, LEVELS: the whole card as plain
│                      values, no xarray and no colours — the sole renderer contract
└── render/        (4) draw a Layout; compute nothing
    ├── colours.py     palettes (colour schemes): a box's family and level → colours
    ├── html.py        render_html() → self-contained page (str)
    ├── payload.py     the page's packed drill-down data
    └── static.py      render_figure() → matplotlib Figure; save_figure()
```

**Colour is derived, and chosen last.** A metric's polarity says which direction
is better. From it and the paired difference, the engine gives each box a family
(`"error"` or `"activity"`) and a signed level. The palette, which the caller
picks when drawing (`to_html(colour_scheme=...)`), turns those into colours. So
the layout carries no colours, and one card can be drawn in any palette. The
number of levels, `LEVELS`, is part of the contract: the scaling has one break
per level and every palette one ramp step per level, and tests check both.

**The dependency rule.** `layout.model` is the only thing the renderers see. They
import `mlwp_scorecards.layout`, whose `__init__` re-exports the model types,
`polarity` for the direction words, and each other. They never import
`layout.engine`, `xarray`, or the input dataset. Python runs a package's
`__init__` before any of its submodules, so `layout/__init__.py` must never import
the engine. `tests/test_layout.py` checks that on the source. If a renderer needs
something the `Layout` does not carry, extend `layout/model.py` rather than
reaching past it.

## Running tests

```bash
uv run python -m pytest                              # everything
uv run python -m pytest -m "not slow and not mpl"    # fast subset, as CI's first job
uv run python -m pytest tests/test_layout.py         # one file
```

Markers:

- `slow` — builds a full ECMWF-scale card (45 x 30 x 15)
- `mpl` — needs matplotlib and the committed baseline images
- `golden` — compares against committed golden artefacts

## Pre-commit

```bash
uv sync --extra test --group dev
pre-commit run --all-files
pre-commit install          # run the hooks before each commit
```

CI runs the same hooks via `.github/workflows/pre-commit.yml`.

## Matplotlib image baselines

The static backend is regression-tested with `pytest-mpl` against committed PNGs of a
small fixture only — never a full-size card.

```bash
uv run pytest --mpl --mpl-generate-path=tests/baseline
```

`matplotlib` is pinned to a minor range in the `test` extra because glyph rasterisation
drifts between releases. Bumping it requires regenerating the baselines.

## Versioning

Versions come from git tags via `setuptools-scm`. CI checks out with `fetch-depth: 0`
so the tag history is available.

## Design document

`PLAN.md` is the source of truth for the input schema, the layout vocabulary and the
rendering approach. Update it alongside behavioural changes.
