# Developing

## Environment

This project uses `uv` for dependency management and local commands.

```bash
uv sync --extra test --group dev
```

For a one-off command, `uv run ...` works without activating the environment.

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
