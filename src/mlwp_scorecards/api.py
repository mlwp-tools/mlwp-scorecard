"""Public API.

Everything the caller needs is passed as plain arguments — coordinate names,
source names, output paths. There is no configuration object to construct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from .aggregate import aggregate
from .colours import SCHEMES, ColourScheme, FixedScaling
from .ingest import (
    CASE_DIM,
    FORECAST_DIM,
    METRIC_DIM,
    VARIABLE_DIM,
    ValidationReport,
    prepare,
)
from .layout import resolve
from .model import Layout

__all__ = [
    "make_scorecard",
    "build_layout",
    "render",
    "DEFAULT_ROWS",
    "DEFAULT_COLUMNS",
]

#: Preferred nesting when ``rows`` is not given, filtered to what the data has.
#: A *preference*, not a requirement: these names get a conventional position on
#: the card if they are present, and any dimension not listed here still lands on
#: an axis -- see ``_infer_axes``.
DEFAULT_ROWS = ("truth_source", VARIABLE_DIM, "level")
#: Preferred nesting when ``columns`` is not given. ``region`` is accepted as a
#: synonym of ``spatial_region`` here only, for datasets that use the short name.
DEFAULT_COLUMNS = ("spatial_region", "region", METRIC_DIM)

_STATIC_SUFFIXES = {".png", ".pdf", ".svg", ".eps", ".jpg", ".jpeg", ".tif", ".tiff"}


def _infer_axes(
    ds: xr.Dataset,
    rows: Sequence[str] | None,
    columns: Sequence[str] | None,
    cell: str,
) -> tuple[list[str], list[str]]:
    """Choose row and column nesting when the caller did not.

    ``variable`` and ``metric`` are not dimensions of the input -- ``prepare``
    produces them by splitting the ``{metric}.{variable}`` names -- so they are
    added here rather than read off the dataset. The excluded names are the ones
    the rendering consumes rather than lays out, and are the same set ``prepare``
    exempts from its unassigned-dimension check; both are built from the
    constants in :mod:`~mlwp_scorecards.ingest` so the two cannot drift apart.
    """
    available = ({str(d) for d in ds.dims} | {VARIABLE_DIM, METRIC_DIM}) - {
        cell,
        FORECAST_DIM,
        CASE_DIM,
    }

    if rows is not None and columns is not None:
        return list(rows), list(columns)

    r = [d for d in (rows if rows is not None else DEFAULT_ROWS) if d in available]
    c = [
        d
        for d in (columns if columns is not None else DEFAULT_COLUMNS)
        if d in available and d not in r
    ]
    leftover = sorted(available - set(r) - set(c))
    if columns is None:
        c += leftover
    else:
        r += leftover
    if not r or not c:
        raise ValueError(
            f"could not split dimensions {sorted(available)} into rows and columns; "
            "pass rows=[...] and columns=[...] explicitly"
        )
    return r, c


def build_layout(
    data: xr.Dataset,
    *,
    control: str,
    experiment: str,
    rows: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    cell: str = "lead_time",
    truth_source: str | Sequence[str] | None = None,
    select: Mapping[str, Any] | None = None,
    metric_polarity: Mapping[str, str] | None = None,
    scheme: str | ColourScheme = "cvd",
    title: str = "",
    subtitle: str = "",
    bootstrap: str = "moving-block",
    block_length: int | None = None,
    n_resamples: int = 2000,
    confidence_levels: Sequence[float] = (0.68, 0.95, 0.997),
    seed: int = 0,
    strict: bool = False,
    return_validation_report: bool = False,
) -> Layout | tuple[Layout, ValidationReport]:
    """Resolve a verification dataset into a ready-to-render :class:`Layout`.

    Parameters
    ----------
    data : xr.Dataset
        Verification statistics: one variable per ``{metric}.{variable}`` pair.
    control, experiment : str
        Members of ``forecast_source``. The card colours
        ``experiment - control``.
    rows, columns : sequence of str, optional
        Coordinate names to nest on each axis, outermost first. Inferred when omitted.
    cell : str, optional
        Coordinate drawn inside each cell, normally ``"lead_time"``.
    truth_source : str or sequence of str, optional
        Restrict to these truth sources.
    select : mapping, optional
        Further coordinate subsetting, applied before layout.
    metric_polarity : mapping, optional
        Polarity for metrics not in the built-in table, e.g.
        ``{"my_score": "higher_is_better"}``.
    scheme : str or ColourScheme, optional
        ``"cvd"`` (default, colour-vision-safe) or ``"ecmwf"``.
    bootstrap : {"moving-block", "iid"}, optional
        How to resample forecast cases. Consecutive forecasts share a weather
        system, so an iid resample marks about 44% of truly-null cells as
        significant against a nominal 5%; moving-block brings that to roughly 8%.
        Used only when the input carries an ``init_time`` dimension.
    block_length : int, optional
        Block length in forecast **cases**, not hours. Derived from the
        initialisation cadence when omitted, and the choice is reported.
    n_resamples : int, optional
    confidence_levels : sequence of float, optional
        Fractions, so ``0.95`` rather than ``95``.
    seed : int, optional
        Fixed by default: two runs on the same file must agree, and an OS-seeded
        default would flip borderline significance markings between them.
    strict : bool, optional
        Treat warnings as failures.
    return_validation_report : bool, optional
        Also return the report rather than only raising on failure.

    Returns
    -------
    Layout, or (Layout, ValidationReport)
    """
    sch = SCHEMES[scheme] if isinstance(scheme, str) else scheme

    subset: dict[str, Any] = dict(select or {})
    if truth_source is not None:
        subset["truth_source"] = (
            list(truth_source)
            if isinstance(truth_source, (list, tuple))
            else [truth_source]
        )

    row_dims, col_dims = _infer_axes(data, rows, columns, cell)

    cube = prepare(
        data,
        row_dims=row_dims,
        column_dims=col_dims,
        cell_dim=cell,
        strict=strict,
    )
    agg = aggregate(
        cube,
        control=control,
        experiment=experiment,
        subset=subset or None,
        bootstrap=bootstrap,
        block_length=block_length,
        n_resamples=n_resamples,
        confidence_levels=confidence_levels,
        seed=seed,
    )
    layout = resolve(
        cube,
        agg=agg,
        control=control,
        experiment=experiment,
        row_dims=row_dims,
        column_dims=col_dims,
        cell_dim=cell,
        scheme=sch,
        scaling=FixedScaling(),
        metric_polarity=metric_polarity,
        title=title,
        subtitle=subtitle,
    )
    if return_validation_report:
        return layout, cube.report
    return layout


def render(
    layout: Layout,
    path: str | Path,
    *,
    scheme: str | ColourScheme | None = None,
    dpi: int = 200,
) -> Path:
    """Render a resolved layout to one file, by suffix."""
    sch = (
        SCHEMES[layout.scheme_name]
        if scheme is None
        else (SCHEMES[scheme] if isinstance(scheme, str) else scheme)
    )
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        from .render.html import render_html

        return render_html(layout, path, scheme=sch)
    if suffix in _STATIC_SUFFIXES:
        from .render.static import render_static

        return render_static(layout, path, scheme=sch, dpi=dpi)
    raise ValueError(
        f"don't know how to render {suffix!r}; use .html, or "
        f"{', '.join(sorted(_STATIC_SUFFIXES))}"
    )


def make_scorecard(
    data: xr.Dataset,
    output: str | Path | Sequence[str | Path],
    *,
    control: str,
    experiment: str,
    dpi: int = 200,
    **kwargs: Any,
) -> list[Path]:
    """Build a scorecard and write it to one or more files.

    Output format follows each path's suffix: ``.html`` for the interactive page,
    ``.png``/``.pdf``/``.svg`` for the static figure.

    Parameters
    ----------
    data : xr.Dataset
        Verification statistics.
    output : path or sequence of paths
    control, experiment : str
        Members of the prediction-source coordinate.
    dpi : int, optional
        Raster resolution for PNG output.
    **kwargs
        Forwarded to :func:`build_layout`.

    Returns
    -------
    list of Path

    Examples
    --------
    >>> make_scorecard(ds, ["card.html", "card.png"],
    ...                control="IFS-HRES", experiment="GraphCast")  # doctest: +SKIP
    """
    paths = [output] if isinstance(output, (str, Path)) else list(output)
    layout = build_layout(data, control=control, experiment=experiment, **kwargs)
    assert isinstance(layout, Layout)
    return [render(layout, p, dpi=dpi) for p in paths]
