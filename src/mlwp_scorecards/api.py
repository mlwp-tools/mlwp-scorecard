"""Public API.

Everything the caller needs is passed as plain arguments — coordinate names,
source names, output paths. There is no configuration object to construct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from .colours import SCHEMES, ColourScheme, FixedScaling
from .ingest import ValidationReport, prepare
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
DEFAULT_ROWS = ("truth_source", "variable", "level")
#: Preferred nesting when ``columns`` is not given.
DEFAULT_COLUMNS = ("spatial_region", "region", "metric")

_STATIC_SUFFIXES = {".png", ".pdf", ".svg", ".eps", ".jpg", ".jpeg", ".tif", ".tiff"}


def _infer_axes(
    ds: xr.Dataset,
    rows: Sequence[str] | None,
    columns: Sequence[str] | None,
    cell: str,
    prediction_dim: str,
    stat_dim: str,
) -> tuple[list[str], list[str]]:
    """Choose row and column nesting when the caller did not."""
    available = {str(d) for d in ds.dims} | {"variable"}
    available -= {cell, prediction_dim, stat_dim}

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
    prediction_dim: str = "prediction_source",
    stat_dim: str = "stat",
    metric_dim: str | None = "metric",
    strict: bool = False,
    return_validation_report: bool = False,
) -> Layout | tuple[Layout, ValidationReport]:
    """Resolve a verification dataset into a ready-to-render :class:`Layout`.

    Parameters
    ----------
    data : xr.Dataset
        Verification statistics: one score variable per physical variable.
    control, experiment : str
        Members of ``prediction_dim``. The card colours ``experiment - control``.
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

    row_dims, col_dims = _infer_axes(
        data, rows, columns, cell, prediction_dim, stat_dim
    )

    cube = prepare(
        data,
        row_dims=row_dims,
        column_dims=col_dims,
        cell_dim=cell,
        stat_dim=stat_dim,
        strict=strict,
    )
    layout = resolve(
        cube,
        control=control,
        experiment=experiment,
        row_dims=row_dims,
        column_dims=col_dims,
        cell_dim=cell,
        prediction_dim=prediction_dim,
        stat_dim=stat_dim,
        metric_dim=metric_dim,
        scheme=sch,
        scaling=FixedScaling(),
        metric_polarity=metric_polarity,
        subset=subset or None,
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
