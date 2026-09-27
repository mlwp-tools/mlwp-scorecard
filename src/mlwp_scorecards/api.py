"""Public API.

Everything the caller needs is passed as plain arguments — coordinate names,
source names, output paths. There is no configuration object to construct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from .aggregate import aggregate, expand_selection, resolve_sources
from .colours import SCHEMES, ColourScheme, FixedScaling
from .ingest import (
    CASE_DIM,
    FORECAST_DIM,
    METRIC_DIM,
    VARIABLE_DIM,
    ValidationReport,
    prepare,
    split_name,
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
_HTML_SUFFIXES = {".html", ".htm"}


def _is_many(value: Any) -> bool:
    """Whether a ``select=`` value picks several members (keeping the dimension)
    rather than one (which drops it, unless the dimension was placed)."""
    return value is Ellipsis or isinstance(value, (list, tuple, np.ndarray))


def _select_names(data: xr.Dataset, dim: str, value: Any) -> xr.Dataset:
    """Select along ``variable`` or ``metric``, which are halves of the data-variable
    names rather than dimensions of the input.

    Both are always on an axis, so a single value keeps them at length one. Order
    follows the selection, because :func:`~mlwp_scorecards.ingest.prepare` orders
    each by first appearance among the data variables.
    """
    half = 0 if dim == METRIC_DIM else 1
    names = [str(n) for n in data.data_vars]
    parts = {n: split_name(n)[half] for n in names}
    available = list(dict.fromkeys(parts[n] for n in names))
    if value is Ellipsis or not _is_many(value):
        values = [value]
    else:
        values = list(value)
    chosen = expand_selection(dim, values, available)
    ordered = [n for v in chosen for n in names if parts[n] == v]
    return data[ordered]


def _apply_selection(
    data: xr.Dataset,
    select: Mapping[str, Any] | None,
    relative_to: str | None,
    placed: set[str],
) -> tuple[xr.Dataset, tuple[str, ...]]:
    """Apply ``select=`` to the dataset, before any layout is inferred.

    One rule for every key: a single value picks that member and drops the
    dimension -- unless the caller placed the dimension on ``rows``, ``columns`` or
    ``cell``, where it is kept at length one; a list keeps the dimension, subset in
    the order given, with ``...`` for the rest; a slice keeps it, as ``ds.sel``
    would. ``forecast_source`` differs only in that the baseline always stays in
    the data and is never part of a ``...``.

    Returns
    -------
    data : xr.Dataset
    sources : tuple of str
        The forecast sources to show, in order.
    """
    chosen = dict(select or {})
    source_sel = chosen.pop(FORECAST_DIM, ...)
    dims = {str(d) for d in data.dims}
    unknown = sorted(set(chosen) - dims - {VARIABLE_DIM, METRIC_DIM})
    if unknown:
        raise KeyError(
            f"select= names {unknown}, which are not dimensions of the dataset "
            f"(have: {sorted(dims | {VARIABLE_DIM, METRIC_DIM})})"
        )
    for dim, value in chosen.items():
        if dim in (VARIABLE_DIM, METRIC_DIM):
            data = _select_names(data, dim, value)
        elif isinstance(value, slice):
            data = data.sel({dim: value})
        elif _is_many(value):
            values = [value] if value is Ellipsis else list(value)
            data = data.sel({dim: expand_selection(dim, values, data[dim].values)})
        elif dim in placed:
            data = data.sel({dim: [value]})
        else:
            data = data.sel({dim: value}, drop=True)

    if FORECAST_DIM not in data.dims:
        raise KeyError(f"{FORECAST_DIM!r} is not a dimension of the dataset")
    sources = resolve_sources(source_sel, relative_to, data.coords[FORECAST_DIM].values)
    return data, sources


def _infer_axes(
    ds: xr.Dataset,
    rows: Sequence[str] | None,
    columns: Sequence[str] | None,
    cell: str,
    n_sources: int = 1,
) -> tuple[list[str], list[str]]:
    """Choose row and column nesting when the caller did not.

    ``variable`` and ``metric`` are not dimensions of the input -- ``prepare``
    produces them by splitting the ``{metric}.{variable}`` names -- so they are
    added here rather than read off the dataset. The excluded names are the ones
    the rendering consumes rather than lays out, and are the same set ``prepare``
    exempts from its unassigned-dimension check; both are built from the
    constants in :mod:`~mlwp_scorecards.ingest` so the two cannot drift apart.

    ``forecast_source`` is consumed by differencing when there is one forecast
    source, and laid out when there are several -- outermost on the rows unless
    placed elsewhere, one block of rows per source.
    """
    consumed = {cell, CASE_DIM} | ({FORECAST_DIM} if n_sources == 1 else set())
    available = ({str(d) for d in ds.dims} | {VARIABLE_DIM, METRIC_DIM}) - consumed

    if rows is not None and columns is not None:
        return list(rows), list(columns)

    default_rows = DEFAULT_ROWS
    if n_sources > 1 and FORECAST_DIM not in (columns or ()):
        default_rows = (FORECAST_DIM,) + DEFAULT_ROWS
    r = [d for d in (rows if rows is not None else default_rows) if d in available]
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
    relative_to: str | None = None,
    select: Mapping[str, Any] | None = None,
    cases: str = "common",
    rows: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    cell: str = "lead_time",
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
    relative_to : str
        The member of ``forecast_source`` every forecast source is compared with;
        the card colours ``forecast - baseline``. ``None`` will mean a card of
        absolute scores with no baseline, which is not implemented yet and raises.
    select : mapping, optional
        Selection along coordinates, one rule for every key, e.g.
        ``select=dict(forecast_source=["GraphCast", ...], truth_source="analysis",
        spatial_region=["europe", "n.hem"])``:

        - a **single value** picks that member and drops the dimension, unless
          the dimension is named in ``rows``, ``columns`` or ``cell``, where it is
          kept at length one;
        - a **list** keeps the dimension, subset **in the order given** -- which
          is the order it is drawn in -- with at most one ``...`` for every other
          value, in coordinate order;
        - a **slice** keeps the dimension, as ``ds.sel`` would.

        ``forecast_source`` defaults to every source but the baseline, and a
        ``...`` in it never includes the baseline; naming the baseline there is
        an error. With several forecast sources, ``forecast_source`` becomes a
        layout axis. ``variable`` and ``metric`` may be selected too, and always
        stay on an axis.
    cases : {"common", "pairwise"}, optional
        Which forecast cases each comparison rests on. ``"common"``: only those
        every selected source and the baseline scored, so rows are comparable with
        one another. ``"pairwise"``: those each forecast source shares with the
        baseline. Identical when there is one forecast source.
    rows, columns : sequence of str, optional
        Coordinate names to nest on each axis, outermost first. Inferred when omitted.
    cell : str, optional
        Coordinate drawn inside each cell, normally ``"lead_time"``.
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

    Raises
    ------
    NotImplementedError
        If ``relative_to`` is None: the absolute-score card is not built yet.
    """
    if relative_to is None:
        options = (
            [str(s) for s in data.coords[FORECAST_DIM].values]
            if FORECAST_DIM in data.coords
            else []
        )
        raise NotImplementedError(
            "a card of absolute scores (relative_to=None) is not implemented yet; "
            "name the baseline with relative_to= (--relative-to on the command line)"
            + (f", one of: {', '.join(options)}" if options else "")
        )
    # Applied to the dataset before anything is inferred, so a dropped dimension
    # needs no place on the card. Only the caller's own rows/columns count as
    # placing a dimension: inferred axes have not been chosen yet, and could not
    # decide this without the selection deciding them in turn.
    placed = set(rows or ()) | set(columns or ()) | {cell}
    data, sources = _apply_selection(data, select, relative_to, placed)

    sch = SCHEMES[scheme] if isinstance(scheme, str) else scheme
    row_dims, col_dims = _infer_axes(data, rows, columns, cell, len(sources))

    cube = prepare(
        data,
        row_dims=row_dims,
        column_dims=col_dims,
        cell_dim=cell,
        strict=strict,
    )
    agg = aggregate(
        cube,
        forecast_source=sources,
        baseline_source=relative_to,
        cases=cases,
        bootstrap=bootstrap,
        block_length=block_length,
        n_resamples=n_resamples,
        confidence_levels=confidence_levels,
        seed=seed,
    )
    layout = resolve(
        cube,
        agg=agg,
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
    if suffix in _HTML_SUFFIXES:
        from .render.html import render_html

        return render_html(layout, path, scheme=sch)
    if suffix in _STATIC_SUFFIXES:
        from .render.static import render_static

        return render_static(layout, path, scheme=sch, dpi=dpi)
    raise ValueError(
        f"don't know how to render {suffix!r}; use .html, or "
        f"{', '.join(sorted(_STATIC_SUFFIXES))}"
    )


def _output_paths(
    html_path: str | Path | None,
    image_path: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    """Check the requested outputs before any work is done, and order them.

    A suffix that contradicts the argument it was passed to is refused rather than
    re-guessed: ``html_path="card.png"`` is far more likely a slip than a request
    for a PNG, and silently writing one would hide it.
    """
    if html_path is None and image_path is None:
        raise ValueError("nothing to write: pass html_path=, image_path=, or both")
    paths = []
    if html_path is not None:
        path = Path(html_path)
        if path.suffix.lower() not in _HTML_SUFFIXES:
            raise ValueError(f"html_path={str(html_path)!r} does not end in .html")
        paths.append(path)
    if image_path is None:
        images = []
    elif isinstance(image_path, (str, Path)):
        images = [image_path]
    else:
        images = list(image_path)
    for img in images:
        path = Path(img)
        if path.suffix.lower() not in _STATIC_SUFFIXES:
            raise ValueError(
                f"image_path={str(img)!r}: expected one of "
                f"{', '.join(sorted(_STATIC_SUFFIXES))}"
            )
        paths.append(path)
    return paths


def make_scorecard(
    data: xr.Dataset,
    *,
    relative_to: str | None = None,
    select: Mapping[str, Any] | None = None,
    html_path: str | Path | None = None,
    image_path: str | Path | Sequence[str | Path] | None = None,
    dpi: int = 200,
    **kwargs: Any,
) -> list[Path]:
    """Build a scorecard and write the interactive page, static figures, or both.

    Parameters
    ----------
    data : xr.Dataset
        Verification statistics.
    relative_to : str
        The source each is compared with.
    select : mapping, optional
        Selection along coordinates, including which forecast sources to show;
        see :func:`build_layout`.
    html_path : path, optional
        Where to write the self-contained interactive page. Must end in ``.html``.
    image_path : path or sequence of paths, optional
        Where to write the static figure. The format follows each suffix --
        ``.png``, ``.pdf``, ``.svg`` and so on -- so several paths give several
        formats of the same card.
    dpi : int, optional
        Raster resolution for ``image_path``; ignored by the HTML page.
    **kwargs
        Forwarded to :func:`build_layout`.

    Returns
    -------
    list of Path
        The files written: the page first, then the figures in the order given.

    Raises
    ------
    ValueError
        If neither ``html_path`` nor ``image_path`` is given, or a suffix
        contradicts the argument it was passed to. Checked before anything is
        computed.

    Examples
    --------
    >>> make_scorecard(ds, relative_to="IFS-HRES",
    ...                select=dict(forecast_source=["GraphCast", ...]),
    ...                html_path="card.html",
    ...                image_path=["card.png", "card.pdf"])  # doctest: +SKIP
    """
    paths = _output_paths(html_path, image_path)
    layout = build_layout(data, relative_to=relative_to, select=select, **kwargs)
    assert isinstance(layout, Layout)
    return [render(layout, p, dpi=dpi) for p in paths]
