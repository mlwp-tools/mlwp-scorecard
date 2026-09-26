"""Public API.

Everything the caller needs is passed as plain arguments — coordinate names,
source names, output paths. There is no configuration object to construct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from .aggregate import aggregate, resolve_sources
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
_HTML_SUFFIXES = {".html", ".htm"}


def _is_many(value: Any) -> bool:
    """Whether a ``select=`` value picks several members (keeping the dimension)
    rather than one (dropping it)."""
    return isinstance(value, (list, tuple, slice, np.ndarray))


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
    predictions_from: str | Sequence[Any] = (...,),
    relative_to: str | None = None,
    cases: str = "common",
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
    predictions_from : str or sequence, optional
        Members of ``forecast_source`` to show, in order. ``...`` stands for every
        source not otherwise named, in coordinate order, so ``["GraphCast", ...]``
        puts GraphCast first and then all the rest; the default, ``(...,)``, is
        every source but the baseline. With several, ``forecast_source`` becomes
        a layout axis: each row (or column) is one of them compared with the
        baseline.
    relative_to : str
        The member of ``forecast_source`` each is compared with; the card colours
        ``forecast - baseline``. Left out of any ``...``, and an error to name in
        ``predictions_from`` as well. ``None`` will mean a card of absolute scores
        with no baseline, which is not implemented yet and raises.
    cases : {"common", "pairwise"}, optional
        Which forecast cases each comparison rests on. ``"common"``: only those
        every selected source and the baseline scored, so rows are comparable with
        one another. ``"pairwise"``: those each forecast source shares with the
        baseline. Identical when there is one forecast source.
    rows, columns : sequence of str, optional
        Coordinate names to nest on each axis, outermost first. Inferred when omitted.
    cell : str, optional
        Coordinate drawn inside each cell, normally ``"lead_time"``.
    truth_source : str or sequence of str, optional
        Restrict to these truth sources.
    select : mapping, optional
        Further coordinate subsetting. A single value picks that member and drops
        the dimension, so it needs no place on the card:
        ``select={"spatial_region": "europe"}``. A list or slice keeps the
        dimension, subset, and it still has to be on ``rows`` or ``columns``.
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
        raise NotImplementedError(
            "a card of absolute scores (relative_to=None) is not implemented yet; "
            "name the baseline with relative_to="
        )
    if FORECAST_DIM not in data.dims:
        raise KeyError(f"{FORECAST_DIM!r} is not a dimension of the dataset")
    sources = resolve_sources(
        predictions_from, relative_to, data.coords[FORECAST_DIM].values
    )

    sch = SCHEMES[scheme] if isinstance(scheme, str) else scheme

    # A single value picks one member and drops the dimension -- it is taken
    # before the axes are inferred, so it needs no home on the card. A list keeps
    # the dimension, subset, and still has to be placed.
    scalars = {k: v for k, v in (select or {}).items() if not _is_many(v)}
    subset: dict[str, Any] = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in (select or {}).items()
        if _is_many(v)
    }
    if scalars:
        missing = sorted(set(scalars) - {str(d) for d in data.dims})
        if missing:
            raise KeyError(f"select= names {missing}, which are not dimensions")
        data = data.sel(scalars, drop=True)
    if truth_source is not None:
        subset["truth_source"] = (
            list(truth_source)
            if isinstance(truth_source, (list, tuple))
            else [truth_source]
        )

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
    predictions_from: str | Sequence[Any] = (...,),
    relative_to: str | None = None,
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
    predictions_from : str or sequence, optional
        The sources to show, with ``...`` for all the rest; see
        :func:`build_layout`.
    relative_to : str
        The source each is compared with.
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
    >>> make_scorecard(ds, predictions_from=["GraphCast"], relative_to="IFS-HRES",
    ...                html_path="card.html",
    ...                image_path=["card.png", "card.pdf"])  # doctest: +SKIP
    """
    paths = _output_paths(html_path, image_path)
    layout = build_layout(
        data,
        predictions_from=predictions_from,
        relative_to=relative_to,
        **kwargs,
    )
    assert isinstance(layout, Layout)
    return [render(layout, p, dpi=dpi) for p in paths]
