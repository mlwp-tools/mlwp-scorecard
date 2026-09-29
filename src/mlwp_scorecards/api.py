"""Public API.

Everything the caller needs is passed as plain arguments — coordinate names and
source names. There is no configuration object to construct: a
:class:`ScoreCard` is built from the dataset, and gives the card as a matplotlib
figure or an HTML page for the caller to save.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import xarray as xr

from .aggregate import aggregate, expand_selection, resolve_sources
from .colours import SCHEMES, ColourScheme, FixedScaling
from .ingest import (
    CASE_DIM,
    FORECAST_DIM,
    METRIC_DIM,
    VARIABLE_DIM,
    prepare,
    split_name,
)
from .layout import resolve
from .model import Layout

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "ScoreCard",
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


def _is_many(value: Any) -> bool:
    """Tell whether a ``select=`` value picks several members rather than one.

    Several members keep the dimension; one drops it, unless the dimension was
    placed.

    Parameters
    ----------
    value : Any
        One value of the ``select=`` mapping.

    Returns
    -------
    bool
        True for ``...``, a list, a tuple or an array.
    """
    return value is Ellipsis or isinstance(value, (list, tuple, np.ndarray))


def _select_names(data: xr.Dataset, dim: str, value: Any) -> xr.Dataset:
    """Select along ``variable`` or ``metric`` by choosing data variables.

    Both are halves of the ``{metric}.{variable}`` names rather than dimensions of
    the input. Both are always on an axis, so a single value keeps them at length
    one. Order follows the selection, because
    :func:`~mlwp_scorecards.ingest.prepare` orders each by first appearance among
    the data variables.

    Parameters
    ----------
    data : xr.Dataset
        The verification statistics.
    dim : {"variable", "metric"}
        Which half of the names to select on.
    value : Any
        One name, a list of names (with at most one ``...``), or ``...``.

    Returns
    -------
    xr.Dataset
        ``data`` with only the chosen variables, in the order selected.
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
    colour_relative_to: str | None,
    placed: set[str],
) -> tuple[xr.Dataset, tuple[str, ...]]:
    """Apply ``select=`` to the dataset, before any layout is inferred.

    One rule for every key: a single value picks that member and drops the
    dimension -- unless the caller placed the dimension on ``rows``, ``columns`` or
    ``cell``, where it is kept at length one; a list keeps the dimension, subset in
    the order given, with ``...`` for the rest; a slice keeps it, as ``ds.sel``
    would. ``forecast_source`` differs only in that the baseline always stays in
    the data and is never part of a ``...``.

    Parameters
    ----------
    data : xr.Dataset
        The verification statistics.
    select : mapping or None
        The caller's ``select=``.
    colour_relative_to : str or None
        The baseline source, which always stays in the data.
    placed : set of str
        Dimensions the caller named in ``rows``, ``columns`` or ``cell``.

    Returns
    -------
    data : xr.Dataset
        The dataset after selection.
    sources : tuple of str
        The forecast sources to show, in order.

    Raises
    ------
    KeyError
        If ``select=`` names something that is not a dimension, or the dataset
        has no ``forecast_source``.
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
    sources = resolve_sources(
        source_sel, colour_relative_to, data.coords[FORECAST_DIM].values
    )
    return data, sources


def _infer_axes(
    ds: xr.Dataset,
    rows: Sequence[str] | None,
    columns: Sequence[str] | None,
    cell: str,
    n_sources: int = 1,
    lay_out_sources: bool = False,
) -> tuple[list[str], list[str]]:
    """Choose row and column nesting when the caller did not.

    ``variable`` and ``metric`` are not dimensions of the input -- ``prepare``
    produces them by splitting the ``{metric}.{variable}`` names -- so they are
    added here rather than read off the dataset. The excluded names are the ones
    the rendering consumes rather than lays out, and are the same set ``prepare``
    exempts from its unassigned-dimension check; both are built from the
    constants in :mod:`~mlwp_scorecards.ingest` so the two cannot drift apart.

    ``forecast_source`` is consumed by differencing when there is one forecast
    source, and laid out when there are several, or when ``lay_out_sources`` asks
    for it -- outermost on the rows unless placed elsewhere, one block of rows per
    source.

    Parameters
    ----------
    ds : xr.Dataset
        The verification statistics, after selection.
    rows, columns : sequence of str or None
        The caller's nesting, outermost first; None to infer.
    cell : str
        The coordinate drawn inside each cell.
    n_sources : int, optional
        How many forecast sources are shown.
    lay_out_sources : bool, optional
        Put ``forecast_source`` on an axis even for a single forecast source.

    Returns
    -------
    rows : list of str
        Row nesting, outermost first.
    columns : list of str
        Column nesting, outermost first.

    Raises
    ------
    ValueError
        If the dimensions cannot be split so that both axes get at least one.
    """
    laid_out = n_sources > 1 or lay_out_sources
    consumed = {cell, CASE_DIM} | (set() if laid_out else {FORECAST_DIM})
    available = ({str(d) for d in ds.dims} | {VARIABLE_DIM, METRIC_DIM}) - consumed

    if rows is not None and columns is not None:
        return list(rows), list(columns)

    default_rows = DEFAULT_ROWS
    if laid_out and FORECAST_DIM not in (columns or ()):
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


def build_layout(  # numpydoc ignore=PR01
    data: xr.Dataset,
    *,
    colour_relative_to: str | None = None,
    show_values: bool = False,
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
) -> Layout:
    """Resolve a verification dataset into a ready-to-render :class:`Layout`.

    The engine behind :class:`ScoreCard`, and internal: the ``Layout`` is the
    renderer contract, not something a caller needs. The parameters are
    documented on :class:`ScoreCard`, and are not repeated here -- hence the
    ``numpydoc ignore`` on the signature.

    Parameters
    ----------
    data : xr.Dataset
        Verification statistics: one variable per ``{metric}.{variable}`` pair.

    Returns
    -------
    Layout
        The resolved card.
    """
    if colour_relative_to is None and not show_values:
        options = (
            [str(s) for s in data.coords[FORECAST_DIM].values]
            if FORECAST_DIM in data.coords
            else []
        )
        raise ValueError(
            "nothing to show: pass colour_relative_to= to colour by the difference "
            "from a baseline (--colour-relative-to on the command line)"
            + (f", one of: {', '.join(options)}," if options else "")
            + " or show_values=True (--show-values) to print each source's scores"
        )
    # Applied to the dataset before anything is inferred, so a dropped dimension
    # needs no place on the card. Only the caller's own rows/columns count as
    # placing a dimension: inferred axes have not been chosen yet, and could not
    # decide this without the selection deciding them in turn.
    placed = set(rows or ()) | set(columns or ()) | {cell}
    data, sources = _apply_selection(data, select, colour_relative_to, placed)

    sch = SCHEMES[scheme] if isinstance(scheme, str) else scheme
    # With values shown every source is a row of its own -- the baseline too -- so
    # forecast_source is laid out even when there is only one forecast source.
    row_dims, col_dims = _infer_axes(
        data, rows, columns, cell, len(sources), lay_out_sources=show_values
    )

    score, units = prepare(data, row_dims=row_dims, column_dims=col_dims, cell_dim=cell)
    agg = aggregate(
        score,
        forecast_source=sources,
        baseline_source=colour_relative_to,
        cases=cases,
        baseline_row=show_values and colour_relative_to is not None,
        bootstrap=bootstrap,
        block_length=block_length,
        n_resamples=n_resamples,
        confidence_levels=confidence_levels,
        seed=seed,
    )
    return resolve(
        units,
        agg=agg,
        row_dims=row_dims,
        column_dims=col_dims,
        cell_dim=cell,
        scheme=sch,
        scaling=FixedScaling(),
        metric_polarity=metric_polarity,
        title=title,
        subtitle=subtitle,
        show_values=show_values,
    )


class ScoreCard:
    """A scorecard, built from pre-computed verification statistics.

    Building one does all the work -- selection, the paired differences, the
    bootstrap, the layout -- and the result is then drawn with
    :meth:`to_figure` or :meth:`to_html`, for the caller to save. Nothing is
    guessed or quietly relaxed along the way: input the card cannot honestly be
    drawn from raises, naming the choice that would resolve it, and every choice
    that was made -- the resample, the case set -- is printed on the card.

    Parameters
    ----------
    data : xr.Dataset
        Verification statistics: one variable per ``{metric}.{variable}`` pair.
    colour_relative_to : str, optional
        The member of ``forecast_source`` every forecast source is compared with;
        the card colours ``forecast - baseline`` and marks significance. ``None``:
        nothing is compared, every box is neutral, and ``show_values`` must be
        True.
    show_values : bool, optional
        Print each source's own score in its boxes. With a baseline, the baseline
        is also shown as a grey row of its own scores, first, and colours are
        unchanged. ``forecast_source`` is then always on an axis -- outermost on
        the rows unless placed -- even for a single forecast source.
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

        ``forecast_source`` defaults to every source but the baseline (every
        source, with no baseline), and a ``...`` in it never includes the
        baseline; naming the baseline there is an error. With several forecast sources, ``forecast_source`` becomes a
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
    title, subtitle : str, optional
        Printed above the card.
    bootstrap : {"moving-block", "iid"}, optional
        How to resample forecast cases. Consecutive forecasts share a weather
        system, so an iid resample marks about 44% of truly-null cells as
        significant against a nominal 5%; moving-block brings that to roughly 8%.
        Used only when the input carries an ``init_time`` dimension.
    block_length : int, optional
        Block length in forecast **cases**, not hours. Derived from the
        initialisation cadence when omitted: blocks spanning 10 days, which
        needs at least four blocks' worth of cases. With fewer, or no readable
        cadence, building the card raises rather than falling back to ``"iid"``
        -- pass a block length, or ``bootstrap="iid"``, to choose.
    n_resamples : int, optional
        Bootstrap resamples of the forecast cases.
    confidence_levels : sequence of float, optional
        Fractions, so ``0.95`` rather than ``95``.
    seed : int, optional
        Fixed by default: two runs on the same file must agree, and an OS-seeded
        default would flip borderline significance markings between them.

    Raises
    ------
    ValueError
        If ``colour_relative_to`` is None and ``show_values`` is False: the card
        would have nothing on it. Also if the dataset is not in the documented
        shape, or ``block_length`` is omitted and cannot be derived.

    Examples
    --------
    >>> score_card = ScoreCard(ds, colour_relative_to="IFS-HRES",
    ...                        select=dict(forecast_source=["GraphCast", ...]))  # doctest: +SKIP
    >>> score_card.to_figure().savefig("card.png", dpi=200)  # doctest: +SKIP
    >>> Path("card.html").write_text(score_card.to_html())  # doctest: +SKIP
    """

    def __init__(
        self,
        data: xr.Dataset,
        *,
        colour_relative_to: str | None = None,
        show_values: bool = False,
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
    ) -> None:
        self._layout: Layout = build_layout(
            data,
            colour_relative_to=colour_relative_to,
            show_values=show_values,
            select=select,
            cases=cases,
            rows=rows,
            columns=columns,
            cell=cell,
            metric_polarity=metric_polarity,
            scheme=scheme,
            title=title,
            subtitle=subtitle,
            bootstrap=bootstrap,
            block_length=block_length,
            n_resamples=n_resamples,
            confidence_levels=confidence_levels,
            seed=seed,
        )
        self._scheme = SCHEMES[scheme] if isinstance(scheme, str) else scheme

    def __repr__(self) -> str:
        """Name the sources and the card's size.

        Returns
        -------
        str
            E.g. ``<ScoreCard: GraphCast, relative to 'IFS-HRES'; 6 rows x 9
            columns, 378 boxes>``.
        """
        lay = self._layout
        s = lay.stats
        baseline = (
            f", relative to {lay.baseline_source!r}" if lay.baseline_source else ""
        )
        return (
            f"<ScoreCard: {', '.join(lay.forecast_sources)}{baseline}; "
            f"{s.n_rows} rows x {s.n_cols} columns, {s.n_boxes} boxes>"
        )

    def to_figure(self) -> Figure:
        """Draw the card as a matplotlib figure.

        The figure is not registered with ``pyplot`` and the backend is left as
        it was. Save it with ``fig.savefig(path)``, which uses the caller's
        matplotlib settings: set ``pdf.fonttype=42`` and ``svg.fonttype="none"``
        to keep PDF and SVG text selectable.

        Returns
        -------
        matplotlib.figure.Figure
            A new figure on each call.
        """
        from .render.static import render_figure

        return render_figure(self._layout, scheme=self._scheme)

    def to_html(self, *, detail: bool = True) -> str:
        """Render the card as a self-contained interactive HTML page.

        Parameters
        ----------
        detail : bool, optional
            Embed the click-through drill-down data. On a full-size card this is
            the largest thing in the page; pass False for a table-only page.

        Returns
        -------
        str
            The page, to be written to a ``.html`` file.
        """
        from .render.html import render_html

        return render_html(self._layout, scheme=self._scheme, detail=detail)
