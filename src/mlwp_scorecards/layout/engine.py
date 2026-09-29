"""The layout engine: collapsed scores plus a layout declaration in, a :class:`Layout` out.

Stateless: :func:`create_layout` is a pure function of its arguments. The
ordering rule is one global order per dimension, filtered by presence within
each parent branch. Verified against the ECMWF reference: a single variable list
reproduces both its ``an`` and ``ob`` row sequences with no per-branch ordering.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from ..aggregate import Aggregated
from ..colours import ColourScheme, FixedScaling, Polarity, family_of, polarity_of
from ..ingest import FORECAST_DIM, METRIC_DIM, VARIABLE_DIM
from .model import (
    NEUTRAL,
    Cell,
    HeaderCell,
    Key,
    Layout,
    LayoutStats,
    Line,
    Step,
    format_value,
)

__all__ = ["create_layout"]

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def _is_na(v: Any) -> bool:
    """Return whether a coordinate value marks a not-applicable dimension.

    Parameters
    ----------
    v : Any
        A coordinate value.

    Returns
    -------
    bool
        True for None or a float NaN.
    """
    return v is None or (isinstance(v, float) and math.isnan(v))


def _slug(parts: Sequence[Any]) -> str:
    """Join a key's values into an identifier-safe string.

    Parameters
    ----------
    parts : sequence of Any
        The key's coordinate values.

    Returns
    -------
    str
        Each value with runs of non-alphanumerics replaced by ``_`` (``na`` if
        nothing is left), joined with ``-``.
    """
    return "-".join(_SLUG_RE.sub("_", str(p)).strip("_") or "na" for p in parts)


def _label(v: Any) -> str:
    """Return the text a header shows for a coordinate value.

    Parameters
    ----------
    v : Any
        A coordinate value.

    Returns
    -------
    str
        Empty for not-applicable, ``T+h`` for a time delta, an integer for a
        whole float, and ``str(v)`` otherwise.
    """
    if _is_na(v):
        return ""
    if isinstance(v, (np.timedelta64, _dt.timedelta)):
        return _lead_label(v)
    if isinstance(v, (np.floating, float)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _lead_hours(v: Any) -> float:
    """Return a lead time in hours, from a timedelta or a bare number.

    Parameters
    ----------
    v : Any
        A ``numpy.timedelta64``, a ``datetime.timedelta``, or a number already in
        hours.

    Returns
    -------
    float
        The lead time in hours.
    """
    if isinstance(v, np.timedelta64):
        return float(v / np.timedelta64(1, "h"))
    if isinstance(v, _dt.timedelta):
        return v.total_seconds() / 3600.0
    return float(v)


def _lead_label(v: Any) -> str:
    """Return the label for a lead time.

    Parameters
    ----------
    v : Any
        A lead time, in any form :func:`_lead_hours` accepts.

    Returns
    -------
    str
        ``T+`` followed by the hours, such as ``T+24``.
    """
    return f"T+{_lead_hours(v):g}"


def _pct(c: float) -> str:
    """Format a confidence level as a percentage, keeping 99.7% from rounding to 100%.

    Parameters
    ----------
    c : float
        The level as a fraction.

    Returns
    -------
    str
        The level as a percentage, to four significant figures.
    """
    return f"{c * 100:.4g}%"


def _coord_values(cube: xr.DataArray, dim: str) -> list[Any]:
    """Return coordinate values as plain Python objects, with NaN normalised to None.

    NaN cannot be used in a layout key: ``nan != nan``, so two layouts built from
    the same data would compare unequal and ``sel(level=float("nan"))`` could never
    match. None is the not-applicable marker in keys; NaN stays in the data.

    Times are the exception to the unwrapping. ``.item()`` on a ``timedelta64``
    gives a ``datetime.timedelta`` at every resolution *except* nanoseconds,
    where numpy returns a bare int of nanoseconds -- and an int reaching
    :func:`_lead_hours` is indistinguishable from a lead time already given in
    hours, so ``T+6`` came out as ``T+2.16e+13``. Nanoseconds are what a netCDF
    round trip commonly decodes to, so this is the common case, not the exotic
    one. Leave numpy time scalars alone; every consumer below already handles
    them, and they are hashable and comparable so they remain valid keys.

    Parameters
    ----------
    cube : xr.DataArray
        The array whose coordinate to read.
    dim : str
        The dimension.

    Returns
    -------
    list
        The values, in the coordinate's order.
    """
    out = []
    for v in cube.coords[dim].values:
        if not isinstance(v, (np.timedelta64, np.datetime64)):
            v = v.item() if hasattr(v, "item") else v
        out.append(None if isinstance(v, float) and math.isnan(v) else v)
    return out


def _order_key(coord_values: Sequence[Any], order: Sequence[Any] | None):
    """Return a sort key for one dimension's categories.

    The default is the order the coordinate already has in the dataset: the caller
    controls presentation order by ordering their coordinate, which is far less
    surprising than imposing an alphabetical sort on metric or region names.

    Parameters
    ----------
    coord_values : sequence of Any
        The dimension's values, in dataset order.
    order : sequence of Any or None
        An explicit order that replaces the dataset's, if given.

    Returns
    -------
    callable
        A key function: position in the reference order, values absent from it
        last, ties broken by label.
    """
    ref = list(order) if order else list(coord_values)
    pos = {v: i for i, v in enumerate(ref)}
    return lambda v: (pos.get(v, len(pos)), _label(v))


def _resolve_axis(
    present: np.ndarray,
    dims: Sequence[str],
    coords: Mapping[str, list[Any]],
    orders: Mapping[str, Sequence[Any]],
) -> tuple[list[Key], list[list[HeaderCell]]]:
    """Build the ordered leaf keys and per-depth header blocks for one axis.

    ``present`` is a boolean array over the axis dims: True where any data exists.
    Categories with no data anywhere below them are dropped, so the dense cube's
    unused combinations never become rows.

    Parameters
    ----------
    present : np.ndarray
        Boolean, one axis per entry of ``dims``, in that order.
    dims : sequence of str
        The axis dimensions, outermost first.
    coords : mapping of str to list
        Each dimension's values, indexing ``present``.
    orders : mapping of str to sequence
        Explicit category orders, by dimension; see :func:`_order_key`.

    Returns
    -------
    leaves : list of Key
        The leaf keys, in display order.
    headers : list of list of HeaderCell
        The header blocks at each depth, outermost first.
    """
    axis_idx = {d: i for i, d in enumerate(dims)}

    def children(prefix: tuple[Any, ...]) -> list[Any]:
        """Return the next dimension's values that have data below a prefix.

        Parameters
        ----------
        prefix : tuple
            Values of the outermost dimensions, one per depth so far.

        Returns
        -------
        list
            The next dimension's values with any data under ``prefix``, sorted.
        """
        depth = len(prefix)
        dim = dims[depth]
        sel: list[Any] = []
        for v in coords[dim]:
            idx: list[Any] = [slice(None)] * len(dims)
            for d, pv in zip(dims, prefix):
                idx[axis_idx[d]] = coords[d].index(pv)
            idx[axis_idx[dim]] = coords[dim].index(v)
            if present[tuple(idx)].any():
                sel.append(v)
        return sorted(sel, key=_order_key(coords[dim], orders.get(dim)))

    leaves: list[Key] = []

    def walk(prefix: tuple[Any, ...]) -> None:
        """Append every leaf key under a prefix to ``leaves``, depth first.

        Parameters
        ----------
        prefix : tuple
            Values of the outermost dimensions, one per depth so far.
        """
        if len(prefix) == len(dims):
            leaves.append(prefix)
            return
        kids = children(prefix)
        if not kids:
            return
        # a dimension that is NA for this branch contributes a single blank leaf
        for v in kids:
            walk(prefix + (v,))

    walk(())

    # header blocks: maximal runs of equal prefix at each depth
    headers: list[list[HeaderCell]] = []
    for depth, dim in enumerate(dims):
        blocks: list[HeaderCell] = []
        start = 0
        while start < len(leaves):
            pref = leaves[start][: depth + 1]
            stop = start
            while stop < len(leaves) and leaves[stop][: depth + 1] == pref:
                stop += 1
            v = pref[-1]
            blocks.append(
                HeaderCell(
                    dim=dim,
                    key=v,
                    label=_label(v),
                    depth=depth,
                    start=start,
                    span=stop - start,
                    is_na=_is_na(v),
                )
            )
            start = stop
        headers.append(blocks)
    return leaves, headers


def _headers_for(headers: list[list[HeaderCell]], index: int) -> tuple[HeaderCell, ...]:
    """Return the header block covering one line at each depth.

    Parameters
    ----------
    headers : list of list of HeaderCell
        The axis's header blocks, one list per depth.
    index : int
        The line's position on the axis.

    Returns
    -------
    tuple of HeaderCell
        One block per depth, outermost first.
    """
    out = []
    for depth in range(len(headers)):
        for blk in headers[depth]:
            if blk.start <= index < blk.stop:
                out.append(blk)
                break
    return tuple(out)


def create_layout(
    units: Mapping[tuple[str, str], str | None],
    *,
    row_dims: Sequence[str],
    column_dims: Sequence[str],
    cell_dim: str,
    scheme: ColourScheme,
    scaling: FixedScaling,
    metric_polarity: Mapping[str, str] | None = None,
    agg: Aggregated,
    title: str = "",
    subtitle: str = "",
    show_values: bool = False,
) -> Layout:
    """Create the :class:`Layout` for scores already collapsed over cases.

    Differences each forecast source from the baseline, then places everything
    on rows, columns and cells -- order, header spans, each box's colour,
    significance, tooltip and printed value -- with the card's notes and counts.
    The result is the whole card as plain values: a renderer only draws it.
    Where :func:`~mlwp_scorecards.api.build_layout` goes from a dataset to a
    ``Layout``, this goes from the aggregated numbers.

    Parameters
    ----------
    units : mapping of (str, str) to str or None
        Units keyed by ``(metric, variable)``, from
        :func:`~mlwp_scorecards.ingest.prepare`.
    row_dims : sequence of str
        The dimensions nested on the rows, outermost first.
    column_dims : sequence of str
        The dimensions nested on the columns, outermost first.
    cell_dim : str
        The dimension laid out inside each cell.
    scheme : ColourScheme
        The palette whose words go into the tooltips, and whose name the layout
        records.
    scaling : FixedScaling
        Maps each relative difference to a ramp level.
    metric_polarity : mapping of str to str, optional
        Polarities for metrics absent from the built-in table, or overriding it;
        see :func:`~mlwp_scorecards.colours.polarity_of`.
    agg : Aggregated
        The collapse over forecast cases, from
        :func:`~mlwp_scorecards.aggregate.aggregate`. Subsetting and the choice
        of sources happen there, because both must precede the resample. With no
        baseline in it, nothing is compared and every box is neutral.
    title : str, optional
        The card's title.
    subtitle : str, optional
        The card's subtitle.
    show_values : bool, optional
        Print each source's own score in its boxes, and, when there is a baseline,
        show it as a grey row of its own scores, first. Needs ``forecast_source``
        on an axis, and :attr:`Aggregated.baseline_row` when there is a baseline.

    Returns
    -------
    Layout
        The card, with everything a renderer needs decided.

    Raises
    ------
    ValueError
        If ``forecast_source`` is on neither axis while there are several forecast
        sources, or while values are shown: the card would have nowhere to put
        them.
    """
    on_axis = FORECAST_DIM in list(row_dims) + list(column_dims)
    if not on_axis and len(agg.forecast_sources) > 1:
        raise ValueError(
            f"{len(agg.forecast_sources)} forecast sources, but {FORECAST_DIM!r} is "
            f"on neither rows nor columns; add it to one of them"
        )
    if not on_axis and show_values:
        raise ValueError(
            f"show_values=True gives every source a row of its own, but "
            f"{FORECAST_DIM!r} is on neither rows nor columns; add it to one of them"
        )

    def _drop_unplaced_source(da: xr.DataArray | None) -> xr.DataArray | None:
        """Drop the forecast-source dimension when no axis carries it.

        Parameters
        ----------
        da : xr.DataArray or None
            An array from ``agg``.

        Returns
        -------
        xr.DataArray or None
            ``da`` unchanged when the source is on an axis, else squeezed.
        """
        # One source and no axis for it: today's two-source card.
        if da is None or on_axis:
            return da
        return da.squeeze(FORECAST_DIM, drop=True)

    coloured = agg.baseline is not None
    ctl, exp = _drop_unplaced_source(agg.baseline), _drop_unplaced_source(agg.forecast)
    if coloured:
        diff = exp - ctl
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = diff / np.abs(ctl)
    else:
        diff = rel = xr.full_like(exp, np.nan)
    ctl_lo, ctl_hi = _drop_unplaced_source(agg.baseline_lower), _drop_unplaced_source(
        agg.baseline_upper
    )
    exp_lo, exp_hi = _drop_unplaced_source(agg.forecast_lower), _drop_unplaced_source(
        agg.forecast_upper
    )
    dif = {
        c: (_drop_unplaced_source(lo), _drop_unplaced_source(hi))
        for c, (lo, hi) in agg.paired.items()
    }
    counts = _drop_unplaced_source(agg.counts)
    levels = tuple(agg.confidence_levels)
    chart_conf = max(levels) if levels else None

    # The baseline's own row, first along forecast_source: its scores where the
    # forecast sources have theirs, and NaN wherever a comparison would go.
    baseline_key = None
    if show_values and coloured:
        row = agg.baseline_row
        if row is None:
            raise ValueError("show_values=True needs aggregate(..., baseline_row=True)")
        baseline_key = agg.baseline_source

        def _first(da, own=None):
            """Prepend the baseline's own entry along the forecast-source dimension.

            Parameters
            ----------
            da : xr.DataArray or None
                An array with one entry per forecast source.
            own : xr.DataArray, optional
                The baseline's values for this array; NaN where not given.

            Returns
            -------
            xr.DataArray or None
                ``da`` with the baseline's entry first, or None if ``da`` is.
            """
            if da is None:
                return None
            head = own if own is not None else xr.full_like(row.mean, np.nan)
            head = head.expand_dims({FORECAST_DIM: [baseline_key]})
            return xr.concat([head, da], dim=FORECAST_DIM).transpose(*da.dims)

        exp = _first(exp, row.mean)
        exp_lo, exp_hi = _first(exp_lo, row.lower), _first(exp_hi, row.upper)
        counts = _first(counts, row.counts)
        diff, rel, ctl = _first(diff), _first(rel), _first(ctl)
        ctl_lo, ctl_hi = _first(ctl_lo), _first(ctl_hi)
        dif = {c: (_first(lo), _first(hi)) for c, (lo, hi) in dif.items()}

    return _place(
        units=units,
        row_dims=row_dims,
        column_dims=column_dims,
        cell_dim=cell_dim,
        scheme=scheme,
        scaling=scaling,
        metric_polarity=metric_polarity,
        title=title,
        subtitle=subtitle,
        diff=diff,
        rel=rel,
        ctl=ctl,
        exp=exp,
        ctl_lo=ctl_lo,
        ctl_hi=ctl_hi,
        exp_lo=exp_lo,
        exp_hi=exp_hi,
        dif=dif,
        counts=counts,
        levels=levels,
        chart_conf=chart_conf,
        agg=agg,
        coloured=coloured,
        show_values=show_values,
        baseline_key=baseline_key,
    )


def _place(
    *,
    units: Mapping[tuple[str, str], str | None],
    row_dims: Sequence[str],
    column_dims: Sequence[str],
    cell_dim: str,
    scheme: ColourScheme,
    scaling: FixedScaling,
    metric_polarity: Mapping[str, str] | None,
    title: str,
    subtitle: str,
    diff: xr.DataArray,
    rel: xr.DataArray,
    ctl: xr.DataArray | None,
    exp: xr.DataArray,
    ctl_lo: xr.DataArray | None,
    ctl_hi: xr.DataArray | None,
    exp_lo: xr.DataArray | None,
    exp_hi: xr.DataArray | None,
    dif: dict[float, tuple[xr.DataArray, xr.DataArray]],
    counts: xr.DataArray | None,
    levels: tuple[float, ...],
    chart_conf: float | None,
    agg: Aggregated,
    coloured: bool = True,
    show_values: bool = False,
    baseline_key: str | None = None,
) -> Layout:
    """Lay collapsed arrays out on the card. Indifferent to where they came from.

    A cell is *compared* when there is a baseline and it is not the baseline's own
    row: coloured by the difference, as always. Every other cell is neutral: its
    boxes are the source's own score, with nothing marked significant.

    Every array is indexed by ``row_dims``, ``column_dims`` and ``cell_dim``;
    counts may lack some of the axis dimensions.

    Parameters
    ----------
    units : mapping of (str, str) to str or None
        Units keyed by ``(metric, variable)``.
    row_dims : sequence of str
        The dimensions nested on the rows, outermost first.
    column_dims : sequence of str
        The dimensions nested on the columns, outermost first.
    cell_dim : str
        The dimension laid out inside each cell.
    scheme : ColourScheme
        The palette whose words go into the tooltips.
    scaling : FixedScaling
        Maps each relative difference to a ramp level.
    metric_polarity : mapping of str to str or None
        Caller-supplied metric polarities.
    title : str
        The card's title.
    subtitle : str
        The card's subtitle.
    diff : xr.DataArray
        Forecast minus baseline; all NaN when there is no baseline.
    rel : xr.DataArray
        ``diff`` over the baseline's magnitude, not yet signed by polarity.
    ctl : xr.DataArray or None
        The baseline's scores.
    exp : xr.DataArray
        Each forecast source's own scores.
    ctl_lo : xr.DataArray or None
        Lower end of the baseline's own interval.
    ctl_hi : xr.DataArray or None
        Upper end of the baseline's own interval.
    exp_lo : xr.DataArray or None
        Lower end of each forecast source's own interval.
    exp_hi : xr.DataArray or None
        Upper end of each forecast source's own interval.
    dif : dict of float to tuple of xr.DataArray
        The paired difference interval ``(lower, upper)``, by confidence level.
    counts : xr.DataArray or None
        Forecast cases behind each comparison.
    levels : tuple of float
        The confidence levels supplied, ascending.
    chart_conf : float or None
        The level whose paired interval goes into :attr:`Step.value_lower` and
        :attr:`Step.value_upper`: the widest.
    agg : Aggregated
        The aggregation these came from, for its sources and resampling details.
    coloured : bool, optional
        Whether there is a baseline to compare with.
    show_values : bool, optional
        Whether each box prints its source's own score.
    baseline_key : str or None, optional
        The forecast-source entry holding the baseline's own row, if there is one.

    Returns
    -------
    Layout
        The resolved card, with its caveat notes.

    Raises
    ------
    KeyError
        If a layout dimension is not in the arrays, or an array dimension is
        assigned to neither rows, columns nor cell.
    """
    dims = list(row_dims) + list(column_dims)
    for d in dims + [cell_dim]:
        if d not in diff.dims:
            raise KeyError(
                f"{d!r} is not a dimension of the prepared cube {tuple(diff.dims)}"
            )
    unassigned = set(diff.dims) - set(dims) - {cell_dim}
    if unassigned:
        raise KeyError(
            f"dimension(s) {sorted(unassigned)} are assigned to neither rows, columns "
            f"nor cell"
        )

    diff = diff.transpose(*dims, cell_dim)
    rel = rel.transpose(*dims, cell_dim)
    if ctl is not None:
        ctl = ctl.transpose(*dims, cell_dim)
    exp = exp.transpose(*dims, cell_dim)
    if ctl_lo is not None:
        ctl_lo = ctl_lo.transpose(*dims, cell_dim)
        ctl_hi = ctl_hi.transpose(*dims, cell_dim)
    if exp_lo is not None:
        exp_lo = exp_lo.transpose(*dims, cell_dim)
        exp_hi = exp_hi.transpose(*dims, cell_dim)
    dif = {
        c: (lo.transpose(*dims, cell_dim), hi.transpose(*dims, cell_dim))
        for c, (lo, hi) in dif.items()
    }
    if counts is not None:
        counts = counts.transpose(*[d for d in dims if d in counts.dims], cell_dim)

    coords = {d: _coord_values(diff, d) for d in dims}
    orders: dict[str, Sequence[Any]] = {}

    # What there is to draw: the difference on a compared card, and each source's
    # own score wherever a neutral box can stand -- the baseline's row included.
    present = diff if coloured and not show_values else exp
    finite = np.isfinite(present.values)  # (…dims…, cell)
    any_data = finite.any(axis=-1)  # (…dims…)

    n_row = len(row_dims)
    row_present = any_data.any(axis=tuple(range(n_row, len(dims))))
    col_present = any_data.any(axis=tuple(range(n_row)))

    row_keys, row_headers = _resolve_axis(row_present, list(row_dims), coords, orders)
    col_keys, col_headers = _resolve_axis(
        col_present, list(column_dims), coords, orders
    )

    rows = tuple(
        Line(i, k, _headers_for(row_headers, i), _slug(k))
        for i, k in enumerate(row_keys)
    )
    columns = tuple(
        Line(i, k, _headers_for(col_headers, i), _slug(k))
        for i, k in enumerate(col_keys)
    )

    leads_raw = _coord_values(diff, cell_dim)
    lead_times = tuple(_lead_hours(v) for v in leads_raw)
    lead_labels = tuple(_lead_label(v) for v in leads_raw)

    def locate(key: Key, dim_names: Sequence[str]) -> tuple[int, ...]:
        """Return the array indices of a key's coordinate values.

        Parameters
        ----------
        key : Key
            A row or column key.
        dim_names : sequence of str
            The dimensions the key's values belong to, in order.

        Returns
        -------
        tuple of int
            The position of each value along its dimension.
        """
        return tuple(coords[d].index(v) for d, v in zip(dim_names, key))

    diff_v, rel_v, exp_v = diff.values, rel.values, exp.values
    ctl_v = ctl.values if ctl is not None else None
    clo_v = ctl_lo.values if ctl_lo is not None else None
    chi_v = ctl_hi.values if ctl_hi is not None else None
    elo_v = exp_lo.values if exp_lo is not None else None
    ehi_v = exp_hi.values if exp_hi is not None else None
    dif_v = {c: (lo.values, hi.values) for c, (lo, hi) in dif.items()}
    dif_levels = sorted(dif_v)
    chart_lo, chart_hi = dif_v.get(chart_conf, (None, None))
    cnt_v = counts.values if counts is not None else None
    cnt_dims = [d for d in dims if counts is not None and d in counts.dims]

    cells: dict[tuple[Key, Key], Cell] = {}
    n_sat = n_sig = 0

    for rl in rows:
        ri = locate(rl.key, row_dims)
        for cl in columns:
            ci = locate(cl.key, column_dims)
            idx = ri + ci
            if not finite[idx].any():
                continue

            # Both are always on an axis: `prepare` produces them, the subset
            # above cannot drop a dimension (every selector is a list), and the
            # unassigned check rejects any layout that fails to place them.
            pos = dict(zip(dims, rl.key + cl.key))
            metric = str(pos[METRIC_DIM])
            variable = str(pos[VARIABLE_DIM])
            source = str(pos.get(FORECAST_DIM, agg.forecast_sources[0]))
            is_base = baseline_key is not None and source == baseline_key
            compared = coloured and not is_base
            unit = units.get((metric, variable))
            u = f" {unit}" if unit else ""
            pol = polarity_of(metric, metric_polarity)
            fam = family_of(pol) if compared else NEUTRAL

            steps = []
            for k in range(len(lead_times)):

                def _at(arr, k=k, idx=idx):
                    """Read this box's value from an array.

                    Parameters
                    ----------
                    arr : np.ndarray or None
                        Values indexed by the layout dims, then the cell dim.
                    k : int, optional
                        The lead-time index; bound to the current box.
                    idx : tuple of int, optional
                        The row and column indices; bound to the current cell.

                    Returns
                    -------
                    float or None
                        The value, or None if absent or not finite.
                    """
                    if arr is None:
                        return None
                    v = arr[idx + (k,)]
                    return float(v) if np.isfinite(v) else None

                def _n(k=k, rl=rl, cl=cl):
                    """Read this box's case count.

                    Parameters
                    ----------
                    k : int, optional
                        The lead-time index; bound to the current box.
                    rl : Line, optional
                        The row; bound to the current cell.
                    cl : Line, optional
                        The column; bound to the current cell.

                    Returns
                    -------
                    int or None
                        The count, or None if there are no counts or it is not
                        finite.
                    """
                    if cnt_v is None:
                        return None
                    cidx = tuple(
                        coords[d].index(v)
                        for d, v in zip(
                            list(row_dims) + list(column_dims), rl.key + cl.key
                        )
                        if d in cnt_dims
                    )
                    val = cnt_v[cidx + (k,)]
                    return int(val) if np.isfinite(val) else None

                d_ = diff_v[idx + (k,)]
                r_ = rel_v[idx + (k,)]
                own = _at(exp_v)
                if not (np.isfinite(d_) if compared else own is not None):
                    steps.append(
                        Step(
                            lead_time=lead_times[k],
                            value=None,
                            relative=None,
                            baseline=None,
                            forecast=None,
                            baseline_lower=None,
                            baseline_upper=None,
                            forecast_lower=None,
                            forecast_upper=None,
                            value_lower=None,
                            value_upper=None,
                            n=None,
                            level=0,
                            family=fam,
                            significant_at=None,
                            tooltip=f"{lead_labels[k]} no data",
                        )
                    )
                    continue
                text = format_value(own) if show_values and own is not None else ""

                if not compared:
                    # A neutral box: this source's own score, compared with nothing.
                    lo, hi = _at(elo_v), _at(ehi_v)
                    nn = _n()
                    tip = f"{lead_labels[k]} {format_value(own)}{u}"
                    if lo is not None and hi is not None and chart_conf:
                        tip += (
                            f", {_pct(chart_conf)} interval "
                            f"{format_value(lo)} to {format_value(hi)}"
                        )
                    if nn is not None:
                        tip += f" ({nn} cases)"
                    steps.append(
                        Step(
                            lead_time=lead_times[k],
                            value=None,
                            relative=None,
                            baseline=None,
                            forecast=own,
                            baseline_lower=None,
                            baseline_upper=None,
                            forecast_lower=lo,
                            forecast_upper=hi,
                            value_lower=None,
                            value_upper=None,
                            n=nn,
                            level=0,
                            family=NEUTRAL,
                            significant_at=None,
                            tooltip=tip,
                            text=text,
                        )
                    )
                    continue

                signed = -float(r_) if pol is Polarity.LOWER_IS_BETTER else float(r_)
                lvl = scaling.level(signed)
                if scaling.saturated(signed):
                    n_sat += 1

                # Significant when the paired difference interval excludes zero.
                # Without that input nothing is marked: the two sources' marginal
                # intervals cannot answer this, being much wider than the paired one.
                #
                # Levels ascend, so intervals widen; the answer is the highest one
                # that still excludes zero. Stopping at the first failure rather
                # than scanning on keeps a non-nested set of intervals -- analytic
                # bounds from different approximations, say -- from reporting a
                # level whose narrower neighbours do not support it.
                sig_at = None
                for c in dif_levels:
                    lo, hi = _at(dif_v[c][0]), _at(dif_v[c][1])
                    if lo is None or hi is None or not (lo > 0 or hi < 0):
                        break
                    sig_at = c
                if sig_at is not None:
                    n_sig += 1
                d_lo, d_hi = _at(chart_lo), _at(chart_hi)
                nn = _n()
                word = scheme.word(fam, lvl)
                pct = abs(float(r_)) * 100
                score = f" {text}{u}," if text else ""
                tip = f"{lead_labels[k]}{score} {pct:.3g}% {word}"
                if sig_at is not None:
                    tip += f", significant at {_pct(sig_at)}"
                if nn is not None:
                    tip += f" ({nn} cases)"
                steps.append(
                    Step(
                        lead_time=lead_times[k],
                        value=float(d_),
                        relative=signed,
                        baseline=_at(ctl_v),
                        forecast=_at(exp_v),
                        baseline_lower=_at(clo_v),
                        baseline_upper=_at(chi_v),
                        forecast_lower=_at(elo_v),
                        forecast_upper=_at(ehi_v),
                        value_lower=d_lo,
                        value_upper=d_hi,
                        n=nn,
                        level=lvl,
                        family=fam,
                        significant_at=sig_at,
                        tooltip=tip,
                        text=text,
                    )
                )

            cells[(rl.key, cl.key)] = Cell(
                row=rl.index,
                col=cl.index,
                row_key=rl.key,
                col_key=cl.key,
                cell_id=f"{rl.slug}__{cl.slug}",
                metric=metric,
                units=unit,
                steps=tuple(steps),
                forecast_source=source,
                is_baseline=is_base,
            )

    stats = LayoutStats(
        n_rows=len(rows),
        n_cols=len(columns),
        n_cells_possible=len(rows) * len(columns),
        n_cells_present=len(cells),
        n_boxes=len(cells) * len(lead_times),
        n_saturated=n_sat,
        n_significant=n_sig,
    )

    # Sample sizes actually used, for the caveats below. Small or highly
    # autocorrelated samples are the usual reason a card over-claims.
    case_counts = {
        s.n for cell in cells.values() for s in cell.steps if s.n is not None
    }

    notes = []
    if agg.already_means:
        notes.append(
            "The input has no forecast cases: each value is a mean as given, so "
            "there are no intervals and nothing is marked significant."
        )
    block = (
        f"blocks of {agg.block_length} cases"
        if agg.block_length > 1
        else "independent cases"
    )
    if not coloured:
        own_cases = (
            "the forecast cases every source scored, so the rows can be compared"
            if agg.cases == "common" or len(agg.forecast_sources) == 1
            else "each source's own forecast cases, which differ between sources, "
            "so the rows should not be ranked against one another"
        )
        notes.append(
            f"No baseline: each box is its source's own score, averaged over "
            f"{own_cases}. Nothing is compared, so nothing is coloured or marked "
            f"significant."
        )
        if agg.n_resamples:
            notes.append(
                f"Intervals in the tooltips and drill-down are each source's own, "
                f"from a {agg.method} bootstrap over forecast cases ({block}, "
                f"{agg.n_resamples} resamples, seed {agg.seed})."
            )
    elif show_values:
        where = (
            "the same forecast cases the other rows were compared on"
            if agg.cases == "common"
            else "all of its own forecast cases"
        )
        notes.append(
            f"Each box prints its source's own score. The grey rows are "
            f"{agg.baseline_source}'s own scores, over {where}; they are what the "
            f"colours are relative to, and are not compared with anything."
        )
    if coloured and agg.n_resamples:
        sharing = "both sources" if len(agg.forecast_sources) == 1 else "every source"
        notes.append(
            f"Intervals from a {agg.method} bootstrap over forecast cases "
            f"({block}, {agg.n_resamples} resamples, seed {agg.seed}). The "
            f"difference is paired: it is taken per case before averaging, with "
            f"one resample shared by {sharing}, so their common error cancels."
        )
    # With one forecast source the two policies coincide, and saying anything
    # would only add noise to today's card.
    if coloured and len(agg.forecast_sources) > 1:
        if agg.cases == "common":
            notes.append(
                f"Every forecast source is compared with {agg.baseline_source} on "
                f"the same forecast cases: those all of them scored. Rows and "
                f"columns can therefore be compared with one another."
            )
        else:
            notes.append(
                f"Each forecast source is compared with {agg.baseline_source} on "
                f"the cases the two share, which differ between sources. Each "
                f"comparison uses as much data as it can, but they do not rest on "
                f"the same weather and should not be ranked against one another."
            )
    # The package chooses the resample now, so the caveat has to name what it
    # chose -- and the measured cost of that choice, not a general warning.
    if coloured and agg.block_length > 1:
        notes.append(
            "Consecutive forecasts share a weather system. Blocking reduces the "
            "resulting over-marking but does not remove it: measured at roughly "
            "8% false positives against a nominal 5% on AR(1) synthetic data, "
            "and no block length reaches nominal."
        )
    elif coloured and agg.n_resamples:
        notes.append(
            "Cases were resampled independently, which treats consecutive "
            "forecasts as unrelated weather. On AR(1) synthetic data that marks "
            "about 44% of truly-null cells as significant against a nominal 5%, "
            "so read the markings below as optimistic."
        )
    if coloured:
        notes.append(
            f"This card shows {stats.n_boxes} simultaneous comparisons. Isolated "
            "cells mean little; coherent blocks mean a lot."
        )
    if coloured and case_counts:
        lo, hi = min(case_counts), max(case_counts)
        span = f"{lo}" if lo == hi else f"{lo}-{hi}"
        if hi < 30:
            notes.append(
                f"Only {span} forecast cases per cell. A bootstrap over so few is "
                f"weak, so treat any significance marking here as suggestive rather "
                f"than settled."
            )
    if stats.n_significant and stats.n_boxes:
        frac = stats.n_significant / stats.n_boxes
        if frac > 0.75:
            notes.append(
                f"{frac:.0%} of boxes are marked significant. That is high enough to "
                f"be worth double-checking how the interval was resampled, rather "
                f"than read as {frac:.0%} confidence in the result."
            )

    return Layout(
        rows=rows,
        columns=columns,
        row_headers=tuple(tuple(b) for b in row_headers),
        column_headers=tuple(tuple(b) for b in col_headers),
        lead_times=lead_times,
        lead_labels=lead_labels,
        cells=cells,
        row_dims=tuple(row_dims),
        column_dims=tuple(column_dims),
        cell_dim=cell_dim,
        stats=stats,
        title=title,
        subtitle=subtitle,
        baseline_source=agg.baseline_source,
        forecast_sources=agg.forecast_sources,
        cases=agg.cases,
        confidence_levels=levels,
        resampling=agg.method,
        block_length=agg.block_length,
        n_resamples=agg.n_resamples,
        seed=agg.seed,
        scheme_name=scheme.name,
        notes=tuple(notes),
        show_values=show_values,
    )
