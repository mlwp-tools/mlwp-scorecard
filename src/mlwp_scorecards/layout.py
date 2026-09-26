"""Resolve a prepared cube plus a layout declaration into a :class:`Layout`.

The ordering rule is one global order per dimension, filtered by presence within
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

from .aggregate import Aggregated
from .colours import ColourScheme, FixedScaling, Polarity, family_of, polarity_of
from .ingest import FORECAST_DIM, METRIC_DIM, VARIABLE_DIM, PreparedCube
from .model import Cell, HeaderCell, Key, Layout, LayoutStats, Line, Step

__all__ = ["resolve"]

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def _is_na(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _slug(parts: Sequence[Any]) -> str:
    return "-".join(_SLUG_RE.sub("_", str(p)).strip("_") or "na" for p in parts)


def _label(v: Any) -> str:
    if _is_na(v):
        return ""
    if isinstance(v, (np.timedelta64, _dt.timedelta)):
        return _lead_label(v)
    if isinstance(v, (np.floating, float)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _lead_hours(v: Any) -> float:
    """Lead time in hours, from a timedelta or a bare number."""
    if isinstance(v, np.timedelta64):
        return float(v / np.timedelta64(1, "h"))
    if isinstance(v, _dt.timedelta):
        return v.total_seconds() / 3600.0
    return float(v)


def _lead_label(v: Any) -> str:
    return f"T+{_lead_hours(v):g}"


def _pct(c: float) -> str:
    """A confidence level as a percentage, keeping 99.7% from rounding to 100%."""
    return f"{c * 100:.4g}%"


def _coord_values(cube: xr.DataArray, dim: str) -> list[Any]:
    """Coordinate values as plain Python objects, with NaN normalised to None.

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
    """
    out = []
    for v in cube.coords[dim].values:
        if not isinstance(v, (np.timedelta64, np.datetime64)):
            v = v.item() if hasattr(v, "item") else v
        out.append(None if isinstance(v, float) and math.isnan(v) else v)
    return out


def _order_key(coord_values: Sequence[Any], order: Sequence[Any] | None):
    """Sort key for one dimension's categories.

    The default is the order the coordinate already has in the dataset: the caller
    controls presentation order by ordering their coordinate, which is far less
    surprising than imposing an alphabetical sort on metric or region names.
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
    """
    axis_idx = {d: i for i, d in enumerate(dims)}

    def children(prefix: tuple[Any, ...]) -> list[Any]:
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


def _headers_for(
    leaf: Key, headers: list[list[HeaderCell]], index: int
) -> tuple[HeaderCell, ...]:
    out = []
    for depth in range(len(headers)):
        for blk in headers[depth]:
            if blk.start <= index < blk.stop:
                out.append(blk)
                break
    return tuple(out)


def resolve(
    cube: PreparedCube,
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
) -> Layout:
    """Difference each forecast source from the baseline and lay the result out.

    Parameters
    ----------
    agg : Aggregated
        The collapse over forecast cases, from
        :func:`~mlwp_scorecards.aggregate.aggregate`. Subsetting and the choice
        of sources happen there, because both must precede the resample.

    Returns
    -------
    Layout

    Raises
    ------
    ValueError
        If there are several forecast sources and ``forecast_source`` is on
        neither axis: the card would have nowhere to put them.
    """
    on_axis = FORECAST_DIM in list(row_dims) + list(column_dims)
    if not on_axis and len(agg.forecast_sources) > 1:
        raise ValueError(
            f"{len(agg.forecast_sources)} forecast sources, but {FORECAST_DIM!r} is "
            f"on neither rows nor columns; add it to one of them"
        )

    def _placed(da: xr.DataArray | None) -> xr.DataArray | None:
        # One source and no axis for it: today's two-source card.
        if da is None or on_axis:
            return da
        return da.squeeze(FORECAST_DIM, drop=True)

    ctl, exp = _placed(agg.baseline), _placed(agg.forecast)
    diff = exp - ctl
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = diff / np.abs(ctl)
    ctl_lo, ctl_hi = _placed(agg.baseline_lower), _placed(agg.baseline_upper)
    exp_lo, exp_hi = _placed(agg.forecast_lower), _placed(agg.forecast_upper)
    dif = {c: (_placed(lo), _placed(hi)) for c, (lo, hi) in agg.paired.items()}
    counts = _placed(agg.counts)
    levels = tuple(agg.confidence_levels)
    chart_conf = max(levels) if levels else None

    return _lay_out(
        cube=cube,
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
    )


def _lay_out(
    *,
    cube: PreparedCube,
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
    ctl: xr.DataArray,
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
) -> Layout:
    """Lay collapsed arrays out on the card. Indifferent to where they came from."""
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
    ctl = ctl.transpose(*dims, cell_dim)
    exp = exp.transpose(*dims, cell_dim)
    if ctl_lo is not None:
        ctl_lo = ctl_lo.transpose(*dims, cell_dim)
        ctl_hi = ctl_hi.transpose(*dims, cell_dim)
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

    finite = np.isfinite(diff.values)  # (…dims…, cell)
    any_data = finite.any(axis=-1)  # (…dims…)

    n_row = len(row_dims)
    row_present = any_data.any(axis=tuple(range(n_row, len(dims))))
    col_present = any_data.any(axis=tuple(range(n_row)))

    row_keys, row_headers = _resolve_axis(row_present, list(row_dims), coords, orders)
    col_keys, col_headers = _resolve_axis(
        col_present, list(column_dims), coords, orders
    )

    rows = tuple(
        Line(i, k, _headers_for(k, row_headers, i), _slug(k))
        for i, k in enumerate(row_keys)
    )
    columns = tuple(
        Line(i, k, _headers_for(k, col_headers, i), _slug(k))
        for i, k in enumerate(col_keys)
    )

    leads_raw = _coord_values(diff, cell_dim)
    lead_times = tuple(_lead_hours(v) for v in leads_raw)
    lead_labels = tuple(_lead_label(v) for v in leads_raw)

    def locate(key: Key, dim_names: Sequence[str]) -> tuple[int, ...]:
        return tuple(coords[d].index(v) for d, v in zip(dim_names, key))

    diff_v, rel_v, ctl_v, exp_v = diff.values, rel.values, ctl.values, exp.values
    has_ci = ctl_lo is not None
    clo_v = ctl_lo.values if has_ci else None
    chi_v = ctl_hi.values if has_ci else None
    elo_v = exp_lo.values if has_ci else None
    ehi_v = exp_hi.values if has_ci else None
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
            pol = polarity_of(metric, metric_polarity)
            fam = family_of(pol)

            steps = []
            for k in range(len(lead_times)):
                d_ = diff_v[idx + (k,)]
                r_ = rel_v[idx + (k,)]
                if not np.isfinite(d_):
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
                signed = -float(r_) if pol is Polarity.LOWER_IS_BETTER else float(r_)
                lvl = scaling.level(signed)
                if scaling.saturated(signed):
                    n_sat += 1

                def _at(arr, k=k, idx=idx):
                    if arr is None:
                        return None
                    v = arr[idx + (k,)]
                    return float(v) if np.isfinite(v) else None

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
                nn = None
                if cnt_v is not None:
                    cidx = tuple(
                        coords[d].index(v)
                        for d, v in zip(
                            list(row_dims) + list(column_dims), rl.key + cl.key
                        )
                        if d in cnt_dims
                    )
                    val = cnt_v[cidx + (k,)]
                    nn = int(val) if np.isfinite(val) else None
                word = scheme.word(fam, lvl)
                pct = abs(float(r_)) * 100
                tip = f"{lead_labels[k]} {pct:.3g}% {word}"
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
                    )
                )

            cells[(rl.key, cl.key)] = Cell(
                row=rl.index,
                col=cl.index,
                row_key=rl.key,
                col_key=cl.key,
                cell_id=f"{rl.slug}__{cl.slug}",
                metric=metric,
                units=cube.units.get((metric, variable)),
                steps=tuple(steps),
                forecast_source=source,
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
    if agg.n_resamples:
        block = (
            f"blocks of {agg.block_length} cases"
            if agg.block_length > 1
            else "independent cases"
        )
        sharing = "both sources" if len(agg.forecast_sources) == 1 else "every source"
        notes.append(
            f"Intervals from a {agg.method} bootstrap over forecast cases "
            f"({block}, {agg.n_resamples} resamples, seed {agg.seed}). The "
            f"difference is paired: it is taken per case before averaging, with "
            f"one resample shared by {sharing}, so their common error cancels."
        )
    # With one forecast source the two policies coincide, and saying anything
    # would only add noise to today's card.
    if len(agg.forecast_sources) > 1:
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
    if agg.block_length > 1:
        notes.append(
            "Consecutive forecasts share a weather system. Blocking reduces the "
            "resulting over-marking but does not remove it: measured at roughly "
            "8% false positives against a nominal 5% on AR(1) synthetic data, "
            "and no block length reaches nominal."
        )
    elif agg.n_resamples:
        notes.append(
            "Cases were resampled independently, which treats consecutive "
            "forecasts as unrelated weather. On AR(1) synthetic data that marks "
            "about 44% of truly-null cells as significant against a nominal 5%, "
            "so read the markings below as optimistic."
        )
    notes.append(
        f"This card shows {stats.n_boxes} simultaneous comparisons. Isolated cells "
        "mean little; coherent blocks mean a lot."
    )
    if case_counts:
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
    )
