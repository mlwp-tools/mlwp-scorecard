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

from .colours import ColourScheme, FixedScaling, Polarity, family_of, polarity_of
from .ingest import PreparedCube
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


def _coord_values(cube: xr.DataArray, dim: str) -> list[Any]:
    """Coordinate values as plain Python objects, with NaN normalised to None.

    NaN cannot be used in a layout key: ``nan != nan``, so two layouts built from
    the same data would compare unequal and ``sel(level=float("nan"))`` could never
    match. None is the not-applicable marker in keys; NaN stays in the data.
    """
    out = []
    for v in cube.coords[dim].values:
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
    control: str,
    experiment: str,
    row_dims: Sequence[str],
    column_dims: Sequence[str],
    cell_dim: str,
    prediction_dim: str = "prediction_source",
    stat_dim: str = "stat",
    metric_dim: str | None = "metric",
    variable_dim: str = "variable",
    scheme: ColourScheme,
    scaling: FixedScaling,
    metric_polarity: Mapping[str, str] | None = None,
    subset: Mapping[str, Any] | None = None,
    title: str = "",
    subtitle: str = "",
) -> Layout:
    """Difference two prediction sources and lay the result out.

    Returns
    -------
    Layout
    """
    da = cube.score
    counts = cube.counts

    if subset:
        sel = {k: (v if isinstance(v, list) else [v]) for k, v in subset.items()}
        da = da.sel(sel)
        if counts is not None:
            counts = counts.sel({k: v for k, v in sel.items() if k in counts.dims})

    if prediction_dim not in da.dims:
        raise KeyError(f"{prediction_dim!r} is not a dimension of the dataset")
    sources = [str(s) for s in da.coords[prediction_dim].values]
    for role, name in (("control", control), ("experiment", experiment)):
        if name not in sources:
            raise KeyError(
                f"{role}={name!r} is not in {prediction_dim} (have: {sources})"
            )

    has_stat = stat_dim in da.dims
    take = (lambda a, s: a.sel({stat_dim: s})) if has_stat else (lambda a, s: a)

    ctl = take(da.sel({prediction_dim: control}), "mean")
    exp = take(da.sel({prediction_dim: experiment}), "mean")
    diff = exp - ctl
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = diff / np.abs(ctl)

    lower = upper = None
    if has_stat and "lower" in [str(s) for s in da.coords[stat_dim].values]:
        lower = da.sel({prediction_dim: experiment, stat_dim: "lower"})
        upper = da.sel({prediction_dim: experiment, stat_dim: "upper"})

    if counts is not None and prediction_dim in counts.dims:
        counts = counts.sel({prediction_dim: experiment})

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
    if lower is not None:
        lower = lower.transpose(*dims, cell_dim)
        upper = upper.transpose(*dims, cell_dim)
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
    low_v = lower.values if lower is not None else None
    up_v = upper.values if upper is not None else None
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

            metric = "value"
            if metric_dim:
                for d, v in zip(list(row_dims) + list(column_dims), rl.key + cl.key):
                    if d == metric_dim:
                        metric = str(v)
            pol = polarity_of(metric, metric_polarity)
            fam = family_of(pol)

            variable = None
            for d, v in zip(list(row_dims) + list(column_dims), rl.key + cl.key):
                if d == variable_dim:
                    variable = str(v)

            steps = []
            for k in range(len(lead_times)):
                d_ = diff_v[idx + (k,)]
                r_ = rel_v[idx + (k,)]
                if not np.isfinite(d_):
                    steps.append(
                        Step(
                            lead_times[k],
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            0,
                            fam,
                            False,
                            f"{lead_labels[k]} no data",
                        )
                    )
                    continue
                signed = -float(r_) if pol is Polarity.LOWER_IS_BETTER else float(r_)
                lvl = scaling.level(signed)
                if scaling.saturated(signed):
                    n_sat += 1
                lo = float(low_v[idx + (k,)]) if low_v is not None else None
                up = float(up_v[idx + (k,)]) if up_v is not None else None
                sig = False  # paired difference intervals are not yet an input
                if sig:
                    n_sig += 1
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
                if nn is not None:
                    tip += f" ({nn} cases)"
                steps.append(
                    Step(
                        lead_times[k],
                        float(d_),
                        signed,
                        float(ctl_v[idx + (k,)]),
                        float(exp_v[idx + (k,)]),
                        lo,
                        up,
                        nn,
                        lvl,
                        fam,
                        sig,
                        tip,
                    )
                )

            cells[(rl.key, cl.key)] = Cell(
                row=rl.index,
                col=cl.index,
                row_key=rl.key,
                col_key=cl.key,
                cell_id=f"{rl.slug}__{cl.slug}",
                metric=metric,
                units=cube.units.get(variable) if variable else None,
                steps=tuple(steps),
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

    notes = [
        "Forecast cases are autocorrelated: consecutive runs share a weather system, "
        "so nominal confidence overstates certainty.",
        f"This card shows {stats.n_boxes} simultaneous comparisons. Isolated cells mean "
        "little; coherent blocks mean a lot.",
    ]

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
        control=control,
        experiment=experiment,
        confidence=cube.confidence,
        scheme_name=scheme.name,
        notes=tuple(notes),
    )
