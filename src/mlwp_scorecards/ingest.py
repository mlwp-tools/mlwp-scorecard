"""Turn a CF-style verification dataset into the uniform cube the layout engine wants.

Input carries one variable per ``{metric}.{physical_variable}`` pair — the naming
WeatherBench-X's ``AggregationState.metric_values`` emits, so a WBX evaluation
loads with no transformation. Two things follow from putting the metric in the
name rather than on a coordinate: ``units`` can differ per metric (RMSE is in K,
ACC is dimensionless), and coverage can be ragged, since a metric that needs an
ensemble simply has no variable for the fields that lack one.

Each variable also omits the dimensions that do not apply to it (``msl`` has no
``level``). Here every variable is padded with a single NaN element on any
optional dimension it lacks, then stacked into the full ``metric`` x ``variable``
grid, so downstream code sees one array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import xarray as xr

__all__ = [
    "ValidationReport",
    "PreparedCube",
    "prepare",
    "split_name",
    "METRIC_DIM",
    "VARIABLE_DIM",
    "FORECAST_DIM",
    "CASE_DIM",
]

#: The schema's dimension names. Fixed, not configurable: the package reads one
#: documented shape (see README.md), and these are that shape's vocabulary. They
#: live here, in one place, because the same names were previously spelled as
#: string literals in three separate modules and could drift apart.
#:
#: ``metric`` and ``variable`` are the exception in kind: they are *produced* by
#: :func:`prepare` from the ``{metric}.{variable}`` data-variable names rather
#: than read off the input.
METRIC_DIM = "metric"
VARIABLE_DIM = "variable"
FORECAST_DIM = "forecast_source"

#: The forecast cases. One score per case, which is what makes the collapse over
#: them -- and so the interval, and so significance -- the package's to perform.
#: ``init_time`` is WeatherBench-X's name for the axis and this repo's own.
CASE_DIM = "init_time"


def split_name(name: str) -> tuple[str, str]:
    """Split ``{metric}.{variable}`` on the **last** dot.

    Splitting last rather than first is what lets a metric carry parameters in its
    own name, which WeatherBench-X's ``unique_name`` does::

        >>> split_name("rmse.2t")
        ('rmse', '2t')
        >>> split_name("seeps.v1.5.tp")
        ('seeps.v1.5', 'tp')

    Raises
    ------
    ValueError
        If the name has no dot, or either half is empty. Guessing here would
        silently mislabel a card, so it is refused instead.
    """
    metric, dot, variable = name.rpartition(".")
    if not dot or not metric or not variable:
        raise ValueError(
            f"score variable {name!r} is not of the form '{{metric}}.{{variable}}' "
            f"(for example 'rmse.2t'); rename it or pass it as an ancillary"
        )
    return metric, variable


@dataclass
class ValidationReport:
    """Accumulated problems found while reading a dataset."""

    fails: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.fails.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def has_fails(self) -> bool:
        return bool(self.fails)

    def __iadd__(self, other: "ValidationReport") -> "ValidationReport":
        self.fails += other.fails
        self.warnings += other.warnings
        return self

    def __str__(self) -> str:
        lines = [f"ERROR   {m}" for m in self.fails]
        lines += [f"warning {m}" for m in self.warnings]
        return "\n".join(lines) or "no problems found"

    def raise_if_failed(self) -> None:
        if self.has_fails():
            raise ValueError("dataset validation failed:\n" + str(self))


@dataclass(frozen=True, slots=True)
class PreparedCube:
    """A verification dataset flattened to a single score array."""

    #: One score per forecast case: ``(metric, variable, ..., lead_time,
    #: init_time)``. The collapse over cases has not happened yet -- that is
    #: :mod:`~mlwp_scorecards.aggregate`'s job, and the reason this package can
    #: pair the two sources at all.
    score: xr.DataArray
    #: Units keyed by ``(metric, variable)``. Metric-dependent by construction:
    #: RMSE of temperature is in K while its ACC is dimensionless.
    units: dict[tuple[str, str], str | None]
    report: ValidationReport

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(self.score.dims)


def _pad_optional(da: xr.DataArray, optional: Iterable[str]) -> xr.DataArray:
    """Give a variable a single-NaN coordinate for each optional dim it lacks.

    ``msl`` gains ``level = [nan]`` and so becomes one row, rather than being
    broadcast across every pressure level as NaN.
    """
    for dim in optional:
        if dim not in da.dims:
            da = da.expand_dims({dim: [np.nan]})
    return da


def _stack(
    blocks: dict[tuple[str, str], xr.DataArray],
    metrics: Sequence[str],
    variables: Sequence[str],
    template_for: dict[str, xr.DataArray],
) -> xr.DataArray:
    """Assemble the full ``metric`` x ``variable`` grid, NaN where absent.

    Coverage is ragged on purpose — CRPS needs an ensemble, FSS a threshold — so
    absent combinations are filled here and dropped again by the layout engine,
    which already removes categories with no data anywhere below them.
    """
    per_metric = []
    for m in metrics:
        per_var = [
            blocks.get((m, v), xr.full_like(template_for[v], np.nan)) for v in variables
        ]
        per_metric.append(
            xr.concat(
                per_var,
                dim=xr.DataArray(list(variables), dims=VARIABLE_DIM, name=VARIABLE_DIM),
                join="outer",
            )
        )
    return xr.concat(
        per_metric,
        dim=xr.DataArray(list(metrics), dims=METRIC_DIM, name=METRIC_DIM),
        join="outer",
    )


def prepare(
    ds: xr.Dataset,
    *,
    row_dims: Sequence[str],
    column_dims: Sequence[str],
    cell_dim: str,
    strict: bool = False,
) -> PreparedCube:
    """Validate a verification dataset and flatten it to one score array.

    Parameters
    ----------
    ds : xr.Dataset
        CF-style verification statistics, one variable per ``{metric}.{variable}``.
    row_dims, column_dims : sequence of str
        Coordinate names nested on the rows and columns.
    cell_dim : str
        The within-cell coordinate, normally ``"lead_time"``.
    strict : bool, optional
        Promote warnings to failures.

    Notes
    -----
    The schema's other dimension names are fixed, not arguments: see the
    ``*_DIM`` constants above. There is one documented input shape.

    Returns
    -------
    PreparedCube
    """
    report = ValidationReport()

    # `variable` and `metric` are produced here, so finding either already on the
    # input means the caller has a differently-shaped dataset -- a flat cube,
    # most likely. Say that, rather than letting `split_name` further down
    # complain that a variable name has no dot in it.
    already = [d for d in (VARIABLE_DIM, METRIC_DIM) if d in ds.dims]
    if already:
        report.fail(
            f"dataset already has {' and '.join(repr(d) for d in already)} as "
            f"dimension(s). This package takes one data variable per "
            f"'{{metric}}.{{variable}}' name -- 'rmse.2t', 'crps.z' -- and builds "
            f"those two dimensions itself. See the Input section of README.md."
        )
        report.raise_if_failed()

    scores = [str(n) for n in ds.data_vars]
    if not scores:
        report.fail("dataset has no score variables")
        report.raise_if_failed()

    # First-appearance order, never a set: `test_determinism` renders the same
    # card in a fresh process with a different PYTHONHASHSEED and demands the same
    # bytes, and set iteration order would break that.
    parsed = {s: split_name(s) for s in scores}
    metrics: list[str] = []
    variables: list[str] = []
    for m, v in parsed.values():
        if m not in metrics:
            metrics.append(m)
        if v not in variables:
            variables.append(v)

    produced = {VARIABLE_DIM, METRIC_DIM}
    wanted = list(row_dims) + list(column_dims) + [cell_dim]
    required = [d for d in wanted if d not in produced]

    present = set(ds.dims)
    for d in required:
        if d not in present:
            raise KeyError(
                f"{d!r} is named in the layout but is not a dimension of the dataset "
                f"(has: {sorted(present)})"
            )

    # Every dimension must be placed somewhere. Silently leaving one out would
    # collapse it without saying so, which is exactly the kind of quiet wrongness
    # a decision artefact must not have. The two exempt names are the ones the
    # rendering consumes rather than lays out: `forecast_source` is collapsed by
    # differencing (or laid out, when it is named above, for several forecast
    # sources), `init_time` by the bootstrap.
    unassigned = present - set(wanted) - {CASE_DIM, FORECAST_DIM}
    if unassigned:
        # Not "or subset them away": `select=` is applied inside `resolve`, after
        # this check, and it keeps the dimension at length 1 rather than dropping
        # it -- so it is no remedy at all here. Dropping the dimension on the
        # dataset is.
        raise KeyError(
            f"dimension(s) {sorted(unassigned)} are assigned to neither rows, "
            f"columns nor cell; name them in rows=[...] or columns=[...], or pick "
            f"one value before calling: ds.sel({sorted(unassigned)[0]}=...)"
        )

    # Dimensions carried by some variables but not others are optional and get a
    # single NaN element on the variables that lack them -- computed over ALL such
    # dimensions, not just the ones on the layout, so the concat below always aligns.
    all_var_dims = {str(d) for s in scores for d in ds[s].dims}
    optional = [d for d in all_var_dims if any(d not in ds[s].dims for s in scores)]

    units: dict[tuple[str, str], str | None] = {}
    padded: dict[tuple[str, str], xr.DataArray] = {}
    template_for: dict[str, xr.DataArray] = {}
    for s in scores:
        da = _pad_optional(ds[s], optional)
        units[parsed[s]] = da.attrs.get("units")
        padded[parsed[s]] = da.rename(None)
        template_for.setdefault(parsed[s][1], padded[parsed[s]])

    score = _stack(padded, metrics, variables, template_for)

    if strict and report.warnings:
        report.fails.extend(report.warnings)
        report.warnings.clear()
    report.raise_if_failed()

    return PreparedCube(score, units, report)
