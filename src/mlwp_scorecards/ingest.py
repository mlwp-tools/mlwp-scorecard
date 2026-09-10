"""Turn a CF-style verification dataset into the uniform cube the layout engine wants.

Input carries one score variable per physical variable, each omitting the
dimensions that do not apply to it (``msl`` has no ``level``). That shape is what
makes raggedness need no sentinel value. Here every variable is padded with a
single NaN element on any optional dimension it lacks, then concatenated along a
new ``variable`` dimension, so downstream code sees one array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import xarray as xr

__all__ = ["ValidationReport", "PreparedCube", "prepare"]

#: Suffixes identifying a case-count variable when ``ancillary_variables`` is absent.
COUNT_SUFFIXES = ("_number_of_cases", "_n", "_popul", "_count")


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

    score: xr.DataArray
    counts: xr.DataArray | None
    units: dict[str, str | None]
    long_names: dict[str, str | None]
    confidence: float | None
    report: ValidationReport

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(self.score.dims)


def _split_variables(ds: xr.Dataset) -> tuple[list[str], dict[str, str]]:
    """Partition data variables into scores and their case-count ancillaries.

    Association follows CF's ``ancillary_variables`` attribute where present, with
    a suffix convention as a fallback.
    """
    declared: dict[str, str] = {}
    ancillary: set[str] = set()
    for name, da in ds.data_vars.items():
        for anc in str(da.attrs.get("ancillary_variables", "")).split():
            if anc in ds.data_vars:
                declared[str(name)] = anc
                ancillary.add(anc)

    scores = [str(n) for n in ds.data_vars if n not in ancillary]
    # suffix fallback for anything not already paired
    remaining = [s for s in scores if s not in declared]
    for s in list(remaining):
        for suf in COUNT_SUFFIXES:
            cand = f"{s}{suf}"
            if cand in ds.data_vars:
                declared[s] = cand
                ancillary.add(cand)
                break
    scores = [s for s in scores if s not in ancillary]
    return scores, declared


def _pad_optional(da: xr.DataArray, optional: Iterable[str]) -> xr.DataArray:
    """Give a variable a single-NaN coordinate for each optional dim it lacks.

    ``msl`` gains ``level = [nan]`` and so becomes one row, rather than being
    broadcast across every pressure level as NaN.
    """
    for dim in optional:
        if dim not in da.dims:
            da = da.expand_dims({dim: [np.nan]})
    return da


def prepare(
    ds: xr.Dataset,
    *,
    row_dims: Sequence[str],
    column_dims: Sequence[str],
    cell_dim: str,
    stat_dim: str = "stat",
    variable_dim: str = "variable",
    strict: bool = False,
) -> PreparedCube:
    """Validate a verification dataset and flatten it to one score array.

    Parameters
    ----------
    ds : xr.Dataset
        CF-style verification statistics, one score variable per physical variable.
    row_dims, column_dims : sequence of str
        Coordinate names nested on the rows and columns.
    cell_dim : str
        The within-cell coordinate, normally ``"lead_time"``.
    stat_dim : str, optional
        Coordinate carrying ``mean``/``lower``/``upper``.
    variable_dim : str, optional
        Name for the dimension created by stacking the score variables.
    strict : bool, optional
        Promote warnings to failures.

    Returns
    -------
    PreparedCube
    """
    report = ValidationReport()
    scores, count_of = _split_variables(ds)
    if not scores:
        report.fail("dataset has no score variables")
        report.raise_if_failed()

    wanted = list(row_dims) + list(column_dims) + [cell_dim]
    # `variable` is produced here, not required of the input
    required = [d for d in wanted if d != variable_dim]

    present = set(ds.dims)
    for d in required:
        if d not in present:
            raise KeyError(
                f"{d!r} is named in the layout but is not a dimension of the dataset "
                f"(has: {sorted(present)})"
            )

    # Every dimension must be placed somewhere. Silently leaving one out would
    # collapse it without saying so, which is exactly the kind of quiet wrongness
    # a decision artefact must not have.
    unassigned = present - set(wanted) - {stat_dim, "prediction_source"}
    if unassigned:
        raise KeyError(
            f"dimension(s) {sorted(unassigned)} are assigned to neither rows, columns "
            f"nor cell; name them in rows=[...] or columns=[...], or subset them away"
        )

    # Dimensions carried by some variables but not others are optional and get a
    # single NaN element on the variables that lack them -- computed over ALL such
    # dimensions, not just the ones on the layout, so the concat below always aligns.
    all_var_dims = {str(d) for s in scores for d in ds[s].dims}
    optional = [d for d in all_var_dims if any(d not in ds[s].dims for s in scores)]

    units, long_names = {}, {}
    padded = []
    for s in scores:
        da = _pad_optional(ds[s], optional)
        units[s] = da.attrs.get("units")
        long_names[s] = da.attrs.get("long_name") or da.attrs.get("standard_name")
        padded.append(da.rename(None))

    score = xr.concat(
        padded,
        dim=xr.DataArray(scores, dims=variable_dim, name=variable_dim),
        join="outer",
    )

    counts = None
    if count_of:
        cpad = []
        for s in scores:
            cname = count_of.get(s)
            if cname is None:
                base = (
                    ds[scores[0]].isel({stat_dim: 0}, drop=True)
                    if stat_dim in ds[scores[0]].dims
                    else ds[scores[0]]
                )
                cpad.append(xr.full_like(_pad_optional(base, optional), np.nan))
            else:
                cpad.append(_pad_optional(ds[cname], optional).rename(None))
        counts = xr.concat(
            cpad,
            dim=xr.DataArray(scores, dims=variable_dim, name=variable_dim),
            join="outer",
        )
    else:
        report.warn("no case-count variable found; tooltips will omit sample sizes")

    if stat_dim not in score.dims:
        report.warn(
            f"no {stat_dim!r} dimension: treating values as the mean, with no interval"
        )

    conf = ds.coords.get("confidence")
    confidence = float(conf.values) if conf is not None and conf.size == 1 else None

    if strict and report.warnings:
        report.fails.extend(report.warnings)
        report.warnings.clear()
    report.raise_if_failed()

    return PreparedCube(score, counts, units, long_names, confidence, report)
