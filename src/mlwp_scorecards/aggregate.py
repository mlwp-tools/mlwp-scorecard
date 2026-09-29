"""Collapse forecast cases: the mean, its bootstrap interval, and the paired one.

This is the second of the two collapses a scorecard rests on. The first — over
the gridpoints of a spatial region, within a single forecast case — is a
deterministic reduction of that case's error field, needs the raw fields, and
happens upstream. The second is a *sample*: N weather situations drawn from the
population of possible ones, whose mean score estimates the expected score with a
standard error. That is what happens here.

Two things make the result honest rather than merely computed:

**The resample is drawn once and shared.** Every forecast source and the
baseline are scored on the same weather, so most of their error is common and
cancels when the difference is taken per case. Within each replicate they are
compared on the same cases, so ``boot(forecast) - boot(baseline)`` *is* the
paired bootstrap — measured 0.9x to 3.1x tighter than treating the two as
independent, and the only thing that can decide significance. Sharing one
resample across several forecast sources also keeps their rows consistent with
one another.

**Consecutive forecasts share a weather system.** An iid resample over cases
treats them as independent and badly overstates certainty: on AR(1) synthetic
data at phi=0.75 it marks 44% of truly-null cells as significant against a
nominal 5%. A moving-block resample brings that to roughly 8%. No block length
reaches nominal, which is why the card says which resample produced it.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import xarray as xr

from .ingest import CASE_DIM, FORECAST_DIM, PreparedCube, ValidationReport

__all__ = [
    "Aggregated",
    "aggregate",
    "resample_indices",
    "bootstrap_mean",
    "resolve_sources",
    "expand_selection",
    "CASE_POLICIES",
]

#: Boxes per chunk when bootstrapping. The working set is
#: ``chunk * n_resamples * 8`` bytes per array, so 4000 keeps a full-size card
#: under ~200 MB instead of the ~1 GB an unchunked pass would take.
CHUNK = 4000

#: Target decorrelation span for a derived block length. Ten days is comfortably
#: past the synoptic timescale, and lands on blocks of 20 for 12-hourly
#: initialisations and 40 for 6-hourly -- inside the flat region of the measured
#: false-positive curve (12.5% / 8.2% / 9.8% at blocks of 16 / 24 / 32).
BLOCK_TARGET = np.timedelta64(10, "D")

METHODS = ("moving-block", "iid")

#: Which forecast cases a comparison rests on. ``common``: those every selected
#: source *and* the baseline scored, so rows answer the same question.
#: ``pairwise``: those each forecast source shares with the baseline, so each row
#: uses as much data as it can but rows are no longer comparable with each other.
CASE_POLICIES = ("common", "pairwise")


@dataclass(frozen=True, slots=True)
class SourceSummary:
    """One source's own scores, collapsed over cases: for the baseline's grey row.

    No ``forecast_source`` dimension: it describes a single source.

    Attributes
    ----------
    mean : xr.DataArray
        The source's mean score over its cases.
    lower, upper : xr.DataArray or None
        Its own interval at the widest confidence level; None with no cases to
        resample.
    counts : xr.DataArray
        Forecast cases behind each mean.
    """

    mean: xr.DataArray
    lower: xr.DataArray | None
    upper: xr.DataArray | None
    counts: xr.DataArray


@dataclass(frozen=True, slots=True)
class Aggregated:
    """Forecast sources against a baseline, collapsed over cases, ready to lay out.

    Every array here is derived from the same stacked cube by a reduction along
    ``init_time`` alone, so they share coordinate objects rather than merely
    comparing equal. The layout engine indexes them positionally against one
    another and that is what makes it safe.

    Every array carries a ``forecast_source`` dimension, one entry per forecast
    source -- :attr:`baseline` included, because under ``cases="pairwise"`` the
    baseline's mean is taken over a different set of cases for each of them.

    With no baseline, every ``baseline*`` field is None and nothing is paired:
    each forecast source is summarised on its own.

    Attributes
    ----------
    baseline : xr.DataArray or None
        The baseline's mean score, over the cases each comparison rests on.
    forecast : xr.DataArray
        Each forecast source's mean score.
    baseline_lower, baseline_upper : xr.DataArray or None
        The baseline's own interval, at the widest confidence level.
    forecast_lower, forecast_upper : xr.DataArray or None
        Each forecast source's own interval, at the widest confidence level.
    paired : dict of float to (xr.DataArray, xr.DataArray)
        The paired-difference interval, lower and upper, per confidence level.
    counts : xr.DataArray or None
        Forecast cases behind each cell.
    confidence_levels : tuple of float
        The levels in ``paired``, as fractions.
    baseline_source : str
        The baseline's name; empty with no baseline.
    forecast_sources : tuple of str
        The forecast sources, in order.
    cases : {"common", "pairwise"}
        Which cases each comparison rests on.
    baseline_row : SourceSummary or None
        The baseline's own scores, when shown as a row of their own.
    method : str
        The resampling: ``"moving-block"`` or ``"iid"``.
    block_length : int
        Block length in forecast cases.
    n_resamples : int
        Bootstrap resamples drawn.
    seed : int
        The seed the resamples were drawn from; -1 when a generator was passed.
    """

    baseline: xr.DataArray | None
    forecast: xr.DataArray
    #: Each source's own interval, at the widest confidence level. For the
    #: drill-down chart only -- it is far wider than the paired interval below
    #: and cannot decide significance.
    baseline_lower: xr.DataArray | None
    baseline_upper: xr.DataArray | None
    forecast_lower: xr.DataArray | None
    forecast_upper: xr.DataArray | None
    #: The paired difference interval, one entry per confidence level.
    paired: dict[float, tuple[xr.DataArray, xr.DataArray]]
    #: Forecast cases behind each cell: those the comparison is paired on (see
    #: :data:`CASE_POLICIES`), which is the sample the interval actually rests
    #: on. None when the input supplied no count and there are no cases to count.
    counts: xr.DataArray | None
    confidence_levels: tuple[float, ...]
    baseline_source: str = ""
    forecast_sources: tuple[str, ...] = ()
    cases: str = "common"
    #: The baseline's own scores, for showing it as a row of its own: over the
    #: common cases under ``cases="common"`` (what every row was compared on), and
    #: over all of its own cases under ``"pairwise"``. Only when asked for.
    baseline_row: SourceSummary | None = None
    #: How the interval was produced. Carried to the card, because a
    #: significance claim cannot be checked without it.
    method: str = "moving-block"
    block_length: int = 1
    n_resamples: int = 0
    seed: int = 0

    @property
    def has_intervals(self) -> bool:
        """Report whether there is a paired interval to decide significance.

        Returns
        -------
        bool
            False with no baseline, or no ``init_time`` to resample.
        """
        return bool(self.paired)


def _same(a: Any, b: Any) -> bool:
    """Compare coordinate values, tolerating types that cannot be compared.

    Parameters
    ----------
    a, b : Any
        The two values.

    Returns
    -------
    bool
        True if they compare equal; False if not, or if comparing them raises.
    """
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def expand_selection(
    dim: str,
    values: Sequence[Any],
    available: Sequence[Any],
    *,
    exclude: Sequence[Any] = (),
) -> list[Any]:
    """Expand one ``select=`` list along ``dim``, keeping the order given.

    ``...`` stands for every value of the coordinate not otherwise named (and not
    in ``exclude``), in coordinate order, so ``["europe", ...]`` is Europe first
    and then all the rest.

    Parameters
    ----------
    dim : str
        The dimension, for error messages.
    values : sequence
        The values asked for, with at most one ``...``.
    available : sequence
        The coordinate's values, in coordinate order.
    exclude : sequence, optional
        Values an ``...`` never expands to.

    Returns
    -------
    list
        The values, with ``...`` replaced by the rest of the coordinate.

    Raises
    ------
    KeyError
        If a value is not on the coordinate.
    ValueError
        If ``...`` appears more than once or a value is repeated.
    """
    items = list(values)
    if sum(1 for v in items if v is Ellipsis) > 1:
        raise ValueError(f"select=dict({dim}=...) may contain '...' at most once")
    named = [v for v in items if v is not Ellipsis]
    have = list(available)
    shown = [str(h) for h in have]
    for v in named:
        if not any(_same(v, h) for h in have):
            raise KeyError(
                f"select=dict({dim}=...): {v!r} is not in {dim} (have: {shown})"
            )
    dupes = sorted({str(v) for i, v in enumerate(named) if v in named[:i]})
    if dupes:
        raise ValueError(f"select=dict({dim}=...) repeats {dupes}")
    rest = [
        h
        for h in have
        if not any(_same(h, v) for v in named) and not any(_same(h, e) for e in exclude)
    ]
    out: list[Any] = []
    for v in items:
        out.extend(rest if v is Ellipsis else [v])
    return out


def resolve_sources(
    forecast_source: Any,
    colour_relative_to: str | None,
    available: Sequence[str],
) -> tuple[str, ...]:
    """Expand and validate the forecast sources to show.

    ``...`` stands for every source not otherwise named, in the order of the
    ``forecast_source`` coordinate, so ``["GraphCast", ...]`` is GraphCast first
    and then all the rest. The baseline is left out of that expansion
    automatically, since it is what every forecast source is compared against
    rather than one of them; naming it explicitly is an error.

    Parameters
    ----------
    forecast_source : str, ``...``, or sequence of str and at most one ``...``
        The forecast sources asked for.
    colour_relative_to : str or None
        The baseline.
    available : sequence of str
        The members of ``forecast_source``, in coordinate order.

    Returns
    -------
    tuple of str
        The forecast sources to show, in order, without the baseline.

    Raises
    ------
    KeyError
        If a named source is not in ``forecast_source``.
    ValueError
        If ``...`` appears more than once, a source is repeated, the baseline is
        also named as a forecast source, or nothing is left to show.
    """
    have = [str(s) for s in available]
    if isinstance(forecast_source, slice):
        raise ValueError(
            f"select=dict({FORECAST_DIM}=...) takes source names, not a slice"
        )
    if isinstance(forecast_source, str) or forecast_source is Ellipsis:
        items = [forecast_source]
    else:
        items = list(forecast_source)

    if colour_relative_to is not None and colour_relative_to not in have:
        raise KeyError(
            f"colour_relative_to={colour_relative_to!r} is not in {FORECAST_DIM} (have: {have})"
        )
    if colour_relative_to is not None and colour_relative_to in items:
        raise ValueError(
            f"colour_relative_to={colour_relative_to!r} is also named in "
            f"select=dict({FORECAST_DIM}=...); the baseline is what every forecast "
            f"source is compared against, so it cannot also be one of them"
        )
    sources = expand_selection(FORECAST_DIM, items, have, exclude=[colour_relative_to])
    if not sources:
        raise ValueError(
            f"select=dict({FORECAST_DIM}=...) leaves no forecast source to show"
        )
    return tuple(str(s) for s in sources)


def resample_indices(
    n_case: int,
    *,
    n_resamples: int,
    method: str = "moving-block",
    block_length: int = 1,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the case indices for each bootstrap replicate.

    Drawn **once** by the caller and reused for every source; that sharing is
    what makes the difference interval a paired one.

    Parameters
    ----------
    n_case : int
        Number of forecast cases.
    n_resamples : int
        Number of replicates.
    method : {"moving-block", "iid"}, optional
        How cases are drawn.
    block_length : int, optional
        Block length in cases. A block of 1, or one covering every case, falls
        back to an iid draw.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        Integer case indices, ``(n_resamples, n_case)``.

    Raises
    ------
    ValueError
        If ``method`` is not one of :data:`METHODS`.
    """
    if method not in METHODS:
        raise ValueError(f"bootstrap must be one of {METHODS}, got {method!r}")
    if method == "iid" or block_length <= 1 or block_length >= n_case:
        return rng.integers(0, n_case, (n_resamples, n_case))
    n_block = int(math.ceil(n_case / block_length))
    starts = rng.integers(0, n_case - block_length + 1, (n_resamples, n_block))
    idx = starts[:, :, None] + np.arange(block_length)
    return idx.reshape(n_resamples, -1)[:, :n_case]


def _weights(idx: np.ndarray, n_case: int) -> np.ndarray:
    """Count how often each case is drawn in each replicate.

    Parameters
    ----------
    idx : np.ndarray
        Case indices, ``(n_resamples, n_case)``, from :func:`resample_indices`.
    n_case : int
        Number of forecast cases.

    Returns
    -------
    np.ndarray
        Multiplicity of each case in each replicate, ``(n_resamples, n_case)``.
    """
    w = np.zeros((idx.shape[0], n_case))
    rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
    np.add.at(w, (rows, idx.ravel()), 1.0)
    return w


def bootstrap_mean(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute the bootstrap distribution of the mean, ``(series, n_resamples)``.

    A bootstrap mean is a weighted mean, so every replicate of every series is
    one BLAS call rather than a fancy-index gather. At full card scale that is
    3 s instead of 433 s, and it agrees with ``nanmean(x[:, idx])`` to 3e-15.

    Parameters
    ----------
    x : np.ndarray
        ``(series, case)``. NaN marks a case that did not score.
    weights : np.ndarray
        ``(n_resamples, case)`` from :func:`_weights`.

    Returns
    -------
    np.ndarray
        The mean of each series in each replicate, ``(series, n_resamples)``;
        NaN where a replicate drew only missing cases.
    """
    finite = np.isfinite(x)
    wt = weights.T
    with np.errstate(invalid="ignore", divide="ignore"):
        # 0/0 where a replicate drew only missing cases: that must stay NaN. A
        # cell with no data silently becoming a perfect score is the failure
        # this package exists to avoid.
        return (np.where(finite, x, 0.0) @ wt) / (finite @ wt)


def _percentile_bounds(
    boot: np.ndarray, levels: Sequence[float]
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Take percentile interval bounds at each level from one bootstrap distribution.

    Parameters
    ----------
    boot : np.ndarray
        Bootstrap distribution, replicates along the last axis.
    levels : sequence of float
        Confidence levels, as fractions.

    Returns
    -------
    dict of float to tuple of np.ndarray
        ``(lower, upper)`` for each level; NaN where the distribution is all NaN.
    """
    half = [(1 - c) / 2 * 100 for c in levels]
    with warnings.catch_warnings():
        # A cell whose every case is missing gives an all-NaN distribution, and
        # NaN bounds are the right answer for it -- not a condition to report.
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.nanpercentile(boot, [*half, *(100 - a for a in half)], axis=-1)
    n = len(levels)
    return {c: (q[i], q[i + n]) for i, c in enumerate(levels)}


def derive_block_length(cases: np.ndarray, report: ValidationReport) -> int:
    """Derive a block length in **forecast cases** from the initialisation cadence.

    The cadence is visible in the data; the decorrelation time is not. So this
    targets a fixed span rather than pretending to estimate one, and says what it
    chose -- a block length silently picked for you is not something a reader can
    check.

    Parameters
    ----------
    cases : np.ndarray
        The ``init_time`` coordinate values.
    report : ValidationReport
        Where to warn when falling back to an iid resample.

    Returns
    -------
    int
        Cases per block, covering :data:`BLOCK_TARGET`; 1 (an iid resample) when
        the cadence cannot be read or there are too few cases for such blocks.
    """
    n_case = len(cases)
    if n_case < 2 or not np.issubdtype(np.asarray(cases).dtype, np.datetime64):
        report.warn(
            f"block length 1 (an iid resample): {CASE_DIM!r} is not a time axis, "
            f"so the initialisation cadence cannot be read. Consecutive forecasts "
            f"share weather, so the intervals below are optimistic; pass "
            f"block_length= if you know the cadence"
        )
        return 1
    spacing = np.median(np.diff(np.sort(np.asarray(cases))))
    # dividing two timedelta64s gives a bare float, and an infinite one means a
    # zero cadence -- duplicate initialisation times, which say nothing about
    # decorrelation
    span = BLOCK_TARGET / spacing
    if not np.isfinite(span) or span <= 0:
        return 1
    block = max(1, int(math.ceil(span)))
    if n_case < 4 * block:
        report.warn(
            f"block length 1 (an iid resample): {n_case} forecast cases is too "
            f"few for blocks of {block}. The intervals below are optimistic, "
            f"because consecutive forecasts share weather"
        )
        return 1
    return block


def aggregate(
    cube: PreparedCube,
    *,
    forecast_source: str | Sequence[str],
    baseline_source: str | None,
    cases: str = "common",
    baseline_row: bool = False,
    bootstrap: str = "moving-block",
    block_length: int | None = None,
    n_resamples: int = 2000,
    confidence_levels: Sequence[float] = (0.68, 0.95, 0.997),
    seed: int | np.random.Generator = 0,
) -> Aggregated:
    """Collapse a per-case cube over its forecast cases, against a baseline.

    Parameters
    ----------
    cube : PreparedCube
        From :func:`~mlwp_scorecards.ingest.prepare`.
    forecast_source : str or sequence of str
        Members of ``forecast_source`` to compare with the baseline; ``...`` is
        expanded as in :func:`resolve_sources`.
    baseline_source : str or None
        The member every one of them is differenced against: the card colours
        ``forecast - baseline``. None: nothing is compared, and each source is
        summarised on its own.
    cases : {"common", "pairwise"}, optional
        Which forecast cases each comparison rests on; see :data:`CASE_POLICIES`.
        The two agree when there is one forecast source. With no baseline,
        ``"common"`` is the cases every selected source scored and ``"pairwise"``
        each source's own.
    baseline_row : bool, optional
        Also summarise the baseline on its own, for showing it as a row; see
        :attr:`Aggregated.baseline_row`.
    bootstrap : {"moving-block", "iid"}, optional
        How the forecast cases are resampled.
    block_length : int, optional
        In forecast **cases**, not hours. Derived from the initialisation cadence
        when omitted.
    n_resamples : int, optional
        Number of bootstrap replicates.
    confidence_levels : sequence of float, optional
        Interval levels, as fractions strictly between 0 and 1.
    seed : int or np.random.Generator, optional
        Seed for the resample, or a generator to draw it from (recorded as -1).

    Returns
    -------
    Aggregated
        Means, intervals, paired difference intervals and case counts, with how
        they were produced.

    Raises
    ------
    ValueError
        If ``cases`` or ``bootstrap`` is not a known policy, or a confidence
        level is not a fraction.
    KeyError
        If the dataset has no ``forecast_source`` dimension, or a source is not
        in it.
    """
    report = cube.report
    da = cube.score
    if cases not in CASE_POLICIES:
        raise ValueError(f"cases must be one of {CASE_POLICIES}, got {cases!r}")

    # Coordinate selection has already happened, on the dataset, before the cube
    # was built: resampling data that is then discarded would be wasteful and wrong.
    if FORECAST_DIM not in da.dims:
        raise KeyError(f"{FORECAST_DIM!r} is not a dimension of the dataset")
    sources = resolve_sources(
        forecast_source, baseline_source, da.coords[FORECAST_DIM].values
    )

    has_base = baseline_source is not None
    fc_da = da.sel({FORECAST_DIM: list(sources)})
    raw_base = da.sel({FORECAST_DIM: baseline_source}, drop=True) if has_base else None
    # Pair the sources before anything is averaged: a case the baseline or the
    # forecast source lacks is dropped from both. Otherwise each mean is taken
    # over that source's own cases, the difference compares different weather,
    # and a card can show a confidently significant difference the paired data
    # does not contain. Under "common", a case any one source lacks is dropped
    # from all of them -- which, with no baseline, is still what keeps the rows
    # comparable with one another.
    ok = np.isfinite(fc_da)
    if has_base:
        ok = ok & np.isfinite(raw_base)
    if cases == "common":
        ok = ok.all(FORECAST_DIM)
    fc_da = fc_da.where(ok)
    base_da = None
    if has_base:
        # The baseline gets a forecast_source dimension of its own: under
        # "pairwise" it is masked differently for each forecast source.
        base_da, fc_da = xr.broadcast(raw_base.where(ok), fc_da)
        base_da = base_da.transpose(*fc_da.dims)
    # The baseline as a row of its own: over the cases every row was compared on
    # when those are common to all, and over all of its own cases otherwise.
    row_da = None
    if baseline_row and has_base:
        row_da = raw_base.where(ok) if cases == "common" else raw_base
    provenance = dict(
        baseline_source=baseline_source or "", forecast_sources=sources, cases=cases
    )

    levels = tuple(sorted(float(c) for c in confidence_levels))
    if not all(0.0 < c < 1.0 for c in levels):
        raise ValueError(
            f"confidence_levels must be fractions strictly between 0 and 1 "
            f"(0.95, not 95); got {levels}"
        )

    # No case axis: the values are already means, so there is nothing to resample.
    if CASE_DIM not in da.dims:
        report.warn(
            f"no {CASE_DIM!r} dimension: values are read as already-collapsed "
            f"means, with no interval and nothing marked significant"
        )
        empty = xr.full_like(fc_da, np.nan)
        row = None
        if row_da is not None:
            row = SourceSummary(row_da, None, None, xr.full_like(row_da, np.nan))
        return Aggregated(
            baseline=base_da,
            forecast=fc_da,
            baseline_lower=None,
            baseline_upper=None,
            forecast_lower=None,
            forecast_upper=None,
            paired={},
            counts=empty,
            confidence_levels=(),
            baseline_row=row,
            **provenance,
        )

    if isinstance(seed, np.random.Generator):
        rng, seed_out = seed, -1
    else:
        rng, seed_out = np.random.default_rng(seed), int(seed)

    case_times = da.coords[CASE_DIM].values
    n_case = len(case_times)
    block = (
        derive_block_length(case_times, report)
        if block_length is None and bootstrap == "moving-block"
        else max(1, int(block_length or 1))
    )
    idx = resample_indices(
        n_case,
        n_resamples=n_resamples,
        method=bootstrap,
        block_length=block,
        rng=rng,
    )
    weights = _weights(idx, n_case)

    # Flatten to (series, case) so the bootstrap is one matmul per chunk.
    fc_da = fc_da.transpose(..., CASE_DIM)
    shape = fc_da.shape[:-1]
    dims = fc_da.dims[:-1]
    coords = {d: fc_da.coords[d] for d in dims if d in fc_da.coords}
    f_flat = fc_da.values.reshape(-1, n_case)
    if has_base:
        base_da = base_da.transpose(..., CASE_DIM)
        b_flat = base_da.values.reshape(-1, n_case)
    n_series = f_flat.shape[0]

    widest = levels[-1]
    out = {k: np.empty(n_series) for k in ("b_lo", "b_hi", "f_lo", "f_hi")}
    paired_flat = {c: (np.empty(n_series), np.empty(n_series)) for c in levels}

    for lo in range(0, n_series, CHUNK):
        hi = min(lo + CHUNK, n_series)
        f_boot = bootstrap_mean(f_flat[lo:hi], weights)
        fb = _percentile_bounds(f_boot, [widest])[widest]
        out["f_lo"][lo:hi], out["f_hi"][lo:hi] = fb
        if not has_base:
            continue
        b_boot = bootstrap_mean(b_flat[lo:hi], weights)
        # The paired difference shares the resample, so this subtraction *is* the
        # paired bootstrap. It is not the same as differencing two independent
        # ones, which would be far wider and quite wrong.
        d_boot = f_boot - b_boot

        bb = _percentile_bounds(b_boot, [widest])[widest]
        out["b_lo"][lo:hi], out["b_hi"][lo:hi] = bb
        for c, (dlo, dhi) in _percentile_bounds(d_boot, levels).items():
            paired_flat[c][0][lo:hi] = dlo
            paired_flat[c][1][lo:hi] = dhi

    def _wrap(flat: np.ndarray) -> xr.DataArray:
        """Restore flattened series to the cube's non-case dimensions.

        Parameters
        ----------
        flat : np.ndarray
            One value per series.

        Returns
        -------
        xr.DataArray
            The values on the cube's dimensions and coordinates, less ``init_time``.
        """
        return xr.DataArray(flat.reshape(shape), dims=dims, coords=coords)

    with np.errstate(invalid="ignore"):
        mean_f = fc_da.mean(CASE_DIM, skipna=True)
        mean_b = base_da.mean(CASE_DIM, skipna=True) if has_base else None
    # The cases each comparison is paired on: after the masking above, a forecast
    # value is finite exactly where its pair is.
    counts = np.isfinite(fc_da).sum(CASE_DIM)

    return Aggregated(
        baseline=mean_b,
        forecast=mean_f,
        baseline_lower=_wrap(out["b_lo"]) if has_base else None,
        baseline_upper=_wrap(out["b_hi"]) if has_base else None,
        forecast_lower=_wrap(out["f_lo"]),
        forecast_upper=_wrap(out["f_hi"]),
        paired=(
            {c: (_wrap(v[0]), _wrap(v[1])) for c, v in paired_flat.items()}
            if has_base
            else {}
        ),
        counts=counts,
        confidence_levels=levels,
        method=bootstrap,
        block_length=block,
        n_resamples=n_resamples,
        seed=seed_out,
        baseline_row=(
            _summarise(row_da, weights, widest) if row_da is not None else None
        ),
        **provenance,
    )


def _summarise(da: xr.DataArray, weights: np.ndarray, conf: float) -> SourceSummary:
    """Summarise one source on its own: mean, bootstrap interval, case count.

    Parameters
    ----------
    da : xr.DataArray
        The source's per-case scores, with an ``init_time`` dimension.
    weights : np.ndarray
        ``(n_resamples, case)`` from :func:`_weights`, shared with the comparison.
    conf : float
        Confidence level of the interval.

    Returns
    -------
    SourceSummary
        Collapsed over ``init_time``.
    """
    da = da.transpose(..., CASE_DIM)
    shape, dims = da.shape[:-1], da.dims[:-1]
    coords = {d: da.coords[d] for d in dims if d in da.coords}
    flat = da.values.reshape(-1, da.shape[-1])
    lo = np.empty(flat.shape[0])
    hi = np.empty(flat.shape[0])
    for start in range(0, flat.shape[0], CHUNK):
        stop = min(start + CHUNK, flat.shape[0])
        boot = bootstrap_mean(flat[start:stop], weights)
        lo[start:stop], hi[start:stop] = _percentile_bounds(boot, [conf])[conf]

    def _wrap(a: np.ndarray) -> xr.DataArray:
        """Restore flattened series to the source's non-case dimensions.

        Parameters
        ----------
        a : np.ndarray
            One value per series.

        Returns
        -------
        xr.DataArray
            The values on the source's dimensions and coordinates, less ``init_time``.
        """
        return xr.DataArray(a.reshape(shape), dims=dims, coords=coords)

    with np.errstate(invalid="ignore"):
        mean = da.mean(CASE_DIM, skipna=True)
    return SourceSummary(mean, _wrap(lo), _wrap(hi), np.isfinite(da).sum(CASE_DIM))
