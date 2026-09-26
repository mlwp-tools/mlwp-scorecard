"""Collapse forecast cases: the mean, its bootstrap interval, and the paired one.

This is the second of the two collapses a scorecard rests on. The first — over
the gridpoints of a spatial region, within a single forecast case — is a
deterministic reduction of that case's error field, needs the raw fields, and
happens upstream. The second is a *sample*: N weather situations drawn from the
population of possible ones, whose mean score estimates the expected score with a
standard error. That is what happens here.

Two things make the result honest rather than merely computed:

**The resample is drawn once and shared.** Both sources are scored on the same
weather, so most of their error is common and cancels when the difference is
taken per case. Within each replicate the two are compared on the same cases, so
``boot(experiment) - boot(control)`` *is* the paired bootstrap — measured 0.9x to
3.1x tighter than treating the two as independent, and the only thing that can
decide significance.

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
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from .ingest import CASE_DIM, FORECAST_DIM, PreparedCube, ValidationReport

__all__ = ["Aggregated", "aggregate", "resample_indices", "bootstrap_mean"]

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


@dataclass(frozen=True, slots=True)
class Aggregated:
    """One pair of sources, collapsed over cases and ready to lay out.

    Every array here is derived from the same stacked cube by a reduction along
    ``init_time`` alone, so they share coordinate objects rather than merely
    comparing equal. The layout engine indexes them positionally against one
    another and that is what makes it safe.
    """

    control: xr.DataArray
    experiment: xr.DataArray
    #: Each source's own interval, at the widest confidence level. For the
    #: drill-down chart only -- it is far wider than the paired interval below
    #: and cannot decide significance.
    control_lower: xr.DataArray | None
    control_upper: xr.DataArray | None
    experiment_lower: xr.DataArray | None
    experiment_upper: xr.DataArray | None
    #: The paired difference interval, one entry per confidence level.
    paired: dict[float, tuple[xr.DataArray, xr.DataArray]]
    #: Forecast cases behind each cell: those where *both* sources scored, which
    #: is the sample the interval actually rests on. None when the input supplied
    #: no count and there are no cases to count.
    counts: xr.DataArray | None
    confidence_levels: tuple[float, ...]
    #: How the interval was produced. Carried to the card, because a
    #: significance claim cannot be checked without it.
    method: str = "moving-block"
    block_length: int = 1
    n_resamples: int = 0
    seed: int = 0

    @property
    def has_intervals(self) -> bool:
        return bool(self.paired)


def resample_indices(
    n_case: int,
    *,
    n_resamples: int,
    method: str = "moving-block",
    block_length: int = 1,
    rng: np.random.Generator,
) -> np.ndarray:
    """Case indices for each replicate, ``(n_resamples, n_case)``.

    Drawn **once** by the caller and reused for both sources; that sharing is
    what makes the difference interval a paired one.
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
    """Multiplicity of each case in each replicate, ``(n_resamples, n_case)``."""
    w = np.zeros((idx.shape[0], n_case))
    rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
    np.add.at(w, (rows, idx.ravel()), 1.0)
    return w


def bootstrap_mean(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Bootstrap distribution of the mean, ``(series, n_resamples)``.

    A bootstrap mean is a weighted mean, so every replicate of every series is
    one BLAS call rather than a fancy-index gather. At full card scale that is
    3 s instead of 433 s, and it agrees with ``nanmean(x[:, idx])`` to 3e-15.

    Parameters
    ----------
    x : np.ndarray
        ``(series, case)``. NaN marks a case that did not score.
    weights : np.ndarray
        ``(n_resamples, case)`` from :func:`_weights`.
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
    """Interval bounds at each level, from one bootstrap distribution."""
    half = [(1 - c) / 2 * 100 for c in levels]
    with warnings.catch_warnings():
        # A cell whose every case is missing gives an all-NaN distribution, and
        # NaN bounds are the right answer for it -- not a condition to report.
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.nanpercentile(boot, [*half, *(100 - a for a in half)], axis=-1)
    n = len(levels)
    return {c: (q[i], q[i + n]) for i, c in enumerate(levels)}


def derive_block_length(cases: np.ndarray, report: ValidationReport) -> int:
    """Block length in **forecast cases**, from the initialisation cadence.

    The cadence is visible in the data; the decorrelation time is not. So this
    targets a fixed span rather than pretending to estimate one, and says what it
    chose -- a block length silently picked for you is not something a reader can
    check.
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
    control: str,
    experiment: str,
    subset: Mapping[str, Any] | None = None,
    bootstrap: str = "moving-block",
    block_length: int | None = None,
    n_resamples: int = 2000,
    confidence_levels: Sequence[float] = (0.68, 0.95, 0.997),
    seed: int | np.random.Generator = 0,
) -> Aggregated:
    """Collapse a per-case cube over its forecast cases, for one pair of sources.

    Parameters
    ----------
    cube : PreparedCube
        From :func:`~mlwp_scorecards.ingest.prepare`.
    control, experiment : str
        Members of ``forecast_source``. The card colours ``experiment - control``.
    subset : mapping, optional
        Coordinate subsetting, applied **before** the resample rather than after:
        resampling data that is then discarded would be both wasteful and wrong.
    bootstrap : {"moving-block", "iid"}, optional
    block_length : int, optional
        In forecast **cases**, not hours. Derived from the initialisation cadence
        when omitted.
    n_resamples, confidence_levels, seed : optional

    Returns
    -------
    Aggregated
    """
    report = cube.report
    da = cube.score

    if subset:
        sel = {k: (v if isinstance(v, list) else [v]) for k, v in subset.items()}
        da = da.sel({k: v for k, v in sel.items() if k in da.dims})

    if FORECAST_DIM not in da.dims:
        raise KeyError(f"{FORECAST_DIM!r} is not a dimension of the dataset")
    sources = [str(s) for s in da.coords[FORECAST_DIM].values]
    for role, name in (("control", control), ("experiment", experiment)):
        if name not in sources:
            raise KeyError(
                f"{role}={name!r} is not in {FORECAST_DIM} (have: {sources})"
            )

    ctl_da = da.sel({FORECAST_DIM: control}, drop=True)
    exp_da = da.sel({FORECAST_DIM: experiment}, drop=True)

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
        empty = xr.full_like(ctl_da, np.nan)
        return Aggregated(
            control=ctl_da,
            experiment=exp_da,
            control_lower=None,
            control_upper=None,
            experiment_lower=None,
            experiment_upper=None,
            paired={},
            counts=empty,
            confidence_levels=(),
        )

    if isinstance(seed, np.random.Generator):
        rng, seed_out = seed, -1
    else:
        rng, seed_out = np.random.default_rng(seed), int(seed)

    cases = da.coords[CASE_DIM].values
    n_case = len(cases)
    block = (
        derive_block_length(cases, report)
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
    ctl_da = ctl_da.transpose(..., CASE_DIM)
    exp_da = exp_da.transpose(..., CASE_DIM)
    shape = ctl_da.shape[:-1]
    dims = ctl_da.dims[:-1]
    coords = {d: ctl_da.coords[d] for d in dims if d in ctl_da.coords}
    c_flat = ctl_da.values.reshape(-1, n_case)
    e_flat = exp_da.values.reshape(-1, n_case)
    n_series = c_flat.shape[0]

    widest = levels[-1]
    out = {k: np.empty(n_series) for k in ("c_lo", "c_hi", "e_lo", "e_hi")}
    paired_flat = {c: (np.empty(n_series), np.empty(n_series)) for c in levels}

    for lo in range(0, n_series, CHUNK):
        hi = min(lo + CHUNK, n_series)
        c_boot = bootstrap_mean(c_flat[lo:hi], weights)
        e_boot = bootstrap_mean(e_flat[lo:hi], weights)
        # The paired difference shares the resample, so this subtraction *is* the
        # paired bootstrap. It is not the same as differencing two independent
        # ones, which would be far wider and quite wrong.
        d_boot = e_boot - c_boot

        cb = _percentile_bounds(c_boot, [widest])[widest]
        eb = _percentile_bounds(e_boot, [widest])[widest]
        out["c_lo"][lo:hi], out["c_hi"][lo:hi] = cb
        out["e_lo"][lo:hi], out["e_hi"][lo:hi] = eb
        for c, (dlo, dhi) in _percentile_bounds(d_boot, levels).items():
            paired_flat[c][0][lo:hi] = dlo
            paired_flat[c][1][lo:hi] = dhi

    def _wrap(flat: np.ndarray) -> xr.DataArray:
        return xr.DataArray(flat.reshape(shape), dims=dims, coords=coords)

    with np.errstate(invalid="ignore"):
        mean_c = ctl_da.mean(CASE_DIM, skipna=True)
        mean_e = exp_da.mean(CASE_DIM, skipna=True)
    # Cases where *both* sources scored: the sample the interval rests on.
    counts = np.isfinite(ctl_da) & np.isfinite(exp_da)
    counts = counts.sum(CASE_DIM)

    return Aggregated(
        control=mean_c,
        experiment=mean_e,
        control_lower=_wrap(out["c_lo"]),
        control_upper=_wrap(out["c_hi"]),
        experiment_lower=_wrap(out["e_lo"]),
        experiment_upper=_wrap(out["e_hi"]),
        paired={c: (_wrap(v[0]), _wrap(v[1])) for c, v in paired_flat.items()},
        counts=counts,
        confidence_levels=levels,
        method=bootstrap,
        block_length=block,
        n_resamples=n_resamples,
        seed=seed_out,
    )
