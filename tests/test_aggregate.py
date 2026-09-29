"""The collapse over forecast cases, and the bootstrap that gives it an interval.

The performance argument for this module rests entirely on a bootstrap mean being
expressible as a matmul, so that identity is asserted here rather than assumed.
So is the property the whole design exists for: that a resample shared between
the two sources gives a far tighter interval than treating them as independent.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from mlwp_scorecards.aggregate import (
    _weights,
    aggregate,
    bootstrap_mean,
    resample_indices,
)
from mlwp_scorecards.ingest import prepare


def _per_case(n_case=60, n_lead=4, seed=0, rho=0.9, drift=0.15):
    """Two sources scored on the same cases, with most of their error shared.

    `rho` is how much of each case's error is common weather rather than model
    skill. It is the reason pairing helps at all, so it is what these tests turn
    on.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 1.0, (n_case, n_lead))
    ctl = 2.0 + rho * common + (1 - rho) * rng.normal(0, 1, (n_case, n_lead))
    exp = ctl + drift + (1 - rho) * rng.normal(0, 1, (n_case, n_lead))
    lead = np.arange(1, n_lead + 1).astype("timedelta64[D]")
    # Daily, so the default 60 cases take the default 10-day blocks.
    init = np.datetime64("2024-01-01") + np.arange(n_case) * np.timedelta64(24, "h")
    return xr.Dataset(
        {
            "rmse.2t": xr.DataArray(
                np.stack([ctl, exp]),
                dims=["forecast_source", "init_time", "lead_time"],
                coords=dict(
                    forecast_source=["ctl", "exp"], init_time=init, lead_time=lead
                ),
                attrs=dict(units="K"),
            )
        }
    )


def _score(ds):
    score, _ = prepare(
        ds, row_dims=["variable"], column_dims=["metric"], cell_dim="lead_time"
    )
    return score


# --------------------------------------------------------------------------- #
# the identity the performance argument rests on
# --------------------------------------------------------------------------- #
def test_the_matmul_is_exactly_nanmean_of_the_gathered_resample():
    """433 s becomes 3 s at card scale, so this had better be the same number.

    A bootstrap mean is a weighted mean, the weights being how many times each
    case was drawn. Nothing else in the suite pins that, and if it drifts every
    interval on every card is quietly wrong.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 0.2, (50, 400))
    x[:, -40:] = np.where(rng.random((50, 40)) < 0.4, np.nan, x[:, -40:])
    idx = rng.integers(0, 400, (300, 400))

    with np.errstate(invalid="ignore"):
        reference = np.nanmean(x[:, idx], axis=2)
    fast = bootstrap_mean(x, _weights(idx, 400))

    assert fast.shape == reference.shape
    assert np.allclose(fast, reference, equal_nan=True, rtol=0, atol=1e-12)


def test_a_replicate_of_only_missing_cases_is_nan_not_zero():
    """0/0 must stay NaN. A cell with no data becoming a perfect score is the
    exact failure this package exists to avoid, and it would look plausible."""
    x = np.full((1, 8), np.nan)
    x[0, :2] = 1.0
    idx = np.tile(np.array([2, 3, 4, 5, 6, 7, 2, 3]), (4, 1))  # never draws 0 or 1
    out = bootstrap_mean(x, _weights(idx, 8))
    assert np.isnan(out).all()


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def test_moving_block_draws_contiguous_runs():
    """Consecutive forecasts share a weather system, so the resample must keep
    runs of them together; that is the whole difference from iid."""
    idx = resample_indices(
        60,
        n_resamples=50,
        method="moving-block",
        block_length=6,
        rng=np.random.default_rng(0),
    )
    assert idx.shape == (50, 60)
    steps = np.diff(idx.reshape(50, 10, 6), axis=2)
    assert (steps == 1).all(), "cases within a block must be consecutive"


def test_iid_and_a_block_of_one_are_the_same_thing():
    a = resample_indices(30, n_resamples=20, method="iid", rng=np.random.default_rng(7))
    b = resample_indices(
        30,
        n_resamples=20,
        method="moving-block",
        block_length=1,
        rng=np.random.default_rng(7),
    )
    assert (a == b).all()


def test_an_unknown_resample_method_is_refused():
    with pytest.raises(ValueError, match="moving-block"):
        resample_indices(
            10, n_resamples=5, method="jackknife", rng=np.random.default_rng(0)
        )


# --------------------------------------------------------------------------- #
# why the package does this at all
# --------------------------------------------------------------------------- #
def test_the_paired_interval_is_tighter_than_treating_the_sources_as_independent():
    """The reason the difference cannot be reconstructed from the two marginals.

    Both sources see the same weather, so most of their error is shared and
    cancels when the difference is taken per case. Combining the marginals in
    quadrature keeps that shared error instead of cancelling it, and so marks far
    less than the data supports.
    """
    agg = aggregate(
        _score(_per_case(rho=0.95)),
        baseline_source="ctl",
        forecast_source="exp",
        n_resamples=500,
        confidence_levels=(0.95,),
        seed=0,
    )
    paired_lo, paired_hi = agg.paired[0.95]
    paired = (paired_hi - paired_lo).values
    naive = 2 * np.hypot(
        (agg.baseline_upper - agg.baseline_lower).values / 2,
        (agg.forecast_upper - agg.forecast_lower).values / 2,
    )
    assert (paired < naive).all(), f"paired {paired}, naive {naive}"
    assert (naive / paired).min() > 1.5


def test_intervals_nest_with_the_confidence_level():
    agg = aggregate(
        _score(_per_case()),
        baseline_source="ctl",
        forecast_source="exp",
        n_resamples=500,
        confidence_levels=(0.68, 0.95, 0.997),
        seed=0,
    )
    assert agg.confidence_levels == (0.68, 0.95, 0.997)
    for lower, upper in zip((0.68, 0.95), (0.95, 0.997)):
        assert (
            agg.paired[upper][0].values <= agg.paired[lower][0].values + 1e-12
        ).all()
        assert (
            agg.paired[upper][1].values >= agg.paired[lower][1].values - 1e-12
        ).all()


def test_the_mean_is_the_plain_mean_over_cases():
    """No resampling involved: the colour of a box comes from the data, not the
    bootstrap. Only the interval around it is resampled."""
    ds = _per_case()
    agg = aggregate(
        _score(ds), baseline_source="ctl", forecast_source="exp", n_resamples=50, seed=0
    )
    want = ds["rmse.2t"].sel(forecast_source="ctl").mean("init_time")
    assert np.allclose(agg.baseline.values.ravel(), want.values.ravel())


def test_counts_are_the_cases_where_both_sources_scored():
    """The paired count, not the experiment's: it is the sample the interval and
    the significance actually rest on."""
    ds = _per_case(n_case=20)
    ds["rmse.2t"][0, :4] = np.nan  # control missing four cases
    ds["rmse.2t"][1, 3:6] = np.nan  # experiment missing three, one overlapping
    agg = aggregate(
        _score(ds),
        baseline_source="ctl",
        forecast_source="exp",
        n_resamples=50,
        seed=0,
        bootstrap="iid",  # 20 cases: too few for the default blocks
    )
    assert set(np.unique(agg.counts.values)) == {20 - 6}


def test_the_means_use_only_the_cases_both_sources_scored():
    """A case one source lacks must drop out of *both* means, not only the count.

    The sources agree exactly wherever both scored, and the control is bad on
    exactly the cases the experiment is missing. Averaging each source over its
    own cases would compare different weather and report a large, confidently
    significant difference that the paired data does not contain.
    """
    ds = _per_case(n_case=40)
    ds["rmse.2t"][:] = 1.0
    ds["rmse.2t"][0, 20:] = 5.0  # control: bad on the second half
    ds["rmse.2t"][1, 20:] = np.nan  # experiment: missing the second half
    agg = aggregate(
        _score(ds),
        baseline_source="ctl",
        forecast_source="exp",
        n_resamples=200,
        seed=0,
    )

    assert np.allclose(agg.baseline.values, 1.0)
    assert np.allclose(agg.forecast.values, 1.0)
    for lo, hi in agg.paired.values():
        assert np.allclose(lo.values, 0.0) and np.allclose(hi.values, 0.0)
    assert set(np.unique(agg.counts.values)) == {20}


# --------------------------------------------------------------------------- #
# reproducibility and provenance
# --------------------------------------------------------------------------- #
def test_the_same_seed_gives_the_same_interval_and_a_different_one_does_not():
    """`test_determinism` asserts byte-identical HTML, which an OS-seeded RNG
    would break intermittently and in a way that looks like a rendering bug."""
    score = _score(_per_case())
    kw = dict(
        baseline_source="ctl",
        forecast_source="exp",
        n_resamples=200,
        confidence_levels=(0.95,),
    )
    a = aggregate(score, seed=0, **kw).paired[0.95][0].values
    b = aggregate(score, seed=0, **kw).paired[0.95][0].values
    c = aggregate(score, seed=1, **kw).paired[0.95][0].values
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_the_block_length_is_derived_from_the_cadence_and_recorded():
    """A block length silently chosen for you is not something a reader can
    check, so it is derived from the initialisation cadence and carried out."""
    score = _score(_per_case(n_case=120))  # daily -> 10 days is 10 cases
    agg = aggregate(
        score, baseline_source="ctl", forecast_source="exp", n_resamples=50, seed=0
    )
    assert agg.method == "moving-block"
    assert agg.block_length == 10
    assert agg.n_resamples == 50 and agg.seed == 0


@pytest.mark.parametrize(
    "init_time, match",
    [
        (np.arange(12), "not a time axis"),
        (np.full(60, np.datetime64("2024-01-01", "ns")), "repeats values"),
        (None, "12 forecast cases is too few"),
    ],
)
def test_no_derivable_block_length_is_refused_not_quietly_iid(init_time, match):
    """An iid resample marks far too much as significant, so falling back to it
    has to be the caller's choice -- and the message says how to make it."""
    ds = _per_case(n_case=12 if init_time is None else len(init_time))
    if init_time is not None:
        ds = ds.assign_coords(init_time=init_time)
    kw = dict(baseline_source="ctl", forecast_source="exp", n_resamples=50, seed=0)
    with pytest.raises(ValueError, match=rf"(?s){match}.*bootstrap='iid'"):
        aggregate(_score(ds), **kw)
    # and choosing it explicitly is accepted
    assert aggregate(_score(ds), bootstrap="iid", **kw).block_length == 1


def test_a_percentage_confidence_level_is_refused():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        aggregate(
            _score(_per_case()),
            baseline_source="ctl",
            forecast_source="exp",
            confidence_levels=(68, 95),
        )


def test_no_case_dimension_means_the_values_are_already_means():
    """The magnitude-only card: a documented input shape, not a failure."""
    ds = _per_case()
    collapsed = ds.mean("init_time", keep_attrs=True)
    score = _score(collapsed)
    agg = aggregate(score, baseline_source="ctl", forecast_source="exp", seed=0)
    assert agg.already_means
    assert agg.paired == {}
    assert agg.confidence_levels == ()
    assert agg.baseline_lower is None
