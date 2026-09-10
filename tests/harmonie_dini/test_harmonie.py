"""Assertions against the real HARMONIE-AROME DINI / AIFS data.

Skipped unless the local datasets have been built, so the suite stays runnable
without credentials. See README.md in this folder.

Unlike the synthetic tests these cannot assert an exact answer -- neither model is
known to be right -- so they check the things that would be wrong if the pipeline
were broken: that valid times line up, that units were converted, that the
regridding left no holes, and that the result is not a whitewash.
"""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from common import OUT  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (OUT / "verification.zarr").exists(),
    reason="HARMONIE/AIFS datasets not built; see tests/harmonie_dini/README.md",
)

SOURCES = ("aifs", "harmonie-arome")


@pytest.fixture(scope="module")
def summary():
    return xr.open_zarr(OUT / "verification.zarr")


def card(summary, control: str, experiment: str):
    from mlwp_scorecards import build_layout

    return build_layout(
        summary,
        control=control,
        experiment=experiment,
        rows=["truth_source", "variable"],
        columns=["metric"],
        cell="lead_time",
    )


def test_both_sources_are_present(summary):
    assert list(summary["prediction_source"].values) == list(SOURCES)


def test_scoring_aligns_forecast_and_truth_valid_times():
    """The load-bearing check: a forecast equal to the truth must score zero.

    Everything else rests on ``valid = init + lead`` selecting the right analysis.
    Feeding the truth back in as a perfect forecast isolates that lookup from any
    question about either model.
    """
    from verify import per_case_scores

    truth = xr.open_zarr(OUT / "truth.zarr")
    dini = xr.open_zarr(OUT / "forecast.zarr")
    weights = np.cos(np.deg2rad(truth["lat"].values))

    valid = dini["init_time"].values[:, None] + dini["lead_time"].values[None, :]
    index = {t: i for i, t in enumerate(truth["time"].values)}
    perfect = np.stack(
        [np.stack([truth["t2m"].values[index[v]] for v in row]) for row in valid]
    )
    oracle = dini.copy()
    oracle["t2m"] = (("init_time", "lead_time", "y", "x"), perfect)

    scores = per_case_scores(oracle, truth, "t2m", weights)
    assert np.allclose(scores["rmse"], 0.0), "truth scored against itself is not zero"
    assert np.allclose(scores["mae"], 0.0)

    # and a deliberate one-step offset must not score zero, or the test above
    # would pass even if every lead were reading the same field
    shifted = oracle.copy()
    shifted["t2m"] = (("init_time", "lead_time", "y", "x"), np.roll(perfect, 1, axis=1))
    assert not np.allclose(per_case_scores(shifted, truth, "t2m", weights)["rmse"], 0.0)


def test_aifs_regridding_left_no_holes():
    """NaNs here almost always mean a longitude-convention mismatch."""
    aifs = xr.open_zarr(OUT / "aifs.zarr")
    for var in aifs.data_vars:
        finite = np.isfinite(aifs[var].values).mean()
        assert finite > 0.99, f"{var} is only {finite:.1%} finite after interpolation"


def test_aifs_temperature_was_converted_to_kelvin():
    """AIFS reports degrees Celsius; a missed conversion is a 273 K offset."""
    aifs = xr.open_zarr(OUT / "aifs.zarr")
    assert aifs["t2m"].attrs["units"] == "K"
    assert 230 < float(aifs["t2m"].mean()) < 320


def test_aifs_and_harmonie_share_the_comparison_axes():
    aifs = xr.open_zarr(OUT / "aifs.zarr")
    dini = xr.open_zarr(OUT / "forecast.zarr")
    assert (aifs["init_time"].values == dini["init_time"].values).all()
    assert (aifs["lead_time"].values == dini["lead_time"].values).all()
    assert aifs["t2m"].shape == dini["t2m"].shape


def test_errors_are_physically_plausible(summary):
    """Both models should be wrong by a believable amount, not by orders of magnitude.

    Catches unit and alignment failures that leave the arrays the right shape:
    a Celsius/kelvin slip would put t2m RMSE near 273, not near 1.
    """
    bounds = {
        "t2m": (0.05, 5.0),
        "pres_seasurface": (5.0, 500.0),
        "wind_speed_10m": (0.1, 5.0),
    }
    for var, (lo, hi) in bounds.items():
        v = summary[var].sel(truth_source="dini-analysis", metric="rmse", stat="mean")
        assert np.all((v.values > lo) & (v.values < hi)), f"{var}: {v.values}"


def test_errors_grow_with_lead_time(summary):
    """Forecast error must grow as the forecast ages, for both models."""
    for source in SOURCES:
        v = (
            summary["wind_speed_10m"]
            .sel(
                truth_source="dini-analysis",
                prediction_source=source,
                metric="rmse",
                stat="mean",
            )
            .values
        )
        assert v[-1] > v[0], f"{source} error did not grow with lead time: {v}"


def test_confidence_intervals_bracket_the_mean(summary):
    """`lower <= mean <= upper` is the one thing the package asks of the input."""
    for var in ("t2m", "pres_seasurface", "wind_speed_10m"):
        v = summary[var].sel(truth_source="dini-analysis")
        lo = v.sel(stat="lower").values
        mid = v.sel(stat="mean").values
        hi = v.sel(stat="upper").values
        assert np.all(lo <= mid) and np.all(mid <= hi), var


def test_the_card_reports_a_mixed_result(summary):
    """HARMONIE vs AIFS is not a whitewash, and the card must show that.

    A card that came out uniformly one colour would usually mean the comparison
    had collapsed -- a unit or alignment bug -- rather than a real result.
    """
    lay = card(summary, "aifs", "harmonie-arome")
    signs = {
        np.sign(s.relative)
        for _, _, cell in lay.iter_cells()
        for s in cell.steps
        if s.relative
    }
    assert signs == {-1.0, 1.0}, f"expected wins on both sides, got {signs}"


def test_mslp_favours_aifs_at_long_lead(summary):
    """The clearest real signal: AIFS takes over on mean sea level pressure."""
    lay = card(summary, "aifs", "harmonie-arome")
    cell = lay.sel(
        truth_source="dini-analysis", variable="pres_seasurface", metric="rmse"
    )
    assert cell.steps[0].relative > 0, "HARMONIE should lead at +6 h"
    assert cell.steps[-1].relative < 0, "AIFS should lead by +36 h"


def test_swapping_control_and_experiment_flips_the_card(summary):
    """Roles are arguments, so the reverse card must be the mirror image."""
    a = card(summary, "aifs", "harmonie-arome")
    b = card(summary, "harmonie-arome", "aifs")
    for (_, _, ca), (_, _, cb) in zip(a.iter_cells(), b.iter_cells()):
        assert ca.row_key == cb.row_key and ca.col_key == cb.col_key
        for sa, sb in zip(ca.steps, cb.steps):
            assert np.sign(sa.relative) == -np.sign(sb.relative)


def test_case_counts_are_the_number_of_initialisations(summary):
    n = summary["t2m_number_of_cases"].values
    assert set(np.unique(n)) == {summary.attrs["n_initialisations"]}


def test_renders_both_formats(summary, tmp_path):
    from mlwp_scorecards import make_scorecard

    outs = make_scorecard(
        summary,
        [tmp_path / "c.html", tmp_path / "c.png"],
        control="aifs",
        experiment="harmonie-arome",
        rows=["truth_source", "variable"],
        columns=["metric"],
        cell="lead_time",
        title="HARMONIE-AROME vs AIFS",
    )
    for p in outs:
        assert p.stat().st_size > 2000
    assert "harmonie-arome" in outs[0].read_text()
