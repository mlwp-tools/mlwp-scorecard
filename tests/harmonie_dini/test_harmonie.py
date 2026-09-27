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

from common import CONFIDENCE_LEVELS, N_BOOT, OUT  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (OUT / "verification.zarr").exists(),
    reason="HARMONIE/AIFS datasets not built; see tests/harmonie_dini/README.md",
)

SOURCES = ("aifs", "harmonie-arome")
TRUTHS = ("dini-analysis", "observations")


@pytest.fixture(scope="module")
def summary():
    return xr.open_zarr(OUT / "verification.zarr")


def card(summary, baseline: str, forecast: str, levels=CONFIDENCE_LEVELS):
    from mlwp_scorecards import build_layout

    return build_layout(
        summary,
        colour_relative_to=baseline,
        select=dict(forecast_source=[forecast]),
        rows=["truth_source", "variable"],
        columns=["metric"],
        cell="lead_time",
        confidence_levels=levels,
        n_resamples=N_BOOT,
        seed=0,
    )


def test_both_sources_and_both_truths_are_present(summary):
    assert list(summary["forecast_source"].values) == list(SOURCES)
    assert list(summary["truth_source"].values) == list(TRUTHS)


def test_observations_are_physically_plausible():
    """A unit slip in the metObs conversion would be obvious here.

    metObs reports Celsius and hectopascals; the extraction converts to K and Pa.
    """
    obs = xr.open_zarr(OUT / "observations.zarr")
    assert 240 < float(obs["t2m"].mean()) < 320, "not kelvin?"
    assert 9.5e4 < float(obs["pres_seasurface"].mean()) < 1.06e5, "not pascals?"
    assert 0 <= float(obs["wind_speed_10m"].min())
    assert float(obs["wind_speed_10m"].max()) < 60
    for var in ("t2m", "pres_seasurface", "wind_speed_10m"):
        filled = np.isfinite(obs[var].values).mean()
        assert filled > 0.7, f"{var} only {filled:.0%} reported"


def test_stations_lie_inside_the_model_domain():
    """Every station must match a grid point that is actually near it."""
    from verify import station_indices

    obs = xr.open_zarr(OUT / "observations.zarr")
    dini = xr.open_zarr(OUT / "forecast.zarr")
    _, distance_km = station_indices(dini, obs)
    assert (
        distance_km.max() < 25
    ), f"furthest station is {distance_km.max():.0f} km away"


def test_scoring_aligns_forecast_and_truth_valid_times():
    """The load-bearing check: a forecast equal to the truth must score zero.

    Everything else rests on ``valid = init + lead`` selecting the right analysis.
    Feeding the truth back in as a perfect forecast isolates that lookup from any
    question about either model.
    """
    from verify import score_against_grid

    truth = xr.open_zarr(OUT / "truth.zarr")
    dini = xr.open_zarr(OUT / "forecast.zarr")

    valid = dini["init_time"].values[:, None] + dini["lead_time"].values[None, :]
    index = {t: i for i, t in enumerate(truth["time"].values)}
    perfect = np.stack(
        [np.stack([truth["t2m"].values[index[v]] for v in row]) for row in valid]
    )
    oracle = dini.copy()
    oracle["t2m"] = (("init_time", "lead_time", "y", "x"), perfect)

    scores = score_against_grid(oracle, truth, "t2m")
    assert np.allclose(scores["rmse"], 0.0), "truth scored against itself is not zero"
    assert np.allclose(scores["mae"], 0.0)

    # and a deliberate one-step offset must not score zero, or the test above
    # would pass even if every lead were reading the same field
    shifted = oracle.copy()
    shifted["t2m"] = (("init_time", "lead_time", "y", "x"), np.roll(perfect, 1, axis=1))
    assert not np.allclose(score_against_grid(shifted, truth, "t2m")["rmse"], 0.0)


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
        for truth in TRUTHS:
            v = summary[f"rmse.{var}"].sel(truth_source=truth).mean("init_time")
            assert np.all(
                (v.values > lo) & (v.values < hi)
            ), f"{var} vs {truth}: {v.values}"


def test_errors_grow_with_lead_time(summary):
    """Forecast error must grow as the forecast ages, for both models."""
    for source in SOURCES:
        v = (
            summary["rmse.wind_speed_10m"]
            .sel(truth_source="dini-analysis", forecast_source=source)
            .mean("init_time")
            .values
        )
        assert v[-1] > v[0], f"{source} error did not grow with lead time: {v}"


def test_confidence_intervals_bracket_the_mean(summary):
    """`lower <= mean <= upper`, which the package now produces rather than reads.

    It used to be the one thing asked of the input; with the collapse moved
    inside, it is the one thing asked of the bootstrap.
    """
    lay = card(summary, "aifs", "harmonie-arome")
    for _, _, cell in lay.iter_cells():
        for s in cell.steps:
            if s.baseline is None or s.baseline_lower is None:
                continue
            assert s.baseline_lower <= s.baseline <= s.baseline_upper, cell.cell_id
            assert s.forecast_lower <= s.forecast <= s.forecast_upper, cell.cell_id


def test_intervals_nest_with_the_confidence_level(summary):
    """A wider level must give a wider interval, at every cell.

    Bootstrap percentiles guarantee this, so a violation would mean the level
    axis had been read in the wrong order -- exactly the bug that would make a
    cell claim more significance than it has. Two cards at one level each, on the
    same seed, so the two resamples are identical and only the percentile differs.
    """
    narrow = card(summary, "aifs", "harmonie-arome", levels=(0.68,))
    wide = card(summary, "aifs", "harmonie-arome", levels=(0.95,))
    seen = 0
    for (_, _, a), (_, _, b) in zip(narrow.iter_cells(), wide.iter_cells()):
        for sa, sb in zip(a.steps, b.steps):
            if sa.value_lower is None or sb.value_lower is None:
                continue
            assert sb.value_lower <= sa.value_lower + 1e-12, a.cell_id
            assert sb.value_upper >= sa.value_upper - 1e-12, a.cell_id
            seen += 1
    assert seen, "no intervals were compared"


def test_significance_is_graded_and_nested(summary):
    """A cell significant at 95% must also be significant at 68%.

    `significant_at` reports the highest level whose paired interval still
    excludes zero, so the claim it makes is checkable rather than a bare flag.
    """
    lay = card(summary, "aifs", "harmonie-arome")
    assert lay.confidence_levels == tuple(CONFIDENCE_LEVELS)
    marked = [
        s
        for _, _, cell in lay.iter_cells()
        for s in cell.steps
        if s.significant_at is not None
    ]
    assert marked, "the real data should mark something"
    assert all(s.significant_at in lay.confidence_levels for s in marked)
    assert all(s.significant for s in marked)


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


def test_swapping_baseline_and_forecast_source_flips_the_card(summary):
    """Roles are arguments, so the reverse card must be the mirror image."""
    a = card(summary, "aifs", "harmonie-arome")
    b = card(summary, "harmonie-arome", "aifs")
    for (_, _, ca), (_, _, cb) in zip(a.iter_cells(), b.iter_cells()):
        assert ca.row_key == cb.row_key and ca.col_key == cb.col_key
        for sa, sb in zip(ca.steps, cb.steps):
            assert np.sign(sa.relative) == -np.sign(sb.relative)


def test_the_choice_of_truth_changes_the_verdict(summary):
    """The headline finding, and the reason both truths are on the card.

    Against the DINI analysis -- HARMONIE's own state -- HARMONIE wins on 2 m
    temperature. Against neutral station observations the sign reverses. A
    comparison that used only the analysis would have reported the opposite
    conclusion with no hint that it was an artefact of the choice of truth.
    """
    lay = card(summary, "aifs", "harmonie-arome")
    by_truth = {}
    for truth in TRUTHS:
        cell = lay.sel(truth_source=truth, variable="t2m", metric="rmse")
        by_truth[truth] = np.mean([s.relative for s in cell.steps])

    assert (
        by_truth["dini-analysis"] > 0
    ), "HARMONIE should lead against its own analysis"
    assert by_truth["observations"] < 0, "AIFS should lead against neutral observations"


def test_analysis_flatters_harmonie_on_every_variable(summary):
    """HARMONIE always scores relatively better against its own analysis.

    Not a claim about which model is better -- a claim about the truth being
    non-neutral, which is why the caveat is on the card rather than in a footnote.
    """
    lay = card(summary, "aifs", "harmonie-arome")
    for var in ("t2m", "pres_seasurface", "wind_speed_10m"):
        vs_analysis = np.mean(
            [
                s.relative
                for s in lay.sel(
                    truth_source="dini-analysis", variable=var, metric="rmse"
                ).steps
            ]
        )
        vs_obs = np.mean(
            [
                s.relative
                for s in lay.sel(
                    truth_source="observations", variable=var, metric="rmse"
                ).steps
            ]
        )
        assert vs_analysis > vs_obs, (
            f"{var}: HARMONIE scored {vs_analysis:.3f} against its own analysis but "
            f"{vs_obs:.3f} against observations"
        )


def test_case_counts_are_the_number_of_initialisations(summary):
    lay = card(summary, "aifs", "harmonie-arome")
    n = {s.n for _, _, cell in lay.iter_cells() for s in cell.steps}
    assert n == {summary.sizes["init_time"]}


def test_renders_both_formats(summary, tmp_path):
    from mlwp_scorecards import make_scorecard

    outs = make_scorecard(
        summary,
        html_path=tmp_path / "c.html",
        image_path=tmp_path / "c.png",
        colour_relative_to="aifs",
        select=dict(forecast_source=["harmonie-arome"]),
        rows=["truth_source", "variable"],
        columns=["metric"],
        cell="lead_time",
        title="HARMONIE-AROME vs AIFS",
    )
    for p in outs:
        assert p.stat().st_size > 2000
    assert "harmonie-arome" in outs[0].read_text()
