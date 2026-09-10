"""Assertions against the real HARMONIE-AROME DINI / AIFS data.

Skipped unless the local datasets have been built, so the suite stays runnable
without credentials. See README.md in this folder.

Unlike the synthetic tests these cannot assert an exact answer, only its shape: an
operational model must beat persistence, and the persistence error must show the
diurnal signature that proves valid times were aligned correctly.
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

SOURCES = ("persistence", "aifs", "harmonie-arome")


@pytest.fixture(scope="module")
def summary():
    return xr.open_zarr(OUT / "verification.zarr")


def card(summary, control: str, experiment: str):
    from mlwp_scorecards import build_layout

    return build_layout(
        summary, control=control, experiment=experiment,
        rows=["truth_source", "variable"], columns=["metric"], cell="lead_time",
    )


def test_all_three_sources_are_present(summary):
    assert list(summary["prediction_source"].values) == list(SOURCES)


def test_persistence_is_constant_and_equals_the_analysis_at_init():
    truth = xr.open_zarr(OUT / "truth.zarr")
    pers = xr.open_zarr(OUT / "persistence.zarr")
    v = pers["t2m"].values
    assert np.allclose(v, v[:, :1], equal_nan=True), "persistence varies with lead time"
    at_init = truth["t2m"].sel(time=pers["init_time"].values).values
    assert np.allclose(v[:, 0], at_init, equal_nan=True)


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


@pytest.mark.parametrize("model", ["harmonie-arome", "aifs"])
def test_real_models_beat_persistence_everywhere(summary, model):
    """A model that lost to persistence at these lead times would be broken."""
    lay = card(summary, "persistence", model)
    losses = [
        (lay.rows[r].key[-1], cell.metric, s.lead_time)
        for r, _, cell in lay.iter_cells()
        for s in cell.steps
        if s.relative is not None and s.relative <= 0
    ]
    assert not losses, f"persistence beat {model} in {len(losses)} places: {losses[:5]}"


def test_persistence_error_shows_the_diurnal_cycle(summary):
    """Persistence is best a whole number of days out.

    This structure only appears if forecast and truth valid times were aligned
    correctly, so it is a stronger check than any magnitude assertion.
    """
    v = summary["t2m"].sel(truth_source="dini-analysis", prediction_source="persistence",
                           metric="rmse", stat="mean")
    leads = (summary["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    at = dict(zip(leads, v.values))
    assert at[24] < at[12], "persistence should recover at +24 h"
    assert at[24] < at[36], "and degrade again by +36 h"


def test_the_card_reports_a_mixed_result(summary):
    """HARMONIE vs AIFS is not a whitewash, and the card must show that.

    A card that came out uniformly one colour would mean the comparison had
    collapsed -- usually a unit or alignment bug rather than a real result.
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
    cell = lay.sel(truth_source="dini-analysis", variable="pres_seasurface",
                   metric="rmse")
    assert cell.steps[0].relative > 0, "HARMONIE should lead at +6 h"
    assert cell.steps[-1].relative < 0, "AIFS should lead by +36 h"


def test_case_counts_are_the_number_of_initialisations(summary):
    n = summary["t2m_number_of_cases"].values
    assert set(np.unique(n)) == {summary.attrs["n_initialisations"]}


def test_renders_both_formats(summary, tmp_path):
    from mlwp_scorecards import make_scorecard

    outs = make_scorecard(
        summary, [tmp_path / "c.html", tmp_path / "c.png"],
        control="aifs", experiment="harmonie-arome",
        rows=["truth_source", "variable"], columns=["metric"], cell="lead_time",
        title="HARMONIE-AROME vs AIFS",
    )
    for p in outs:
        assert p.stat().st_size > 2000
    assert "harmonie-arome" in outs[0].read_text()


def test_one_dataset_yields_every_pairwise_card(summary, tmp_path):
    """Roles are arguments, so three sources give three cards from one file."""
    from mlwp_scorecards import make_scorecard

    pairs = [("aifs", "harmonie-arome"), ("persistence", "harmonie-arome"),
             ("persistence", "aifs")]
    for control, experiment in pairs:
        out = make_scorecard(
            summary, tmp_path / f"{experiment}-vs-{control}.html",
            control=control, experiment=experiment,
            rows=["truth_source", "variable"], columns=["metric"], cell="lead_time",
        )[0]
        assert out.stat().st_size > 2000
