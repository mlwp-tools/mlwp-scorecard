"""The WeatherBench 2 example still builds, and says what the README says it does.

The README's image is drawn from scores that `docs/example/score_weatherbench2.py`
writes to the gitignored `tmp/wb2/`. They are not committed, so this runs where
they have been made and skips otherwise, like the HARMONIE tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlwp_scorecards import ScoreCard

SCORES = Path(__file__).resolve().parents[1] / "tmp/wb2/wb2_graphcast_vs_hres_2020.nc"


@pytest.fixture(scope="module")
def score_card():
    if not SCORES.exists():
        pytest.skip("no WeatherBench 2 scores: run docs/example/score_weatherbench2.py")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    ds = xr.open_dataset(SCORES)
    return ScoreCard(
        ds,
        baseline="ifs-hres",
        rows=["metric", "variable"],
        columns=["spatial_region"],
        n_resamples=50,
    )


def test_the_example_builds_with_the_default_resample(score_card):
    """732 twelve-hourly cases: enough for the default blocks, no iid needed."""
    lay = score_card._layout
    assert lay.resampling == "moving-block" and lay.block_length > 1
    assert lay.stats.n_rows == 3 and lay.stats.n_cols == 5
    assert lay.stats.n_boxes == 3 * 5 * 10


def test_graphcast_is_better_than_hres_in_most_boxes(score_card):
    """What the README says the picture shows, as WeatherBench 2 reports."""
    steps = [s for _, _, cell in score_card._layout.iter_cells() for s in cell.steps]
    better = sum(s.level > 0 for s in steps)
    assert better > 0.8 * len(steps), (better, len(steps))
