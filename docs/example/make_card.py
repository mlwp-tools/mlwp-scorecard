"""Draw the README's example scorecard from the WeatherBench 2 scores.

Reads ``tmp/wb2/wb2_graphcast_vs_hres_2020.nc``, which ``score_weatherbench2.py``
writes (run that first), and writes the image and the interactive page under
``docs/images/``. Those two are committed; the scores are not.

    uv run --extra netcdf --extra static python docs/example/make_card.py
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from mlwp_scorecards import ScoreCard
from mlwp_scorecards.render.static import save_figure

HERE = Path(__file__).resolve().parent
SCORES = HERE.parents[1] / "tmp" / "wb2" / "wb2_graphcast_vs_hres_2020.nc"
IMAGES = HERE.parent / "images"


def main() -> None:
    """Build the card and write ``scorecard.png`` and ``scorecard.html``."""
    if not SCORES.exists():
        raise SystemExit(f"no scores at {SCORES}: run score_weatherbench2.py first")
    ds = xr.open_dataset(SCORES)
    score_card = ScoreCard(
        ds,
        baseline="ifs-hres",
        # One metric: as the outer row label it spans the variables once,
        # rather than repeating as a header row under every region.
        rows=["metric", "variable"],
        columns=["spatial_region"],
        title="GraphCast vs IFS HRES, verified against ERA5",
        subtitle=(
            f"WeatherBench 2, 2020: {ds.sizes['init_time']} initialisations "
            "(00 and 12 UTC), lead times 1-10 days, 5.625° grid"
        ),
    )
    IMAGES.mkdir(exist_ok=True)
    save_figure(score_card.to_figure(), IMAGES / "scorecard.png", dpi=160)
    (IMAGES / "scorecard.html").write_text(score_card.to_html(), encoding="utf-8")
    print(repr(score_card))


if __name__ == "__main__":
    main()
