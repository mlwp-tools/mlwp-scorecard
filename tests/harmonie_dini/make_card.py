"""Render the HARMONIE-AROME vs AIFS scorecard.

A limited-area physics model against a global data-driven one, both scored against
the same analysis. Which source is the baseline and which the forecast source is
an argument, not something baked into the data.

With no spatial regions and no pressure levels this is the simplest card shape:
one grouping on each axis, variable by metric, lead time inside the cell.
"""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr
from common import OUT

from mlwp_scorecards import ScoreCard
from mlwp_scorecards.render.static import save_figure

PAIRS = {
    "harmonie-vs-aifs": ("aifs", "harmonie-arome"),
    "aifs-vs-harmonie": ("harmonie-arome", "aifs"),
}


def render(ds: xr.Dataset, name: str, baseline: str, forecast: str, scheme: str):
    """Render one pairwise card in all three formats."""
    leads = ds["lead_time"].values / np.timedelta64(1, "h")
    n_init = ds.attrs.get("n_initialisations", "?")
    axes = dict(
        rows=["truth_source", "variable"],
        columns=["metric"],
        cell="lead_time",
    )
    kwargs = dict(
        colour_relative_to=baseline,
        select=dict(forecast_source=[forecast]),
        scheme=scheme,
        **axes,
        # a handful of initialisations: far too few for blocks
        bootstrap="iid",
        title=f"{forecast} vs {baseline}",
        subtitle=(
            f"{n_init} initialisations, 2026-09-04 to 09-05, lead times "
            f"+{leads[0]:.0f} h to +{leads[-1]:.0f} h. Verified twice: against the "
            f"DINI analysis, which is HARMONIE's own state, and against "
            f"{ds.attrs.get('n_stations', '?')} DMI stations, which are neutral."
        ),
    )
    score_card = ScoreCard(ds, **kwargs)
    html = OUT / f"{name}.html"
    html.write_text(score_card.to_html(), encoding="utf-8")
    fig = score_card.to_figure()
    written = [html] + [save_figure(fig, OUT / f"{name}.{s}") for s in ("png", "pdf")]
    layout = score_card._layout

    print(f"\n{name}: {layout.stats.n_rows} rows x {layout.stats.n_cols} columns")
    for p in written:
        print(f"  wrote {p.name}  {p.stat().st_size / 1024:.0f} kB")

    # `Step.relative` is oriented by polarity: positive always means the
    # forecast source is better, whichever direction of the raw metric that is.
    print(f"  % better than {baseline}, by lead time")
    print("    " + " " * 26 + "".join(f"{h:>7.0f}h" for h in leads))
    for r, _, cell in layout.iter_cells():
        var = layout.rows[r].key[-1]
        vals = "".join(f"{s.relative * 100:>8.1f}" for s in cell.steps)
        print(f"    {var:<20} {cell.metric:<5}{vals}")
    return layout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pair",
        default="harmonie-vs-aifs",
        choices=[*PAIRS, "all"],
        help="which comparison to render",
    )
    ap.add_argument("--scheme", default="cvd", choices=("cvd", "ecmwf"))
    args = ap.parse_args()

    ds = xr.open_zarr(OUT / "verification.zarr")
    names = list(PAIRS) if args.pair == "all" else [args.pair]
    for name in names:
        baseline, forecast = PAIRS[name]
        render(ds, name, baseline, forecast, args.scheme)

    print("\ncaveats carried in the dataset attributes:")
    for k, v in sorted(ds.attrs.items()):
        if k.startswith("caveat_"):
            print(f"  - {v}")


if __name__ == "__main__":
    main()
