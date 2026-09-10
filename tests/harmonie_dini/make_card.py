"""Render scorecards from the HARMONIE-AROME / AIFS / persistence verification.

One verification dataset, several cards: which source is control and which is
experiment is an argument, not something baked into the data. The default pair is
the interesting one -- the limited-area physics model against the global
data-driven one -- with persistence available as a floor.

With no spatial regions and no pressure levels this is the simplest card shape:
one grouping on each axis, variable by metric, lead time inside the cell.
"""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

from common import OUT
from mlwp_scorecards import build_layout, make_scorecard

PAIRS = {
    "harmonie-vs-aifs": ("aifs", "harmonie-arome"),
    "harmonie-vs-persistence": ("persistence", "harmonie-arome"),
    "aifs-vs-persistence": ("persistence", "aifs"),
}


def render(ds: xr.Dataset, name: str, control: str, experiment: str, scheme: str):
    """Render one pairwise card in all three formats."""
    leads = ds["lead_time"].values / np.timedelta64(1, "h")
    n_init = ds.attrs.get("n_initialisations", "?")
    axes = dict(
        rows=["truth_source", "variable"], columns=["metric"], cell="lead_time",
    )
    kwargs = dict(
        control=control, experiment=experiment, scheme=scheme, **axes,
        title=f"{experiment} vs {control}",
        subtitle=(
            f"{n_init} initialisations, 2026-09-04 to 09-05, verified against the "
            f"HARMONIE-AROME DINI analysis. Lead times +{leads[0]:.0f} h to "
            f"+{leads[-1]:.0f} h."
        ),
    )
    outputs = [OUT / f"{name}.{ext}" for ext in ("html", "png", "pdf")]
    written = make_scorecard(ds, outputs, **kwargs)
    layout = build_layout(ds, control=control, experiment=experiment, **axes)

    print(f"\n{name}: {layout.stats.n_rows} rows x {layout.stats.n_cols} columns")
    for p in written:
        print(f"  wrote {p.name}  {p.stat().st_size / 1024:.0f} kB")

    # `Step.relative` is oriented by polarity: positive always means the
    # experiment is better, whichever direction of the raw metric that is.
    print(f"  % better than {control}, by lead time")
    print("    " + " " * 26 + "".join(f"{h:>7.0f}h" for h in leads))
    for r, _, cell in layout.iter_cells():
        var = layout.rows[r].key[-1]
        vals = "".join(f"{s.relative * 100:>8.1f}" for s in cell.steps)
        print(f"    {var:<20} {cell.metric:<5}{vals}")
    return layout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", default="harmonie-vs-aifs",
                    choices=[*PAIRS, "all"], help="which comparison to render")
    ap.add_argument("--scheme", default="cvd", choices=("cvd", "ecmwf"))
    args = ap.parse_args()

    ds = xr.open_zarr(OUT / "verification.zarr")
    names = list(PAIRS) if args.pair == "all" else [args.pair]
    for name in names:
        control, experiment = PAIRS[name]
        render(ds, name, control, experiment, args.scheme)

    print("\ncaveats carried in the dataset attributes:")
    for k in ("caveat_truth", "caveat_regridding", "caveat_sample"):
        print(f"  - {ds.attrs[k]}")


if __name__ == "__main__":
    main()
