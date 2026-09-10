"""Report what the paired difference interval actually decides.

Contrasts the paired interval against the naive one you would get by treating the
two sources as independent, and shows which cells the card can now mark.

    uv run python tests/harmonie_dini/check_significance.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from common import OUT

from mlwp_scorecards import build_layout


def main() -> None:
    ds = xr.open_zarr(OUT / "verification.zarr")
    leads = (ds["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    conf = float(ds["confidence"].values)
    print(f"interval: {conf:.0%}  ({(1 - conf) / 2:.1%} .. {1 - (1 - conf) / 2:.1%})")

    print(
        "\nwidth of the difference interval, paired vs treating the two "
        "sources as independent"
    )
    print(f"  {'variable':<16} {'lead':>5} {'paired':>10} {'naive':>10} {'ratio':>7}")
    for var in ("t2m", "pres_seasurface", "wind_speed_10m"):
        d = ds[f"{var}_difference"].sel(
            truth_source="observations",
            control_source="aifs",
            experiment_source="harmonie-arome",
            metric="rmse",
        )
        m = ds[var].sel(truth_source="observations", metric="rmse")
        paired = (d.sel(stat="upper") - d.sel(stat="lower")).values
        # independent: add the two half-widths in quadrature
        hw = (
            lambda s: (  # noqa: E731
                m.sel(prediction_source=s, stat="upper")
                - m.sel(prediction_source=s, stat="lower")
            ).values
            / 2
        )
        naive = 2 * np.hypot(hw("aifs"), hw("harmonie-arome"))
        for i, h in enumerate(leads):
            if i % 2:
                continue
            print(
                f"  {var:<16} {h:>4d}h {paired[i]:>10.4g} {naive[i]:>10.4g} "
                f"{naive[i] / paired[i]:>6.1f}x"
            )

    print("\ncells the card can now mark, by truth source")
    for truth in ds["truth_source"].values:
        lay = build_layout(
            ds,
            control="aifs",
            experiment="harmonie-arome",
            rows=["truth_source", "variable"],
            columns=["metric"],
            cell="lead_time",
            truth_source=str(truth),
        )
        total = lay.stats.n_boxes
        sig = lay.stats.n_significant
        print(f"  {truth:<16} {sig}/{total} boxes significant ({sig / total:.0%})")
        for _, _, cell in lay.iter_cells():
            marks = "".join("*" if s.significant else "." for s in cell.steps)
            var = lay.rows[cell.row].key[-1]
            print(f"      {var:<16} {cell.metric:<5} {marks}")


if __name__ == "__main__":
    main()
