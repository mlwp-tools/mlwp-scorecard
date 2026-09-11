"""Report what the paired difference interval actually decides.

Contrasts the paired interval against the naive one you would get by treating the
two sources as independent, and shows which cells the card can mark.

Both now come out of the package rather than the input, so this reads them off
the resolved `Layout` instead of the dataset.

    uv run python tests/harmonie_dini/check_significance.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from common import CONFIDENCE_LEVELS, N_BOOT, OUT

from mlwp_scorecards import build_layout


def main() -> None:
    ds = xr.open_zarr(OUT / "verification.zarr")
    levels = list(CONFIDENCE_LEVELS)
    conf = max(levels)
    print(f"levels: {', '.join(f'{c:.4g}' for c in levels)}")
    print(f"widths below are at {conf:.0%}, which is what the drill-down draws")

    def card(truth: str):
        return build_layout(
            ds,
            control="aifs",
            experiment="harmonie-arome",
            rows=["truth_source", "variable"],
            columns=["metric"],
            truth_source=truth,
            confidence_levels=levels,
            n_resamples=N_BOOT,
            seed=0,
        )

    print(
        "\nwidth of the difference interval, paired vs treating the two "
        "sources as independent"
    )
    print(f"  {'variable':<16} {'lead':>5} {'paired':>10} {'naive':>10} {'ratio':>7}")
    lay = card("observations")
    for _, _, cell in lay.iter_cells():
        if cell.metric != "rmse":
            continue
        var = lay.rows[cell.row].key[-1]
        for i, s in enumerate(cell.steps):
            if i % 2 or s.value_lower is None:
                continue
            paired = s.value_upper - s.value_lower
            # independent: add the two half-widths in quadrature
            naive = 2 * np.hypot(
                (s.control_upper - s.control_lower) / 2,
                (s.experiment_upper - s.experiment_lower) / 2,
            )
            print(
                f"  {var:<16} {s.lead_time:>4.0f}h {paired:>10.4g} {naive:>10.4g} "
                f"{naive / paired:>6.1f}x"
            )

    print("\ncells the card can mark, by truth source")
    for truth in ds["truth_source"].values:
        lay = card(str(truth))
        total, sig = lay.stats.n_boxes, lay.stats.n_significant
        print(f"  {truth:<16} {sig}/{total} boxes significant ({sig / total:.0%})")
        key = ", ".join(f"{i + 1}={c:.4g}" for i, c in enumerate(levels))
        print(f"      (. = not significant, {key})")
        for _, _, cell in lay.iter_cells():
            marks = "".join(
                "."
                if s.significant_at is None
                else str(levels.index(s.significant_at) + 1)
                for s in cell.steps
            )
            var = lay.rows[cell.row].key[-1]
            print(f"      {var:<16} {cell.metric:<5} {marks}")


if __name__ == "__main__":
    main()
