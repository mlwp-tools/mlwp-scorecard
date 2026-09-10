"""Time a single field read, to size the full extraction before starting it.

Zarr chunks here are whole spatial fields, so a chunk is the smallest unit that can
be transferred and the total cost is (fields needed) x (this).
"""

from __future__ import annotations

import time

from common import GRID_STRIDE, LEAD_STEP_H, N_INIT, VARIABLES
from inspect_store import list_analysis_times, open_store


def main() -> None:
    times = list_analysis_times()
    ds = open_store(times[0])

    t0 = time.perf_counter()
    field = ds["t2m"].isel(time=0).values
    dt = time.perf_counter() - t0
    mb = field.nbytes / 1e6
    print(f"one field: {dt:.2f}s for {mb:.1f} MB uncompressed ({mb / dt:.1f} MB/s)")

    n_lead = len(range(0, 37, LEAD_STEP_H))
    n_truth = len(times) * len(VARIABLES)
    n_fcst = N_INIT * n_lead * len(VARIABLES)
    total = n_truth + n_fcst
    print(f"\nplanned reads:")
    print(f"  truth      {len(times)} analysis times x {len(VARIABLES)} vars = {n_truth}")
    print(f"  forecasts  {N_INIT} inits x {n_lead} leads x {len(VARIABLES)} vars = {n_fcst}")
    print(f"  total      {total} fields, ~{total * mb / 1000:.1f} GB uncompressed")
    print(f"  estimate   {total * dt / 60:.1f} min at the rate above")
    sub = field[::GRID_STRIDE, ::GRID_STRIDE]
    print(f"\nafter stride {GRID_STRIDE}: {sub.shape} = {sub.nbytes / 1e6:.2f} MB per field")
    print(f"  stored total ~{total * sub.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
