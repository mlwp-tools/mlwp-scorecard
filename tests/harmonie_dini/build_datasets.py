"""Build the local zarr datasets for the HARMONIE-AROME DINI side of the comparison.

    truth.zarr     (time, y, x)                    the analysis at every analysis time
    forecast.zarr  (init_time, lead_time, y, x)    the HARMONIE-AROME forecast

Initialisations are chosen to be shared with AIFS (6-hourly) and fully verifiable,
i.e. a DINI analysis exists at every valid time out to the longest lead.
"""

from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
from common import (
    DINI_VARIABLES,
    GRID_STRIDE,
    INIT_STEP_H,
    N_INIT,
    OUT,
    iso,
    lead_times,
)
from inspect_store import list_analysis_times, open_store


def choose_initialisations(analysis_times: list[str]) -> list[str]:
    """Pick the oldest initialisations that AIFS also has and truth can verify.

    Returns
    -------
    list of str
        Analysis-time directory names, oldest first.
    """
    stamps = {t: np.datetime64(iso(t)) for t in analysis_times}
    available = set(stamps.values())
    leads = lead_times()

    chosen = []
    for name, t in sorted(stamps.items(), key=lambda kv: kv[1]):
        hour = t.astype("datetime64[h]").astype(int) % 24
        if hour % INIT_STEP_H:  # AIFS only initialises 6-hourly
            continue
        if not np.isin(t + leads, list(available)).all():
            continue  # no truth at some valid time
        chosen.append(name)
        if len(chosen) == N_INIT:
            break
    return chosen


def _read_field(args: tuple[str, str, int]) -> tuple[tuple[str, str, int], np.ndarray]:
    """Read one (analysis time, variable, time index) field, strided."""
    analysis_time, var, itime = args
    ds = open_store(analysis_time)
    return (analysis_time, var, itime), ds[var].isel(time=itime).values[
        ::GRID_STRIDE, ::GRID_STRIDE
    ]


def _run(jobs: list[tuple[str, str, int]], workers: int, label: str) -> dict:
    """Fetch many fields concurrently; S3 reads are IO bound."""
    out: dict = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (key, arr) in enumerate(pool.map(_read_field, jobs), start=1):
            out[key] = arr
            if i % 40 == 0 or i == len(jobs):
                print(
                    f"  {label}: {i}/{len(jobs)} fields "
                    f"({i / (time.perf_counter() - t0):.1f}/s)",
                    flush=True,
                )
    return out


def build(force: bool = False, workers: int = 8) -> None:
    """Extract and write the DINI truth and forecast datasets."""
    OUT.mkdir(parents=True, exist_ok=True)
    times = list_analysis_times()
    inits = choose_initialisations(times)
    leads = lead_times()
    lead_h = (leads / np.timedelta64(1, "h")).astype(int)

    print(f"{len(times)} analysis times available")
    print(f"initialisations ({len(inits)}, shared with AIFS and fully verifiable):")
    for t in inits:
        print(f"    {t}")
    print(f"lead times: {list(lead_h)} h")

    template = open_store(times[0])
    y = template["y"].values[::GRID_STRIDE]
    x = template["x"].values[::GRID_STRIDE]
    lat = template["lat"].values[::GRID_STRIDE, ::GRID_STRIDE]
    lon = template["lon"].values[::GRID_STRIDE, ::GRID_STRIDE]
    attrs = {v: dict(template[v].attrs) for v in DINI_VARIABLES}
    print(f"grid {len(y)}x{len(x)} (stride {GRID_STRIDE})")

    coords_xy = dict(
        y=y,
        x=x,
        lat=(("y", "x"), lat, {"units": "degrees_north", "standard_name": "latitude"}),
        lon=(("y", "x"), lon, {"units": "degrees_east", "standard_name": "longitude"}),
    )

    # ---- truth: the analysis at every analysis time ------------------------
    truth_path = OUT / "truth.zarr"
    if force or not truth_path.exists():
        print("\nreading truth (analysis at every analysis time)")
        jobs = [(t, v, 0) for t in times for v in DINI_VARIABLES]
        got = _run(jobs, workers, "truth")
        truth = xr.Dataset(
            {
                v: (
                    ("time", "y", "x"),
                    np.stack([got[(t, v, 0)] for t in times]),
                    attrs[v],
                )
                for v in DINI_VARIABLES
            },
            coords=dict(
                time=np.array([np.datetime64(iso(t)) for t in times]), **coords_xy
            ),
            attrs=dict(
                title="HARMONIE-AROME DINI analysis (truth)",
                source="s3://harmonie-zarr/dini/control",
                note="lead 0 of each analysis time's forecast, i.e. the analysis itself",
                grid_stride=GRID_STRIDE,
            ),
        )
        _write(truth, truth_path)
    else:
        print(f"\n{truth_path} exists, skipping (use --force to rebuild)")
    truth = xr.open_zarr(truth_path)

    # ---- forecast ----------------------------------------------------------
    fcst_path = OUT / "forecast.zarr"
    if force or not fcst_path.exists():
        print("\nreading HARMONIE-AROME forecasts")
        jobs = [(t, v, int(h)) for t in inits for v in DINI_VARIABLES for h in lead_h]
        got = _run(jobs, workers, "forecast")
        fcst = xr.Dataset(
            {
                v: (
                    ("init_time", "lead_time", "y", "x"),
                    np.stack(
                        [np.stack([got[(t, v, int(h))] for h in lead_h]) for t in inits]
                    ),
                    attrs[v],
                )
                for v in DINI_VARIABLES
            },
            coords=dict(
                init_time=np.array([np.datetime64(iso(t)) for t in inits]),
                lead_time=leads,
                **coords_xy,
            ),
            attrs=dict(
                title="HARMONIE-AROME DINI control forecast",
                source="s3://harmonie-zarr/dini/control",
                grid_stride=GRID_STRIDE,
            ),
        )
        _write(fcst, fcst_path)
    else:
        print(f"\n{fcst_path} exists, skipping")
    fcst = xr.open_zarr(fcst_path)

    print("\nsummary")
    for name, d in (("truth", truth), ("forecast", fcst)):
        print(f"  {name:12s} {dict(d.sizes)}")


def _write(ds: xr.Dataset, path) -> None:
    if path.exists():
        shutil.rmtree(path)
    ds.to_zarr(path)
    mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {path}  {mb:.0f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    build(force=args.force, workers=args.workers)


if __name__ == "__main__":
    main()
