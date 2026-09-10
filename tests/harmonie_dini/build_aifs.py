"""Extract AIFS and put it on the DINI grid, so the two models can be compared.

    aifs.zarr   (init_time, lead_time, y, x)   AIFS interpolated to the DINI points

AIFS is global at 0.25 deg; DINI is a ~2 km limited-area Lambert grid. Comparing
them means putting one on the other's points, and this interpolates the coarse
model onto the fine grid. That is the conventional direction and keeps the truth
untouched, but it is not neutral: it gives AIFS credit for detail it never
resolved, while penalising nothing for the smoothness. Any conclusion drawn from
the resulting card carries that caveat.

Two traps handled here, both silent if missed:

* AIFS reports temperature in degrees Celsius, DINI in kelvin.
* DINI longitudes run 316..400 deg (a 0-360 convention that wraps past 360);
  AIFS uses -180..180. Interpolating without folding puts every eastern DINI
  point outside the source grid, yielding NaN over half the domain.
"""

from __future__ import annotations

import argparse
import shutil

import numpy as np
import xarray as xr
from common import AIFS_REPO, AIFS_VARIABLES, OUT, to_180
from inspect_aifs import open_aifs


def build(force: bool = False) -> None:
    """Extract AIFS for the DINI initialisations and regrid it."""
    out_path = OUT / "aifs.zarr"
    if out_path.exists() and not force:
        print(f"{out_path} exists, skipping (use --force to rebuild)")
        return

    fcst = xr.open_zarr(OUT / "forecast.zarr")
    init_times = fcst["init_time"].values
    leads = fcst["lead_time"].values
    lat2d = fcst["lat"].values
    lon2d = to_180(fcst["lon"].values)

    print(f"initialisations : {[str(t)[:16] for t in init_times]}")
    print(f"lead times      : {(leads / np.timedelta64(1, 'h')).astype(int)} h")
    print(
        f"target grid     : {lat2d.shape}, "
        f"lat {lat2d.min():.2f}..{lat2d.max():.2f}, "
        f"lon {lon2d.min():.2f}..{lon2d.max():.2f} (folded to -180..180)"
    )

    aifs = open_aifs()
    missing = [t for t in init_times if t not in set(aifs["init_time"].values)]
    if missing:
        raise SystemExit(f"AIFS lacks initialisations: {missing}")

    # Pointwise interpolation: naming the target dims (y, x) makes xarray pair the
    # lat and lon arrays elementwise rather than forming their outer product.
    target_lat = xr.DataArray(lat2d, dims=("y", "x"))
    target_lon = xr.DataArray(lon2d, dims=("y", "x"))

    sub = aifs[list(AIFS_VARIABLES)].sel(init_time=init_times, lead_time=leads)
    # AIFS latitude runs north to south; interp needs it ascending.
    if sub["latitude"].values[0] > sub["latitude"].values[-1]:
        sub = sub.isel(latitude=slice(None, None, -1))

    # Crop to the DINI bounding box first: interpolating the full global field
    # would pull 61x721x1440 chunks for a domain covering a few percent of them.
    pad = 1.0
    sub = sub.sel(
        latitude=slice(lat2d.min() - pad, lat2d.max() + pad),
        longitude=slice(lon2d.min() - pad, lon2d.max() + pad),
    )
    print(
        f"source subset   : lat {sub.sizes['latitude']}, lon {sub.sizes['longitude']}"
    )

    print("loading and interpolating ...", flush=True)
    sub = sub.load()
    regridded = sub.interp(latitude=target_lat, longitude=target_lon, method="linear")

    data = {}
    for src, (dini_name, offset) in AIFS_VARIABLES.items():
        arr = regridded[src].transpose("init_time", "lead_time", "y", "x").values
        if offset:
            arr = arr + offset  # degC -> K
        units = "K" if offset else aifs[src].attrs.get("units")
        data[dini_name] = (
            ("init_time", "lead_time", "y", "x"),
            arr,
            dict(units=units, long_name=src, source_variable=src),
        )

    ds = xr.Dataset(
        data,
        coords=dict(
            init_time=init_times,
            lead_time=leads,
            y=fcst["y"].values,
            x=fcst["x"].values,
            lat=(("y", "x"), lat2d),
            lon=(("y", "x"), fcst["lon"].values),
        ),
        attrs=dict(
            title="ECMWF AIFS single forecast, interpolated to the DINI grid",
            source=f"arraylake {AIFS_REPO}",
            regridding="bilinear from 0.25 deg global onto the DINI Lambert points",
            caveat=(
                "the coarse model is interpolated onto the fine grid; this is the "
                "conventional direction but is not neutral between the two models"
            ),
        ),
    )

    if out_path.exists():
        shutil.rmtree(out_path)
    ds.to_zarr(out_path)
    mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out_path}  {mb:.0f} MB  {dict(ds.sizes)}")

    finite = {v: float(np.isfinite(ds[v].values).mean()) for v in ds.data_vars}
    print(
        "finite fraction after regridding:", {k: f"{v:.1%}" for k, v in finite.items()}
    )
    if min(finite.values()) < 0.99:
        print(
            "  WARNING: NaNs after interpolation usually mean a longitude "
            "convention mismatch"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    build(force=p.parse_args().force)


if __name__ == "__main__":
    main()
