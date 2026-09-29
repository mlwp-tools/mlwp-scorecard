"""Compute the scores behind the README's example scorecard.

Downloads GraphCast and IFS HRES forecasts and ERA5 for 2020 from WeatherBench
2's public bucket (over HTTPS, no credentials needed), computes the
area-weighted RMSE of each forecast against ERA5 for every initialisation,
lead time and region, and writes the result to
``tmp/wb2/wb2_graphcast_vs_hres_2020.nc`` (gitignored; about 0.7 MB).

    uv run --extra netcdf python docs/example/score_weatherbench2.py

Why this is a separate script: ``mlwp-scorecards`` draws scorecards from
scores it is given -- one score per forecast case -- and never computes them
itself; normally that is done beforehand, with a tool such as mxalign. This
script is that step for the example, and ``make_card.py`` then draws the card
from its output.

It needs ``fsspec`` and ``aiohttp`` for the HTTPS reads, and downloads about
0.3 GB: the stores are chunked as whole global fields, so the download does
not shrink with the region scored. Downloads are cached in ``tmp/wb2/`` too,
so a rerun does not fetch them again.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

BUCKET = "https://storage.googleapis.com/weatherbench2/datasets"
FORECASTS = {
    "ifs-hres": f"{BUCKET}/hres/2016-2022-0012-64x32_equiangular_conservative.zarr",
    "graphcast": (
        f"{BUCKET}/graphcast/2020/"
        "date_range_2019-11-16_2021-02-01_12_hours-64x32_equiangular_conservative.zarr"
    ),
}
TRUTH = f"{BUCKET}/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

#: Scored name -> (WeatherBench 2 field, units).
VARIABLES = {
    "t2m": ("2m_temperature", "K"),
    "msl": ("mean_sea_level_pressure", "Pa"),
    "ws10": ("10m_wind_speed", "m s-1"),
}

INITS = np.arange(
    np.datetime64("2020-01-01T00"),
    np.datetime64("2021-01-01T00"),
    np.timedelta64(12, "h"),
).astype("datetime64[ns]")
LEADS = (np.arange(1, 11) * np.timedelta64(24, "h")).astype("timedelta64[ns]")

#: Region -> (lat min, lat max, lon min, lon max), degrees; lon in [-180, 180).
REGIONS = {
    "global": (-90.0, 90.0, -180.0, 180.0),
    "n.extratropics": (20.0, 90.0, -180.0, 180.0),
    "tropics": (-20.0, 20.0, -180.0, 180.0),
    "s.extratropics": (-90.0, -20.0, -180.0, 180.0),
    "europe": (35.0, 72.0, -12.5, 42.5),
}

CACHE = Path(__file__).resolve().parents[2] / "tmp" / "wb2"
#: The scores: in the gitignored cache rather than the repo, because they are
#: large and remade from public data by running this script.
OUT = CACHE / "wb2_graphcast_vs_hres_2020.nc"


def _cached(name: str, load) -> xr.DataArray:
    """Return a field from the local cache, loading and caching it on a miss.

    Parameters
    ----------
    name : str
        The cache file's stem.
    load : callable
        Returns the field as a loaded ``xr.DataArray``.

    Returns
    -------
    xr.DataArray
        The field.
    """
    path = CACHE / f"{name}.nc"
    if path.exists():
        return xr.open_dataarray(path).load()
    CACHE.mkdir(parents=True, exist_ok=True)
    da = load()
    da.to_netcdf(path)
    return da


def forecast(source: str, field: str) -> xr.DataArray:
    """Load one forecast field at the scored initialisations and lead times.

    Parameters
    ----------
    source : str
        A key of :data:`FORECASTS`.
    field : str
        The WeatherBench 2 field name.

    Returns
    -------
    xr.DataArray
        ``(time, prediction_timedelta, longitude, latitude)``.
    """

    def load() -> xr.DataArray:
        """Read the field from the bucket.

        Returns
        -------
        xr.DataArray
            The loaded field.
        """
        # Recent xarray decodes CF timedeltas only when asked; without this the
        # lead times arrive as bare integers.
        ds = xr.open_zarr(FORECASTS[source], consolidated=True, decode_timedelta=True)
        da = ds[field].sel(time=INITS, prediction_timedelta=LEADS)
        print(f"  downloading {source} {field} {dict(da.sizes)}")
        return da.load()

    return _cached(f"{source}_{field}", load)


def truth(field: str) -> xr.DataArray:
    """Load one ERA5 field over every valid time the forecasts need.

    Parameters
    ----------
    field : str
        The WeatherBench 2 field name.

    Returns
    -------
    xr.DataArray
        ``(time, longitude, latitude)``.
    """

    def load() -> xr.DataArray:
        """Read the field from the bucket.

        Returns
        -------
        xr.DataArray
            The loaded field.
        """
        ds = xr.open_zarr(TRUTH, consolidated=True)
        first, last = INITS[0] + LEADS[0], INITS[-1] + LEADS[-1]
        da = ds[field].sel(time=slice(first, last))
        print(f"  downloading era5 {field} {dict(da.sizes)}")
        return da.load()

    return _cached(f"era5_{field}", load)


def region_weights(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    """Build cos(latitude) area weights, zero outside each region.

    Parameters
    ----------
    lat, lon : xr.DataArray
        The grid's coordinates, longitude in ``[0, 360)``.

    Returns
    -------
    xr.DataArray
        ``(spatial_region, longitude, latitude)``.
    """
    lon180 = ((lon + 180) % 360) - 180
    out = []
    for lat0, lat1, lon0, lon1 in REGIONS.values():
        inside = (lat >= lat0) & (lat <= lat1) & (lon180 >= lon0) & (lon180 <= lon1)
        out.append(np.cos(np.deg2rad(lat)) * inside)
    return xr.concat(out, dim=xr.DataArray(list(REGIONS), dims="spatial_region"))


def rmse(fc: xr.DataArray, obs: xr.DataArray, w: xr.DataArray) -> xr.DataArray:
    """Area-weighted RMSE per case, lead time and region.

    Parameters
    ----------
    fc : xr.DataArray
        ``(time, prediction_timedelta, longitude, latitude)``.
    obs : xr.DataArray
        The truth at each forecast's valid time, same dimensions.
    w : xr.DataArray
        From :func:`region_weights`.

    Returns
    -------
    xr.DataArray
        ``(spatial_region, time, prediction_timedelta)``.
    """
    sq = (fc - obs) ** 2
    mse = (sq * w).sum(("longitude", "latitude")) / w.sum(("longitude", "latitude"))
    return np.sqrt(mse)


def main() -> None:
    """Score every source and variable and write the netCDF."""
    variables = {}
    for name, (field, units) in VARIABLES.items():
        era5 = truth(field)
        valid = (INITS[:, None] + LEADS[None, :]).ravel()
        obs = era5.sel(time=valid).values.reshape(
            len(INITS), len(LEADS), *era5.shape[1:]
        )
        per_source = []
        for source in FORECASTS:
            fc = forecast(source, field).transpose(
                "time", "prediction_timedelta", "longitude", "latitude"
            )
            o = fc.copy(data=obs)
            w = region_weights(fc.latitude, fc.longitude)
            per_source.append(rmse(fc, o, w))
        da = xr.concat(
            per_source, dim=xr.DataArray(list(FORECASTS), dims="forecast_source")
        )
        da = da.rename(time="init_time", prediction_timedelta="lead_time")
        da = da.transpose("forecast_source", "spatial_region", "lead_time", "init_time")
        da.attrs = {"units": units, "long_name": f"area-weighted RMSE of {field}"}
        variables[f"rmse.{name}"] = da.astype("float32")

    ds = xr.Dataset(variables)
    ds.attrs = {
        "title": "GraphCast and IFS HRES vs ERA5, 2020: per-case area-weighted RMSE",
        "source": "WeatherBench 2, gs://weatherbench2 (64x32 equiangular, conservative)",
        "forecasts": "; ".join(f"{k}: {v}" for k, v in FORECASTS.items()),
        "truth": f"era5: {TRUTH}",
        "created_by": "docs/example/score_weatherbench2.py",
        "created": dt.date.today().isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT, encoding={v: {"zlib": True} for v in ds.data_vars})
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
