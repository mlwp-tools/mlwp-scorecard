"""Fetch DMI station observations as a second, neutral truth source.

    observations.zarr   (time, station)   station observations at the valid times

The DINI analysis is HARMONIE's own state, so it structurally favours HARMONIE.
Station observations are neutral between the two models, which is why they are
worth the extra work: they are the only truth here that neither model produced.

Source: https://opendataapi.dmi.dk/v2/metObs -- open, no API key required. (The
old ``dmigw.govcloud.dk`` endpoint was retired on 2026-06-30 and no longer
resolves.)

Only the valid times the forecasts actually need are fetched, which is the union
of ``init + lead`` over the five initialisations -- ten times, not the whole
period.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np
import xarray as xr
from common import METOBS_BASE, METOBS_PARAMETERS, OBS_BBOX, OBS_WINDOW_MIN, OUT


def fetch(parameter: str, when: np.datetime64, *, limit: int = 100000) -> list[dict]:
    """Fetch one parameter at one valid time, within a short window.

    Stations report every 10 minutes, so a narrow window either side of the valid
    time picks up one report per station without needing exact timestamps.

    Returns
    -------
    list of dict
        GeoJSON features.
    """
    half = np.timedelta64(OBS_WINDOW_MIN, "m")
    lo = np.datetime_as_string(when - half, unit="s") + "Z"
    hi = np.datetime_as_string(when + half, unit="s") + "Z"
    params = {
        "parameterId": parameter,
        "bbox": ",".join(str(v) for v in OBS_BBOX),
        "datetime": f"{lo}/{hi}",
        "limit": limit,
    }
    url = (
        f"{METOBS_BASE}/collections/observation/items?{urllib.parse.urlencode(params)}"
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read()).get("features", [])


def valid_times(fcst: xr.Dataset) -> np.ndarray:
    """Every valid time the forecasts need, deduplicated and sorted."""
    valid = fcst["init_time"].values[:, None] + fcst["lead_time"].values[None, :]
    return np.unique(valid.ravel())


def build(force: bool = False) -> None:
    """Fetch observations and write them as a station dataset."""
    out_path = OUT / "observations.zarr"
    if out_path.exists() and not force:
        print(f"{out_path} exists, skipping (use --force to rebuild)")
        return

    fcst = xr.open_zarr(OUT / "forecast.zarr")
    times = valid_times(fcst)
    print(f"{len(times)} valid times: {str(times[0])[:16]} .. {str(times[-1])[:16]}")
    print(f"bbox {OBS_BBOX}, +/-{OBS_WINDOW_MIN} min window")

    # station -> (lon, lat); value[(parameter, time, station)] -> float
    coords: dict[str, tuple[float, float]] = {}
    values: dict[str, dict] = defaultdict(dict)

    for dini_name, (parameter, scale, offset) in METOBS_PARAMETERS.items():
        n = 0
        for when in times:
            for feat in fetch(parameter, when):
                p = feat["properties"]
                sid = p["stationId"]
                coords.setdefault(sid, tuple(feat["geometry"]["coordinates"]))
                v = p.get("value")
                if v is None:
                    continue
                # keep the report closest to the valid time
                key = (when, sid)
                prev = values[dini_name].get(key)
                obs_t = np.datetime64(p["observed"][:19])
                if prev is None or abs(obs_t - when) < abs(prev[1] - when):
                    values[dini_name][key] = (v * scale + offset, obs_t)
                n += 1
        print(f"  {dini_name:<16} <- {parameter:<16} {n} reports")

    stations = sorted(coords)
    sidx = {s: i for i, s in enumerate(stations)}
    tidx = {t: i for i, t in enumerate(times)}
    print(f"\n{len(stations)} stations reporting")

    data = {}
    for dini_name in METOBS_PARAMETERS:
        arr = np.full((len(times), len(stations)), np.nan)
        for (when, sid), (val, _) in values[dini_name].items():
            arr[tidx[when], sidx[sid]] = val
        data[dini_name] = (("time", "station"), arr)
        filled = np.isfinite(arr).mean()
        print(f"  {dini_name:<16} {filled:.1%} of (time, station) filled")

    lon = np.array([coords[s][0] for s in stations])
    lat = np.array([coords[s][1] for s in stations])
    ds = xr.Dataset(
        data,
        coords=dict(
            time=times,
            station=np.array(stations),
            lat=("station", lat, {"units": "degrees_north"}),
            lon=("station", lon, {"units": "degrees_east"}),
        ),
        attrs=dict(
            title="DMI station observations",
            source=f"{METOBS_BASE} (open, no API key)",
            note=(
                "a truth neither model produced, unlike the DINI analysis; "
                "point measurements, not grid means"
            ),
            window_minutes=OBS_WINDOW_MIN,
        ),
    )
    for name, (_, scale, offset) in METOBS_PARAMETERS.items():
        ds[name].attrs.update(
            units={"t2m": "K", "pres_seasurface": "Pa", "wind_speed_10m": "m s-1"}[
                name
            ],
            conversion=f"metObs value * {scale} + {offset}",
        )

    if out_path.exists():
        shutil.rmtree(out_path)
    ds.to_zarr(out_path)
    print(f"\nwrote {out_path}  {dict(ds.sizes)}")

    print("\nsanity check (should be physically plausible):")
    for name in METOBS_PARAMETERS:
        v = ds[name].values
        print(
            f"  {name:<16} min {np.nanmin(v):>10.2f}  mean {np.nanmean(v):>10.2f}  "
            f"max {np.nanmax(v):>10.2f}  {ds[name].attrs['units']}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    build(force=p.parse_args().force)


if __name__ == "__main__":
    main()
