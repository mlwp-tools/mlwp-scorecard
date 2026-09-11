"""Score HARMONIE-AROME and AIFS against two independent truths.

Produces the per-case scores the scorecard package consumes:

    {metric}.{var}(truth_source, forecast_source, lead_time, init_time)

Two truth sources, which is the point of the exercise:

* **dini-analysis** -- gridded, over the whole DINI domain. It is HARMONIE's own
  state, so it structurally favours HARMONIE: the model is being checked against
  the field it was initialised from, sharing its physics and orography.
* **observations** -- DMI station reports. Neutral between the two models, since
  neither produced them, but only ~60 points, only over Denmark, and a 2 m
  thermometer measures something a grid mean does not.

Neither is "the" truth. Putting both on one card is the honest presentation, and
is what the ``truth_source`` dimension is for -- the ECMWF reference card stacks
its ``an`` and ``ob`` blocks the same way.

The two collapses, and which one is this script's:

1. **Over space** -- grid points (area-weighted by cos(latitude)) or stations
   (equal weight) -- giving one number per forecast case. Deterministic; no
   sampling uncertainty attaches here, and it needs the fields, so it happens
   here. **This is all this script does.**
2. **Over forecast cases.** The five initialisations reduce to a mean, and *that*
   is the sample an interval describes. It belongs to the package, which needs
   the per-case numbers to pair the two sources against the same weather.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from common import METRICS, OUT

#: Scored fields. Wind speed is derived from u/v, which both models carry.
SCORED = ("t2m", "pres_seasurface", "wind_speed_10m")

UNITS = {"t2m": "K", "pres_seasurface": "Pa", "wind_speed_10m": "m s-1"}
LONG_NAMES = {
    "t2m": "2 metre temperature",
    "pres_seasurface": "Mean sea level pressure",
    "wind_speed_10m": "10 metre wind speed",
}


def field(ds: xr.Dataset, var: str) -> np.ndarray:
    """Return one scored field, deriving wind speed where needed."""
    if var == "wind_speed_10m":
        return np.hypot(ds["u10m"].values, ds["v10m"].values)
    return ds[var].values


def _align_truth(
    pred: xr.Dataset, truth_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map each (init, lead) cell onto the truth time axis.

    Returns
    -------
    have : np.ndarray
        ``(init, lead)`` boolean, True where a truth field exists.
    where : np.ndarray
        ``(init, lead)`` index into ``truth_times``; meaningless where not ``have``.
    """
    valid = pred["init_time"].values[:, None] + pred["lead_time"].values[None, :]
    index = {t: i for i, t in enumerate(truth_times)}
    have = np.isin(valid, truth_times)
    where = np.zeros(valid.shape, dtype=int)
    for i, j in zip(*np.where(have)):
        where[i, j] = index[valid[i, j]]
    return have, where


def _reduce(err: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Collapse the trailing spatial axis into one number per (case, lead)."""
    allnan = np.isnan(err).all(axis=-1)
    w = weights / weights.sum()
    with np.errstate(invalid="ignore"):
        mse = np.nansum(w * err**2, axis=-1)
        mae = np.nansum(w * np.abs(err), axis=-1)
    return {
        "rmse": np.where(allnan, np.nan, np.sqrt(mse)),
        "mae": np.where(allnan, np.nan, mae),
    }


def score_against_grid(
    pred: xr.Dataset, truth: xr.Dataset, var: str
) -> dict[str, np.ndarray]:
    """Score on the model grid, area-weighted by cos(latitude)."""
    have, where = _align_truth(pred, truth["time"].values)
    t_all = field(truth, var)  # (time, y, x)
    t_full = np.where(have[..., None, None], t_all[where], np.nan)
    err = field(pred, var) - t_full  # (init, lead, y, x)
    weights = np.broadcast_to(
        np.cos(np.deg2rad(truth["lat"].values)), err.shape[-2:]
    ).ravel()
    return _reduce(err.reshape(err.shape[:2] + (-1,)), weights)


def station_indices(
    model: xr.Dataset, obs: xr.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest model grid point to each station, and the distance to it.

    Nearest neighbour rather than bilinear: on the strided ~16 km grid the
    interpolation weights would smooth more than the comparison warrants. The
    displacement is returned so it can be judged rather than assumed negligible.

    Returns
    -------
    flat_index : np.ndarray
        ``(station,)`` index into the flattened ``(y, x)`` grid.
    distance_km : np.ndarray
        ``(station,)`` great-circle distance to that grid point.
    """
    from scipy.spatial import cKDTree

    def xyz(lat, lon):
        la, lo = np.deg2rad(lat), np.deg2rad(lon)
        return np.stack(
            [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1
        )

    grid = xyz(model["lat"].values.ravel(), model["lon"].values.ravel())
    pts = xyz(obs["lat"].values, obs["lon"].values)
    chord, idx = cKDTree(grid).query(pts)
    # chord length on the unit sphere -> great-circle distance
    return idx, 2 * np.arcsin(np.clip(chord / 2, 0, 1)) * 6371.0


def score_against_stations(
    pred: xr.Dataset, obs: xr.Dataset, var: str, flat_index: np.ndarray
) -> dict[str, np.ndarray]:
    """Score at station points, sampling the model at the nearest grid point."""
    have, where = _align_truth(pred, obs["time"].values)
    o_all = obs[var].values  # (time, station)
    o_full = np.where(have[..., None], o_all[where], np.nan)  # (init, lead, station)

    p = field(pred, var)  # (init, lead, y, x)
    p_at = p.reshape(p.shape[:2] + (-1,))[..., flat_index]  # (init, lead, station)

    return _reduce(p_at - o_full, np.ones(o_full.shape[-1]))


def build_summary() -> xr.Dataset:
    """Score both prediction sources against both truths."""
    analysis = xr.open_zarr(OUT / "truth.zarr")
    obs = xr.open_zarr(OUT / "observations.zarr")
    sources = {
        "aifs": xr.open_zarr(OUT / "aifs.zarr"),
        "harmonie-arome": xr.open_zarr(OUT / "forecast.zarr"),
    }
    leads = sources["harmonie-arome"]["lead_time"].values
    inits = sources["harmonie-arome"]["init_time"].values
    truths = ["dini-analysis", "observations"]

    flat_index, distance_km = station_indices(sources["harmonie-arome"], obs)
    print(
        f"{len(flat_index)} stations matched to grid points; displacement "
        f"median {np.median(distance_km):.1f} km, max {distance_km.max():.1f} km"
    )

    names = list(sources)
    n_case = len(inits)

    # One score per forecast case. The collapse over cases -- the mean, its
    # interval, the paired difference and the case count -- is the package's, so
    # none of it happens here. What remains is the collapse over *space*, which
    # needs the fields and so cannot move.
    out: dict[str, xr.DataArray] = {}
    for var in SCORED:
        raw = np.full(
            (len(truths), len(names), len(METRICS), len(leads), n_case), np.nan
        )
        for si, (name, pred) in enumerate(sources.items()):
            scored = {
                "dini-analysis": score_against_grid(pred, analysis, var),
                "observations": score_against_stations(pred, obs, var, flat_index),
            }
            for ti, truth_name in enumerate(truths):
                for mi, metric in enumerate(METRICS):
                    raw[ti, si, mi] = scored[truth_name][metric].T  # (lead, case)

        coords = dict(
            truth_source=truths,
            forecast_source=names,
            metric=list(METRICS),
            lead_time=leads,
            init_time=inits,
        )
        scored_da = xr.DataArray(raw, dims=list(coords), coords=coords)
        for metric in METRICS:
            out[f"{metric}.{var}"] = scored_da.sel(
                metric=metric, drop=True
            ).assign_attrs(
                units=UNITS[var],
                long_name=f"{LONG_NAMES[var]} {metric.upper()}",
            )

    ds = xr.Dataset(out)
    ds.attrs.update(
        title="HARMONIE-AROME DINI vs ECMWF AIFS",
        caveat_analysis=(
            "the DINI analysis is HARMONIE's own state and favours it: same physics, "
            "same orography, and the field HARMONIE was initialised from"
        ),
        caveat_observations=(
            f"station observations are neutral between the models but cover only "
            f"{obs.sizes['station']} points over Denmark, and a point measurement is "
            f"not a grid mean; models are sampled at the nearest grid point, median "
            f"{np.median(distance_km):.1f} km away"
        ),
        caveat_regridding="AIFS is interpolated from 0.25 deg onto the ~2 km DINI grid",
        caveat_sample=(
            "5 initialisations spanning 24 h; the bootstrap is iid over them and is "
            "therefore optimistic, since consecutive runs share weather"
        ),
        n_initialisations=int(sources["harmonie-arome"].sizes["init_time"]),
        n_stations=int(obs.sizes["station"]),
    )
    return ds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT / "verification.zarr"))
    args = ap.parse_args()

    ds = build_summary()
    p = Path(args.out)
    if p.exists():
        shutil.rmtree(p)
    ds.to_zarr(p)

    leads = (ds["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    for var in SCORED:
        print(f"\n{var} RMSE ({UNITS[var]})")
        for truth in ds["truth_source"].values:
            v = ds[f"rmse.{var}"].sel(truth_source=truth).mean("init_time")
            print(f"  vs {truth}")
            print("      " + "".join(f"{s:>17s}" for s in ds["forecast_source"].values))
            for i, h in enumerate(leads):
                row = "".join(
                    f"{float(v[si, i]):>17.4g}"
                    for si in range(ds.sizes["forecast_source"])
                )
                print(f"    {h:>3d}h" + row)

    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
