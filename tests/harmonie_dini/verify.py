"""Score HARMONIE-AROME and AIFS against two independent truths.

Produces the verification summary the scorecard package consumes:

    <var>(truth_source, prediction_source, metric, lead_time, stat)
    <var>_number_of_cases(truth_source, prediction_source, metric, lead_time)

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

The two collapses, in both cases:

1. **Over space** -- grid points (area-weighted by cos(latitude)) or stations
   (equal weight) -- giving one number per forecast case. Deterministic; no
   sampling uncertainty attaches here.
2. **Over forecast cases.** The five initialisations reduce to a mean, and *that*
   is the sample whose uncertainty ``lower``/``upper`` describe.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from common import CONFIDENCE, METRICS, N_BOOT, OUT

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


def resample(n_case: int, rng: np.random.Generator) -> np.ndarray:
    """One set of bootstrap case indices, ``(boot, case)``.

    Drawn **once** and reused for every source and every metric. That is what makes
    the difference interval below a *paired* one: within each replicate the two
    sources are compared on the same weather, so the shared component of their
    error cancels instead of adding.
    """
    return rng.integers(0, n_case, (N_BOOT, n_case))


def summarise(per_case: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Collapse cases: the mean, its bootstrap interval, and the case count.

    Parameters
    ----------
    per_case : np.ndarray
        ``(case, lead_time)`` -- one score per forecast case.
    idx : np.ndarray
        ``(boot, case)`` from :func:`resample`.
    """
    with np.errstate(invalid="ignore"):
        boot = np.nanmean(per_case[idx], axis=1)  # (boot, lead_time)
        mean = np.nanmean(per_case, axis=0)  # (lead_time,)
    a = (1 - CONFIDENCE) / 2 * 100
    return dict(
        mean=mean,
        lower=np.nanpercentile(boot, a, axis=0),
        upper=np.nanpercentile(boot, 100 - a, axis=0),
        n=np.sum(np.isfinite(per_case), axis=0),
    )


def summarise_difference(
    experiment: np.ndarray, control: np.ndarray, idx: np.ndarray
) -> dict[str, np.ndarray]:
    """The paired difference and its interval.

    The difference is taken **per case, before any averaging**, and the same
    resample is applied to it. Differencing two independently-bootstrapped means
    would give a far wider and quite wrong interval: the two models are run on the
    same weather, so most of their error is shared and cancels.

    Parameters
    ----------
    experiment, control : np.ndarray
        ``(case, lead_time)`` per-case scores for the two sources.
    idx : np.ndarray
        ``(boot, case)`` from :func:`resample`.
    """
    d = experiment - control  # (case, lead_time) -- paired
    with np.errstate(invalid="ignore"):
        boot = np.nanmean(d[idx], axis=1)  # (boot, lead_time)
        mean = np.nanmean(d, axis=0)
    a = (1 - CONFIDENCE) / 2 * 100
    return dict(
        mean=mean,
        lower=np.nanpercentile(boot, a, axis=0),
        upper=np.nanpercentile(boot, 100 - a, axis=0),
        n=np.sum(np.isfinite(d), axis=0),
    )


def build_summary() -> xr.Dataset:
    """Score both prediction sources against both truths."""
    analysis = xr.open_zarr(OUT / "truth.zarr")
    obs = xr.open_zarr(OUT / "observations.zarr")
    sources = {
        "aifs": xr.open_zarr(OUT / "aifs.zarr"),
        "harmonie-arome": xr.open_zarr(OUT / "forecast.zarr"),
    }
    leads = sources["harmonie-arome"]["lead_time"].values
    truths = ["dini-analysis", "observations"]
    stats = ["mean", "lower", "upper"]
    rng = np.random.default_rng(0)

    flat_index, distance_km = station_indices(sources["harmonie-arome"], obs)
    print(
        f"{len(flat_index)} stations matched to grid points; displacement "
        f"median {np.median(distance_km):.1f} km, max {distance_km.max():.1f} km"
    )

    n_case = int(sources["harmonie-arome"].sizes["init_time"])
    idx = resample(n_case, rng)
    names = list(sources)

    out: dict[str, xr.DataArray] = {}
    for var in SCORED:
        shape = (len(truths), len(sources), len(METRICS), len(leads))
        vals = np.full(shape + (len(stats),), np.nan)
        cnts = np.zeros(shape, dtype=np.int64)

        # per-case scores kept, so the difference below can be paired
        per_case: dict[tuple[str, str, str], np.ndarray] = {}

        for si, (name, pred) in enumerate(sources.items()):
            scored = {
                "dini-analysis": score_against_grid(pred, analysis, var),
                "observations": score_against_stations(pred, obs, var, flat_index),
            }
            for ti, truth_name in enumerate(truths):
                for mi, metric in enumerate(METRICS):
                    series = scored[truth_name][metric]
                    per_case[(name, truth_name, metric)] = series
                    s = summarise(series, idx)
                    for ki, key in enumerate(stats):
                        vals[ti, si, mi, :, ki] = s[key]
                    cnts[ti, si, mi] = s["n"]

        coords = dict(
            truth_source=truths,
            prediction_source=names,
            metric=list(METRICS),
            lead_time=leads,
            stat=stats,
        )
        out[var] = xr.DataArray(
            vals,
            dims=list(coords),
            coords=coords,
            attrs=dict(
                units=UNITS[var],
                long_name=LONG_NAMES[var],
                ancillary_variables=f"{var}_number_of_cases {var}_difference",
            ),
        )
        out[f"{var}_number_of_cases"] = xr.DataArray(
            cnts,
            dims=[d for d in coords if d != "stat"],
            coords={k: v for k, v in coords.items() if k != "stat"},
            attrs=dict(standard_name="number_of_observations"),
        )

        # ---- the paired difference, for every ordered pair of sources ---------
        # Sparse by construction: the diagonal is zero and is left NaN.
        dshape = (len(truths), len(names), len(names), len(METRICS), len(leads))
        dvals = np.full(dshape + (len(stats),), np.nan)
        for ci, ctl_name in enumerate(names):
            for ei, exp_name in enumerate(names):
                if ci == ei:
                    continue
                for ti, truth_name in enumerate(truths):
                    for mi, metric in enumerate(METRICS):
                        s = summarise_difference(
                            per_case[(exp_name, truth_name, metric)],
                            per_case[(ctl_name, truth_name, metric)],
                            idx,
                        )
                        for ki, key in enumerate(stats):
                            dvals[ti, ci, ei, mi, :, ki] = s[key]

        dcoords = dict(
            truth_source=truths,
            control_source=names,
            experiment_source=names,
            metric=list(METRICS),
            lead_time=leads,
            stat=stats,
        )
        out[f"{var}_difference"] = xr.DataArray(
            dvals,
            dims=list(dcoords),
            coords=dcoords,
            attrs=dict(
                units=UNITS[var],
                long_name=f"{LONG_NAMES[var]}: experiment minus control",
                note=(
                    "paired: the difference is taken per forecast case before "
                    "averaging, and one bootstrap resample is shared by both "
                    "sources, so their common error cancels"
                ),
            ),
        )

    ds = xr.Dataset(out)
    ds.coords["confidence"] = CONFIDENCE
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
            v = ds[var].sel(truth_source=truth, metric="rmse", stat="mean")
            print(f"  vs {truth}")
            print(
                "      " + "".join(f"{s:>17s}" for s in ds["prediction_source"].values)
            )
            for i, h in enumerate(leads):
                row = "".join(
                    f"{float(v[si, i]):>17.4g}"
                    for si in range(ds.sizes["prediction_source"])
                )
                print(f"    {h:>3d}h" + row)

    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
