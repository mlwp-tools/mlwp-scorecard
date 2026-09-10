"""Score HARMONIE-AROME, AIFS and persistence against the DINI analysis.

Reads the local zarr datasets and produces the verification summary the scorecard
package consumes, in the schema documented in ``PLAN.md``:

    <var>(truth_source, prediction_source, metric, lead_time, stat)
    <var>_number_of_cases(truth_source, prediction_source, metric, lead_time)

All three sources are scored, so one file yields any pairwise card.

No spatial region grouping and no pressure levels, as requested: one implicit
global domain (the DINI area) and single-level variables only.

The two collapses:

1. **Over space, within each forecast case.** The grid reduces to one number per
   case, area-weighted by cos(latitude). Deterministic; no sampling uncertainty.
2. **Over forecast cases.** The five initialisations reduce to a mean, and *that*
   is the sample whose uncertainty ``lower``/``upper`` describe.

The truth is HARMONIE's own analysis, which is a real home advantage for DINI: it
is the state DINI was initialised from and is consistent with DINI's own physics
and orography. AIFS is being judged against a competitor's analysis. That is
recorded in the dataset attributes and repeated on the card.
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


def per_case_scores(
    pred: xr.Dataset, truth: xr.Dataset, var: str, weights: np.ndarray
) -> dict[str, np.ndarray]:
    """Collapse space, giving one number per (case, lead time) per metric.

    Returns
    -------
    dict of str to np.ndarray
        Metric name -> ``(init_time, lead_time)``.
    """
    valid = pred["init_time"].values[:, None] + pred["lead_time"].values[None, :]
    have = np.isin(valid, truth["time"].values)          # (init, lead)

    t_all = field(truth, var)                            # (time, y, x)
    index = {t: i for i, t in enumerate(truth["time"].values)}
    t_full = np.full(valid.shape + t_all.shape[1:], np.nan)
    for i, j in zip(*np.where(have)):
        t_full[i, j] = t_all[index[valid[i, j]]]

    err = field(pred, var) - t_full                      # (init, lead, y, x)
    allnan = np.isnan(err).all(axis=(-2, -1))            # (init, lead)
    w = weights / weights.sum()

    with np.errstate(invalid="ignore"):
        mse = np.nansum(w * err**2, axis=(-2, -1))
        mae = np.nansum(w * np.abs(err), axis=(-2, -1))
    return {
        "rmse": np.where(allnan, np.nan, np.sqrt(mse)),
        "mae": np.where(allnan, np.nan, mae),
    }


def summarise(per_case: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Collapse cases: the mean, its bootstrap interval, and the case count.

    Parameters
    ----------
    per_case : np.ndarray
        ``(case, lead_time)`` -- one score per forecast case.
    """
    n_case = per_case.shape[0]
    idx = rng.integers(0, n_case, (N_BOOT, n_case))      # (boot, case)
    with np.errstate(invalid="ignore"):
        boot = np.nanmean(per_case[idx], axis=1)         # (boot, lead_time)
        mean = np.nanmean(per_case, axis=0)              # (lead_time,)
    a = (1 - CONFIDENCE) / 2 * 100
    return dict(
        mean=mean,
        lower=np.nanpercentile(boot, a, axis=0),
        upper=np.nanpercentile(boot, 100 - a, axis=0),
        n=np.sum(np.isfinite(per_case), axis=0),
    )


def build_summary() -> xr.Dataset:
    """Produce the verification-summary dataset for all three sources."""
    truth = xr.open_zarr(OUT / "truth.zarr")
    sources = {
        "persistence": xr.open_zarr(OUT / "persistence.zarr"),
        "aifs": xr.open_zarr(OUT / "aifs.zarr"),
        "harmonie-arome": xr.open_zarr(OUT / "forecast.zarr"),
    }
    weights = np.cos(np.deg2rad(truth["lat"].values))
    leads = sources["harmonie-arome"]["lead_time"].values
    stats = ["mean", "lower", "upper"]
    rng = np.random.default_rng(0)

    out: dict[str, xr.DataArray] = {}
    for var in SCORED:
        vals = np.full((len(sources), len(METRICS), len(leads), len(stats)), np.nan)
        cnts = np.zeros((len(sources), len(METRICS), len(leads)), dtype=np.int64)
        for si, (name, pred) in enumerate(sources.items()):
            scores = per_case_scores(pred, truth, var, weights)
            for mi, metric in enumerate(METRICS):
                s = summarise(scores[metric], rng)
                for ti, key in enumerate(stats):
                    vals[si, mi, :, ti] = s[key]
                cnts[si, mi] = s["n"]

        coords = dict(
            truth_source=["dini-analysis"],
            prediction_source=list(sources),
            metric=list(METRICS),
            lead_time=leads,
            stat=stats,
        )
        out[var] = xr.DataArray(
            vals[None], dims=list(coords), coords=coords,
            attrs=dict(units=UNITS[var], long_name=LONG_NAMES[var],
                       ancillary_variables=f"{var}_number_of_cases"),
        )
        out[f"{var}_number_of_cases"] = xr.DataArray(
            cnts[None], dims=[d for d in coords if d != "stat"],
            coords={k: v for k, v in coords.items() if k != "stat"},
            attrs=dict(standard_name="number_of_observations"),
        )

    ds = xr.Dataset(out)
    ds.coords["confidence"] = CONFIDENCE
    ds.attrs.update(
        title="HARMONIE-AROME DINI vs ECMWF AIFS, against the DINI analysis",
        truth="HARMONIE-AROME DINI analysis",
        caveat_truth=(
            "the truth is HARMONIE's own analysis, which favours DINI: it is the "
            "state DINI was initialised from and shares its physics and orography"
        ),
        caveat_regridding=(
            "AIFS is interpolated from 0.25 deg onto the ~2 km DINI grid"
        ),
        caveat_sample=(
            "5 initialisations spanning 24 h; the bootstrap is iid over them and "
            "is therefore optimistic, since consecutive runs share weather"
        ),
        n_initialisations=int(sources["harmonie-arome"].sizes["init_time"]),
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
        v = ds[var].sel(truth_source="dini-analysis", metric="rmse", stat="mean")
        print(f"\n{var} RMSE ({UNITS[var]}) against the DINI analysis")
        head = "  " + "".join(f"{s:>16s}" for s in ds["prediction_source"].values)
        print(f"  {'lead':>5}" + head[2:])
        for i, h in enumerate(leads):
            row = "".join(f"{float(v[si, i]):>16.4g}"
                          for si in range(ds.sizes["prediction_source"]))
            print(f"  {h:>4d}h" + row)

    n = ds[f"{SCORED[0]}_number_of_cases"].values
    print(f"\ncases per lead time: {np.unique(n)}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
