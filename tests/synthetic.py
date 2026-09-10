"""Synthetic verification data built from a toy reanalysis.

Produces a gridded "reanalysis", scores two forecast sources against it, and
returns the verification-summary dataset the package consumes.

The two forecast sources are deliberately chosen so the expected answer is known:

* **persistence** — the analysis at initialisation time, held constant. The
  classic trivial baseline; its error grows as the atmosphere decorrelates.
* **drifting persistence** — persistence plus an accumulated Gaussian random
  walk, so its error grows faster with lead time. It is *worse* than the
  baseline, and increasingly so, which is what the resulting scorecard must show.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

__all__ = ["make_reanalysis", "make_forecasts", "make_verification_dataset"]

VARIABLES = {
    "z": dict(
        units="m", long_name="Geopotential height", levels=[500.0, 850.0], scale=60.0
    ),
    "t": dict(units="K", long_name="Temperature", levels=[500.0, 850.0], scale=3.0),
    "msl": dict(
        units="Pa", long_name="Mean sea level pressure", levels=None, scale=400.0
    ),
    "2t": dict(units="K", long_name="2 metre temperature", levels=None, scale=2.5),
}

REGIONS = {
    "n.hem": (20.0, 90.0),
    "tropics": (-20.0, 20.0),
    "s.hem": (-90.0, -20.0),
}

METRICS = ("rmse", "mae", "spread")


def make_reanalysis(
    *, n_case: int = 120, n_lat: int = 24, n_lon: int = 48, seed: int = 0
) -> xr.Dataset:
    """A toy gridded reanalysis: smooth fields evolving with a synoptic timescale.

    Returns
    -------
    xr.Dataset
        Dimensions ``(case, lat, lon)`` per variable-level, in physical units.
    """
    rng = np.random.default_rng(seed)
    lat = np.linspace(-87.5, 87.5, n_lat)
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    case = np.arange(n_case)  # 12-hourly initialisation index

    LON, LAT = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))  # (lat, lon)

    def field(scale: float, n_wave: int = 3) -> np.ndarray:
        """AR(1)-in-time sum of travelling waves -> (case, lat, lon)."""
        out = np.zeros((n_case, n_lat, n_lon))
        phi = 0.85  # ~2-day decorrelation at 12-hourly
        for _ in range(n_wave):
            k = rng.integers(2, 7)
            speed = rng.uniform(0.05, 0.25)
            amp = np.zeros(n_case)
            e = rng.normal(0, 1, n_case)
            amp[0] = e[0]
            for i in range(1, n_case):
                amp[i] = phi * amp[i - 1] + np.sqrt(1 - phi**2) * e[i]
            phase = rng.uniform(0, 2 * np.pi)
            wave = (
                np.cos(k * LON - speed * case[:, None, None] + phase) * np.cos(LAT) ** 2
            )
            out += amp[:, None, None] * wave
        return out / np.sqrt(n_wave) * scale

    data = {}
    for name, meta in VARIABLES.items():
        levels = meta["levels"]
        if levels is None:
            data[name] = xr.DataArray(
                field(meta["scale"]),
                dims=("case", "lat", "lon"),
                coords=dict(case=case, lat=lat, lon=lon),
                attrs=dict(units=meta["units"], long_name=meta["long_name"]),
            )
        else:
            stack = np.stack([field(meta["scale"]) for _ in levels], axis=1)
            data[name] = xr.DataArray(
                stack,
                dims=("case", "level", "lat", "lon"),
                coords=dict(case=case, level=levels, lat=lat, lon=lon),
                attrs=dict(units=meta["units"], long_name=meta["long_name"]),
            )
    return xr.Dataset(data)


def make_forecasts(
    analysis: xr.DataArray, leads: np.ndarray, *, drift: float, seed: int
) -> np.ndarray:
    """Forecast a field by persistence, optionally plus a random walk.

    Parameters
    ----------
    analysis : xr.DataArray
        ``(case, [level,] lat, lon)`` truth.
    leads : np.ndarray
        Lead times in hours.
    drift : float
        Standard deviation of the per-step random-walk increment, as a fraction of
        the field's own standard deviation. ``0`` gives pure persistence.
    seed : int

    Returns
    -------
    np.ndarray
        ``(case, lead_time, [level,] lat, lon)`` forecast values.
    """
    rng = np.random.default_rng(seed)
    a = analysis.values  # (case, [level,] lat, lon)
    sigma = float(np.std(a))
    out = np.empty((a.shape[0], len(leads)) + a.shape[1:])
    for i in range(a.shape[0]):
        walk = np.zeros(a.shape[1:])
        for j in range(len(leads)):
            if drift:
                walk = walk + rng.normal(0, drift * sigma, a.shape[1:])
            out[i, j] = a[i] + walk  # persistence + accumulated walk
    return out


def _weighted_rmse(err: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Area-weighted RMSE over the trailing (lat, lon) axes.

    A case with no truth at all yields NaN, not a spuriously perfect zero --
    ``nansum`` of an all-NaN slice is 0, which would silently count as a flawless
    forecast and keep the case in ``n``.
    """
    allnan = np.isnan(err).all(axis=(-2, -1))
    out = np.sqrt(np.nansum(w * err**2, axis=(-2, -1)) / np.nansum(w))
    return np.where(allnan, np.nan, out)


def _weighted_mae(err: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Area-weighted mean absolute error; all-NaN slices stay NaN."""
    allnan = np.isnan(err).all(axis=(-2, -1))
    out = np.nansum(w * np.abs(err), axis=(-2, -1)) / np.nansum(w)
    return np.where(allnan, np.nan, out)


def make_verification_dataset(
    *,
    n_case: int = 120,
    leads_h: tuple[int, ...] = (24, 48, 72, 96, 120, 144, 168),
    drift: float = 0.06,
    n_boot: int = 300,
    block: int = 8,
    seed: int = 0,
) -> xr.Dataset:
    """Build a complete verification-summary dataset from synthetic data.

    Scores persistence (control) and drifting persistence (experiment) against the
    reanalysis, over three latitude bands and three metrics, and summarises over
    forecast cases with a moving-block bootstrap.

    Returns
    -------
    xr.Dataset
        One score variable per physical variable, plus case-count ancillaries.
    """
    rng = np.random.default_rng(seed + 99)
    ana = make_reanalysis(n_case=n_case, seed=seed)
    leads = np.asarray(leads_h)
    lat = ana["lat"].values
    coslat = np.cos(np.deg2rad(lat))

    stats = ["mean", "lower", "upper"]
    out: dict[str, xr.DataArray] = {}

    for name, meta in VARIABLES.items():
        truth = ana[name]
        has_level = "level" in truth.dims
        levels = truth["level"].values if has_level else [None]

        ctl = make_forecasts(truth, leads, drift=0.0, seed=seed + 1)
        exp = make_forecasts(truth, leads, drift=drift, seed=seed + 2)

        # a case initialised at index i verifies at i + lead/12; beyond the end
        # of the record there is no truth, so it drops out
        valid = np.arange(n_case)[:, None] + (leads[None, :] // 12)  # (case, lead)
        ok = valid < n_case  # (case, lead)

        shape = (2, len(levels), len(REGIONS), len(METRICS), len(leads), len(stats))
        vals = np.full(shape, np.nan)
        cnts = np.full(
            (2, len(levels), len(REGIONS), len(METRICS), len(leads)), 0, dtype=np.int64
        )

        for li, lev in enumerate(levels):
            t = (
                truth.sel(level=lev).values if has_level else truth.values
            )  # (case, lat, lon)
            c = ctl[:, :, li] if has_level else ctl  # (case, lead, lat, lon)
            e = exp[:, :, li] if has_level else exp
            # verify against the truth valid at case+lead
            vi = np.clip(valid, 0, n_case - 1)
            tv = t[vi]  # (case, lead, lat, lon)
            err_c = np.where(ok[..., None, None], c - tv, np.nan)
            err_e = np.where(ok[..., None, None], e - tv, np.nan)

            for ri, (rname, (lo, hi)) in enumerate(REGIONS.items()):
                m = (lat >= lo) & (lat <= hi)
                w = np.broadcast_to(coslat[m][:, None], (m.sum(), len(ana["lon"])))
                for mi, metric in enumerate(METRICS):
                    if metric == "rmse":
                        sc, se = (
                            _weighted_rmse(err_c[..., m, :], w),
                            _weighted_rmse(err_e[..., m, :], w),
                        )
                    elif metric == "mae":
                        sc, se = (
                            _weighted_mae(err_c[..., m, :], w),
                            _weighted_mae(err_e[..., m, :], w),
                        )
                    else:  # spread: dispersion of the forecast field itself
                        with np.errstate(invalid="ignore"), warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            sc = np.nanstd(
                                np.where(ok[..., None, None], c, np.nan)[..., m, :],
                                axis=(-2, -1),
                            )
                            se = np.nanstd(
                                np.where(ok[..., None, None], e, np.nan)[..., m, :],
                                axis=(-2, -1),
                            )
                        sc = np.where(ok, sc, np.nan)
                        se = np.where(ok, se, np.nan)
                    # sc, se: (case, lead)
                    for si, per_case in enumerate((sc, se)):
                        # per_case: (case, lead_time) -- ONE number per forecast case
                        cnts[si, li, ri, mi] = np.sum(np.isfinite(per_case), axis=0)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            vals[si, li, ri, mi, :, 0] = np.nanmean(per_case, axis=0)
                            # moving-block bootstrap over CASES: consecutive runs
                            # share a weather system, so an iid resample would
                            # badly overstate certainty
                            nb = int(np.ceil(n_case / block))
                            st = rng.integers(0, n_case - block + 1, (n_boot, nb))
                            idx = (st[:, :, None] + np.arange(block)).reshape(
                                n_boot, -1
                            )[
                                :, :n_case
                            ]  # (boot, case)
                            b = np.nanmean(per_case[idx], axis=1)  # (boot, lead_time)
                            vals[si, li, ri, mi, :, 1] = np.nanpercentile(
                                b, 2.5, axis=0
                            )
                            vals[si, li, ri, mi, :, 2] = np.nanpercentile(
                                b, 97.5, axis=0
                            )

        dims = (
            ["prediction_source"]
            + (["level"] if has_level else [])
            + ["spatial_region", "metric", "lead_time", "stat"]
        )
        arr = vals if has_level else vals[:, 0]
        cnt = cnts if has_level else cnts[:, 0]
        coords = dict(
            prediction_source=["persistence", "drifting-persistence"],
            spatial_region=list(REGIONS),
            metric=list(METRICS),
            lead_time=leads.astype("timedelta64[h]"),
            stat=stats,
        )
        if has_level:
            coords["level"] = np.asarray(levels, dtype=float)

        out[name] = xr.DataArray(
            arr,
            dims=dims,
            coords=coords,
            attrs=dict(
                units=meta["units"],
                long_name=meta["long_name"],
                ancillary_variables=f"{name}_number_of_cases",
            ),
        )
        out[f"{name}_number_of_cases"] = xr.DataArray(
            cnt,
            dims=[d for d in dims if d != "stat"],
            coords={k: v for k, v in coords.items() if k != "stat"},
            attrs=dict(standard_name="number_of_observations"),
        )

    ds = xr.Dataset(out)
    ds = ds.expand_dims(truth_source=["analysis"])
    ds.coords["confidence"] = 0.95
    ds.attrs.update(control="persistence", experiment="drifting-persistence")
    return ds
