"""Shared configuration for the HARMONIE-AROME DINI vs AIFS experiment.

Real-data counterpart to the synthetic tests. Three prediction sources are scored
against a common truth, the DINI analysis:

* **harmonie-arome** -- DMI's operational limited-area physics model, ~2 km.
* **aifs** -- ECMWF's global data-driven model, 0.25 deg.
* **persistence** -- the analysis at initialisation, held constant. Not a serious
  competitor, but a floor: anything that fails to beat it is broken.

Sources:

    s3://harmonie-zarr/dini/control/<analysis-time>/single_levels.zarr
    arraylake danish-meteorological-institute/ecmwf-aifs-single-forecast-subscription

The two models agree on nothing by default -- initialisation cadence, lead-time
step, grid, units and variable names all differ -- so the constants below record
the intersection that ``check_alignment.py`` establishes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# HARMONIE-AROME DINI
# --------------------------------------------------------------------------- #
BUCKET = "harmonie-zarr"
PREFIX = "dini/control"
STORE = "single_levels.zarr"

AWS_PROFILE = os.environ.get("HARMONIE_AWS_PROFILE", "dmidev-mlflow")
AWS_REGION = os.environ.get("AWS_REGION", "eu-central-1")

#: DINI fields to extract. u10m/v10m are combined into a wind speed at scoring time.
DINI_VARIABLES = ("t2m", "pres_seasurface", "u10m", "v10m")

# --------------------------------------------------------------------------- #
# ECMWF AIFS
# --------------------------------------------------------------------------- #
AIFS_REPO = "danish-meteorological-institute/ecmwf-aifs-single-forecast-subscription"

#: AIFS fields, and the factor/offset taking them to the DINI units.
#: AIFS reports temperature in degrees Celsius; DINI in kelvin.
AIFS_VARIABLES = {
    "temperature_2m": ("t2m", 273.15),
    "pressure_reduced_to_mean_sea_level": ("pres_seasurface", 0.0),
    "wind_u_10m": ("u10m", 0.0),
    "wind_v_10m": ("v10m", 0.0),
}

# --------------------------------------------------------------------------- #
# What can actually be compared
# --------------------------------------------------------------------------- #
#: AIFS initialises 6-hourly, DINI 3-hourly, so only 6-hourly inits are shared.
INIT_STEP_H = 6

#: AIFS carries 6-hourly lead times; DINI hourly. Lead 0 is excluded because
#: persistence is the analysis there, making a relative difference undefined.
LEAD_STEP_H = 6
MAX_LEAD_H = 36

#: Initialisations to score. Chosen by check_alignment.py: shared with AIFS, and
#: with DINI analyses available at every valid time out to MAX_LEAD_H.
N_INIT = 5

#: Grid stride. Zarr chunks are whole spatial fields, so this saves memory and
#: compute but not transfer; verification statistics do not need all 3.1M points.
GRID_STRIDE = 8

#: Local output root. Gitignored -- these are large.
OUT = Path(__file__).resolve().parents[2] / "tmp" / "harmonie"

#: Scored quantities. Both are error metrics: lower is better.
METRICS = ("rmse", "mae")

CONFIDENCE = 0.95
N_BOOT = 2000


def storage_options() -> dict:
    """fsspec options for the DMI S3 endpoint."""
    return {"profile": AWS_PROFILE, "client_kwargs": {"region_name": AWS_REGION}}


def store_path(analysis_time: str) -> str:
    """Bucket-relative path of one analysis time's single-level store."""
    return f"{BUCKET}/{PREFIX}/{analysis_time}/{STORE}"


def iso(analysis_time: str) -> str:
    """``2026-09-04T060000Z`` -> ``2026-09-04T06:00:00``."""
    return (
        f"{analysis_time[:11]}{analysis_time[11:13]}:"
        f"{analysis_time[13:15]}:{analysis_time[15:17]}"
    )


def lead_times() -> np.ndarray:
    """Verification lead times, as a timedelta array."""
    return np.arange(LEAD_STEP_H, MAX_LEAD_H + 1, LEAD_STEP_H).astype("timedelta64[h]")


def to_180(lon: np.ndarray) -> np.ndarray:
    """Fold longitudes onto [-180, 180).

    DINI stores longitude in a 0-360 convention that runs past 360 at the eastern
    edge of the domain; AIFS uses -180..180. Interpolating one onto the other
    without this silently lands every eastern point outside the source grid.
    """
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0
