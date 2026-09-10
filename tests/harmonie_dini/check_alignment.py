"""Work out where AIFS and HARMONIE-AROME DINI can actually be compared.

The two models agree on nothing by default: different initialisation cadences,
different lead-time steps, different grids, different units, different variable
names. This reports the intersection so the extraction plan is grounded rather
than assumed.
"""

from __future__ import annotations

import numpy as np

from common import iso
from inspect_aifs import open_aifs
from inspect_store import list_analysis_times, open_store

#: DINI name -> AIFS name, for directly comparable fields.
DIRECT = {
    "t2m": "temperature_2m",
    "pres_seasurface": "pressure_reduced_to_mean_sea_level",
}
#: Both stores carry 10 m u/v, so wind speed is a clean derived third variable.
WIND = {"dini": ("u10m", "v10m"), "aifs": ("wind_u_10m", "wind_v_10m")}


def main() -> None:
    aifs = open_aifs()
    dini_times = list_analysis_times()
    dini = open_store(dini_times[0])

    a_lead = aifs["lead_time"].values
    d_lead = dini["time"].values - dini["time"].values[0]
    a_h = (a_lead / np.timedelta64(1, "h")).astype(int)
    d_h = (d_lead / np.timedelta64(1, "h")).astype(int)
    print(f"AIFS lead times : {a_h[0]}..{a_h[-1]} h, step {a_h[1] - a_h[0]} ({len(a_h)})")
    print(f"DINI lead times : {d_h[0]}..{d_h[-1]} h, step {d_h[1] - d_h[0]} ({len(d_h)})")

    a_init = aifs["init_time"].values
    d_init = np.array([np.datetime64(iso(t)) for t in dini_times])
    print(f"\nAIFS inits      : {a_init[0]} .. {a_init[-1]} "
          f"(step {(a_init[1] - a_init[0]) / np.timedelta64(1, 'h'):g} h)")
    print(f"DINI inits      : {d_init[0]} .. {d_init[-1]} "
          f"(step {(d_init[1] - d_init[0]) / np.timedelta64(1, 'h'):g} h)")

    # what can actually be compared
    common_init = np.intersect1d(a_init, d_init)
    common_lead = np.intersect1d(a_h, d_h)
    common_lead = common_lead[common_lead > 0]      # lead 0 is the analysis
    print(f"\ncommon inits    : {len(common_init)} -> {[str(t)[:16] for t in common_init]}")
    print(f"common leads    : {list(common_lead)} h")

    # truth must exist at every valid time
    truth_times = d_init
    ok = []
    for t in common_init:
        valid = t + common_lead.astype("timedelta64[h]")
        if np.isin(valid, truth_times).all():
            ok.append(t)
    print(f"fully verifiable: {len(ok)} -> {[str(t)[:16] for t in ok]}")

    print("\nvariables")
    for d, a in DIRECT.items():
        print(f"  {d:18s} <- {a:36s} "
              f"{dini[d].attrs.get('units'):>10s} vs {aifs[a].attrs.get('units')}")
    print(f"  {'wind_speed_10m':18s} <- derived from "
          f"{WIND['dini']} and {WIND['aifs']}")

    print("\ngrids")
    lat, lon = dini["lat"].values, dini["lon"].values
    print(f"  DINI: {lat.shape} Lambert, lat {lat.min():.2f}..{lat.max():.2f}, "
          f"lon {lon.min():.2f}..{lon.max():.2f}")
    print(f"  AIFS: 0.25 deg regular global; will be interpolated onto the DINI points")


if __name__ == "__main__":
    main()
