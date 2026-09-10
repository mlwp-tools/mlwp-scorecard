"""Report what is in the ECMWF AIFS subscription, before extracting anything.

AIFS is ECMWF's data-driven forecast model. Comparing it against HARMONIE-AROME
DINI on a common truth is the comparison this package exists for: a global
data-driven model against a limited-area physics-based one.
"""

from __future__ import annotations


def open_aifs():
    """Open the AIFS forecast subscription read-only."""
    import xarray as xr
    from arraylake import Client

    client = Client()
    repo = client.get_repo(
        "danish-meteorological-institute/ecmwf-aifs-single-forecast-subscription"
    )
    session = repo.readonly_session(branch="main")
    return xr.open_zarr(session.store, zarr_format=3)


def main() -> None:
    ds = open_aifs()
    print(ds)
    print("\n--- coordinates ---")
    for name, c in ds.coords.items():
        vals = c.values
        if vals.ndim == 0:
            print(f"  {name}: {vals}")
        elif vals.size <= 8:
            print(f"  {name} ({vals.size}): {vals}")
        else:
            print(f"  {name} ({vals.size}): {vals[0]} .. {vals[-1]}")

    print("\n--- data variables ---")
    for name, v in ds.data_vars.items():
        print(
            f"  {name:24s} {tuple(v.dims)} {v.shape} "
            f"units={v.attrs.get('units', '?')}"
        )

    if "latitude" in ds.coords and "longitude" in ds.coords:
        lat, lon = ds["latitude"].values, ds["longitude"].values
        print(
            f"\ngrid: lat {lat.min():.2f}..{lat.max():.2f} ({lat.size}), "
            f"lon {lon.min():.2f}..{lon.max():.2f} ({lon.size})"
        )
        if lat.size > 1:
            print(f"      resolution ~{abs(lat[1] - lat[0]):.3f} deg")


if __name__ == "__main__":
    main()
