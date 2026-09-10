"""Report what is in the bucket and in one store, before downloading anything.

Run first: it establishes the analysis-time cadence, the forecast length, and how
much data a single field costs, all of which the extraction plan depends on.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import s3fs
import xarray as xr

from common import AWS_PROFILE, AWS_REGION, BUCKET, PREFIX, iso, storage_options, store_path


def list_analysis_times() -> list[str]:
    """Analysis times present in the bucket, oldest first."""
    env = dict(os.environ, AWS_REGION=AWS_REGION)
    out = subprocess.run(
        ["s5cmd", "--profile", AWS_PROFILE, "ls", f"s3://{BUCKET}/{PREFIX}/"],
        capture_output=True, text=True, check=True, env=env,
    )
    return sorted(ln.split()[-1].rstrip("/") for ln in out.stdout.splitlines() if "DIR" in ln)


def open_store(analysis_time: str) -> xr.Dataset:
    """Open one analysis time's store lazily."""
    fs = s3fs.S3FileSystem(**storage_options())
    return xr.open_zarr(s3fs.S3Map(store_path(analysis_time), s3=fs), consolidated=True)


def main() -> None:
    times = list_analysis_times()
    print(f"{len(times)} analysis times: {times[0]} .. {times[-1]}")
    gap = np.datetime64(iso(times[1])) - np.datetime64(iso(times[0]))
    print(f"analysis interval: {gap / np.timedelta64(1, 'h'):g} h")

    ds = open_store(times[0])
    print(f"\nstore: s3://{store_path(times[0])}")
    print(f"  dims        : {dict(ds.sizes)}")
    print(f"  valid time  : {ds.time.values[0]} .. {ds.time.values[-1]}")
    lead = (ds.time.values - ds.time.values[0]) / np.timedelta64(1, "h")
    print(f"  lead hours  : {lead[0]:g} .. {lead[-1]:g}, step {lead[1] - lead[0]:g}")
    print(f"  3-D vars    : {sorted(v for v in ds.data_vars if ds[v].ndim == 3)}")
    print(f"  one field   : {ds.t2m.isel(time=0).nbytes / 1e6:.1f} MB uncompressed"
          f" ({ds.t2m.dtype})")
    print(f"  chunks      : {ds.t2m.encoding.get('chunks')}")


if __name__ == "__main__":
    main()
