"""Probe the DMI Open Data metObs API before writing anything against it.

Checks reachability, then the collection listing, then one observation record, so
that the assumptions baked into ``build_observations.py`` -- endpoint shape, query
parameters, response format, parameter names and units -- are verified rather than
guessed.

Needs an API key from https://dmiapi.govcloud.dk/, in ``DMI_METOBS_API_KEY``.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request

# The old dmigw.govcloud.dk endpoint was retired on 2026-06-30 and no longer even
# resolves; DMI Open Data now serves from opendataapi.dmi.dk.
BASE = "https://opendataapi.dmi.dk/v2/metObs"
HOST = "opendataapi.dmi.dk"

#: metObs parameters matching the fields both models carry.
WANTED = {
    "temp_dry": "2 m dry-bulb air temperature",
    "pressure_at_sea": "pressure reduced to mean sea level",
    "wind_speed": "10 m mean wind speed",
}


def api_key() -> str | None:
    return os.environ.get("DMI_METOBS_API_KEY")


def get(path: str, **params) -> dict:
    """GET one metObs endpoint and parse the JSON response."""
    key = api_key()
    if key:
        params["api-key"] = key
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    shown = url.replace(key, "<key>") if key else url
    print(f"  GET {shown}")
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def diagnose() -> None:
    """Resolve a few hosts, to tell a blocked host from a wrong hostname."""
    hosts = [
        HOST,
        "dmigw.govcloud.dk",  # retired 2026-06-30
        "dmiapi.govcloud.dk",  # the key-issuing portal
        "s3.eu-central-1.amazonaws.com",  # known good: the DINI bucket
        "pypi.org",  # known good: uv resolves against it
    ]
    for h in hosts:
        try:
            print(f"  {h:<34} {socket.gethostbyname(h)}")
        except OSError as exc:
            print(f"  {h:<34} FAIL: {exc}")


def main() -> None:
    print(f"resolving {HOST} ...")
    try:
        print(f"  -> {socket.gethostbyname(HOST)}")
    except OSError as exc:
        print(f"  cannot resolve {HOST}: {exc}\n")
        print("resolving other hosts, to locate the problem:")
        diagnose()
        raise SystemExit(
            "\nIf the known-good hosts resolve and the DMI ones do not, the host is "
            "blocked here rather than wrong."
        )

    if not api_key():
        print("\nDMI_METOBS_API_KEY is not set; trying anonymously.")

    print("\ncollections")
    try:
        cols = get("collections")
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"  HTTP {exc.code}: {exc.read()[:300].decode(errors='replace')}"
        )
    for c in cols.get("collections", []):
        print(f"  {c.get('id'):<14} {c.get('title', '')}")

    print("\none station")
    st = get("collections/station/items", limit=1)
    feat = st["features"][0]
    print(f"  keys       : {sorted(feat)}")
    print(f"  properties : {sorted(feat['properties'])}")
    print(f"  geometry   : {feat.get('geometry')}")
    print(f"  example    : {json.dumps(feat['properties'], indent=2)[:600]}")

    print("\none observation per wanted parameter")
    for pid, desc in WANTED.items():
        try:
            obs = get("collections/observation/items", parameterId=pid, limit=1)
        except urllib.error.HTTPError as exc:
            print(f"  {pid:<18} HTTP {exc.code}")
            continue
        feats = obs.get("features") or []
        if not feats:
            print(f"  {pid:<18} no records returned")
            continue
        p = feats[0]["properties"]
        print(f"  {pid:<18} {desc}")
        print(
            f"    value={p.get('value')} station={p.get('stationId')} "
            f"observed={p.get('observed')}"
        )

    print("\nbbox + datetime query (the shape the extraction will use)")
    obs = get(
        "collections/observation/items",
        parameterId="temp_dry",
        bbox="7,54,16,58",
        datetime="2026-09-04T06:00:00Z/2026-09-04T06:10:00Z",
        limit=5,
    )
    print(f"  returned {len(obs.get('features', []))} features")
    if obs.get("features"):
        print(f"  {json.dumps(obs['features'][0], indent=2)[:500]}")


if __name__ == "__main__":
    main()
