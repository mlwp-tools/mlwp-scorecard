"""Fetch /api/scorecard responses for the window the linked page shows, in both modes,
for every region the page offers. Saves source/api/scorecard_<mode>_<region>.json.

The page sends `end` as the chosen end date + 1 day (module 9203), so a
2026-08-09..2026-09-08 window is requested as start=2026-08-09&end=2026-09-09.

Run with:  python fetch_api.py
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://owb.brightband.com/api/scorecard?"
OUT = Path(__file__).parent / "source" / "api"
START, END = "2026-08-09", "2026-09-09"
REGIONS = (
    "global",
    "tropics",
    "northern-hemisphere",
    "southern-hemisphere",
    "north-america",
    "europe",
)
MODES = ("det", "prob")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        for region in REGIONS:
            q = urllib.parse.urlencode(
                {"region": region, "mode": mode, "start": START, "end": END}
            )
            with urllib.request.urlopen(BASE + q, timeout=60) as r:
                body = json.load(r)
            path = OUT / f"scorecard_{mode}_{region}.json"
            path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
            print(f"{path.name}: {len(body.get('rows', []))} rows")


if __name__ == "__main__":
    main()
