"""Summarise the saved /api/scorecard responses: row schema, which fields exist (is
there any case count or uncertainty?), coverage per model, and a reconstruction of
one scorecard row with the page's own colour rule (module 6717).

Run with:  python analyse_api.py
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

API = Path(__file__).parent / "source" / "api"
LEADS = (24, 72, 120, 168, 240, 360)  # module 2168, `W`


def relative(value: float, base: float, higher_better: bool) -> float:
    """Module 6717 `c`: (base - value)/|base|, sign flipped for higher-is-better.

    Positive means better than the baseline.
    """
    r = (base - value) / abs(base)
    return -r if higher_better else r


def colour_position(r: float) -> float:
    """Module 6717 `s`: position on the diverging ramp, -1..1, log in |%| from 1 % to 50 %."""
    pct = 100 * abs(r)
    return math.copysign(
        0.0 if pct <= 1 else min(1.0, math.log10(pct) / math.log10(50)), r
    )


def main() -> None:
    body = json.loads((API / "scorecard_det_global.json").read_text())
    rows = body["rows"]
    print("top-level keys:", sorted(body))
    print("row keys:", sorted(rows[0]))
    print("example row:", rows[0])
    print("field value counts:")
    for key in sorted(rows[0]):
        if key != "value":
            print(f"  {key}: {dict(Counter(r[key] for r in rows))}")

    for mode in ("det", "prob"):
        rows = json.loads((API / f"scorecard_{mode}_global.json").read_text())["rows"]
        cover = defaultdict(set)
        for r in rows:
            cover[r["model"]].add((r["variable"], r["lead_time_h"]))
        print(f"\n{mode}: (variable, lead) cells per model")
        for model, cells in sorted(cover.items()):
            print(f"  {model:24s} {len(cells)}")

    # Rebuild the T850 row of the default card: HRES baseline, RMSE, global.
    rows = json.loads((API / "scorecard_det_global.json").read_text())["rows"]
    val = {(r["model"], r["variable"], r["lead_time_h"]): r["value"] for r in rows}
    var = "temperature_850"
    print(f"\n{var}, det, global, baseline hres: value / % better / ramp position")
    for model in sorted({m for m, v, _ in val if v == var}):
        cells = []
        for lead in LEADS:
            v, b = val.get((model, var, lead)), val.get(("hres", var, lead))
            if v is None or b is None:
                cells.append(f"{'—':>22s}")
                continue
            r = relative(v, b, higher_better=False)
            cells.append(f"{v:7.3f} {100 * r:+6.1f}% {colour_position(r):+.2f}")
        print(f"  {model:22s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
