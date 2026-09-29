"""What the drill-down costs on a full-size card.

The reference implementation carries this data as a raw JSON literal: 3.2 MB of
its 7.4 MB. This reports the equivalent here.

    uv run python tests/check_drilldown_scale.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_scale import _big_dataset  # noqa: E402

from mlwp_scorecard.api import build_layout  # noqa: E402
from mlwp_scorecard.render.colours import SCHEMES  # noqa: E402
from mlwp_scorecard.render.html import render_html  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp"
AXES = dict(
    rows=["truth_source", "variable", "level"], columns=["spatial_region", "metric"]
)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ds = _big_dataset()
    layout = build_layout(
        ds, baseline="ctl", select=dict(forecast_source=["exp"]), **AXES
    )
    s = layout.stats
    print(
        f"{s.n_rows} rows x {s.n_cols} columns, {s.n_boxes} boxes, "
        f"{s.n_cells_present} cells"
    )

    sizes = {}
    for detail in (False, True):
        p = OUT / f"big-detail-{int(detail)}.html"
        p.write_text(
            render_html(layout, scheme=SCHEMES["cvd"], detail=detail), encoding="utf-8"
        )
        sizes[detail] = p.stat().st_size

    page = (OUT / "big-detail-1.html").read_text()
    m = re.search(r'id="sc-data">([^<]+)</script>', page)
    payload = len(m.group(1))

    print(f"\n  table only     {sizes[False] / 1e6:>6.2f} MB")
    print(
        f"  with drilldown {sizes[True] / 1e6:>6.2f} MB  "
        f"(payload {payload / 1e6:.2f} MB)"
    )
    print(
        "\n  reference card of this shape: 7.40 MB, of which 3.20 MB is the "
        "same payload as a raw JSON literal"
    )
    print(
        f"  ratio: {7.40e6 / sizes[True]:.1f}x smaller overall, "
        f"{3.20e6 / payload:.1f}x smaller payload"
    )


if __name__ == "__main__":
    main()
