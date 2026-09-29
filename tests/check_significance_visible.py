"""Is the significance channel actually reaching the renderers, and is it legible?

Fill encodes magnitude and border encodes significance, so the border has to be
distinguishable from the fill it sits on. A dark border on a dark fill carries no
information even when it is drawn correctly.

    uv run python tests/check_significance_visible.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import make_verification_dataset  # noqa: E402

from mlwp_scorecards.api import build_layout  # noqa: E402
from mlwp_scorecards.colours import SCHEMES, contrast_ratio  # noqa: E402
from mlwp_scorecards.render.html import render_html  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp"


def main() -> None:
    scheme = SCHEMES["cvd"]

    print("border vs fill contrast, per ramp step")
    print(f"  {'family':<10} {'level':>5} {'fill':>9} {'border':>9} {'ratio':>7}")
    worst = (99.0, None)
    for family in scheme.families:
        for level in (1, 4, 7, 10, 14):
            sw = scheme.swatch(family, level)
            r = contrast_ratio(sw.fill, sw.edge)
            if r < worst[0]:
                worst = (r, (family, level))
            print(f"  {family:<10} {level:>5} {sw.fill:>9} {sw.edge:>9} {r:>6.2f}")
    print(f"\n  worst: {worst[0]:.2f} at {worst[1]}")
    if worst[0] < 1.6:
        print(
            "  -> too low: a significant cell will not look different from an "
            "insignificant one at that end of the ramp"
        )

    # does the marking survive into the page?
    # 80 12-hourly cases: the fewest the default 10-day blocks (20 cases) accept.
    ds = make_verification_dataset(n_case=80, n_boot=200, drift=0.25)
    layout = build_layout(
        ds,
        colour_relative_to="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
    )
    s = layout.stats
    print(f"\nsynthetic card: {s.n_significant}/{s.n_boxes} boxes significant")
    if s.n_significant == 0:
        print("  (no paired difference input, so nothing can be marked -- expected)")

    page = render_html(layout, scheme=scheme)
    (OUT / "sig.html").write_text(page, encoding="utf-8")
    marked = len(re.findall(r'class="b [pnz]\d* sig"', page))
    print(f"  page carries {marked} boxes with the sig class")


if __name__ == "__main__":
    main()
