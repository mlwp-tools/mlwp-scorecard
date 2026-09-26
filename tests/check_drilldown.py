"""Render a card with the drill-down and report what went into the page.

Run directly:

    uv run python tests/check_drilldown.py
"""

from __future__ import annotations

import base64
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import make_verification_dataset  # noqa: E402

from mlwp_scorecards import build_layout, make_scorecard  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ds = make_verification_dataset(n_case=80, n_boot=200, drift=0.25)
    kwargs = dict(
        relative_to="persistence",
        predictions_from=["drifting-persistence"],
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
        title="drifting-persistence vs persistence",
        subtitle="synthetic reanalysis - click any cell for the full series",
    )
    layout = build_layout(ds, **kwargs)
    written = make_scorecard(
        ds, html_path=OUT / "card.html", image_path=OUT / "card.png", **kwargs
    )
    page = written[0].read_text()

    print(f"page {len(page) / 1024:.0f} kB")
    print(f"cells {layout.stats.n_cells_present}, boxes {layout.stats.n_boxes}")

    # the embedded payload must round-trip and line up with the table
    m = re.search(r'id="sc-data">([^<]+)</script>', page)
    if not m:
        raise SystemExit("no drill-down payload in the page")
    raw = gzip.decompress(base64.b64decode(m.group(1)))
    payload = json.loads(raw)
    print(
        f"payload {len(m.group(1)) / 1024:.0f} kB encoded, "
        f"{len(raw) / 1024:.0f} kB raw ({len(raw) / len(m.group(1)):.1f}x)"
    )

    idx = sorted(int(i) for i in re.findall(r'data-i="(\d+)"', page))
    assert idx == list(
        range(len(payload["cells"]))
    ), "data-i does not index the payload"
    print(f"every one of {len(idx)} data-i values resolves")

    first = payload["cells"][0]
    print(f"\nfirst cell: {first['t']}  [{first['m']}, {first['u']}]")
    print(f"  lead     {payload['labels']}")
    for key, label in (("c", "baseline"), ("e", "forecast"), ("d", "difference")):
        if key in first:
            vals = [f"{v:.4g}" if v is not None else "--" for v in first[key]]
            print(f"  {label:<10} {vals}")
    for key, label in (
        ("cl", "baseline lo"),
        ("cu", "baseline hi"),
        ("el", "forecast lo"),
        ("eu", "forecast hi"),
    ):
        if key in first:
            print(f"  {label:<13} present")
    print(f"  cases      {first.get('n')}")

    assert not re.search(r'(?:src|href)\s*=\s*["\']https?://', page), "external request"
    print("\nno external requests")
    for needed in ("<dialog", "DecompressionStream", "showModal"):
        assert needed in page, needed
    print("dialog, lazy inflate and modal open are all present")

    check_javascript(page)
    print(f"\nopen {written[0]}")


def check_javascript(page: str) -> None:
    """Syntax-check the emitted JavaScript, if node is available.

    The chart code cannot be exercised from here, so a syntax error would
    otherwise only show up as a silently dead dialog in the browser.
    """
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    if not scripts:
        raise SystemExit("no inline script emitted")
    if not node:
        print(f"\nnode not found; {len(scripts)} inline script(s) not syntax-checked")
        return

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write("\n".join(scripts))
        tmp = fh.name
    result = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"emitted JavaScript is invalid:\n{result.stderr}")
    lines = sum(s.count("\n") for s in scripts)
    print(f"\nemitted JavaScript parses ({lines} lines, node --check)")


if __name__ == "__main__":
    main()
