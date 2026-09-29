"""The click-through drill-down: payload, wiring, and that it actually draws.

The charts are generated SVG rather than a plotting library, so nothing else
verifies them. ``test_drilldown_actually_draws`` runs the page's own JavaScript
under a minimal DOM shim in node, which is the difference between "it parses" and
"it draws"; it skips when node is unavailable.
"""

from __future__ import annotations

import base64
import gzip
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from mlwp_scorecards import ScoreCard
from mlwp_scorecards.render.payload import build_payload, pack

HARNESS = Path(__file__).parent / "drilldown_harness.mjs"


@pytest.fixture(scope="module")
def page(verification) -> str:
    return ScoreCard(
        verification,
        baseline="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
        title="t",
    ).to_html()


def _payload(page: str) -> dict:
    m = re.search(r'id="sc-data">([^<]+)</script>', page)
    assert m, "no drill-down payload in the page"
    return json.loads(gzip.decompress(base64.b64decode(m.group(1))))


def test_every_cell_carries_an_index_that_resolves(page, layout):
    """Cell identity is an integer, and it must index the payload exactly.

    The reference builds ids by concatenating labels, which breaks on any label
    containing the separator and leaves the click silently doing nothing.
    """
    payload = _payload(page)
    idx = sorted(int(i) for i in re.findall(r'data-i="(\d+)"', page))
    assert idx == list(range(len(payload["cells"])))
    assert len(idx) == layout.stats.n_cells_present


def test_payload_order_matches_iter_cells(layout):
    """The table and the payload must enumerate cells in the same order."""
    payload = build_payload(layout)
    for entry, (_, _, cell) in zip(payload["cells"], layout.iter_cells()):
        assert entry["m"] == cell.metric
        assert entry["u"] == (cell.units or "")


def test_payload_carries_both_sources_with_intervals(layout):
    payload = build_payload(layout)
    first = payload["cells"][0]
    for key in ("c", "cl", "cu", "e", "el", "eu", "d"):
        assert key in first, f"missing {key}"
        assert len(first[key]) == len(layout.lead_times)


def test_payload_is_rounded_and_compresses(layout):
    """Rounding is what makes the text compress; it is not cosmetic."""
    payload = build_payload(layout, precision=4)
    raw = json.dumps(payload, separators=(",", ":")).encode()
    packed = pack(payload)
    assert len(packed) < len(raw), "compression made it bigger"
    for v in payload["cells"][0]["c"]:
        if v is not None:
            assert len(f"{v!r}".replace("-", "").replace(".", "").lstrip("0")) <= 6


def test_payload_is_byte_reproducible(layout):
    assert pack(build_payload(layout)) == pack(build_payload(layout))


def test_payload_refuses_to_emit_bare_nan(layout):
    """`NaN` is not valid JSON and would break JSON.parse in the browser."""
    raw = gzip.decompress(base64.b64decode(pack(build_payload(layout))))
    assert b"NaN" not in raw
    json.loads(raw)  # and it must actually parse


def test_detail_can_be_switched_off(verification):
    """On a full-size card this is the largest thing in the file."""
    score_card = ScoreCard(
        verification,
        baseline="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
    )
    with_, without = score_card.to_html(), score_card.to_html(detail=False)
    # the payload *element*, not the getElementById call that looks for it
    element = re.compile(r'<script[^>]*id="sc-data"')
    assert element.search(with_)
    assert not element.search(without)
    assert "<dialog" not in without
    assert len(without) < len(with_)


def test_page_still_makes_no_external_requests(page):
    """The reference pulls 2.7 MB of Plotly over plain http from an unpinned CDN."""
    assert not re.search(r'(?:src|href)\s*=\s*["\']https?://', page)
    assert "plotly" not in page.lower()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_emitted_javascript_parses(page, tmp_path):
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    js = tmp_path / "page.js"
    js.write_text("\n".join(scripts))
    r = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_drilldown_actually_draws(verification, tmp_path):
    """Run the page's own JavaScript: click a cell, expect two charts.

    Syntax-checking cannot catch a chart that silently draws nothing, and there is
    no other coverage of the SVG generation.
    """
    out = tmp_path / "card.html"
    score_card = ScoreCard(
        verification,
        baseline="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
        title="t",
    )
    out.write_text(score_card.to_html(), encoding="utf-8")
    r = subprocess.run(["node", str(HARNESS), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "dialog opened      : true" in r.stdout
    figures = int(re.search(r"figures\s+:\s+(\d+)", r.stdout).group(1))
    svgs = int(re.search(r"svg nodes drawn\s+:\s+(\d+)", r.stdout).group(1))
    assert figures == 2, r.stdout
    assert svgs > 20, f"only {svgs} svg nodes drawn"
