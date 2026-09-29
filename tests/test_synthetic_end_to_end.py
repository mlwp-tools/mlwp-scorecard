"""End-to-end: synthetic reanalysis -> verification -> scorecard, in both formats.

These assert the *meaning* of the card, not just that files appear. The experiment
is persistence plus an accumulated random walk, so it is worse than persistence by
construction; a card that does not say so is wrong.
"""

from __future__ import annotations

import re

import pytest

from mlwp_scorecards import ScoreCard
from mlwp_scorecards.api import build_layout
from mlwp_scorecards.polarity import Polarity, polarity_of
from mlwp_scorecards.render.static import save_figure


def test_experiment_is_worse_everywhere(layout):
    """A random walk on top of persistence can only degrade an error metric."""
    bad = []
    for _, _, cell in layout.iter_cells():
        if polarity_of(cell.metric) is not Polarity.LOWER_IS_BETTER:
            continue
        for step in cell.steps:
            if step.value is not None and step.level > 0:
                bad.append((cell.cell_id, step.lead_time, step.level))
    assert (
        not bad
    ), f"drifting persistence scored better in {len(bad)} places: {bad[:5]}"


def test_degradation_grows_with_lead_time(layout):
    """The walk accumulates, so the gap widens with lead time."""
    cell = layout.sel(
        truth_source="analysis",
        variable="z",
        level=500.0,
        spatial_region="n.hem",
        metric="rmse",
    )
    diffs = [s.value for s in cell.steps]
    assert all(
        d > 0 for d in diffs
    ), "experiment should have larger error at every lead"
    assert diffs[-1] > diffs[0] * 1.5, f"gap should widen with lead time, got {diffs}"


def test_spread_uses_the_activity_family(layout):
    """Spread is neither better nor worse, so it gets its own colour family."""
    for _, _, cell in layout.iter_cells():
        want = "activity" if cell.metric == "spread" else "error"
        assert all(s.family == want for s in cell.steps), cell.cell_id


def test_case_count_falls_with_lead_time(layout):
    """Cases near the end of the record have no truth to verify against."""
    cell = layout.sel(
        truth_source="analysis",
        variable="z",
        level=500.0,
        spatial_region="n.hem",
        metric="rmse",
    )
    counts = [s.n for s in cell.steps]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] - counts[1] == 2, "12-hourly cases: 2 drop per 24 h step"


def test_surface_variables_produce_one_row_not_many(layout):
    """`msl` has no pressure level, so it is one row with a blank level cell."""
    msl = [r for r in layout.rows if r.key[1] == "msl"]
    assert len(msl) == 1, f"expected one msl row, got {[r.key for r in msl]}"
    assert msl[0].headers[-1].is_na
    assert msl[0].key[-1] is None, "not-applicable level is None, never NaN"
    assert msl[0].headers[-1].label == ""
    z = [r for r in layout.rows if r.key[1] == "z"]
    assert len(z) == 2, "z has two pressure levels"


def test_units_reach_the_cells(layout):
    """CF attributes on the score variable survive to the layout."""
    cell = layout.sel(
        truth_source="analysis",
        variable="t",
        level=500.0,
        spatial_region="n.hem",
        metric="rmse",
    )
    assert cell.units == "K"


def test_renders_html_and_static(verification, tmp_path):
    """Both backends produce output from the same card."""
    score_card = ScoreCard(
        verification,
        colour_relative_to="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
        title="t",
    )
    assert len(score_card.to_html()) > 2000
    fig = score_card.to_figure()
    for suffix in ("png", "pdf"):
        p = save_figure(fig, tmp_path / f"c.{suffix}")
        assert p.stat().st_size > 2000, p


def test_html_has_no_external_requests(verification):
    """No CDN, no analytics, no webfonts: the page must work offline."""
    text = ScoreCard(
        verification,
        colour_relative_to="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
    ).to_html()
    assert not re.search(r'(?:src|href)\s*=\s*["\']https?://', text)
    assert "googletagmanager" not in text


def test_every_column_header_level_is_on_the_page(verification):
    """With three column levels the middle one was dropped: only the outermost
    and leaf header rows were written."""
    score_card = ScoreCard(
        verification,
        colour_relative_to="persistence",
        rows=["variable", "level"],
        columns=["truth_source", "spatial_region", "metric"],
        n_resamples=50,
    )
    lay = score_card._layout
    thead = re.search(r"<thead>(.*?)</thead>", score_card.to_html(), re.S).group(1)
    rows = re.findall(r"<tr>(.*?)</tr>", thead, re.S)
    assert len(rows) == 3
    for depth, blocks in enumerate(lay.column_headers):
        labels = re.findall(r"<th[^>]*>([^<]*)</th>", rows[depth])
        want = [b.label for b in blocks]
        assert [x for x in labels if x] == want, (depth, labels)


def test_html_box_count_matches_the_layout(layout, verification):
    page = ScoreCard(
        verification,
        rows=layout.row_dims,
        columns=layout.column_dims,
        colour_relative_to="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
    ).to_html()
    assert page.count('<i class="b') == layout.stats.n_boxes


def test_unknown_metric_refuses_to_guess(verification, tmp_path):
    """Guessing polarity would produce a confidently backwards card."""
    ds = verification.rename(
        {
            n: str(n).replace("spread.", "wibble.")
            for n in verification.data_vars
            if str(n).startswith("spread.")
        }
    )
    with pytest.raises(KeyError, match="unknown metric"):
        build_layout(
            ds,
            colour_relative_to="persistence",
            select=dict(forecast_source=["drifting-persistence"]),
        )

    lay = build_layout(
        ds,
        colour_relative_to="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
        metric_polarity={"wibble": "lower_is_better"},
    )
    assert lay.stats.n_cells_present > 0
