"""Printed values, the baseline's grey row, and the card with no baseline.

`baseline=` decides the colouring and nothing else; `show_values=True`
prints each source's own score in its boxes. With a baseline the colours must be
exactly what they are without values, plus a grey row of the baseline's own
scores; with none, every box is a number on grey and nothing is significant.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from mlwp_scorecards import ScoreCard
from mlwp_scorecards.api import build_layout
from mlwp_scorecards.layout import NEUTRAL, format_value
from mlwp_scorecards.render.static import save_figure

sys.path.insert(0, str(Path(__file__).parent))
from test_sources import _dataset, _with_gaps  # noqa: E402

#: The hand-built data here is too short for the default blocks, so the
#: resample is chosen explicitly.
KW = dict(n_resamples=200, seed=0, columns=["metric"], bootstrap="iid")
ROWS = ["forecast_source", "variable"]


def _card(ds=None, **kw):
    return build_layout(_dataset() if ds is None else ds, rows=ROWS, **{**KW, **kw})


def _mean(ds, source, var="2t"):
    return ds[f"rmse.{var}"].sel(forecast_source=source).mean("init_time").values


# --------------------------------------------------------------------------- #
# the one formatting helper
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value, text",
    [
        (433.2, "433"),
        (21.37, "21.4"),
        (2.071, "2.07"),
        (0.617, "0.62"),
        (0.4, "0.40"),
        (0.00123, "0.0012"),
        (-12.34, "-12.3"),
        (0.0, "0"),
    ],
)
def test_format_value(value, text):
    assert format_value(value) == text


# --------------------------------------------------------------------------- #
# with a baseline: colours unchanged, plus a grey row of the baseline
# --------------------------------------------------------------------------- #
def test_showing_values_leaves_the_colours_exactly_as_they_were():
    plain = _card(baseline="base")
    shown = _card(baseline="base", show_values=True)
    for _, _, cell in plain.iter_cells():
        other = shown.cells[(cell.row_key, cell.col_key)]
        assert not other.is_baseline
        for a, b in zip(cell.steps, other.steps):
            assert (a.level, a.family, a.significant_at) == (
                b.level,
                b.family,
                b.significant_at,
            )
            assert a.text == "" and b.text == format_value(b.forecast)


def test_the_baseline_is_a_grey_row_of_its_own_scores_first():
    ds = _dataset()
    lay = _card(ds, baseline="base", show_values=True)
    assert lay.rows[0].key[0] == "base"
    base = lay.sel(forecast_source="base", variable="2t", metric="rmse")
    assert base.is_baseline
    assert all(s.family == NEUTRAL and s.level == 0 for s in base.steps)
    assert all(s.significant_at is None and s.value is None for s in base.steps)
    assert np.allclose([s.forecast for s in base.steps], _mean(ds, "base"))
    assert [s.text for s in base.steps] == [
        format_value(s.forecast) for s in base.steps
    ]


def test_one_forecast_source_still_gets_a_baseline_row():
    lay = build_layout(
        _dataset(),
        baseline="base",
        show_values=True,
        select=dict(forecast_source=["a"]),
        **{k: v for k, v in KW.items() if k != "columns"},
    )
    assert lay.row_dims[0] == "forecast_source"
    order = list(dict.fromkeys(r.key[0] for r in lay.rows))
    assert order == ["base", "a"]


def test_values_need_forecast_source_on_an_axis():
    with pytest.raises(ValueError, match="show_values=True"):
        build_layout(
            _dataset(),
            baseline="base",
            show_values=True,
            select=dict(forecast_source=["a"]),
            rows=["variable"],
            columns=["metric"],
            bootstrap="iid",
        )


def test_under_pairwise_cases_the_baseline_row_uses_all_its_own_cases():
    ds = _with_gaps()  # `a` scores one case in three
    common = _card(ds, baseline="base", show_values=True, cases="common")
    pairwise = _card(ds, baseline="base", show_values=True, cases="pairwise")

    def n(lay):
        cell = lay.sel(forecast_source="base", variable="2t", metric="rmse")
        return {s.n for s in cell.steps}

    assert n(common) == {20}
    assert n(pairwise) == {60}
    assert any("all of its own forecast cases" in note for note in pairwise.notes)


# --------------------------------------------------------------------------- #
# no baseline
# --------------------------------------------------------------------------- #
def test_with_no_baseline_every_box_is_its_own_score_on_grey():
    ds = _dataset()
    lay = _card(ds, show_values=True)
    assert not lay.coloured
    assert lay.forecast_sources == ("base", "a", "b", "c")
    assert lay.stats.n_significant == 0
    for _, _, cell in lay.iter_cells():
        assert not cell.is_baseline
        assert all(s.family == NEUTRAL and s.level == 0 for s in cell.steps)
    b = lay.sel(forecast_source="b", variable="2t", metric="rmse")
    assert np.allclose([s.forecast for s in b.steps], _mean(ds, "b"))
    assert any("No baseline" in note for note in lay.notes)


@pytest.mark.parametrize(
    "cases, want_a, want_b", [("common", 20, 20), ("pairwise", 20, 60)]
)
def test_with_no_baseline_the_case_policy_still_applies(cases, want_a, want_b):
    lay = _card(_with_gaps(), show_values=True, cases=cases)

    def n(source):
        cell = lay.sel(forecast_source=source, variable="2t", metric="rmse")
        return {s.n for s in cell.steps}

    assert n("a") == {want_a} and n("b") == {want_b}


def test_neither_a_baseline_nor_values_is_refused():
    with pytest.raises(ValueError, match="nothing to show"):
        _card(show_values=False)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("baseline", ["base", None])
def test_both_backends_render_the_numbers(tmp_path, baseline):
    score_card = ScoreCard(
        _dataset(), baseline=baseline, show_values=True, rows=ROWS, **KW
    )
    page = score_card.to_html()
    png = save_figure(score_card.to_figure(), tmp_path / "c.png")
    lay = _card(baseline=baseline, show_values=True)
    first = lay.sel(forecast_source="a", variable="2t", metric="rmse").steps[0].text
    assert re.search(rf'<i class="b [^"]*" title="[^"]*">{re.escape(first)}</i>', page)
    assert png.stat().st_size > 2000
    if baseline:
        assert 'class="base"' in page
    else:
        assert "There is no" in page and "error metrics" not in page


def test_a_card_with_values_is_byte_reproducible():
    kw = dict(baseline="base", show_values=True, rows=ROWS, **KW)
    assert ScoreCard(_dataset(), **kw).to_html() == (
        ScoreCard(_dataset(), **kw).to_html()
    )
