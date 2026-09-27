"""Several forecast sources against one baseline.

A card with N forecast sources must be exactly N two-source cards stacked on a
``forecast_source`` axis: the same means, the same paired intervals, the same
significance. That is asserted directly, because it is the property that makes
the rows comparable at all. The rest pins the rules around it -- the baseline is
never also a row, and which cases each row rests on is a choice the card states.
"""

from __future__ import annotations

import base64
import gzip
import json
import re

import numpy as np
import pytest
import xarray as xr

from mlwp_scorecards import build_layout, make_scorecard

SOURCES = ("base", "a", "b", "c")


def _dataset(n_case=60, n_lead=4, seed=0):
    """Four sources scored on the same cases, sharing most of their error."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 1.0, (n_case, n_lead))
    offsets = {"base": 0.0, "a": 0.2, "b": -0.15, "c": 0.02}
    scores = np.stack(
        [
            2.0 + 0.9 * common + offsets[s] + 0.1 * rng.normal(0, 1, (n_case, n_lead))
            for s in SOURCES
        ]
    )
    lead = np.arange(1, n_lead + 1).astype("timedelta64[D]")
    init = np.datetime64("2024-01-01") + np.arange(n_case) * np.timedelta64(12, "h")
    coords = dict(forecast_source=list(SOURCES), init_time=init, lead_time=lead)
    dims = ["forecast_source", "init_time", "lead_time"]
    return xr.Dataset(
        {
            "rmse.2t": xr.DataArray(
                scores, dims=dims, coords=coords, attrs=dict(units="K")
            ),
            "rmse.msl": xr.DataArray(
                scores * 50, dims=dims, coords=coords, attrs=dict(units="Pa")
            ),
        }
    )


KW = dict(colour_relative_to="base", n_resamples=300, seed=0)


def _card(ds, sources, **kw):
    return build_layout(
        ds,
        select=dict(forecast_source=sources),
        rows=["forecast_source", "variable"],
        columns=["metric"],
        **{**KW, **kw},
    )


# --------------------------------------------------------------------------- #
# the property the whole feature rests on
# --------------------------------------------------------------------------- #
def test_each_row_is_exactly_the_two_source_card_for_that_source():
    """One resample shared by every pair, so stacking changes nothing."""
    ds = _dataset()
    multi = _card(ds, ["a", "b", "c"])
    for source in ("a", "b", "c"):
        single = build_layout(
            ds,
            select=dict(forecast_source=[source]),
            rows=["variable"],
            columns=["metric"],
            **KW,
        )
        for var in ("2t", "msl"):
            got = multi.sel(forecast_source=source, variable=var, metric="rmse")
            want = single.sel(variable=var, metric="rmse")
            assert got.forecast_source == source
            for g, w in zip(got.steps, want.steps):
                assert g.value == w.value
                assert (g.value_lower, g.value_upper) == (w.value_lower, w.value_upper)
                assert g.significant_at == w.significant_at
                assert g.n == w.n


def test_rows_follow_the_order_the_sources_were_given_in():
    lay = _card(_dataset(), ["c", "a", "b"])
    order = []
    for r in lay.rows:
        if r.key[0] not in order:
            order.append(r.key[0])
    assert order == ["c", "a", "b"]


def test_several_sources_are_placed_outermost_on_the_rows_by_default():
    lay = build_layout(_dataset(), select=dict(forecast_source=["a", "b"]), **KW)
    assert lay.row_dims[0] == "forecast_source"
    assert lay.forecast_sources == ("a", "b")
    assert lay.baseline_source == "base"


def test_one_source_is_consumed_as_before_not_laid_out():
    lay = build_layout(_dataset(), select=dict(forecast_source=["a"]), **KW)
    assert "forecast_source" not in lay.row_dims + lay.column_dims
    assert all(cell.forecast_source == "a" for _, _, cell in lay.iter_cells())


def test_sources_may_go_on_the_columns():
    lay = build_layout(
        _dataset(),
        select=dict(forecast_source=["a", "b"]),
        rows=["variable"],
        columns=["forecast_source", "metric"],
        **KW,
    )
    assert [c.key for c in lay.columns] == [("a", "rmse"), ("b", "rmse")]


# --------------------------------------------------------------------------- #
# what is refused
# --------------------------------------------------------------------------- #
def test_the_baseline_cannot_also_be_a_forecast_source():
    with pytest.raises(ValueError, match="also named in"):
        _card(_dataset(), ["a", "base"])


def test_a_repeated_forecast_source_is_refused():
    with pytest.raises(ValueError, match="repeats"):
        _card(_dataset(), ["a", "b", "a"])


def test_an_unknown_source_is_refused_naming_the_argument():
    with pytest.raises(KeyError, match="'nonsuch' is not in forecast_source"):
        _card(_dataset(), ["a", "nonsuch"])
    with pytest.raises(KeyError, match="colour_relative_to='nonsuch'"):
        _card(_dataset(), ["a"], colour_relative_to="nonsuch")


def test_no_baseline_and_no_values_leaves_nothing_to_show():
    """Neutral boxes with nothing in them would be an empty card: say so, and list
    the sources that could be the baseline."""
    with pytest.raises(ValueError, match="nothing to show") as excinfo:
        _card(_dataset(), ["a"], colour_relative_to=None)
    assert "one of: base, a, b, c" in str(excinfo.value), "should list the options"


# --------------------------------------------------------------------------- #
# the ellipsis
# --------------------------------------------------------------------------- #
def _sources(lay):
    return list(lay.forecast_sources)


def test_the_default_is_every_source_but_the_baseline_in_dataset_order():
    lay = build_layout(_dataset(), **KW)
    assert _sources(lay) == ["a", "b", "c"]


def test_an_ellipsis_fills_in_the_rest_after_the_named_ones():
    assert _sources(_card(_dataset(), ["c", ...])) == ["c", "a", "b"]


def test_an_ellipsis_may_sit_anywhere():
    assert _sources(_card(_dataset(), ["b", ..., "a"])) == ["b", "c", "a"]


def test_an_ellipsis_never_includes_the_baseline():
    ds = _dataset()
    assert _sources(_card(ds, [...], colour_relative_to="b")) == ["base", "a", "c"]


def test_two_ellipses_are_refused():
    with pytest.raises(ValueError, match="at most once"):
        _card(_dataset(), [..., "a", ...])


def test_an_ellipsis_with_nothing_left_to_fill_is_harmless():
    assert _sources(_card(_dataset(), ["a", "b", "c", ...])) == ["a", "b", "c"]


def test_a_card_with_no_forecast_source_is_refused():
    only_base = _dataset().sel(forecast_source=["base"])
    with pytest.raises(ValueError, match="no forecast source"):
        _card(only_base, [...])


def test_several_sources_with_no_axis_for_them_is_refused():
    """Otherwise the card would have to average over them, or pick one."""
    with pytest.raises(ValueError, match="on neither rows nor columns"):
        build_layout(
            _dataset(),
            select=dict(forecast_source=["a", "b"]),
            rows=["variable"],
            columns=["metric"],
            **KW,
        )


def test_an_unknown_case_policy_is_refused():
    with pytest.raises(ValueError, match="cases must be one of"):
        _card(_dataset(), ["a", "b"], cases="everything")


# --------------------------------------------------------------------------- #
# which cases each row rests on
# --------------------------------------------------------------------------- #
def _with_gaps():
    """Source `a` runs one case in three; `b` has every case."""
    ds = _dataset(n_case=60)
    for name in ds.data_vars:
        values = ds[name].values
        values[SOURCES.index("a"), 1::3, :] = np.nan
        values[SOURCES.index("a"), 2::3, :] = np.nan
    return ds


def _counts(lay):
    return {
        src: {
            s.n
            for _, _, cell in lay.iter_cells()
            if cell.forecast_source == src
            for s in cell.steps
        }
        for src in ("a", "b")
    }


def test_common_cases_are_the_ones_every_source_scored():
    counts = _counts(_card(_with_gaps(), ["a", "b"], cases="common"))
    assert counts == {"a": {20}, "b": {20}}


def test_pairwise_cases_are_the_ones_each_shares_with_the_baseline():
    counts = _counts(_card(_with_gaps(), ["a", "b"], cases="pairwise"))
    assert counts == {"a": {20}, "b": {60}}


def test_under_common_cases_every_row_sees_the_same_baseline():
    lay = _card(_with_gaps(), ["a", "b"], cases="common")
    for var in ("2t", "msl"):
        a = lay.sel(forecast_source="a", variable=var, metric="rmse")
        b = lay.sel(forecast_source="b", variable=var, metric="rmse")
        assert [s.baseline for s in a.steps] == [s.baseline for s in b.steps]


def test_the_case_policy_is_stated_on_a_card_with_several_sources():
    common = _card(_with_gaps(), ["a", "b"], cases="common")
    pairwise = _card(_with_gaps(), ["a", "b"], cases="pairwise")
    assert any("same forecast cases" in n for n in common.notes)
    assert any("should not be ranked" in n for n in pairwise.notes)
    single = build_layout(_with_gaps(), select=dict(forecast_source=["b"]), **KW)
    assert not any("forecast cases: those all" in n for n in single.notes)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _payload(page: str) -> dict:
    m = re.search(r'id="sc-data">([^<]+)</script>', page)
    assert m, "no drill-down payload in the page"
    return json.loads(gzip.decompress(base64.b64decode(m.group(1))))


def test_renders_with_each_cell_naming_its_source(tmp_path):
    html, png = make_scorecard(
        _dataset(),
        html_path=tmp_path / "c.html",
        image_path=tmp_path / "c.png",
        select=dict(forecast_source=["a", "b"]),
        **KW,
    )
    assert png.stat().st_size > 2000
    page = html.read_text()
    payload = _payload(page)
    assert payload["control"] == "base"
    assert {cell["s"] for cell in payload["cells"]} == {"a", "b"}
    assert "each forecast source" in page


def test_a_single_source_payload_does_not_repeat_the_source_per_cell(tmp_path):
    (html,) = make_scorecard(
        _dataset(),
        html_path=tmp_path / "c.html",
        select=dict(forecast_source=["a"]),
        **KW,
    )
    assert all("s" not in cell for cell in _payload(html.read_text())["cells"])
