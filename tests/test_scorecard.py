"""The public object: build a card, then take it as a figure or a page."""

from __future__ import annotations

import warnings

import pytest
from test_schema import _dataset, _score

from mlwp_scorecards import ScoreCard

KW = dict(
    colour_relative_to="persistence",
    select=dict(forecast_source=["drifting-persistence"]),
    title="t",
    n_resamples=100,
)


@pytest.fixture(scope="module")
def score_card(verification):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ScoreCard(verification, **KW)


def test_to_figure_is_a_figure_pyplot_does_not_know_about(score_card):
    """Repeated builds must not pile up in pyplot's figure list, and an
    interactive session must keep its backend."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    backend = matplotlib.get_backend()
    before = plt.get_fignums()
    fig = score_card.to_figure()
    assert isinstance(fig, Figure)
    assert plt.get_fignums() == before
    assert matplotlib.get_backend() == backend


def test_to_figure_saves_with_the_callers_own_savefig(score_card, tmp_path):
    p = tmp_path / "c.png"
    score_card.to_figure().savefig(p)
    assert p.stat().st_size > 2000


def test_to_html_is_the_page(score_card):
    page = score_card.to_html()
    assert page.startswith("<!DOCTYPE html>")
    assert page == score_card.to_html()


def test_repr_names_the_sources(score_card):
    r = repr(score_card)
    assert "drifting-persistence" in r and "persistence" in r


def test_validation_warnings_are_issued_not_dropped():
    """Before, a caller saw these only by asking for the report."""
    ds = _dataset(**{"rmse.2t": _score(2.0, units="K")}).mean(
        "init_time", keep_attrs=True
    )
    with pytest.warns(UserWarning, match="already-collapsed means"):
        ScoreCard(
            ds,
            colour_relative_to="ctl",
            select=dict(forecast_source=["exp"]),
            rows=["truth_source", "variable"],
            columns=["metric"],
        )
