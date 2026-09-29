"""The public object: build a card, then take it as a figure or a page."""

from __future__ import annotations

import warnings

import pytest

from mlwp_scorecards import ScoreCard

KW = dict(
    colour_relative_to="persistence",
    select=dict(forecast_source=["drifting-persistence"]),
    title="t",
    n_resamples=100,
)


@pytest.fixture(scope="module")
def score_card(verification):
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


def test_the_palette_is_chosen_when_drawing_not_when_building(score_card):
    """The layout carries no colours, so one card draws in any palette."""
    from mlwp_scorecards.render.colours import ECMWF

    cvd, ecmwf = score_card.to_html(), score_card.to_html(colour_scheme="ecmwf")
    assert cvd != ecmwf
    assert ECMWF.families["error"].positive[-1].fill in ecmwf
    assert score_card.to_html(colour_scheme=ECMWF) == ecmwf
    with pytest.raises(KeyError, match="unknown colour scheme 'nonsuch'"):
        score_card.to_figure(colour_scheme="nonsuch")


def test_repr_names_the_sources(score_card):
    r = repr(score_card)
    assert "drifting-persistence" in r and "persistence" in r


def test_building_a_card_issues_no_warnings(verification):
    """Nothing is quietly relaxed: what the card cannot honestly show raises,
    and every choice made is printed on the card, so there is nothing left to
    warn about."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ScoreCard(verification, **KW)


def test_too_few_cases_for_blocks_is_refused_naming_the_choice(verification):
    """Falling back to an iid resample used to be a warning; it marks far too
    much as significant, so it is now the caller's choice to make."""
    short = verification.isel(init_time=slice(0, 30))
    with pytest.raises(ValueError, match="(?s)30 forecast cases.*bootstrap='iid'"):
        ScoreCard(short, **KW)
    assert "iid" in ScoreCard(short, bootstrap="iid", **KW)._layout.resampling
