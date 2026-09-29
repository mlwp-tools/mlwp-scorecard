"""Metric polarity: an explicit table, a refusal to guess, and the words for it."""

from __future__ import annotations

import pytest

from mlwp_scorecard.polarity import FAMILY_WORDS, Polarity, family_of, polarity_of, word
from mlwp_scorecard.render.colours import SCHEMES


def test_polarity_lookup_and_refusal():
    assert polarity_of("rmse") is Polarity.LOWER_IS_BETTER
    assert polarity_of("acc") is Polarity.HIGHER_IS_BETTER
    assert polarity_of("spread") is Polarity.ACTIVITY
    with pytest.raises(KeyError, match="unknown metric"):
        polarity_of("not_a_metric")
    assert (
        polarity_of("not_a_metric", {"not_a_metric": "higher_is_better"})
        is Polarity.HIGHER_IS_BETTER
    )


def test_words_follow_the_sign_of_the_level():
    assert word("error", 3) == "better" and word("error", -3) == "worse"
    assert word("activity", 1) == "more active"
    assert word("activity", -1) == "less active"
    assert word("error", 0) == "no change"


def test_every_family_has_words_and_a_ramp_in_every_palette():
    families = {family_of(p) for p in Polarity}
    assert families == set(FAMILY_WORDS)
    for scheme in SCHEMES.values():
        assert set(scheme.families) == families
