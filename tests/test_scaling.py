"""Scaling a relative difference onto the layout's signed levels."""

from __future__ import annotations

import pytest

from mlwp_scorecard.layout import LEVELS
from mlwp_scorecard.layout.scaling import FixedScaling


def test_scaling_is_symmetric_and_monotone():
    s = FixedScaling()
    assert s.level(0.0) == 0
    for v in (0.006, 0.03, 0.2, 0.9):
        assert s.level(v) == -s.level(-v)
    levels = [s.level(v) for v in (0.006, 0.03, 0.2, 0.9)]
    assert levels == sorted(levels)


def test_scaling_saturates_rather_than_wrapping():
    s = FixedScaling()
    assert s.level(5.0) == s.level(1.0) == LEVELS
    assert s.saturated(5.0) and not s.saturated(0.1)


def test_breaks_that_do_not_match_the_level_range_are_refused():
    with pytest.raises(ValueError, match=f"needs {LEVELS} breaks"):
        FixedScaling(breaks=(0.1, 0.2))
