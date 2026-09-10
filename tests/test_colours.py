"""Colour semantics, contrast, and the refusal to guess polarity."""

from __future__ import annotations

import pytest

from mlwp_scorecards.colours import (
    SCHEMES,
    FixedScaling,
    Polarity,
    contrast_ratio,
    polarity_of,
)


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_every_swatch_has_legible_foreground(name):
    """A glyph drawn on any fill must stay readable (WCAG AA)."""
    scheme = SCHEMES[name]
    for family in scheme.families:
        for level in range(1, scheme.depth + 1):
            for signed in (level, -level):
                sw = scheme.swatch(family, signed)
                assert contrast_ratio(sw.fill, sw.fg) >= 4.5, (name, family, signed)


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_significant_is_distinguishable_from_not_at_every_ramp_step(name):
    """The border must contrast with *white*, which is the insignificant state.

    Contrast against the fill is the wrong thing to check, and checking it leads
    directly to a broken card: a light border on a dark fill scores well against
    the fill, but an insignificant box already has a white border, so both then
    show a pale ring and significance becomes invisible on exactly the darkest,
    most interesting cells. It also leaves the legend swatches looking borderless
    on a white page.
    """
    scheme = SCHEMES[name]
    for family in scheme.families:
        for level in range(1, scheme.depth + 1):
            for signed in (level, -level):
                sw = scheme.swatch(family, signed)
                r = contrast_ratio(sw.edge, scheme.insignificant_edge)
                assert r >= 3.0, (
                    f"{name}/{family}/{signed}: border is only {r:.2f}:1 against "
                    f"the insignificant border, so the two look alike"
                )


@pytest.mark.parametrize("name", sorted(SCHEMES))
def test_borders_are_darker_than_their_fill(name):
    """A frame should read as a frame, not as a lighter inset."""
    from mlwp_scorecards.colours import _relative_luminance

    scheme = SCHEMES[name]
    for family in scheme.families:
        for level in range(1, scheme.depth + 1):
            sw = scheme.swatch(family, level)
            assert _relative_luminance(sw.edge) < _relative_luminance(sw.fill), (
                name,
                family,
                level,
            )


def test_scaling_is_symmetric_and_monotone():
    s = FixedScaling()
    assert s.level(0.0) == 0
    for v in (0.006, 0.03, 0.2, 0.9):
        assert s.level(v) == -s.level(-v)
    levels = [s.level(v) for v in (0.006, 0.03, 0.2, 0.9)]
    assert levels == sorted(levels)


def test_scaling_saturates_rather_than_wrapping():
    s = FixedScaling()
    assert s.level(5.0) == s.level(1.0) == s.depth
    assert s.saturated(5.0) and not s.saturated(0.1)


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


def test_diverging_ramp_collapses_in_greyscale():
    """Documents *why* colour alone cannot carry the judgement in print.

    A diverging ramp is luminance-symmetric by construction, so the two arms are
    not separable once hue is removed. No palette fixes this; a redundant glyph
    channel is the only remedy.
    """
    scheme = SCHEMES["cvd"]
    from mlwp_scorecards.colours import _relative_luminance

    for family in scheme.families:
        for level in range(1, scheme.depth + 1):
            pos = _relative_luminance(scheme.swatch(family, level).fill)
            neg = _relative_luminance(scheme.swatch(family, -level).fill)
            assert abs(pos - neg) < 0.30, (family, level)
