"""What a metric means: which direction of a difference is an improvement.

Polarity is a fact about the metric, not about colour. The layout engine derives
each box's colour *family* and signed *level* from it; only the renderers turn
those into actual colours. Nothing here infers meaning from a metric's name by
pattern matching: :data:`METRIC_POLARITY` is an explicit lookup table, and an
unknown metric raises rather than defaulting, because a guess produces a
confidently backwards card.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping

__all__ = [
    "Polarity",
    "METRIC_POLARITY",
    "FAMILY_WORDS",
    "polarity_of",
    "family_of",
    "word",
]


class Polarity(StrEnum):
    """Which direction of a difference counts as an improvement."""

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"
    ACTIVITY = "activity"
    NEUTRAL = "neutral"


#: Known metrics. Anything absent must be declared by the caller.
METRIC_POLARITY: dict[str, Polarity] = {
    "rmse": Polarity.LOWER_IS_BETTER,
    "rmsef": Polarity.LOWER_IS_BETTER,
    "mae": Polarity.LOWER_IS_BETTER,
    "mse": Polarity.LOWER_IS_BETTER,
    "bias": Polarity.NEUTRAL,
    "crps": Polarity.LOWER_IS_BETTER,
    "crpss": Polarity.HIGHER_IS_BETTER,
    "acc": Polarity.HIGHER_IS_BETTER,
    "csi": Polarity.HIGHER_IS_BETTER,
    "ets": Polarity.HIGHER_IS_BETTER,
    "spread": Polarity.ACTIVITY,
    "stdev": Polarity.ACTIVITY,
    "activity": Polarity.ACTIVITY,
}

_FAMILY_OF = {
    Polarity.LOWER_IS_BETTER: "error",
    Polarity.HIGHER_IS_BETTER: "error",
    Polarity.NEUTRAL: "error",
    Polarity.ACTIVITY: "activity",
}

#: The words for each family's (negative, positive) direction, for tooltips and
#: legends. A positive level always means better, or more active.
FAMILY_WORDS: dict[str, tuple[str, str]] = {
    "error": ("worse", "better"),
    "activity": ("less active", "more active"),
}


def polarity_of(
    metric: str, overrides: Mapping[str, str | Polarity] | None = None
) -> Polarity:
    """Resolve a metric's polarity.

    Parameters
    ----------
    metric : str
        Bare metric name — the half before the last dot of a
        ``{metric}.{variable}`` data-variable name.
    overrides : mapping, optional
        Caller-supplied polarities, taking precedence over :data:`METRIC_POLARITY`.

    Returns
    -------
    Polarity
        The metric's polarity.

    Raises
    ------
    KeyError
        If the metric is unknown and not overridden. Guessing here would produce a
        confidently backwards scorecard, so it is refused.
    """
    if overrides and metric in overrides:
        return Polarity(overrides[metric])
    if metric in METRIC_POLARITY:
        return METRIC_POLARITY[metric]
    raise KeyError(
        f"unknown metric {metric!r}: cannot tell whether higher or lower is better. "
        f"Pass metric_polarity={{{metric!r}: 'lower_is_better'}} (or 'higher_is_better', "
        f"'activity', 'neutral')."
    )


def family_of(polarity: Polarity) -> str:
    """Return the family a polarity's comparisons belong to.

    Parameters
    ----------
    polarity : Polarity
        The metric's polarity.

    Returns
    -------
    str
        ``"activity"`` for activity metrics, ``"error"`` for everything else: a
        key of :data:`FAMILY_WORDS`, and of each palette's families.
    """
    return _FAMILY_OF[polarity]


def word(family: str, level: int) -> str:
    """Return the word describing a signed level's direction.

    Parameters
    ----------
    family : str
        A key of :data:`FAMILY_WORDS`.
    level : int
        The signed ramp level; only its sign matters.

    Returns
    -------
    str
        The family's positive or negative word, or ``"no change"`` at zero.
    """
    if level == 0:
        return "no change"
    negative, positive = FAMILY_WORDS[family]
    return positive if level > 0 else negative
