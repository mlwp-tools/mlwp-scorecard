"""Colour ramps, scaling, and the metric semantics that drive them.

Two independent facts about a cell get two independent visual channels: the
**fill** encodes magnitude, the **border** encodes significance. Nothing here
infers meaning from a metric's name by pattern matching; :data:`METRIC_POLARITY`
is an explicit lookup table and unknown metrics raise rather than defaulting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

__all__ = [
    "Polarity",
    "Swatch",
    "Ramp",
    "Family",
    "ColourScheme",
    "FixedScaling",
    "SCHEMES",
    "METRIC_POLARITY",
    "polarity_of",
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
    """Colour family for a polarity."""
    return _FAMILY_OF[polarity]


# --------------------------------------------------------------------------- #
# swatches and ramps
# --------------------------------------------------------------------------- #


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def _mix(a: str, b: str, t: float) -> str:
    ra, rb = _hex_to_rgb(a), _hex_to_rgb(b)
    return _rgb_to_hex(tuple(x + (y - x) * t for x, y in zip(ra, rb)))  # type: ignore[arg-type]


def _relative_luminance(h: str) -> float:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in _hex_to_rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two hex colours."""
    la, lb = _relative_luminance(a), _relative_luminance(b)
    lo, hi = sorted((la, lb))
    return (hi + 0.05) / (lo + 0.05)


@dataclass(frozen=True, slots=True)
class Swatch:
    """One step of a ramp: fill, its saturated border, and a legible foreground."""

    fill: str
    edge: str
    fg: str


@dataclass(frozen=True, slots=True)
class Ramp:
    """A one-directional sequence of swatches, weakest first."""

    swatches: tuple[Swatch, ...]

    def __len__(self) -> int:
        return len(self.swatches)

    def __getitem__(self, i: int) -> Swatch:
        return self.swatches[min(max(i, 0), len(self.swatches) - 1)]


def _build_ramp(light: str, dark: str, n: int = 14) -> Ramp:
    """Interpolate ``n`` swatches from ``light`` to ``dark``.

    Fill carries direction and magnitude; the border carries significance.

    **The border is always dark**, and the contrast that matters is against
    *white*, not against the fill. An insignificant box is drawn with a white
    border, so "significant" reads as "framed" and "not significant" as "not
    framed". Flipping the border to light on dark fills -- which looks right if you
    only compare it with the fill it sits on -- makes a significant dark box
    indistinguishable from an insignificant one, since both then show a pale ring,
    and it leaves the legend swatches looking borderless on a white page.

    ``fg`` is a separate question: that is for a glyph drawn *on* the fill, so it
    does flip.
    """
    out = []
    for i in range(n):
        t = i / (n - 1)
        fill = _mix(light, dark, t)
        edge = _mix(fill, "#000000", 0.55)
        fg = "#000000" if contrast_ratio(fill, "#000000") >= 4.5 else "#ffffff"
        out.append(Swatch(fill, edge, fg))
    return Ramp(tuple(out))


@dataclass(frozen=True, slots=True)
class Family:
    """A pair of ramps for the two directions of one kind of metric."""

    key: str
    positive: Ramp
    negative: Ramp
    positive_word: str
    negative_word: str


@dataclass(frozen=True, slots=True)
class ColourScheme:
    """A complete palette: one :class:`Family` per metric family."""

    name: str
    families: Mapping[str, Family]
    neutral: Swatch = Swatch("#f0f0f0", "#bbbbbb", "#000000")
    missing: str = "#f4f4f4"
    insignificant_edge: str = "#ffffff"

    def swatch(self, family: str, level: int) -> Swatch:
        """Return the swatch for a signed ramp level."""
        if level == 0:
            return self.neutral
        fam = self.families[family]
        ramp = fam.positive if level > 0 else fam.negative
        return ramp[abs(level) - 1]

    def word(self, family: str, level: int) -> str:
        """The word describing this direction, for tooltips and legends."""
        fam = self.families[family]
        if level == 0:
            return "no change"
        return fam.positive_word if level > 0 else fam.negative_word

    @property
    def depth(self) -> int:
        """Number of ramp steps per direction."""
        return len(next(iter(self.families.values())).positive)


#: CVD-safe default. RdBu for error-like metrics, BrBG for activity — the
#: reference's purple/green activity pair is not separable under deuteranopia.
CVD = ColourScheme(
    name="cvd",
    families={
        "error": Family(
            "error",
            positive=_build_ramp("#eaf2f8", "#0b3d6b"),
            negative=_build_ramp("#fdeae7", "#7a1710"),
            positive_word="better",
            negative_word="worse",
        ),
        "activity": Family(
            "activity",
            positive=_build_ramp("#f3ece1", "#5c3607"),
            negative=_build_ramp("#e4f2f0", "#01443e"),
            positive_word="more active",
            negative_word="less active",
        ),
    },
)

#: The reference palette, for visual continuity with published ECMWF cards.
ECMWF = ColourScheme(
    name="ecmwf",
    families={
        "error": Family(
            "error",
            positive=_build_ramp("#a6e3fd", "#0043cf"),
            negative=_build_ramp("#fbd5bf", "#c04c26"),
            positive_word="better",
            negative_word="worse",
        ),
        "activity": Family(
            "activity",
            positive=_build_ramp("#dec8e2", "#900090"),
            negative=_build_ramp("#cbe9c5", "#00801b"),
            positive_word="more active",
            negative_word="less active",
        ),
    },
)

SCHEMES: dict[str, ColourScheme] = {"cvd": CVD, "ecmwf": ECMWF}


@dataclass(frozen=True, slots=True)
class FixedScaling:
    """Absolute breakpoints on the magnitude of a relative difference.

    Absolute rather than quantile-based on purpose: quantile scaling makes two
    scorecards from different experiments non-comparable, which defeats the point.
    """

    breaks: tuple[float, ...] = (
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.55,
        0.75,
        1.00,
    )

    def level(self, relative: float | None) -> int:
        """Map a signed relative difference to a signed ramp level."""
        import math

        if relative is None or (isinstance(relative, float) and math.isnan(relative)):
            return 0
        mag = abs(relative)
        idx = 0
        for b in self.breaks:
            if mag >= b:
                idx += 1
            else:
                break
        if idx == 0:
            return 0
        return idx if relative > 0 else -idx

    @property
    def depth(self) -> int:
        return len(self.breaks)

    def saturated(self, relative: float | None) -> bool:
        """True when the value is at or beyond the top break, i.e. off the scale."""
        if relative is None:
            return False
        return abs(relative) >= self.breaks[-1]
