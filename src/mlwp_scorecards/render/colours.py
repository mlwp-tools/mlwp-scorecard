"""Palettes: the actual colours for a box's family and signed level.

The layout says what a box *means* -- its family (``"error"``, ``"activity"``)
and a signed level, derived from the metric's polarity -- and this module says
what that looks like. Two independent facts about a cell get two independent
visual channels: the **fill** encodes magnitude, the **border** encodes
significance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..layout import LEVELS

__all__ = [
    "Swatch",
    "Ramp",
    "Family",
    "ColourScheme",
    "SCHEMES",
    "contrast_ratio",
]


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    """Convert a hex colour to RGB components.

    Parameters
    ----------
    h : str
        A colour as ``#rrggbb`` (the ``#`` is optional).

    Returns
    -------
    tuple of float
        Red, green and blue, each in ``[0, 1]``.
    """
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    """Convert RGB components to a hex colour.

    Parameters
    ----------
    rgb : tuple of float
        Red, green and blue in ``[0, 1]``; values outside are clipped.

    Returns
    -------
    str
        The colour as lowercase ``#rrggbb``.
    """
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def _mix(a: str, b: str, t: float) -> str:
    """Interpolate linearly between two hex colours in RGB.

    Parameters
    ----------
    a : str
        The colour at ``t = 0``, as hex.
    b : str
        The colour at ``t = 1``, as hex.
    t : float
        The fraction of the way from ``a`` to ``b``.

    Returns
    -------
    str
        The mixed colour as hex.
    """
    ra, rb = _hex_to_rgb(a), _hex_to_rgb(b)
    return _rgb_to_hex(tuple(x + (y - x) * t for x, y in zip(ra, rb)))  # type: ignore[arg-type]


def _relative_luminance(h: str) -> float:
    """Return the WCAG relative luminance of a hex colour.

    Parameters
    ----------
    h : str
        A colour as hex.

    Returns
    -------
    float
        Relative luminance, from 0 (black) to 1 (white).
    """

    def lin(c: float) -> float:
        """Linearise one sRGB component.

        Parameters
        ----------
        c : float
            A gamma-encoded component in ``[0, 1]``.

        Returns
        -------
        float
            The linear-light component.
        """
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in _hex_to_rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """Return the WCAG contrast ratio between two hex colours.

    Parameters
    ----------
    a : str
        One colour, as hex.
    b : str
        The other colour, as hex; the order does not matter.

    Returns
    -------
    float
        The ratio, from 1 (identical luminance) to 21 (black on white).
    """
    la, lb = _relative_luminance(a), _relative_luminance(b)
    lo, hi = sorted((la, lb))
    return (hi + 0.05) / (lo + 0.05)


@dataclass(frozen=True, slots=True)
class Swatch:
    """One step of a ramp: fill, its saturated border, and a legible foreground.

    Attributes
    ----------
    fill : str
        The box's fill colour, as hex.
    edge : str
        The border drawn when the box is significant: the fill darkened.
    fg : str
        Black or white, whichever is legible on ``fill``, for text drawn on it.
    """

    fill: str
    edge: str
    fg: str


@dataclass(frozen=True, slots=True)
class Ramp:
    """A one-directional sequence of swatches, weakest first.

    Attributes
    ----------
    swatches : tuple of Swatch
        The steps, from palest to darkest.
    """

    swatches: tuple[Swatch, ...]

    def __len__(self) -> int:
        """Return the number of steps.

        Returns
        -------
        int
            The number of swatches.
        """
        return len(self.swatches)

    def __getitem__(self, i: int) -> Swatch:
        """Return the swatch at a step, clamped to the ends of the ramp.

        Parameters
        ----------
        i : int
            The zero-based step; out-of-range values take the nearest end.

        Returns
        -------
        Swatch
            The swatch at that step.
        """
        return self.swatches[min(max(i, 0), len(self.swatches) - 1)]


def _build_ramp(light: str, dark: str, n: int = LEVELS) -> Ramp:
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

    Parameters
    ----------
    light : str
        The weakest fill, as hex.
    dark : str
        The strongest fill, as hex.
    n : int, optional
        The number of swatches: one per level, by default.

    Returns
    -------
    Ramp
        ``n`` swatches from ``light`` to ``dark``, both ends included.
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
    """A pair of ramps for the two directions of one kind of metric.

    The words for the two directions are not a matter of palette, and live in
    :data:`mlwp_scorecards.polarity.FAMILY_WORDS`.

    Attributes
    ----------
    key : str
        The family's name, as returned by
        :func:`~mlwp_scorecards.polarity.family_of`.
    positive : Ramp
        The ramp for positive levels: better, or more active.
    negative : Ramp
        The ramp for negative levels: worse, or less active.
    """

    key: str
    positive: Ramp
    negative: Ramp


@dataclass(frozen=True, slots=True)
class ColourScheme:
    """A complete palette: one :class:`Family` per metric family.

    Attributes
    ----------
    name : str
        The scheme's name, as a key of :data:`SCHEMES`.
    families : mapping of str to Family
        The families, keyed by :attr:`Family.key`.
    neutral : Swatch, optional
        The swatch for level zero and for boxes compared with nothing.
    missing : str, optional
        The fill for a box with no data.
    insignificant_edge : str, optional
        The border of a box that is not significant.
    """

    name: str
    families: Mapping[str, Family]
    neutral: Swatch = Swatch("#f0f0f0", "#bbbbbb", "#000000")
    missing: str = "#f4f4f4"
    insignificant_edge: str = "#ffffff"

    def swatch(self, family: str, level: int) -> Swatch:
        """Return the swatch for a signed ramp level.

        Parameters
        ----------
        family : str
            The colour family's key.
        level : int
            The signed ramp level; its sign picks the ramp, its magnitude the step.

        Returns
        -------
        Swatch
            The swatch, or :attr:`neutral` at level zero.
        """
        if level == 0:
            return self.neutral
        fam = self.families[family]
        ramp = fam.positive if level > 0 else fam.negative
        return ramp[abs(level) - 1]

    @property
    def depth(self) -> int:
        """Number of ramp steps per direction.

        Returns
        -------
        int
            The length of the first family's positive ramp.
        """
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
        ),
        "activity": Family(
            "activity",
            positive=_build_ramp("#f3ece1", "#5c3607"),
            negative=_build_ramp("#e4f2f0", "#01443e"),
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
        ),
        "activity": Family(
            "activity",
            positive=_build_ramp("#dec8e2", "#900090"),
            negative=_build_ramp("#cbe9c5", "#00801b"),
        ),
    },
)

SCHEMES: dict[str, ColourScheme] = {"cvd": CVD, "ecmwf": ECMWF}
