"""How far along the ramp a difference lands: a relative difference to a signed level."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import LEVELS

__all__ = ["FixedScaling"]


@dataclass(frozen=True, slots=True)
class FixedScaling:
    """Absolute breakpoints on the magnitude of a relative difference.

    Absolute rather than quantile-based on purpose: quantile scaling makes two
    scorecards from different experiments non-comparable, which defeats the point.
    There is one break per level, so levels run from ``-LEVELS`` to ``LEVELS``.

    Attributes
    ----------
    breaks : tuple of float, optional
        Ascending thresholds on ``abs(relative)``; reaching the ``i``-th (from 1)
        gives level ``i``, and below the first gives level zero.
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

    def __post_init__(self) -> None:
        """Refuse a set of breaks that does not match the layout's level range.

        Raises
        ------
        ValueError
            If there is not exactly one break per level.
        """
        if len(self.breaks) != LEVELS:
            raise ValueError(
                f"FixedScaling needs {LEVELS} breaks, one per level; got "
                f"{len(self.breaks)}"
            )

    def level(self, relative: float | None) -> int:
        """Map a signed relative difference to a signed ramp level.

        Parameters
        ----------
        relative : float or None
            The signed relative difference, positive meaning better (or more
            active). None or NaN means no difference.

        Returns
        -------
        int
            The number of breaks reached, carrying the sign of ``relative``.
        """
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

    def saturated(self, relative: float | None) -> bool:
        """Return whether the value is at or beyond the top break, i.e. off the scale.

        Parameters
        ----------
        relative : float or None
            The signed relative difference; None is never saturated.

        Returns
        -------
        bool
            True when ``abs(relative)`` reaches the last break.
        """
        if relative is None:
            return False
        return abs(relative) >= self.breaks[-1]
