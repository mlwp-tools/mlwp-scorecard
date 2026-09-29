"""The card's layout: the types a renderer draws, and the engine that makes them.

- :mod:`~mlwp_scorecard.layout.model` -- the resolved layout types
  (:class:`Layout`, :class:`Cell`, :class:`Step`, ...). The sole renderer
  contract; standard library only.
- :mod:`~mlwp_scorecard.layout.engine` -- :func:`create_layout`, which turns
  scores collapsed over cases into a :class:`Layout`.
- :mod:`~mlwp_scorecard.layout.scaling` -- how far along the ramp a relative
  difference lands: one of ``±LEVELS``.

A layout carries no colours: each box has a family and a signed level, derived
from the metric's polarity, and the renderers choose the palette.

Only the model is re-exported here, and the engine must never be imported from
this file: renderers import ``mlwp_scorecard.layout``, and Python runs this
file before any submodule, so importing the engine here would hand every
renderer xarray and the aggregation code. ``tests/test_layout.py`` checks it.
"""

from __future__ import annotations

from .model import (
    LEVELS,
    NEUTRAL,
    Cell,
    HeaderCell,
    Key,
    Layout,
    LayoutStats,
    Line,
    Step,
    format_value,
)

__all__ = [
    "Layout",
    "Cell",
    "Step",
    "Line",
    "HeaderCell",
    "LayoutStats",
    "Key",
    "LEVELS",
    "NEUTRAL",
    "format_value",
]
