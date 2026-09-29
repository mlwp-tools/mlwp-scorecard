"""The card's layout: the types a renderer draws, and the engine that makes them.

- :mod:`~mlwp_scorecards.layout.model` -- the resolved layout types
  (:class:`Layout`, :class:`Cell`, :class:`Step`, ...). The sole renderer
  contract; standard library only.
- :mod:`~mlwp_scorecards.layout.engine` -- :func:`create_layout`, which turns
  scores collapsed over cases into a :class:`Layout`.

Only the model is re-exported here, and the engine must never be imported from
this file: renderers import ``mlwp_scorecards.layout``, and Python runs this
file before any submodule, so importing the engine here would hand every
renderer xarray and the aggregation code. ``tests/test_layout.py`` checks it.
"""

from __future__ import annotations

from .model import (
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
    "NEUTRAL",
    "format_value",
]
