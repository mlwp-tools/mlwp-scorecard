"""Rendering backends.

Modules in this package may import :mod:`mlwp_scorecards.layout` (the layout
types), :mod:`mlwp_scorecards.polarity` (the words for each direction) and each
other -- :mod:`~mlwp_scorecards.render.colours` holds the palettes -- only. They must not import
:mod:`mlwp_scorecards.layout.engine` or ``xarray``, or read the raw input dataset:
the :class:`~mlwp_scorecards.layout.model.Layout` is the sole renderer contract,
and anything a backend needs that is absent from it belongs in
``layout/model.py`` rather than in a backend-specific code path.

Backends are imported lazily so that an HTML-only install need not pull in
matplotlib.
"""
