"""Rendering backends.

Modules in this package may import :mod:`mlwp_scorecards.model`,
:mod:`mlwp_scorecards.colours` and :mod:`mlwp_scorecards.render.geometry` only.
They must not import ``xarray`` or read a raw
:class:`~mlwp_scorecards.spec.ScorecardSpec`: the
:class:`~mlwp_scorecards.model.Layout` is the sole renderer contract, and anything
a backend needs that is absent from it belongs in ``model.py`` rather than in a
backend-specific code path.

Backends are imported lazily so that an HTML-only install need not pull in
matplotlib.
"""
