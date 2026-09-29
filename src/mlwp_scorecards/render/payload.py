"""Compact drill-down payload for the interactive HTML page.

Clicking a cell shows two charts, which need numbers the table itself does not
carry: the baseline and forecast series with their confidence intervals. That is
~8 arrays per cell, and on a full-size card it is the single largest thing on the
page -- 3.2 MB of the reference implementation's 7.4 MB is exactly this, stored as
a raw JSON literal at full float precision.

Three things keep it small here:

* **Columnar, not per-cell objects.** One array per quantity per cell, not a list
  of records with repeated keys.
* **Rounded to 4 significant figures.** A chart cannot show more, and rounding is
  what makes the text compress well.
* **gzip, then base64.** Inflated lazily in the browser via ``DecompressionStream``,
  which works from ``file://``.

Cells are addressed by integer index, enumerated once here and written into the
table as ``data-i``. Labels never participate in identity -- the reference builds
ids by concatenating labels, which breaks on any label containing ``_``.
"""

from __future__ import annotations

import base64
import gzip
import json

from ..layout import Layout

__all__ = ["build_payload", "pack", "payload_for"]

#: Short keys: this repeats once per cell, so the names matter.
_SERIES = (
    ("c", "baseline"),
    ("cl", "baseline_lower"),
    ("cu", "baseline_upper"),
    ("e", "forecast"),
    ("el", "forecast_lower"),
    ("eu", "forecast_upper"),
    ("d", "relative"),
    # the paired difference in the metric's own units, with its interval
    ("v", "value"),
    ("vl", "value_lower"),
    ("vu", "value_upper"),
    ("n", "n"),
)


def _round(values: list, precision: int) -> list:
    """Round to a fixed number of significant figures, keeping None as null.

    Parameters
    ----------
    values : list
        Numbers to round; None entries are passed through.
    precision : int
        Significant figures to keep.

    Returns
    -------
    list
        The rounded values, as floats or None.
    """
    out = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            out.append(float(f"%.{precision}g" % v))
    return out


def build_payload(layout: Layout, *, precision: int = 4) -> dict:
    """Assemble the drill-down data as a plain, JSON-safe dict.

    Parameters
    ----------
    layout : Layout
        The card whose populated cells are exported.
    precision : int, optional
        Significant figures kept in each series; case counts are not rounded.

    Returns
    -------
    dict
        ``lead`` (hours), ``labels``, and ``cells``: one entry per populated cell,
        in the same order as :meth:`~mlwp_scorecards.layout.model.Layout.iter_cells`.
    """
    cells = []
    for _, _, cell in layout.iter_cells():
        entry: dict[str, object] = {
            "t": " / ".join(str(k) for k in cell.row_key if k is not None)
            + "  ·  "
            + " / ".join(str(k) for k in cell.col_key if k is not None),
            "u": cell.units or "",
            "m": cell.metric,
        }
        # Only when rows differ in source: with one, the page-level name covers
        # every cell and repeating it per cell would only grow the payload.
        if len(layout.forecast_sources) > 1:
            entry["s"] = cell.forecast_source
        for short, attr in _SERIES:
            values = [getattr(s, attr) for s in cell.steps]
            if all(v is None for v in values):
                continue
            entry[short] = values if attr == "n" else _round(values, precision)
        cells.append(entry)

    return {
        "lead": list(layout.lead_times),
        "labels": list(layout.lead_labels),
        # Keys kept from the two-source card so its page stays byte-identical.
        "control": layout.baseline_source,
        "experiment": layout.forecast_label,
        "confidence": layout.confidence,
        "cells": cells,
    }


def pack(payload: dict) -> str:
    """Serialise, compress and base64-encode a payload.

    ``mtime=0`` keeps the output byte-reproducible; ``allow_nan=False`` refuses to
    emit bare ``NaN``, which is not valid JSON and would break ``JSON.parse``.

    Parameters
    ----------
    payload : dict
        A JSON-safe dict, as from :func:`build_payload`.

    Returns
    -------
    str
        The gzipped JSON, base64-encoded.
    """
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    return base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode()


def payload_for(layout: Layout, *, precision: int = 4) -> tuple[str, int, int]:
    """Build and pack in one step.

    Parameters
    ----------
    layout : Layout
        The card whose populated cells are exported.
    precision : int, optional
        Significant figures kept in each series.

    Returns
    -------
    encoded : str
        The packed payload, as from :func:`pack`.
    raw_bytes : int
        Size of the JSON before compression, for reporting.
    packed_bytes : int
        Size of ``encoded`` after compression and base64, for reporting.
    """
    payload = build_payload(layout, precision=precision)
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    encoded = base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode()
    return encoded, len(raw), len(encoded)
