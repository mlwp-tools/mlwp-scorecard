"""Layout engine: ordering, raggedness, spans, indexing."""

from __future__ import annotations

import pytest

from mlwp_scorecards import build_layout


def test_rows_follow_dataset_coordinate_order(verification, layout):
    """Presentation order is the caller's order, not alphabetical.

    With the metric folded into the variable name there is no metric coordinate
    to order, so the order that must be honoured is first appearance among the
    data variables -- which for this dataset is rmse, mae, spread, deliberately
    not alphabetical.
    """
    want = []
    for name in verification.data_vars:
        metric = str(name).rsplit(".", 1)[0]
        if metric != "number_of_forecasts" and metric not in want:
            want.append(metric)
    got = []
    for c in layout.columns:
        if c.key[1] not in got:
            got.append(str(c.key[1]))
    assert got == want


def test_header_spans_cover_every_line_exactly_once(layout):
    """Every depth's blocks are contiguous and tile the axis."""
    for axis, headers, n in (
        ("rows", layout.row_headers, layout.stats.n_rows),
        ("columns", layout.column_headers, layout.stats.n_cols),
    ):
        for depth, blocks in enumerate(headers):
            assert sum(b.span for b in blocks) == n, (axis, depth)
            covered = []
            for b in blocks:
                covered.extend(range(b.start, b.stop))
            assert covered == list(range(n)), (axis, depth)


def test_line_headers_are_never_ragged(layout):
    """Raggedness is absorbed here so no renderer has to branch on it."""
    for line in layout.rows:
        assert len(line.headers) == layout.row_depth
    for line in layout.columns:
        assert len(line.headers) == layout.column_depth


def test_reordering_rows_reorders_the_nesting(verification):
    a = build_layout(
        verification,
        colour_relative_to="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
    )
    b = build_layout(
        verification,
        colour_relative_to="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
        rows=["truth_source", "level", "variable"],
        columns=["spatial_region", "metric"],
    )
    assert a.row_dims != b.row_dims
    assert [r.key for r in a.rows] != [r.key for r in b.rows]
    assert a.stats.n_cells_present == b.stats.n_cells_present


def test_unknown_dimension_is_rejected_clearly(verification):
    with pytest.raises(KeyError, match="not a dimension"):
        build_layout(
            verification,
            colour_relative_to="persistence",
            select=dict(forecast_source=["drifting-persistence"]),
            rows=["nonsuch"],
            columns=["metric"],
        )


def test_unassigned_dimension_is_rejected_clearly(verification):
    """Silently dropping a dimension would average over it without saying so."""
    with pytest.raises(KeyError, match="assigned to neither"):
        build_layout(
            verification,
            colour_relative_to="persistence",
            select=dict(forecast_source=["drifting-persistence"]),
            rows=["variable"],
            columns=["metric"],
        )


def test_unknown_source_is_rejected_clearly(verification):
    with pytest.raises(KeyError, match="'nonsuch' is not in forecast_source"):
        build_layout(
            verification,
            colour_relative_to="persistence",
            select=dict(forecast_source=["nonsuch"]),
        )


@pytest.mark.parametrize("unit", ["h", "s", "ms", "us", "ns"])
def test_lead_times_are_labelled_the_same_at_every_time_resolution(unit):
    """Six hours is `T+6` however the coordinate was stored.

    Nanoseconds is the case that broke: `.item()` on a `timedelta64` returns a
    `datetime.timedelta` at every coarser resolution but a bare int of
    nanoseconds at `ns`, and an int is indistinguishable from a lead time
    already given in hours -- so every box was labelled `T+2.16e+13`. It is also
    the resolution a netCDF round trip commonly produces, so it is the common
    case rather than an exotic one, and it survived because no test had ever
    asserted a label.
    """
    import numpy as np
    import xarray as xr

    lead = np.arange(6, 25, 6).astype("timedelta64[h]").astype(f"timedelta64[{unit}]")
    values = np.outer([1.0, 1.1], np.linspace(1.0, 2.0, len(lead)))
    ds = xr.Dataset(
        {
            "rmse.2t": xr.DataArray(
                values,
                dims=["forecast_source", "lead_time"],
                coords=dict(forecast_source=["ctl", "exp"], lead_time=lead),
                attrs=dict(units="K"),
            )
        }
    )
    layout = build_layout(
        ds, colour_relative_to="ctl", select=dict(forecast_source=["exp"])
    )

    assert layout.lead_labels == ("T+6", "T+12", "T+18", "T+24")
    assert layout.lead_times == (6.0, 12.0, 18.0, 24.0)
    assert layout.sel(variable="2t", metric="rmse").steps[0].tooltip.startswith("T+6 ")
