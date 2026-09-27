"""`select=`: one rule for every coordinate.

A single value picks a member and drops the dimension, unless the caller placed
that dimension on the card; a list keeps it, in the order given, with `...` for
the rest; a slice keeps it as `ds.sel` would. `forecast_source` follows the same
rule, except that the baseline is never part of a `...` and may not be named.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlwp_scorecards import build_layout

ROWS = ["truth_source", "variable", "level"]
COLUMNS = ["spatial_region", "metric"]


def _card(ds, **kw):
    kw.setdefault("n_resamples", 50)
    return build_layout(ds, relative_to="persistence", **kw)


def _order(lay, dim):
    """The distinct values of `dim` along whichever axis carries it, in order."""
    for lines, dims in ((lay.rows, lay.row_dims), (lay.columns, lay.column_dims)):
        if dim in dims:
            i = dims.index(dim)
            return list(dict.fromkeys(line.key[i] for line in lines))
    raise AssertionError(f"{dim} is on neither axis")


# --------------------------------------------------------------------------- #
# lists: subset, and the order they are drawn in
# --------------------------------------------------------------------------- #
def test_a_list_sets_the_order_along_any_axis(verification):
    lay = _card(
        verification,
        rows=ROWS,
        columns=COLUMNS,
        select=dict(spatial_region=["tropics", "n.hem"]),
    )
    assert _order(lay, "spatial_region") == ["tropics", "n.hem"]


def test_an_ellipsis_fills_in_the_rest_in_coordinate_order(verification):
    lay = _card(
        verification,
        rows=ROWS,
        columns=COLUMNS,
        select=dict(spatial_region=["s.hem", ...]),
    )
    assert _order(lay, "spatial_region") == ["s.hem", "n.hem", "tropics"]


def test_variable_and_metric_may_be_selected_and_ordered(verification):
    lay = _card(
        verification,
        rows=ROWS,
        columns=COLUMNS,
        select=dict(variable=["2t", "z"], metric=["mae", "rmse"]),
    )
    assert _order(lay, "variable") == ["2t", "z"]
    assert _order(lay, "metric") == ["mae", "rmse"]


def test_a_single_variable_stays_on_its_axis(verification):
    """`variable` and `metric` always need a place, so one value is kept."""
    lay = _card(verification, rows=ROWS, columns=COLUMNS, select=dict(variable="z"))
    assert _order(lay, "variable") == ["z"]


def test_two_ellipses_are_refused(verification):
    with pytest.raises(ValueError, match="at most once"):
        _card(
            verification,
            rows=ROWS,
            columns=COLUMNS,
            select=dict(spatial_region=[..., "n.hem", ...]),
        )


def test_a_value_not_on_the_coordinate_is_refused(verification):
    with pytest.raises(KeyError, match="'europe' is not in spatial_region"):
        _card(
            verification,
            rows=ROWS,
            columns=COLUMNS,
            select=dict(spatial_region=["europe"]),
        )


# --------------------------------------------------------------------------- #
# single values: dropped, unless the caller placed the dimension
# --------------------------------------------------------------------------- #
def test_a_single_value_drops_the_dimension_when_axes_are_inferred(verification):
    lay = _card(verification, select=dict(spatial_region="n.hem"))
    assert "spatial_region" not in lay.row_dims + lay.column_dims


def test_a_single_value_on_a_placed_dimension_keeps_it(verification):
    lay = _card(
        verification, rows=ROWS, columns=COLUMNS, select=dict(spatial_region="n.hem")
    )
    assert _order(lay, "spatial_region") == ["n.hem"]


def test_a_single_value_on_the_cell_dimension_keeps_it(verification):
    lead = verification["lead_time"].values[1]
    lay = _card(verification, rows=ROWS, columns=COLUMNS, select=dict(lead_time=lead))
    assert len(lay.lead_times) == 1


def test_a_single_truth_drops_the_truth_block_unless_placed(verification):
    inferred = _card(verification, select=dict(truth_source="analysis"))
    assert "truth_source" not in inferred.row_dims
    placed = _card(
        verification, rows=ROWS, columns=COLUMNS, select=dict(truth_source="analysis")
    )
    assert _order(placed, "truth_source") == ["analysis"]


def test_dropping_and_placing_agree_on_the_numbers(verification):
    dropped = _card(
        verification,
        rows=["truth_source", "variable", "level"],
        columns=["metric"],
        select=dict(spatial_region="n.hem"),
    )
    placed = _card(
        verification, rows=ROWS, columns=COLUMNS, select=dict(spatial_region="n.hem")
    )
    a = dropped.sel(truth_source="analysis", variable="z", level=500.0, metric="rmse")
    b = placed.sel(
        truth_source="analysis",
        variable="z",
        level=500.0,
        spatial_region="n.hem",
        metric="rmse",
    )
    assert [s.value for s in a.steps] == [s.value for s in b.steps]


# --------------------------------------------------------------------------- #
# slices
# --------------------------------------------------------------------------- #
def test_a_slice_of_init_time_uses_fewer_cases(verification):
    cases = verification["init_time"].values
    full = _card(verification, rows=ROWS, columns=COLUMNS)
    half = _card(
        verification,
        rows=ROWS,
        columns=COLUMNS,
        select=dict(init_time=slice(cases[0], cases[len(cases) // 2 - 1])),
    )

    def first_n(lay):
        cell = lay.sel(
            truth_source="analysis",
            variable="z",
            level=500.0,
            spatial_region="n.hem",
            metric="rmse",
        )
        return cell.steps[0].n

    assert first_n(half) < first_n(full)


# --------------------------------------------------------------------------- #
# forecast_source
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [..., [...], None])
def test_the_baseline_is_left_out_of_every_source_automatically(verification, value):
    select = None if value is None else dict(forecast_source=value)
    lay = _card(verification, select=select)
    assert lay.forecast_sources == ("drifting-persistence",)
    assert lay.baseline_source == "persistence"


def test_one_source_by_name_is_the_two_source_card(verification):
    a = _card(verification, select=dict(forecast_source="drifting-persistence"))
    b = _card(verification, select=dict(forecast_source=["drifting-persistence"]))
    assert "forecast_source" not in a.row_dims + a.column_dims
    assert [c.key for c in a.columns] == [c.key for c in b.columns]


def test_naming_the_baseline_as_a_forecast_source_is_refused(verification):
    with pytest.raises(ValueError, match="also named in"):
        _card(verification, select=dict(forecast_source=["persistence"]))


def test_a_slice_of_forecast_sources_is_refused(verification):
    with pytest.raises(ValueError, match="not a slice"):
        _card(verification, select=dict(forecast_source=slice(None)))


def test_select_leaves_the_input_dataset_alone(verification):
    before = verification.sizes
    _card(verification, select=dict(spatial_region="n.hem", variable=["z"]))
    assert verification.sizes == before
    assert np.array_equal(
        verification["spatial_region"].values, ["n.hem", "tropics", "s.hem"]
    )
