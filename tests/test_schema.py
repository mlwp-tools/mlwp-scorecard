"""The input schema: dotted names, reserved dimensions, and what is optional.

These cover the things the schema exists to make possible — a metric-dependent
unit, ragged metric coverage, an arbitrary grouping dimension — and the ways the
naming convention could quietly go wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from mlwp_scorecards import build_layout
from mlwp_scorecards.ingest import split_name

N_CASE = 40
LEAD = np.array([24, 48], dtype="timedelta64[h]")
INIT = np.datetime64("2024-01-01") + np.arange(N_CASE) * np.timedelta64(12, "h")


def _score(value: float, *, gap: float = 0.2, noise: float = 0.05, seed=0, **attrs):
    """One score variable: per-case values for two sources, `gap` apart.

    `noise` is the part of the difference that is *not* shared between the two
    sources, so it is what decides how significant the gap comes out.
    """
    rng = np.random.default_rng(seed)
    shape = (1, N_CASE, len(LEAD))
    ctl = value + rng.normal(0.0, 0.3 * abs(value) + 0.1, shape)
    exp = ctl + gap + rng.normal(0.0, noise, shape)
    return xr.DataArray(
        np.stack([ctl, exp], axis=1),
        dims=["truth_source", "forecast_source", "init_time", "lead_time"],
        coords=dict(
            truth_source=["analysis"],
            forecast_source=["ctl", "exp"],
            init_time=INIT,
            lead_time=LEAD,
        ),
        attrs=attrs,
    )


def _dataset(**variables):
    return xr.Dataset(variables)


def _card(ds, **kwargs):
    kwargs.setdefault("n_resamples", 300)
    return build_layout(
        ds,
        control="ctl",
        experiment="exp",
        rows=["truth_source", "variable"],
        columns=["metric"],
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# the split rule
# --------------------------------------------------------------------------- #
def test_split_takes_the_last_dot_so_a_metric_may_carry_parameters():
    assert split_name("rmse.2t") == ("rmse", "2t")
    assert split_name("seeps.v1.5.tp") == ("seeps.v1.5", "tp")


@pytest.mark.parametrize("name", ["rmse", ".2t", "rmse.", ""])
def test_a_name_that_is_not_metric_dot_variable_is_refused(name):
    """Guessing would mislabel the card, so it raises instead."""
    with pytest.raises(ValueError, match="metric"):
        split_name(name)


# --------------------------------------------------------------------------- #
# what the dotted naming exists for
# --------------------------------------------------------------------------- #
def test_units_are_per_metric_not_per_variable():
    """ACC of a temperature is dimensionless; RMSE of it is in K.

    With the metric on a coordinate there was one `units` attribute per variable
    and the dimensionless metric was silently labelled K on both drill-down axes.
    """
    ds = _dataset(
        **{"rmse.2t": _score(2.0, units="K"), "acc.2t": _score(0.8, units="1")}
    )
    lay = _card(ds, metric_polarity={"acc": "higher_is_better"})
    units = {
        lay.rows[r].key + lay.columns[c].key: cell.units
        for r, c, cell in lay.iter_cells()
    }
    assert units[("analysis", "2t", "rmse")] == "K"
    assert units[("analysis", "2t", "acc")] == "1"


def test_metric_coverage_may_be_ragged():
    """CRPS needs an ensemble, so it simply has no variable for the field without one.

    A metric coordinate would force the full cross product and NaN-fill the gap;
    here the empty crossing is dropped from the card instead.
    """
    ds = _dataset(
        **{
            "rmse.2t": _score(2.0, units="K"),
            "rmse.msl": _score(90.0, units="Pa"),
            "crps.2t": _score(1.0, units="K"),
        }
    )
    lay = _card(ds)
    present = {
        (lay.rows[r].key[-1], lay.columns[c].key[-1]) for r, c, _ in lay.iter_cells()
    }
    assert present == {("2t", "rmse"), ("msl", "rmse"), ("2t", "crps")}


def test_a_weatherbenchx_shaped_dataset_ingests_unchanged():
    """The naming is WeatherBench-X's, so its output should need no transformation.

    ``AggregationState.metric_values`` builds each output name as
    ``f'{metric_name}.{var_name}'``, and ``Aggregator.reduce_dims`` is a required
    argument, so leaving ``init_time`` out of it keeps the per-case axis this
    package wants. Both conventions are replicated here rather than imported, so
    this test cannot break on an unrelated WBX release.
    """
    names = ["rmse.2t", "mae.2t", "rmse.10u", "bias.10u"]
    ds = _dataset(**{n: _score(1.5, units="K") for n in names})
    lay = _card(ds, metric_polarity={"bias": "lower_is_better"})
    assert {lay.rows[r].key[-1] for r in range(lay.stats.n_rows)} == {"2t", "10u"}
    assert [c.key[-1] for c in lay.columns] == ["rmse", "mae", "bias"]


def test_a_variable_may_omit_a_dimension_that_does_not_apply_to_it():
    """`msl` has no pressure level, so it is one row with a blank level cell
    rather than seven rows of NaN."""
    levels = np.array([500.0, 850.0])
    ds = _dataset(
        **{
            "rmse.z": _score(40.0, units="m").expand_dims(level=levels),
            "rmse.msl": _score(90.0, units="Pa"),
        }
    )
    lay = build_layout(
        ds,
        control="ctl",
        experiment="exp",
        rows=["truth_source", "variable", "level"],
        columns=["metric"],
        n_resamples=100,
    )
    by_var: dict[str, list] = {}
    for r, _, cell in lay.iter_cells():
        by_var.setdefault(lay.rows[r].key[1], []).append(cell)
    assert len(by_var["z"]) == 2 and len(by_var["msl"]) == 1
    assert by_var["msl"][0].units == "Pa"
    assert [r.key[-1] for r in lay.rows if r.key[1] == "msl"] == [None]


# --------------------------------------------------------------------------- #
# reserved names vs. everything else
# --------------------------------------------------------------------------- #
def _seasonal():
    """A dataset whose only groupings are names the package has never heard of."""
    rng = np.random.default_rng(0)
    coords = dict(
        forecast_source=["ctl", "exp"],
        season=["DJF", "JJA"],
        threshold=[1.0, 10.0],
        lead_time=LEAD,
        init_time=INIT,
    )
    a = rng.uniform(1.0, 3.0, tuple(len(v) for v in coords.values()))
    return xr.Dataset(
        {
            "rmse.tp": xr.DataArray(
                a, dims=list(coords), coords=coords, attrs=dict(units="mm")
            )
        }
    )


def test_an_unreserved_dimension_is_just_an_axis():
    """`season` and `threshold` mean nothing to the package, and that is the point.

    Only the reserved names are consumed; anything else nests on the card exactly
    like `level` or `spatial_region` do, with no registration anywhere.
    """
    lay = build_layout(
        _seasonal(),
        control="ctl",
        experiment="exp",
        rows=["season", "variable"],
        columns=["threshold", "metric"],
        n_resamples=100,
    )
    assert [r.key for r in lay.rows] == [("DJF", "tp"), ("JJA", "tp")]
    assert [c.key for c in lay.columns] == [(1.0, "rmse"), (10.0, "rmse")]


def test_an_unplaced_dimension_is_an_error_naming_a_remedy_that_works():
    """Averaging over it silently would look entirely normal and be wrong."""
    ds = _seasonal()
    with pytest.raises(KeyError, match=r"ds\.sel\(season=\.\.\.\)") as excinfo:
        build_layout(
            ds, control="ctl", experiment="exp", rows=["variable"], columns=["metric"]
        )
    assert "season" in str(excinfo.value) and "threshold" in str(excinfo.value)

    lay = build_layout(
        ds.sel(season="DJF", threshold=1.0),
        control="ctl",
        experiment="exp",
        rows=["variable"],
        columns=["metric"],
        n_resamples=100,
    )
    assert [r.key for r in lay.rows] == [("tp",)]


def test_init_time_and_forecast_source_need_no_home_on_an_axis():
    """Both are consumed rather than laid out -- one by the bootstrap, one by
    differencing -- so neither trips the unassigned-dimension check."""
    lay = _card(_dataset(**{"rmse.2t": _score(2.0, units="K")}), n_resamples=100)
    assert lay.stats.n_cells_present == 1


def test_inference_places_unknown_dimensions_on_the_columns_in_order():
    """Documented behaviour, and it has to be deterministic: `test_determinism`
    renders the same card in a fresh process and demands identical bytes."""
    lay = build_layout(_seasonal(), control="ctl", experiment="exp", n_resamples=100)
    assert lay.row_dims == ("variable",)
    assert lay.column_dims == ("metric", "season", "threshold")


@pytest.mark.parametrize("reserved", ["variable", "metric"])
def test_a_produced_name_cannot_be_an_input_dimension(reserved):
    """`variable` and `metric` are outputs of the name split, never inputs."""
    ds = _dataset(**{"rmse.2t": _score(2.0, units="K")}).expand_dims({reserved: [0]})
    with pytest.raises(ValueError, match=rf"already has '{reserved}'"):
        _card(ds)


# --------------------------------------------------------------------------- #
# what the package computes, now that it computes it
# --------------------------------------------------------------------------- #
def test_significance_is_reported_at_the_highest_level_that_holds():
    """"Significant at 99.7%" and "significant at 68%" are different claims.

    Two variables with the same gap between the sources but different unshared
    noise: the quiet one clears every level, the noisy one less.
    """
    ds = _dataset(
        **{
            "rmse.quiet": _score(2.0, gap=0.2, noise=0.02, seed=1, units="K"),
            "rmse.noisy": _score(2.0, gap=0.2, noise=2.5, seed=2, units="K"),
        }
    )
    lay = _card(ds, confidence_levels=(0.68, 0.95, 0.997), n_resamples=800)
    at = {
        lay.rows[r].key[-1]: [s.significant_at for s in cell.steps]
        for r, _, cell in lay.iter_cells()
    }
    assert set(at["quiet"]) == {0.997}
    quiet = [x or 0.0 for x in at["quiet"]]
    noisy = [x or 0.0 for x in at["noisy"]]
    assert all(n <= q for n, q in zip(noisy, quiet))
    assert min(noisy) < 0.997, f"the noisy variable should clear less: {at}"


def test_swapping_control_and_experiment_negates_the_card():
    """The paired difference is antisymmetric, and now by construction rather
    than by a code path that had to repair a half-filled input."""
    ds = _dataset(**{"rmse.2t": _score(2.0, units="K")})
    kw = dict(
        rows=["truth_source", "variable"], columns=["metric"], n_resamples=300, seed=0
    )
    a = build_layout(ds, control="ctl", experiment="exp", **kw)
    b = build_layout(ds, control="exp", experiment="ctl", **kw)
    for sa, sb in zip(
        a.sel(truth_source="analysis", variable="2t", metric="rmse").steps,
        b.sel(truth_source="analysis", variable="2t", metric="rmse").steps,
    ):
        assert sa.value == pytest.approx(-sb.value)
        assert sa.value_lower == pytest.approx(-sb.value_upper)
        assert sa.value_upper == pytest.approx(-sb.value_lower)
        assert sa.significant_at == sb.significant_at


def test_a_percentage_confidence_level_is_refused():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        _card(_dataset(**{"rmse.2t": _score(2.0)}), confidence_levels=(68, 95))


def test_a_dataset_with_no_init_time_is_read_as_already_collapsed():
    """Values that are already means are a legitimate input: you get a card with
    no error bars. Requiring the axis would make a caller fabricate one just to
    say "these are the numbers"."""
    ds = _dataset(**{"rmse.2t": _score(2.0, units="K")}).mean(
        "init_time", keep_attrs=True
    )
    lay, report = _card(ds, return_validation_report=True)

    assert any("already-collapsed means" in w for w in report.warnings), report
    assert lay.confidence_levels == ()
    steps = [s for _, _, cell in lay.iter_cells() for s in cell.steps]
    assert steps and all(s.value is not None for s in steps)
    assert all(s.control_lower is None and s.significant_at is None for s in steps)


def test_the_minimal_readme_example_renders_as_documented():
    """Two variables, one metric, and nothing else: no `truth_source`, no
    `level`, no `spatial_region`. This is the second listing in README.md."""
    rng = np.random.default_rng(0)

    def series(scale: float, units: str):
        a = rng.uniform(0.9, 1.3, (2, N_CASE, len(LEAD))) * scale
        return xr.DataArray(
            a,
            dims=["forecast_source", "init_time", "lead_time"],
            coords=dict(
                forecast_source=["IFS-HRES", "GraphCast"],
                init_time=INIT,
                lead_time=LEAD,
            ),
            attrs=dict(units=units),
        )

    ds = _dataset(**{"rmse.2t": series(1.2, "K"), "rmse.msl": series(80.0, "Pa")})
    lay = build_layout(ds, control="IFS-HRES", experiment="GraphCast", n_resamples=200)
    assert lay.row_dims == ("variable",)
    assert lay.column_dims == ("metric",)
    assert [r.key for r in lay.rows] == [("2t",), ("msl",)]
    assert [c.key for c in lay.columns] == [("rmse",)]
    assert lay.lead_labels == ("T+24", "T+48")
    assert lay.sel(variable="msl", metric="rmse").units == "Pa"
