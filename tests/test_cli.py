"""CLI behaviour."""

from __future__ import annotations

import pytest

from mlwp_scorecards.cli import main


@pytest.fixture(scope="module")
def netcdf(tmp_path_factory, request):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from synthetic import make_verification_dataset

    ds = make_verification_dataset(n_case=80, n_boot=40, drift=0.25, seed=3)
    ds["lead_time"] = ds["lead_time"].astype("timedelta64[ns]")
    p = tmp_path_factory.mktemp("cli") / "v.nc"
    ds.to_netcdf(p)
    return p


def test_renders_both_formats(netcdf, tmp_path):
    out_html, out_png = tmp_path / "c.html", tmp_path / "c.png"
    rc = main(
        [
            str(netcdf),
            "--baseline",
            "persistence",
            "--select",
            "forecast_source=drifting-persistence",
            "--html-path",
            str(out_html),
            "--image-path",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_html.stat().st_size > 2000
    assert out_png.stat().st_size > 2000


def test_too_few_cases_for_blocks_is_refused_naming_the_flag(netcdf, tmp_path):
    """An iid resample is the caller's choice, and the error says how to make it."""
    import xarray as xr
    from loguru import logger

    short = tmp_path / "short.nc"
    with xr.open_dataset(netcdf) as ds:
        ds.isel(init_time=slice(0, 30)).to_netcdf(short)
    out = tmp_path / "c.html"
    argv = [str(short), "--baseline", "persistence", "--html-path", str(out)]

    logged = []
    sink = logger.add(lambda m: logged.append(m.record["message"]), level="ERROR")
    try:
        assert main(argv) == 1
    finally:
        logger.remove(sink)
    assert not out.exists()
    assert any("--bootstrap iid" in m for m in logged), logged
    assert main(argv + ["--bootstrap", "iid", "--n-resamples", "50"]) == 0


@pytest.fixture(scope="module")
def four_sources(tmp_path_factory):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_sources import _dataset

    p = tmp_path_factory.mktemp("cli") / "four.nc"
    _dataset().to_netcdf(p)
    return p


@pytest.mark.parametrize(
    "select",
    [
        ["--select", "forecast_source=a,b,c"],
        ["--select", "forecast_source=c,..."],
        [],  # the default: every source but the baseline
    ],
)
def test_several_forecast_sources_give_a_block_of_rows_each(
    four_sources, tmp_path, select
):
    out = tmp_path / "c.html"
    argv = [str(four_sources), "--baseline", "base", "--html-path", str(out)]
    argv += ["--cases", "pairwise", "--n-resamples", "50", "--bootstrap", "iid"]
    assert main(argv + select) == 0
    assert out.read_text().count('<i class="b') == 3 * 2 * 4  # sources x vars x leads


def _boxes(netcdf, tmp_path, *select, rows="truth_source,variable,level"):
    out = tmp_path / "c.html"
    argv = [str(netcdf), "--baseline", "persistence", "--html-path", str(out)]
    argv += ["--rows", rows, "--columns", "spatial_region,metric"]
    for s in select:
        argv += ["--select", s]
    assert main(argv + ["--n-resamples", "50"]) == 0
    return out.read_text().count('<i class="b')


def test_select_values_are_cast_to_the_coordinate_type(netcdf, tmp_path):
    """`level=500` must reach xarray as 500.0, not the string '500'."""
    every = _boxes(netcdf, tmp_path)
    one_level = _boxes(netcdf, tmp_path, "level=500", "variable=z,t")
    assert 0 < one_level < every


def test_select_parsing():
    """No comma is one value; commas a list; a trailing comma a list of one."""
    import numpy as np
    import xarray as xr

    from mlwp_scorecards.cli import _parse_select

    ds = xr.Dataset(
        coords=dict(
            level=[500.0, 850.0],
            region=["europe", "n.hem"],
            init_time=np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
        )
    )
    got = _parse_select(
        [
            "level=500",
            "region=europe,",
            "forecast_source=c, ...",
            "init_time=2024-01-02",
            "variable=2t",
        ],
        ds,
    )
    assert got == {
        "level": 500.0,
        "region": ["europe"],
        "forecast_source": ["c", ...],
        "init_time": np.datetime64("2024-01-02"),
        "variable": "2t",
    }
    assert isinstance(got["level"], float)


@pytest.mark.parametrize(
    "select", [["nope"], ["=1"], ["level="], ["level=abc"], ["level=1", "level=2"]]
)
def test_a_malformed_select_exits_nonzero(netcdf, tmp_path, select):
    out = tmp_path / "c.html"
    argv = [str(netcdf), "--baseline", "persistence", "--html-path", str(out)]
    for s in select:
        argv += ["--select", s]
    assert main(argv) == 1
    assert not out.exists()


def test_open_opens_every_file_written(netcdf, tmp_path, monkeypatch):
    import mlwp_scorecards.cli as cli

    opened = []
    monkeypatch.setattr(cli, "_open", opened.append)
    html, png = tmp_path / "c.html", tmp_path / "c.png"
    argv = [str(netcdf), "--baseline", "persistence", "--open"]
    argv += ["--n-resamples", "50"]
    assert main(argv + ["--html-path", str(html), "--image-path", str(png)]) == 0
    assert opened == [html, png]


def test_neither_a_baseline_nor_values_is_a_clear_error(netcdf, tmp_path):
    """Both are optional, but a card with neither would have nothing on it."""
    out = tmp_path / "c.html"
    assert main([str(netcdf), "--html-path", str(out)]) == 1
    assert not out.exists()


@pytest.mark.parametrize("baseline", [["--baseline", "persistence"], []])
def test_show_values_renders_with_and_without_a_baseline(netcdf, tmp_path, baseline):
    out_html, out_png = tmp_path / "c.html", tmp_path / "c.png"
    argv = [str(netcdf), "--show-values", "--n-resamples", "50"]
    argv += ["--html-path", str(out_html), "--image-path", str(out_png)] + baseline
    assert main(argv) == 0
    assert out_html.stat().st_size > 2000 and out_png.stat().st_size > 2000


def test_selecting_a_dimension_the_dataset_lacks_exits_nonzero(netcdf, tmp_path):
    out = tmp_path / "c.html"
    argv = [str(netcdf), "--baseline", "persistence", "--html-path", str(out)]
    assert main(argv + ["--select", "nonsuch=1"]) == 1
    assert not out.exists()


def test_explicit_axes_are_honoured(netcdf, tmp_path):
    out = tmp_path / "c.html"
    rc = main(
        [
            str(netcdf),
            "--baseline",
            "persistence",
            "--select",
            "forecast_source=drifting-persistence",
            "--rows",
            "truth_source,variable,level",
            "--columns",
            "metric,spatial_region",
            "--html-path",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


@pytest.mark.parametrize(
    "flag, name",
    [("--image-path", "c.txt"), ("--image-path", "c.html"), ("--html-path", "c.png")],
)
def test_an_output_suffix_that_contradicts_its_flag_exits_nonzero(
    netcdf, tmp_path, flag, name
):
    """A usage error should be a clear message and an exit code, not a traceback."""
    rc = main(
        [
            str(netcdf),
            "--baseline",
            "persistence",
            "--select",
            "forecast_source=drifting-persistence",
            flag,
            str(tmp_path / name),
        ]
    )
    assert rc == 1
    assert not (tmp_path / name).exists()


def test_no_output_at_all_exits_nonzero(netcdf):
    rc = main(
        [
            str(netcdf),
            "--baseline",
            "persistence",
            "--select",
            "forecast_source=drifting-persistence",
        ]
    )
    assert rc == 1


@pytest.fixture(scope="module")
def zarr_store(tmp_path_factory):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from synthetic import make_verification_dataset

    ds = make_verification_dataset(n_case=80, n_boot=40, drift=0.25, seed=3)
    p = tmp_path_factory.mktemp("cli") / "v.zarr"
    ds.to_zarr(p)
    return p


def test_reads_zarr_as_well_as_netcdf(zarr_store, tmp_path):
    """Both documented input formats, not just the one the other tests use."""
    out = tmp_path / "c.html"
    rc = main(
        [
            str(zarr_store),
            "--baseline",
            "persistence",
            "--select",
            "forecast_source=drifting-persistence",
            "--html-path",
            str(out),
        ]
    )
    assert rc == 0
    assert out.stat().st_size > 2000


def test_unknown_input_format_exits_nonzero(tmp_path):
    """A CSV or a mistyped path should say so, not surface as whatever error
    the engine xarray guessed happens to raise."""
    bad = tmp_path / "v.csv"
    bad.write_text("not,a,dataset\n")
    out = tmp_path / "c.html"
    rc = main(
        [
            str(bad),
            "--baseline",
            "persistence",
            "--select",
            "forecast_source=drifting-persistence",
            "--html-path",
            str(out),
        ]
    )
    assert rc == 1
    assert not out.exists()
