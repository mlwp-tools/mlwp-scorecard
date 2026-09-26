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

    ds = make_verification_dataset(n_case=32, n_boot=40, drift=0.25, seed=3)
    ds["lead_time"] = ds["lead_time"].astype("timedelta64[ns]")
    p = tmp_path_factory.mktemp("cli") / "v.nc"
    ds.to_netcdf(p)
    return p


def test_renders_both_formats(netcdf, tmp_path):
    out_html, out_png = tmp_path / "c.html", tmp_path / "c.png"
    rc = main(
        [
            str(netcdf),
            "--relative-to",
            "persistence",
            "--predictions-from",
            "drifting-persistence",
            "--html-path",
            str(out_html),
            "--image-path",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_html.stat().st_size > 2000
    assert out_png.stat().st_size > 2000


def test_forecast_source_is_repeatable(tmp_path):
    """Several forecast sources give one block of rows each."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_sources import _dataset

    p = tmp_path / "v.nc"
    _dataset().to_netcdf(p)
    out = tmp_path / "c.html"
    argv = [str(p), "--relative-to", "base", "--html-path", str(out)]
    for source in ("a", "b", "c"):
        argv += ["--predictions-from", source]
    assert main(argv + ["--cases", "pairwise", "--n-resamples", "50"]) == 0
    assert out.read_text().count('<i class="b') == 3 * 2 * 4  # sources x vars x leads


def test_predictions_from_accepts_an_ellipsis_and_defaults_to_all(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_sources import _dataset

    p = tmp_path / "v.nc"
    _dataset().to_netcdf(p)
    common = [str(p), "--relative-to", "base", "--n-resamples", "50"]
    for extra in (["--predictions-from", "c", "--predictions-from", "..."], []):
        out = tmp_path / f"c{len(extra)}.html"
        assert main(common + extra + ["--html-path", str(out)]) == 0
        assert out.read_text().count('<i class="b') == 3 * 2 * 4


def test_validate_only_needs_no_output(netcdf):
    rc = main(
        [
            str(netcdf),
            "--relative-to",
            "persistence",
            "--predictions-from",
            "drifting-persistence",
            "--validate-only",
        ]
    )
    assert rc == 0


def test_validate_only_writes_nothing(netcdf, tmp_path):
    out = tmp_path / "c.html"
    rc = main(
        [
            str(netcdf),
            "--relative-to",
            "persistence",
            "--predictions-from",
            "drifting-persistence",
            "--html-path",
            str(out),
            "--validate-only",
        ]
    )
    assert rc == 0
    assert not out.exists()


def test_explicit_axes_are_honoured(netcdf, tmp_path):
    out = tmp_path / "c.html"
    rc = main(
        [
            str(netcdf),
            "--relative-to",
            "persistence",
            "--predictions-from",
            "drifting-persistence",
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
            "--relative-to",
            "persistence",
            "--predictions-from",
            "drifting-persistence",
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
            "--relative-to",
            "persistence",
            "--predictions-from",
            "drifting-persistence",
        ]
    )
    assert rc == 1


@pytest.fixture(scope="module")
def zarr_store(tmp_path_factory):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from synthetic import make_verification_dataset

    ds = make_verification_dataset(n_case=32, n_boot=40, drift=0.25, seed=3)
    p = tmp_path_factory.mktemp("cli") / "v.zarr"
    ds.to_zarr(p)
    return p


def test_reads_zarr_as_well_as_netcdf(zarr_store, tmp_path):
    """Both documented input formats, not just the one the other tests use."""
    out = tmp_path / "c.html"
    rc = main(
        [
            str(zarr_store),
            "--relative-to",
            "persistence",
            "--predictions-from",
            "drifting-persistence",
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
            "--relative-to",
            "persistence",
            "--predictions-from",
            "drifting-persistence",
            "--html-path",
            str(out),
        ]
    )
    assert rc == 1
    assert not out.exists()
