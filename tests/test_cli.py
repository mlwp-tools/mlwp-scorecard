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
            "--control",
            "persistence",
            "--experiment",
            "drifting-persistence",
            "-o",
            str(out_html),
            "-o",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_html.stat().st_size > 2000
    assert out_png.stat().st_size > 2000


def test_validate_only_writes_nothing(netcdf, tmp_path):
    out = tmp_path / "c.html"
    rc = main(
        [
            str(netcdf),
            "--control",
            "persistence",
            "--experiment",
            "drifting-persistence",
            "-o",
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
            "--control",
            "persistence",
            "--experiment",
            "drifting-persistence",
            "--rows",
            "truth_source,variable,level",
            "--columns",
            "metric,spatial_region",
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_unknown_output_format_exits_nonzero(netcdf, tmp_path):
    """A usage error should be a clear message and an exit code, not a traceback."""
    rc = main(
        [
            str(netcdf),
            "--control",
            "persistence",
            "--experiment",
            "drifting-persistence",
            "-o",
            str(tmp_path / "c.txt"),
        ]
    )
    assert rc == 1
    assert not (tmp_path / "c.txt").exists()


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
            "--control",
            "persistence",
            "--experiment",
            "drifting-persistence",
            "-o",
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
            "--control",
            "persistence",
            "--experiment",
            "drifting-persistence",
            "-o",
            str(out),
        ]
    )
    assert rc == 1
    assert not out.exists()
