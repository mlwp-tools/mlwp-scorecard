"""Behaviour at the size of a real card.

Asserts behaviour, never pixels: a full ECMWF-shaped card is 45 x 30 x 15 and the
question is whether it renders at all, quickly enough, at a legible size.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import xarray as xr

from mlwp_scorecards import build_layout, make_scorecard


def _big_dataset(n_var=9, n_level=5, n_region=10, n_metric=3, n_lead=15) -> xr.Dataset:
    """A card-shaped dataset: 45 rows x 30 columns x 15 lead times."""
    rng = np.random.default_rng(0)
    leads = np.arange(1, n_lead + 1) * 24
    coords = dict(
        truth_source=["analysis"],
        prediction_source=["ctl", "exp"],
        level=np.linspace(50, 1000, n_level),
        spatial_region=[f"r{i}" for i in range(n_region)],
        metric=["rmse", "mae", "spread"][:n_metric],
        lead_time=leads.astype("timedelta64[h]"),
        stat=["mean", "lower", "upper"],
    )
    dims = list(coords)
    shape = tuple(len(v) for v in coords.values())
    out = {}
    for i in range(n_var):
        base = rng.uniform(1, 5, shape)
        base[:, 1] = base[:, 0] * (1 + rng.uniform(-0.2, 0.2, base[:, 0].shape))
        name = f"v{i}"
        out[name] = xr.DataArray(
            base, dims=dims, coords=coords, attrs=dict(units="K", long_name=name)
        )
    return xr.Dataset(out)


@pytest.mark.slow
def test_full_size_card_renders_quickly(tmp_path):
    ds = _big_dataset()
    t0 = time.perf_counter()
    layout = build_layout(
        ds,
        control="ctl",
        experiment="exp",
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
    )
    t_layout = time.perf_counter() - t0

    assert layout.stats.n_rows == 45
    assert layout.stats.n_cols == 30
    assert layout.stats.n_boxes == 45 * 30 * 15
    assert t_layout < 10, f"layout took {t_layout:.1f}s"

    t0 = time.perf_counter()
    html, png = make_scorecard(
        ds,
        [tmp_path / "b.html", tmp_path / "b.png"],
        control="ctl",
        experiment="exp",
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
    )
    t_render = time.perf_counter() - t0
    assert t_render < 60, f"render took {t_render:.1f}s"

    # the reference card of this shape is 7.4 MB; ours should be far under
    assert html.stat().st_size < 3_000_000, html.stat().st_size
    assert png.stat().st_size > 10_000


@pytest.mark.slow
def test_full_size_html_box_count(tmp_path):
    ds = _big_dataset()
    layout = build_layout(
        ds,
        control="ctl",
        experiment="exp",
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
    )
    p = make_scorecard(
        ds,
        tmp_path / "b.html",
        control="ctl",
        experiment="exp",
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
    )[0]
    assert p.read_text().count('<i class="b') == layout.stats.n_boxes
