"""Behaviour at the size of a real card.

Asserts behaviour, never pixels: a full ECMWF-shaped card is 45 x 30 x 15 and the
question is whether it renders at all, quickly enough, at a legible size.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import xarray as xr

from mlwp_scorecards import ScoreCard
from mlwp_scorecards.api import build_layout
from mlwp_scorecards.render.static import save_figure


def _big_dataset(
    n_var=9, n_level=5, n_region=10, n_metric=3, n_lead=15, n_case=120
) -> xr.Dataset:
    """A card-shaped dataset: 45 rows x 30 columns x 15 lead times.

    Per-case, so the package does the collapse: 324 kB x ``n_case``, which at the
    default is ~40 MB in and ~80 MB once stacked. That is the largest thing in
    the suite, which is why the case count is a parameter.
    """
    rng = np.random.default_rng(0)
    leads = np.arange(1, n_lead + 1) * 24
    metrics = ["rmse", "mae", "spread"][:n_metric]
    coords = dict(
        truth_source=["analysis"],
        forecast_source=["ctl", "exp"],
        level=np.linspace(50, 1000, n_level),
        spatial_region=[f"r{i}" for i in range(n_region)],
        lead_time=leads.astype("timedelta64[h]"),
        init_time=np.datetime64("2024-01-01")
        + np.arange(n_case) * np.timedelta64(12, "h"),
    )
    dims = list(coords)
    shape = tuple(len(v) for v in coords.values())
    out = {}
    for i in range(n_var):
        for metric in metrics:
            base = rng.uniform(1, 5, shape)
            # the experiment tracks the control case by case, which is what makes
            # the paired interval tighter than the two marginals
            base[:, 1] = base[:, 0] * (1 + rng.uniform(-0.2, 0.2, base[:, 0].shape))
            out[f"{metric}.v{i}"] = xr.DataArray(
                base,
                dims=dims,
                coords=coords,
                attrs=dict(units="K", long_name=f"v{i} {metric}"),
            )
    return xr.Dataset(out)


@pytest.mark.slow
def test_full_size_card_renders_quickly(tmp_path):
    ds = _big_dataset()
    t0 = time.perf_counter()
    layout = build_layout(
        ds,
        colour_relative_to="ctl",
        select=dict(forecast_source=["exp"]),
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
        n_resamples=500,
    )
    t_layout = time.perf_counter() - t0

    assert layout.stats.n_rows == 45
    assert layout.stats.n_cols == 30
    assert layout.stats.n_boxes == 45 * 30 * 15
    assert t_layout < 10, f"layout took {t_layout:.1f}s"

    t0 = time.perf_counter()
    score_card = ScoreCard(
        ds,
        colour_relative_to="ctl",
        select=dict(forecast_source=["exp"]),
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
        n_resamples=500,
    )
    page = score_card.to_html()
    png = save_figure(score_card.to_figure(), tmp_path / "b.png")
    t_render = time.perf_counter() - t0
    assert t_render < 60, f"render took {t_render:.1f}s"

    # the reference card of this shape is 7.4 MB; ours should be far under
    assert len(page.encode()) < 3_000_000, len(page.encode())
    assert png.stat().st_size > 10_000


@pytest.mark.slow
def test_full_size_html_box_count():
    ds = _big_dataset()
    score_card = ScoreCard(
        ds,
        colour_relative_to="ctl",
        select=dict(forecast_source=["exp"]),
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
        n_resamples=200,
    )
    assert score_card.to_html().count('<i class="b') == score_card._layout.stats.n_boxes


@pytest.mark.slow
def test_the_bootstrap_is_affordable_at_full_scale():
    """The collapse is now the package's cost, so it is the package's to bound.

    400 cases x 2000 resamples over 20,250 boxes. Written as a gather this is
    ~430 s; written as a matmul, ~3 s. The budget below is what separates those
    two implementations, not a tight fit around the fast one.
    """
    ds = _big_dataset(n_case=400)
    t0 = time.perf_counter()
    layout = build_layout(
        ds,
        colour_relative_to="ctl",
        select=dict(forecast_source=["exp"]),
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
        n_resamples=2000,
    )
    elapsed = time.perf_counter() - t0
    assert layout.stats.n_boxes == 45 * 30 * 15
    assert elapsed < 60, f"aggregation + layout took {elapsed:.1f}s"
