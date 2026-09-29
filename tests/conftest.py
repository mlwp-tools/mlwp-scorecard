"""Shared fixtures.

The synthetic data is built in :mod:`synthetic`: a toy gridded reanalysis, scored
against persistence (baseline) and persistence-plus-a-random-walk (forecast source).
The expected answer is therefore known in advance — the forecast source must be worse, and
increasingly so with lead time — which is what makes these tests meaningful rather
than merely self-consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import make_verification_dataset  # noqa: E402

ROWS = ["truth_source", "variable", "level"]
COLUMNS = ["spatial_region", "metric"]


@pytest.fixture(scope="session")
def verification() -> "xr.Dataset":  # noqa: F821
    """A small verification-summary dataset with a known answer."""
    return make_verification_dataset(n_case=48, drift=0.25, seed=3)


@pytest.fixture(scope="session")
def layout(verification):
    """The resolved layout for :func:`verification`."""
    from mlwp_scorecards.api import build_layout

    return build_layout(
        verification,
        colour_relative_to="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
        rows=ROWS,
        columns=COLUMNS,
        title="test card",
        n_resamples=200,
    )
