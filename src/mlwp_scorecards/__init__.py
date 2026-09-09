"""Weather forecasting scorecards from pre-computed verification statistics.

A scorecard compares two prediction sources, each already scored against a common
truth source, and colours the difference between their scores.

Examples
--------
>>> import xarray as xr
>>> from mlwp_scorecards import make_scorecard
>>> ds = xr.open_dataset("verification_summary.nc")        # doctest: +SKIP
>>> make_scorecard(ds, ["card.html", "card.png"],          # doctest: +SKIP
...                control="IFS-HRES", experiment="GraphCast")
"""

from importlib.metadata import version

from .api import DEFAULT_COLUMNS, DEFAULT_ROWS, build_layout, make_scorecard, render
from .colours import SCHEMES, Polarity
from .ingest import ValidationReport
from .model import Cell, Layout, Line, Step

__all__ = [
    "__version__",
    "make_scorecard",
    "build_layout",
    "render",
    "Layout",
    "Cell",
    "Line",
    "Step",
    "Polarity",
    "SCHEMES",
    "ValidationReport",
    "DEFAULT_ROWS",
    "DEFAULT_COLUMNS",
]
__version__ = version("mlwp-scorecards")
