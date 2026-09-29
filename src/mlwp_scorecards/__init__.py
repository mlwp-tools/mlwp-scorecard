"""Weather forecasting scorecards from pre-computed verification statistics.

A scorecard compares one or more forecast sources with a baseline source, each
already scored against a common truth source, and colours the paired difference
between their scores.

Examples
--------
>>> import xarray as xr
>>> from pathlib import Path
>>> from mlwp_scorecards import ScoreCard
>>> ds = xr.open_dataset("verification_summary.nc")        # doctest: +SKIP
>>> score_card = ScoreCard(ds, baseline="IFS-HRES",   # doctest: +SKIP
...                        select=dict(forecast_source=["GraphCast"]))
>>> score_card.to_figure().savefig("card.png", dpi=200)        # doctest: +SKIP
>>> Path("card.html").write_text(score_card.to_html())         # doctest: +SKIP
"""

from importlib.metadata import version

from .api import DEFAULT_COLUMNS, DEFAULT_ROWS, ScoreCard
from .render.colours import SCHEMES

__all__ = [
    "__version__",
    "ScoreCard",
    "SCHEMES",
    "DEFAULT_ROWS",
    "DEFAULT_COLUMNS",
]
__version__ = version("mlwp-scorecards")
