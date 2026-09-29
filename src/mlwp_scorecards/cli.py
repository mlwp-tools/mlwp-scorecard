"""CLI for rendering scorecards from a verification-summary dataset."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from .api import ScoreCard

#: The only input formats. Anything else is refused by name rather than handed
#: to ``xr.open_dataset`` to be sniffed: a mistyped path or a CSV should say so,
#: not surface as whatever error the guessed engine happens to raise.
_INPUT_SUFFIXES = {".nc", ".nc4", ".cdf", ".zarr"}

_STATIC_SUFFIXES = {".png", ".pdf", ".svg", ".eps", ".jpg", ".jpeg", ".tif", ".tiff"}
_HTML_SUFFIXES = {".html", ".htm"}


def _output_paths(
    html_path: str | Path | None,
    image_path: Sequence[str | Path] | None,
) -> list[Path]:
    """Check the requested outputs before any work is done, and order them.

    A suffix that contradicts the flag it was passed to is refused rather than
    re-guessed: ``--html-path card.png`` is far more likely a slip than a request
    for a PNG, and silently writing one would hide it.
    """
    if html_path is None and not image_path:
        raise ValueError(
            "nothing to write: pass --html-path, --image-path, or both "
            "(or --validate-only)"
        )
    paths = []
    if html_path is not None:
        path = Path(html_path)
        if path.suffix.lower() not in _HTML_SUFFIXES:
            raise ValueError(f"--html-path {str(html_path)!r} does not end in .html")
        paths.append(path)
    for img in image_path or []:
        path = Path(img)
        if path.suffix.lower() not in _STATIC_SUFFIXES:
            raise ValueError(
                f"--image-path {str(img)!r}: expected one of "
                f"{', '.join(sorted(_STATIC_SUFFIXES))}"
            )
        paths.append(path)
    return paths


def _write(score_card: ScoreCard, path: Path, *, dpi: int) -> Path:
    """Write one output, its format by suffix (already checked by
    :func:`_output_paths`)."""
    if path.suffix.lower() in _HTML_SUFFIXES:
        path.write_text(score_card.to_html(), encoding="utf-8")
        return path
    from .render.static import save_figure

    return save_figure(score_card.to_figure(), path, dpi=dpi)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(
        prog="mlwp.make_scorecard",
        description=(
            "Render a weather forecasting scorecard from pre-computed verification "
            "statistics. Compares forecast sources with a baseline source, all "
            "scored against a common truth."
        ),
    )
    p.add_argument(
        "dataset",
        help=f"verification summary ({', '.join(sorted(_INPUT_SUFFIXES))})",
    )
    p.add_argument(
        "--select",
        action="append",
        metavar="DIM=V1,V2",
        help=(
            "select along a coordinate; repeatable. One value picks it and drops the "
            "dimension (unless it is in --rows/--columns); several, comma-separated, "
            "keep it in that order, with '...' for the rest; a trailing comma makes "
            "a one-value list. e.g. --select forecast_source=GraphCast,... "
            "--select level=500 --select spatial_region=europe,n.hem"
        ),
    )
    p.add_argument(
        "--colour-relative-to",
        metavar="NAME",
        help=(
            "colour each box by its difference from this baseline source, and mark "
            "significance. Leave it out, with --show-values, for a card of each "
            "source's own scores"
        ),
    )
    p.add_argument(
        "--show-values",
        action="store_true",
        help=(
            "print each source's own score in its boxes; with a baseline, also show "
            "the baseline as a grey row of its own scores"
        ),
    )
    p.add_argument(
        "--cases",
        default="common",
        choices=("common", "pairwise"),
        help=(
            "forecast cases each comparison rests on: those every source scored "
            "(common, the default), or those each shares with the baseline"
        ),
    )
    p.add_argument(
        "--html-path",
        metavar="PATH",
        help="write the interactive page here (.html)",
    )
    p.add_argument(
        "--image-path",
        action="append",
        metavar="PATH",
        help="write the static figure here; repeatable. Format from the suffix: "
        ".png, .pdf, .svg, ...",
    )
    p.add_argument("--rows", help="comma-separated coordinates to nest on the rows")
    p.add_argument(
        "--columns", help="comma-separated coordinates to nest on the columns"
    )
    p.add_argument(
        "--cell", default="lead_time", help="coordinate drawn inside each cell"
    )
    p.add_argument(
        "--metric-polarity",
        action="append",
        metavar="NAME=POLARITY",
        help="polarity for an unknown metric, e.g. my_score=higher_is_better",
    )
    p.add_argument(
        "--scheme",
        default="cvd",
        choices=("cvd", "ecmwf"),
        help="colour scheme (default: cvd, colour-vision-safe)",
    )
    p.add_argument(
        "--bootstrap",
        default="moving-block",
        choices=("moving-block", "iid"),
        help=(
            "how to resample forecast cases. Consecutive forecasts share weather, "
            "so iid over-marks significance badly (default: moving-block)"
        ),
    )
    p.add_argument(
        "--block-length",
        type=int,
        default=None,
        metavar="N",
        help="block length in forecast CASES, not hours (default: from the cadence)",
    )
    p.add_argument("--n-resamples", type=int, default=2000, metavar="N")
    p.add_argument(
        "--confidence-level",
        action="append",
        type=float,
        metavar="C",
        help="a fraction such as 0.95; repeatable (default: 0.68 0.95 0.997)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="fixed by default, so two runs on one file agree",
    )
    p.add_argument("--title", default="")
    p.add_argument("--subtitle", default="")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--strict", action="store_true", help="treat warnings as errors")
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="report problems and exit without rendering",
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="open each file written, in the system's default viewer",
    )
    return p


def _open(path: Path) -> None:
    """Open a file in the system's default viewer, without waiting for it."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform == "win32":  # pragma: no cover
        os.startfile(path)  # noqa: S606
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _cast(ds: xr.Dataset, dim: str, token: str) -> Any:
    """A command-line string as a value of the ``dim`` coordinate.

    Selection compares against the coordinate's own values, so ``level=500`` has to
    arrive as a number and ``init_time=2024-01-01`` as a date, not as strings.
    """
    if token == "...":
        return ...
    if dim not in ds.coords:  # `variable` and `metric` are names, not coordinates
        return token
    kind = ds[dim].dtype.kind
    if kind in "iu":
        return int(token)
    if kind == "f":
        return float(token)
    if kind == "M":
        return np.datetime64(token)
    if kind == "m":
        return pd.Timedelta(token).to_timedelta64()
    return token


def _parse_select(items: Sequence[str], ds: xr.Dataset) -> dict[str, Any]:
    """``DIM=V1,V2`` items as a ``select=`` mapping.

    No comma is a single value; commas make a list, and a trailing comma makes a
    list of one, which keeps the dimension where a single value would drop it.

    Raises
    ------
    ValueError
        If an item is not ``DIM=VALUE``, names a dimension twice, or a value does
        not parse as the coordinate's type.
    """
    out: dict[str, Any] = {}
    for item in items:
        dim, eq, raw = item.partition("=")
        dim = dim.strip()
        if not eq or not dim or not raw.strip():
            raise ValueError(f"--select {item!r}: expected DIM=VALUE[,VALUE...]")
        if dim in out:
            raise ValueError(f"--select names {dim!r} twice")
        try:
            if "," in raw:
                tokens = [t.strip() for t in raw.split(",") if t.strip()]
                out[dim] = [_cast(ds, dim, t) for t in tokens]
            else:
                out[dim] = _cast(ds, dim, raw.strip())
        except ValueError as e:
            raise ValueError(f"--select {item!r}: {e}") from None
    return out


def _build(
    ds: xr.Dataset,
    args: argparse.Namespace,
    select: dict[str, Any],
    polarity: dict[str, str],
) -> ScoreCard:
    """Build the :class:`ScoreCard` the parsed command line asks for."""
    return ScoreCard(
        ds,
        colour_relative_to=args.colour_relative_to,
        show_values=args.show_values,
        select=select or None,
        cases=args.cases,
        rows=_split(args.rows),
        columns=_split(args.columns),
        cell=args.cell,
        bootstrap=args.bootstrap,
        block_length=args.block_length,
        n_resamples=args.n_resamples,
        seed=args.seed,
        **(
            {"confidence_levels": tuple(args.confidence_level)}
            if args.confidence_level
            else {}
        ),
        metric_polarity=polarity or None,
        scheme=args.scheme,
        title=args.title,
        subtitle=args.subtitle,
        strict=args.strict,
    )


@logger.catch
def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Returns
    -------
    int
        0 on success, 1 if the dataset failed validation.
    """
    args = build_parser().parse_args(argv)

    # Checked before the dataset is even opened: a mistyped suffix should not cost
    # a full bootstrap first. Validating writes nothing, so it needs no output.
    outputs = []
    if not args.validate_only:
        try:
            outputs = _output_paths(args.html_path, args.image_path)
        except ValueError as e:
            logger.error(str(e))
            return 1

    path = Path(args.dataset)
    suffix = path.suffix.lower()
    if suffix not in _INPUT_SUFFIXES:
        logger.error(
            f"cannot read {path.name}: expected one of "
            f"{', '.join(sorted(_INPUT_SUFFIXES))}"
        )
        return 1
    ds = xr.open_zarr(path) if suffix == ".zarr" else xr.open_dataset(path)

    polarity = {}
    for item in args.metric_polarity or []:
        name, _, pol = item.partition("=")
        polarity[name] = pol

    # A bad selection is a usage error: say so and exit, rather than a traceback.
    try:
        select = _parse_select(args.select or [], ds)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # The card issues what it has to say about the data as UserWarnings; on the
    # command line they belong in the log with everything else. Anything else
    # caught is passed on untouched. A dataset that fails validation raises.
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score_card = _build(ds, args, select, polarity)
    except (KeyError, ValueError) as e:
        logger.error(e.args[0] if e.args else str(e))
        return 1
    for w in caught:
        if w.category is UserWarning:
            logger.warning(str(w.message))
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    s = score_card._layout.stats
    logger.info(
        f"{s.n_rows} rows x {s.n_cols} columns, {s.n_cells_present} populated, "
        f"{s.n_boxes} boxes"
    )
    if args.validate_only:
        return 0

    written = []
    for out in outputs:
        written.append(_write(score_card, out, dpi=args.dpi))
        logger.info(f"wrote {written[-1]}")
    if args.open:
        for path in written:
            _open(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
