"""CLI for rendering scorecards from a verification-summary dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import xarray as xr
from loguru import logger

from .api import _output_paths, build_layout, render

#: The only input formats. Anything else is refused by name rather than handed
#: to ``xr.open_dataset`` to be sniffed: a mistyped path or a CSV should say so,
#: not surface as whatever error the guessed engine happens to raise.
_INPUT_SUFFIXES = {".nc", ".nc4", ".cdf", ".zarr"}


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
        "--predictions-from",
        action="append",
        metavar="NAME",
        help=(
            "forecast source to show; repeatable, in order. '...' stands for every "
            "source not otherwise named (default: all but the baseline)"
        ),
    )
    p.add_argument(
        "--relative-to",
        required=True,
        metavar="NAME",
        help="the baseline every forecast source is compared with",
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
        "--truth-source",
        action="append",
        help="restrict to this truth source; repeatable",
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
    return p


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


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

    layout, report = build_layout(
        ds,
        predictions_from=[
            ... if s == "..." else s for s in (args.predictions_from or ["..."])
        ],
        relative_to=args.relative_to,
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
        truth_source=args.truth_source,
        metric_polarity=polarity or None,
        scheme=args.scheme,
        title=args.title,
        subtitle=args.subtitle,
        strict=args.strict,
        return_validation_report=True,
    )
    for w in report.warnings:
        logger.warning(w)
    if report.has_fails():
        for f in report.fails:
            logger.error(f)
        return 1

    s = layout.stats
    logger.info(
        f"{s.n_rows} rows x {s.n_cols} columns, {s.n_cells_present} populated, "
        f"{s.n_boxes} boxes"
    )
    if args.validate_only:
        return 0

    for out in outputs:
        logger.info(f"wrote {render(layout, out, dpi=args.dpi)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
