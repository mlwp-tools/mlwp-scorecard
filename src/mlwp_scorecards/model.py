"""Resolved layout types — the renderer contract.

A :class:`Layout` is the whole resolved scorecard: ordered rows, ordered columns,
the header cells for each nesting depth, the lead times, and a :class:`Cell` for
every populated crossing.

Renderers consume these and nothing else. In particular they never see ``xarray``
and never re-derive colour semantics: every :class:`Step` already carries the
signed ramp ``level`` and its colour ``family``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

Key = tuple[Any, ...]
"""Coordinate values identifying one row or column, in nesting order."""

#: The colour family of a box that is not compared with anything: a baseline's
#: own row, or every box on a card with no baseline. Drawn in the scheme's
#: neutral grey.
NEUTRAL = "neutral"


def format_value(x: float) -> str:
    """Format a score as printed in a box.

    About three significant figures, never an exponent for the sizes scores come
    in.

    The one place a printed value is formatted, so the HTML page and the figure
    cannot disagree, and output stays byte-reproducible.

    Fixed decimals down to 0.1, so a column of values lines up (``0.40`` beside
    ``0.46``, not ``0.4``); only below that does it keep two significant figures.

    Parameters
    ----------
    x : float
        The score.

    Returns
    -------
    str
        The score as text.

    Examples
    --------
    >>> [format_value(v) for v in (433.2, 21.37, 2.071, 0.617, 0.4, 0.00123, -12.34)]
    ['433', '21.4', '2.07', '0.62', '0.40', '0.0012', '-12.3']
    """
    mag = abs(x)
    if mag >= 100:
        return f"{x:.0f}"
    if mag >= 10:
        return f"{x:.1f}"
    if mag >= 0.1:
        return f"{x:.2f}"
    return f"{x:.2g}"


@dataclass(frozen=True, slots=True)
class HeaderCell:
    """One label block on a row or column axis.

    Attributes
    ----------
    dim : str
        The dimension this block labels.
    key : Any
        The coordinate value it labels; None where the dimension does not apply.
    label : str
        The text shown; empty where the dimension does not apply.
    depth : int
        The nesting depth on its axis, 0 outermost.
    start : int
        The index of the first line (row or leaf column) it covers.
    span : int
        The number of consecutive lines it covers.
    tooltip : str or None, optional
        Hover text for the label, if any.
    tint : str or None, optional
        A background colour for the label, if any.
    is_na : bool, optional
        Whether the dimension does not apply to this branch, as for the level
        of a surface variable.
    """

    dim: str
    key: Any
    label: str
    depth: int
    start: int
    span: int
    tooltip: str | None = None
    tint: str | None = None
    is_na: bool = False

    @property
    def stop(self) -> int:
        """One past the last line this block covers.

        Returns
        -------
        int
            ``start + span``.
        """
        return self.start + self.span


@dataclass(frozen=True, slots=True)
class Line:
    """One physical row, or one leaf column.

    Attributes
    ----------
    index : int
        Its position on the axis.
    key : Key
        Its coordinate values, one per axis dimension, in nesting order.
    headers : tuple of HeaderCell
        The header block covering it at each depth, outermost first.
    slug : str
        ``key`` made safe for an identifier, used to build :attr:`Cell.cell_id`.
    """

    index: int
    key: Key
    headers: tuple[HeaderCell, ...]
    slug: str


@dataclass(frozen=True, slots=True)
class Step:
    """One lead time inside a :class:`Cell` — one drawn box.

    Every score is a mean over forecast cases; every field that is None means
    "not available here", never zero.

    Attributes
    ----------
    lead_time : float
        The lead time, in hours.
    value : float or None
        The difference, forecast minus baseline, in the metric's units. None when
        the box is compared with nothing or has no data.
    relative : float or None
        ``value`` relative to the baseline's magnitude, signed so that positive
        means better (or more active) whatever the metric's polarity.
    baseline : float or None
        The baseline's score.
    forecast : float or None
        The forecast source's own score.
    baseline_lower : float or None
        Lower end of the baseline's own interval, at the widest level.
    baseline_upper : float or None
        Upper end of the baseline's own interval, at the widest level.
    forecast_lower : float or None
        Lower end of the forecast source's own interval, at the widest level.
    forecast_upper : float or None
        Upper end of the forecast source's own interval, at the widest level.
    value_lower : float or None
        Lower end of the paired difference interval, at the widest level.
    value_upper : float or None
        Upper end of the paired difference interval, at the widest level.
    n : int or None
        The number of forecast cases the comparison rests on.
    level : int
        The signed ramp level that picks the fill; 0 for a neutral box.
    family : str
        The colour family, or :data:`NEUTRAL` when compared with nothing.
    significant_at : float or None
        The highest confidence level whose paired interval excludes zero, or
        None when none does.
    tooltip : str
        Hover text describing the box.
    text : str, optional
        The formatted score printed in the box.
    """

    lead_time: float
    value: float | None
    relative: float | None
    baseline: float | None
    forecast: float | None
    baseline_lower: float | None
    baseline_upper: float | None
    forecast_lower: float | None
    forecast_upper: float | None
    #: The *paired* difference interval. Not derivable from the two marginals
    #: above: both sources are scored on the same cases, so their errors are
    #: correlated and this is much tighter. It is what decides significance.
    value_lower: float | None
    value_upper: float | None
    n: int | None
    level: int
    family: str
    #: The **tightest** confidence level whose paired interval excludes zero, or
    #: None when even the widest one does not. Graded rather than boolean because
    #: "significant at 99.7%" and "significant at 68%" are different claims, and a
    #: card that shows only the second is over-claiming.
    significant_at: float | None
    tooltip: str
    #: The number printed in the box: this source's own score, formatted once
    #: here by :func:`format_value`. Empty unless the card shows values.
    text: str = ""

    @property
    def significant(self) -> bool:
        """Whether the paired interval excludes zero at any level supplied.

        Returns
        -------
        bool
            True when :attr:`significant_at` is set.
        """
        return self.significant_at is not None

    @property
    def has_data(self) -> bool:
        """Whether there is anything to draw: a difference, or a score of its own.

        A grey box on an uncoloured card, or in the baseline's row, has a score but
        no difference, so ``value is None`` alone would wrongly mark it missing.

        Returns
        -------
        bool
            True when there is a difference or a score of its own.
        """
        return self.value is not None or self.forecast is not None

    @property
    def has_intervals(self) -> bool:
        """Whether either source carries a confidence interval.

        Returns
        -------
        bool
            True when the baseline or the forecast has a lower bound.
        """
        return self.baseline_lower is not None or self.forecast_lower is not None


@dataclass(frozen=True, slots=True)
class Cell:
    """One row crossed with one leaf column.

    Attributes
    ----------
    row : int
        The row's index.
    col : int
        The leaf column's index.
    row_key : Key
        The row's coordinate values.
    col_key : Key
        The column's coordinate values.
    cell_id : str
        A string identifier built from the row and column slugs.
    metric : str
        The metric scored in this cell.
    units : str or None
        The metric's units for this variable, if known.
    steps : tuple of Step
        One box per lead time, in lead-time order.
    forecast_source : str, optional
        The forecast source this cell compares with the baseline.
    is_baseline : bool, optional
        Whether this is the baseline's own row.
    """

    row: int
    col: int
    row_key: Key
    col_key: Key
    cell_id: str
    metric: str
    units: str | None
    steps: tuple[Step, ...]
    #: The forecast source this cell compares with the baseline. Explicit rather
    #: than left in a row or column key, so a renderer never has to find it there.
    forecast_source: str = ""
    #: The baseline's own row, shown in grey when the card shows values. Its boxes
    #: are its own scores; nothing is compared or marked significant in it.
    is_baseline: bool = False

    @property
    def has_data(self) -> bool:
        """Whether any of its boxes has something to draw.

        Returns
        -------
        bool
            True when any step's :attr:`Step.has_data` is.
        """
        return any(s.has_data for s in self.steps)


@dataclass(frozen=True, slots=True)
class LayoutStats:
    """Counts describing a resolved layout.

    Attributes
    ----------
    n_rows : int
        The number of rows.
    n_cols : int
        The number of leaf columns.
    n_cells_possible : int
        Rows times columns.
    n_cells_present : int
        The number of crossings holding a :class:`Cell`.
    n_boxes : int
        The number of boxes in those cells: one per lead time each.
    n_saturated : int
        The number of boxes at or beyond the top of the colour scale.
    n_significant : int
        The number of boxes marked significant at any level.
    """

    n_rows: int
    n_cols: int
    n_cells_possible: int
    n_cells_present: int
    n_boxes: int
    n_saturated: int
    n_significant: int

    @property
    def n_tests(self) -> int:
        """Number of simultaneous comparisons the card displays.

        Returns
        -------
        int
            Populated cells times the boxes in each.
        """
        return self.n_cells_present * (self.n_boxes // max(self.n_cells_present, 1))


@dataclass(frozen=True, slots=True)
class Layout:
    """A fully resolved scorecard, ready to render.

    Attributes
    ----------
    rows : tuple of Line
        The rows, top to bottom.
    columns : tuple of Line
        The leaf columns, left to right.
    row_headers : tuple of tuple of HeaderCell
        The row header blocks, one tuple per depth, outermost first.
    column_headers : tuple of tuple of HeaderCell
        The column header blocks, one tuple per depth, outermost first.
    lead_times : tuple of float
        The lead times inside every cell, in hours.
    lead_labels : tuple of str
        Their labels, such as ``T+24``.
    cells : mapping of (Key, Key) to Cell
        The populated crossings, keyed by ``(row_key, col_key)``.
    row_dims : tuple of str
        The dimensions nested on the rows, outermost first.
    column_dims : tuple of str
        The dimensions nested on the columns, outermost first.
    cell_dim : str
        The dimension laid out inside each cell: the lead time.
    stats : LayoutStats
        Counts describing the layout.
    title : str, optional
        The card's title.
    subtitle : str, optional
        The card's subtitle.
    baseline_source : str, optional
        The source every other is compared with; empty when there is none.
    forecast_sources : tuple of str, optional
        The sources compared with the baseline.
    cases : str, optional
        Which forecast cases each comparison rests on: ``common`` or ``pairwise``.
    confidence_levels : tuple of float, optional
        Every confidence level the data supplied, ascending.
    resampling : str, optional
        The bootstrap method that produced the intervals.
    block_length : int, optional
        The bootstrap block length, in forecast cases.
    n_resamples : int, optional
        The number of bootstrap resamples; 0 when none were drawn.
    seed : int, optional
        The bootstrap's random seed.
    scheme_name : str, optional
        The name of the colour scheme to draw with.
    notes : tuple of str, optional
        Caveats to print on the card.
    show_values : bool, optional
        Whether each box prints its source's own score.
    """

    rows: tuple[Line, ...]
    columns: tuple[Line, ...]
    row_headers: tuple[tuple[HeaderCell, ...], ...]
    column_headers: tuple[tuple[HeaderCell, ...], ...]
    lead_times: tuple[float, ...]
    lead_labels: tuple[str, ...]
    cells: Mapping[tuple[Key, Key], Cell]
    row_dims: tuple[str, ...]
    column_dims: tuple[str, ...]
    cell_dim: str
    stats: LayoutStats
    title: str = ""
    subtitle: str = ""
    baseline_source: str = ""
    forecast_sources: tuple[str, ...] = ()
    #: Which forecast cases each comparison rests on: ``common`` or ``pairwise``.
    cases: str = "common"
    #: Every confidence level the data supplied, ascending.
    confidence_levels: tuple[float, ...] = field(default_factory=tuple)
    #: How the intervals were produced. On the card, not only in a docstring: a
    #: significance claim cannot be checked without knowing what was resampled,
    #: how many times, and from what seed.
    resampling: str = ""
    block_length: int = 1
    n_resamples: int = 0
    seed: int = 0
    scheme_name: str = "cvd"
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: Whether each box prints its source's own score (:attr:`Step.text`).
    show_values: bool = False

    @property
    def coloured(self) -> bool:
        """Whether boxes are coloured by the difference from a baseline.

        When not, every box is neutral and nothing is marked significant.

        Returns
        -------
        bool
            True when there is a baseline.
        """
        return bool(self.baseline_source)

    @property
    def confidence(self) -> float | None:
        """The level the drill-down error bars are drawn at.

        The widest supplied, which is the most conservative choice: the interval a
        reader sees should not be narrower than the evidence for it.

        Returns
        -------
        float or None
            The widest level, or None when no intervals were supplied.
        """
        return max(self.confidence_levels) if self.confidence_levels else None

    @property
    def forecast_label(self) -> str:
        """Return what the legend calls the compared side.

        The source's name, or, when each row or column is a different source, a
        phrase saying so.

        Returns
        -------
        str
            The label.
        """
        if len(self.forecast_sources) == 1:
            return self.forecast_sources[0]
        return "each forecast source"

    # ---- shape -------------------------------------------------------------
    @property
    def row_depth(self) -> int:
        """Number of dimensions nested on the rows.

        Returns
        -------
        int
            The length of :attr:`row_dims`.
        """
        return len(self.row_dims)

    @property
    def column_depth(self) -> int:
        """Number of dimensions nested on the columns.

        Returns
        -------
        int
            The length of :attr:`column_dims`.
        """
        return len(self.column_dims)

    # ---- label access ------------------------------------------------------
    def sel(self, **coords: Any) -> Cell | None:
        """Return the cell at the given coordinate values.

        Parameters
        ----------
        **coords
            One value per row and column dimension, in any order.

        Returns
        -------
        Cell or None
            None when the crossing exists but holds no data.

        Raises
        ------
        KeyError
            If a coordinate is missing, unknown, or names no row or column.
        """
        missing = set(self.row_dims + self.column_dims) - set(coords)
        if missing:
            raise KeyError(f"missing coordinates: {sorted(missing)}")
        extra = set(coords) - set(self.row_dims + self.column_dims)
        if extra:
            raise KeyError(f"not layout dimensions: {sorted(extra)}")
        rkey = tuple(coords[d] for d in self.row_dims)
        ckey = tuple(coords[d] for d in self.column_dims)
        return self[rkey, ckey]

    def __getitem__(self, keys: tuple[Key, Key]) -> Cell | None:
        """Return the cell at a row key and a column key.

        Parameters
        ----------
        keys : tuple of (Key, Key)
            The row's and the column's coordinate values.

        Returns
        -------
        Cell or None
            None when the crossing exists but holds no data.

        Raises
        ------
        KeyError
            If either key names no row or column.
        """
        rkey, ckey = keys
        rkey, ckey = tuple(rkey), tuple(ckey)
        if rkey not in {r.key for r in self.rows}:
            raise KeyError(f"no such row: {rkey}")
        if ckey not in {c.key for c in self.columns}:
            raise KeyError(f"no such column: {ckey}")
        return self.cells.get((rkey, ckey))

    def row(self, **coords: Any) -> Line:
        """Return the row line at the given coordinate values.

        Parameters
        ----------
        **coords
            One value per row dimension.

        Returns
        -------
        Line
            The row.

        Raises
        ------
        KeyError
            If a row dimension is missing, or the values name no row.
        """
        key = tuple(coords[d] for d in self.row_dims)
        for line in self.rows:
            if line.key == key:
                return line
        raise KeyError(f"no such row: {key}")

    def column(self, **coords: Any) -> Line:
        """Return the column line at the given coordinate values.

        Parameters
        ----------
        **coords
            One value per column dimension.

        Returns
        -------
        Line
            The leaf column.

        Raises
        ------
        KeyError
            If a column dimension is missing, or the values name no column.
        """
        key = tuple(coords[d] for d in self.column_dims)
        for line in self.columns:
            if line.key == key:
                return line
        raise KeyError(f"no such column: {key}")

    # ---- positional access, for renderers ----------------------------------
    def isel(self, *, row: int, col: int) -> Cell | None:
        """Return the cell at the given row and column positions.

        Parameters
        ----------
        row : int
            The row's index.
        col : int
            The leaf column's index.

        Returns
        -------
        Cell or None
            None when the crossing holds no data.
        """
        return self.cells.get((self.rows[row].key, self.columns[col].key))

    def iter_cells(self) -> Iterator[tuple[int, int, Cell]]:
        """Yield ``(row, col, cell)`` for every populated crossing, in reading order.

        Yields
        ------
        tuple of (int, int, Cell)
            The row index, the column index, and the cell there.
        """
        for r, rl in enumerate(self.rows):
            for c, cl in enumerate(self.columns):
                cell = self.cells.get((rl.key, cl.key))
                if cell is not None:
                    yield r, c, cell
