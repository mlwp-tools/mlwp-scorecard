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


@dataclass(frozen=True, slots=True)
class HeaderCell:
    """One label block on a row or column axis."""

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
        """One past the last line this block covers."""
        return self.start + self.span


@dataclass(frozen=True, slots=True)
class Line:
    """One physical row, or one leaf column."""

    index: int
    key: Key
    headers: tuple[HeaderCell, ...]
    slug: str


@dataclass(frozen=True, slots=True)
class Step:
    """One lead time inside a :class:`Cell` — one drawn box."""

    lead_time: float
    value: float | None
    relative: float | None
    control: float | None
    experiment: float | None
    control_lower: float | None
    control_upper: float | None
    experiment_lower: float | None
    experiment_upper: float | None
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

    @property
    def significant(self) -> bool:
        """Whether the paired interval excludes zero at any level supplied."""
        return self.significant_at is not None

    @property
    def has_intervals(self) -> bool:
        """Whether either source carries a confidence interval."""
        return self.control_lower is not None or self.experiment_lower is not None


@dataclass(frozen=True, slots=True)
class Cell:
    """One row crossed with one leaf column."""

    row: int
    col: int
    row_key: Key
    col_key: Key
    cell_id: str
    metric: str
    units: str | None
    steps: tuple[Step, ...]

    @property
    def has_data(self) -> bool:
        return any(s.value is not None for s in self.steps)


@dataclass(frozen=True, slots=True)
class LayoutStats:
    """Counts describing a resolved layout."""

    n_rows: int
    n_cols: int
    n_cells_possible: int
    n_cells_present: int
    n_boxes: int
    n_saturated: int
    n_significant: int

    @property
    def n_tests(self) -> int:
        """Number of simultaneous comparisons the card displays."""
        return self.n_cells_present * (self.n_boxes // max(self.n_cells_present, 1))


@dataclass(frozen=True, slots=True)
class Layout:
    """A fully resolved scorecard, ready to render."""

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
    control: str = ""
    experiment: str = ""
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

    @property
    def confidence(self) -> float | None:
        """The level the drill-down error bars are drawn at.

        The widest supplied, which is the most conservative choice: the interval a
        reader sees should not be narrower than the evidence for it.
        """
        return max(self.confidence_levels) if self.confidence_levels else None

    # ---- shape -------------------------------------------------------------
    @property
    def row_depth(self) -> int:
        return len(self.row_dims)

    @property
    def column_depth(self) -> int:
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
        rkey, ckey = keys
        rkey, ckey = tuple(rkey), tuple(ckey)
        if rkey not in {r.key for r in self.rows}:
            raise KeyError(f"no such row: {rkey}")
        if ckey not in {c.key for c in self.columns}:
            raise KeyError(f"no such column: {ckey}")
        return self.cells.get((rkey, ckey))

    def row(self, **coords: Any) -> Line:
        """Return the row line at the given coordinate values."""
        key = tuple(coords[d] for d in self.row_dims)
        for line in self.rows:
            if line.key == key:
                return line
        raise KeyError(f"no such row: {key}")

    def column(self, **coords: Any) -> Line:
        """Return the column line at the given coordinate values."""
        key = tuple(coords[d] for d in self.column_dims)
        for line in self.columns:
            if line.key == key:
                return line
        raise KeyError(f"no such column: {key}")

    # ---- positional access, for renderers ----------------------------------
    def isel(self, *, row: int, col: int) -> Cell | None:
        """Return the cell at the given row and column positions."""
        return self.cells.get((self.rows[row].key, self.columns[col].key))

    def iter_cells(self) -> Iterator[tuple[int, int, Cell]]:
        """Yield ``(row, col, cell)`` for every populated crossing, in reading order."""
        for r, rl in enumerate(self.rows):
            for c, cl in enumerate(self.columns):
                cell = self.cells.get((rl.key, cl.key))
                if cell is not None:
                    yield r, c, cell
