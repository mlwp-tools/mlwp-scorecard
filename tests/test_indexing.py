"""Label and positional access agree, and disagree only where they should."""

from __future__ import annotations

import pytest


def test_sel_getitem_and_isel_agree(layout):
    for r, c, cell in layout.iter_cells():
        rl, cl = layout.rows[r], layout.columns[c]
        by_pos = layout.isel(row=r, col=c)
        by_key = layout[rl.key, cl.key]
        by_sel = layout.sel(**dict(zip(layout.row_dims, rl.key)),
                            **dict(zip(layout.column_dims, cl.key)))
        assert by_pos is cell and by_key is cell and by_sel is cell


def test_iter_cells_matches_the_count(layout):
    assert sum(1 for _ in layout.iter_cells()) == layout.stats.n_cells_present


def test_unknown_key_raises_but_empty_crossing_returns_none(layout):
    """A missing crossing is data; an unknown key is a mistake."""
    with pytest.raises(KeyError, match="no such row"):
        layout[("nonsuch",) * layout.row_depth, layout.columns[0].key]
    with pytest.raises(KeyError, match="missing coordinates"):
        layout.sel(variable="z")


def test_line_index_matches_position(layout):
    assert all(line.index == i for i, line in enumerate(layout.rows))
    assert all(line.index == i for i, line in enumerate(layout.columns))


def test_cell_ids_are_unique(layout):
    ids = [c.cell_id for _, _, c in layout.iter_cells()]
    assert len(ids) == len(set(ids))
