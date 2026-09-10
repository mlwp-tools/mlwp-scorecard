"""Layout engine: ordering, raggedness, spans, indexing."""

from __future__ import annotations

import pytest

from mlwp_scorecards import build_layout


def test_rows_follow_dataset_coordinate_order(verification, layout):
    """Presentation order is the caller's coordinate order, not alphabetical."""
    want = [str(v) for v in verification["metric"].values]
    got = []
    for c in layout.columns:
        if c.key[1] not in got:
            got.append(str(c.key[1]))
    assert got == want


def test_header_spans_cover_every_line_exactly_once(layout):
    """Every depth's blocks are contiguous and tile the axis."""
    for axis, headers, n in (
        ("rows", layout.row_headers, layout.stats.n_rows),
        ("columns", layout.column_headers, layout.stats.n_cols),
    ):
        for depth, blocks in enumerate(headers):
            assert sum(b.span for b in blocks) == n, (axis, depth)
            covered = []
            for b in blocks:
                covered.extend(range(b.start, b.stop))
            assert covered == list(range(n)), (axis, depth)


def test_line_headers_are_never_ragged(layout):
    """Raggedness is absorbed here so no renderer has to branch on it."""
    for line in layout.rows:
        assert len(line.headers) == layout.row_depth
    for line in layout.columns:
        assert len(line.headers) == layout.column_depth


def test_reordering_rows_reorders_the_nesting(verification):
    a = build_layout(
        verification,
        control="persistence",
        experiment="drifting-persistence",
        rows=["truth_source", "variable", "level"],
        columns=["spatial_region", "metric"],
    )
    b = build_layout(
        verification,
        control="persistence",
        experiment="drifting-persistence",
        rows=["truth_source", "level", "variable"],
        columns=["spatial_region", "metric"],
    )
    assert a.row_dims != b.row_dims
    assert [r.key for r in a.rows] != [r.key for r in b.rows]
    assert a.stats.n_cells_present == b.stats.n_cells_present


def test_unknown_dimension_is_rejected_clearly(verification):
    with pytest.raises(KeyError, match="not a dimension"):
        build_layout(
            verification,
            control="persistence",
            experiment="drifting-persistence",
            rows=["nonsuch"],
            columns=["metric"],
        )


def test_unassigned_dimension_is_rejected_clearly(verification):
    """Silently dropping a dimension would average over it without saying so."""
    with pytest.raises(KeyError, match="assigned to neither"):
        build_layout(
            verification,
            control="persistence",
            experiment="drifting-persistence",
            rows=["variable"],
            columns=["metric"],
        )


def test_unknown_source_is_rejected_clearly(verification):
    with pytest.raises(KeyError, match="experiment="):
        build_layout(verification, control="persistence", experiment="nonsuch")
