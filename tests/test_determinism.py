"""Rendered output must be byte-reproducible.

Without this, any snapshot test is noise and any diff between two runs is
unreadable.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_html_is_byte_identical_across_renders(verification, tmp_path):
    from mlwp_scorecards import make_scorecard

    kw = dict(
        colour_relative_to="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
        title="t",
    )
    a = make_scorecard(verification, html_path=tmp_path / "a.html", **kw)[0]
    b = make_scorecard(verification, html_path=tmp_path / "b.html", **kw)[0]
    assert a.read_bytes() == b.read_bytes()


def test_html_is_identical_in_a_fresh_process(tmp_path):
    """Catches hash-ordering and any accidental use of the clock."""
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(tmp_path.parent.parent)!r})
        sys.path.insert(0, {str((__import__("pathlib").Path(__file__).parent))!r})
        from synthetic import make_verification_dataset
        from mlwp_scorecards import make_scorecard
        ds = make_verification_dataset(n_case=32, drift=0.25, seed=3)
        make_scorecard(ds, html_path=sys.argv[1], colour_relative_to="persistence",
                       select=dict(forecast_source=["drifting-persistence"]), title="t",
                       n_resamples=100)
        """
    )
    outs = []
    for i, seed in enumerate((1, 2)):
        out = tmp_path / f"p{i}.html"
        subprocess.run(
            [sys.executable, "-c", script, str(out)],
            check=True,
            env={"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"},
        )
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]


def test_layout_resolution_is_stable(verification):
    from mlwp_scorecards import build_layout

    kw = dict(
        colour_relative_to="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
    )
    a, b = build_layout(verification, **kw), build_layout(verification, **kw)
    assert [r.key for r in a.rows] == [r.key for r in b.rows]
    assert [c.key for c in a.columns] == [c.key for c in b.columns]
    assert a.stats == b.stats
