"""Rendered output must be byte-reproducible.

Without this, any snapshot test is noise and any diff between two runs is
unreadable.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_html_is_byte_identical_across_renders(verification):
    from mlwp_scorecard import ScoreCard

    kw = dict(
        baseline="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
        title="t",
    )
    a = ScoreCard(verification, **kw).to_html()
    b = ScoreCard(verification, **kw).to_html()
    assert a == b


def test_svg_is_byte_identical_across_renders(verification, tmp_path):
    from mlwp_scorecard import ScoreCard
    from mlwp_scorecard.render.static import save_figure

    kw = dict(
        baseline="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
        title="t",
    )
    a = save_figure(ScoreCard(verification, **kw).to_figure(), tmp_path / "a.svg")
    b = save_figure(ScoreCard(verification, **kw).to_figure(), tmp_path / "b.svg")
    assert a.read_bytes() == b.read_bytes()


def test_html_is_identical_in_a_fresh_process(tmp_path):
    """Catches hash-ordering and any accidental use of the clock."""
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(tmp_path.parent.parent)!r})
        sys.path.insert(0, {str((__import__("pathlib").Path(__file__).parent))!r})
        from synthetic import make_verification_dataset
        from mlwp_scorecard import ScoreCard
        ds = make_verification_dataset(n_case=80, drift=0.25, seed=3)
        score_card = ScoreCard(ds, baseline="persistence",
                               select=dict(forecast_source=["drifting-persistence"]),
                               title="t", n_resamples=100)
        Path(sys.argv[1]).write_text(score_card.to_html(), encoding="utf-8")
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
    from mlwp_scorecard.api import build_layout

    kw = dict(
        baseline="persistence",
        select=dict(forecast_source=["drifting-persistence"]),
    )
    a, b = build_layout(verification, **kw), build_layout(verification, **kw)
    assert [r.key for r in a.rows] == [r.key for r in b.rows]
    assert [c.key for c in a.columns] == [c.key for c in b.columns]
    assert a.stats == b.stats
