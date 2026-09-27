"""Static matplotlib backend: PNG, SVG and PDF.

Everything is drawn into a *single* ``Axes`` in a top-left-origin point space that
mirrors the CSS box model, so geometry is shared with the HTML backend. All boxes
become one ``PatchCollection`` rather than one artist each, which keeps a
full-size card to a single draw pass.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..colours import ColourScheme
from ..model import Layout

__all__ = ["Geometry", "render_figure", "render_static"]


@dataclass(frozen=True, slots=True)
class Geometry:
    """Point-space dimensions. Fonts are never scaled below ``font_pt``."""

    box_w: float = 6.0
    box_h: float = 11.0
    box_gap: float = 1.0
    cell_pad: float = 3.0
    row_h: float = 15.0
    label_w: float = 62.0
    head_h: float = 17.0
    font_pt: float = 7.0
    title_pt: float = 12.0

    def cell_w(self, n_steps: int) -> float:
        return n_steps * (self.box_w + self.box_gap) - self.box_gap + 2 * self.cell_pad


#: Boxes wide enough for a printed value, and the size of that value.
VALUES_BOX_W = 22.0
VALUES_PT = 5.2


def _label_widths(layout: Layout, geom: Geometry) -> list[float]:
    """Width for each row-label column, from its longest label."""
    out = []
    for depth in range(layout.row_depth):
        longest = max((len(h.label) for h in layout.row_headers[depth]), default=1)
        out.append(max(26.0, longest * geom.font_pt * 0.62 + 10))
    return out


def render_figure(
    layout: Layout, *, scheme: ColourScheme, geometry: Geometry | None = None
):
    """Draw ``layout`` into a new matplotlib ``Figure``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    g = geometry or Geometry()
    if layout.show_values and geometry is None:
        g = replace(g, box_w=VALUES_BOX_W)
    n_step = len(layout.lead_times)
    lab_w = _label_widths(layout, g)
    cw = g.cell_w(n_step)
    head_h = g.head_h * layout.column_depth
    title_h = 34.0 if layout.title else 8.0
    legend_h = 46.0

    # Only families actually on the card get a legend entry: a spread ramp on a
    # card with no spread metric is noise.
    used = {s.family for _, _, cell in layout.iter_cells() for s in cell.steps}
    families = [f for k, f in scheme.families.items() if k in used] or list(
        scheme.families.values()
    )
    if not layout.coloured:
        families = []  # nothing is coloured, so a colour legend describes nothing

    span = (
        f"Each cell is {n_step} lead times, {layout.lead_labels[0]} to "
        f"{layout.lead_labels[-1]}, earliest on the left"
    )
    if layout.coloured:
        foot = (
            f"{layout.forecast_label} vs {layout.baseline_source}. {span}; "
            f"intensity is the difference relative to {layout.baseline_source}."
        )
        if layout.show_values:
            foot += (
                f" Numbers are each source's own score; grey rows are "
                f"{layout.baseline_source}'s."
            )
        caveat = (
            f"{layout.stats.n_boxes} simultaneous comparisons, and forecast cases are "
            f"autocorrelated: isolated cells mean little, coherent blocks mean a lot."
        )
    else:
        foot = f"No baseline: each number is its source's own score. {span}."
        caveat = "Nothing is compared, so nothing is coloured or marked significant."

    # The table alone does not set the width: a long title or footnote would be
    # clipped by a figure sized only from the grid.
    def _text_w(text: str, pt: float) -> float:
        return len(text) * pt * 0.56 + 8

    legend_w = 8.0 + sum(
        _text_w(f.negative_word, g.font_pt)
        + _text_w(f.positive_word, g.font_pt)
        + (len(f.negative) // max(1, len(f.negative) // 6) + 2) * 9 * 2
        + 26
        for f in families
    )
    if layout.stats.n_significant:
        legend_w += _text_w("significant", g.font_pt) + _text_w("not", g.font_pt) + 40
    W = max(
        sum(lab_w) + cw * layout.stats.n_cols,
        _text_w(layout.title, g.title_pt),
        _text_w(layout.subtitle, g.font_pt),
        _text_w(foot, g.font_pt - 1),
        _text_w(caveat, g.font_pt - 1),
        legend_w,
    )
    H = title_h + head_h + g.row_h * layout.stats.n_rows + legend_h

    fig = plt.figure(figsize=(W / 72.0, H / 72.0), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # inverted: row 0 at the top, as in the HTML
    ax.set_axis_off()

    x0 = sum(lab_w)
    y0 = title_h + head_h

    if layout.title:
        ax.text(
            4, 14, layout.title, fontsize=g.title_pt, fontweight="bold", va="center"
        )
        if layout.subtitle:
            ax.text(
                4, 27, layout.subtitle, fontsize=g.font_pt, color="#5b6470", va="center"
            )

    # ---- column headers -----------------------------------------------------
    for depth in range(layout.column_depth):
        top = title_h + depth * g.head_h
        for blk in layout.column_headers[depth]:
            bx = x0 + blk.start * cw
            bw = blk.span * cw
            ax.add_patch(
                Rectangle(
                    (bx, top),
                    bw,
                    g.head_h,
                    facecolor="#eef1f4",
                    edgecolor="#ffffff",
                    linewidth=0.8,
                    zorder=1,
                )
            )
            ax.text(
                bx + bw / 2,
                top + g.head_h / 2,
                blk.label,
                ha="center",
                va="center",
                fontsize=g.font_pt,
                zorder=2,
            )

    # ---- row headers: one label per block, vertically centred ---------------
    for depth in range(layout.row_depth):
        left = sum(lab_w[:depth])
        for blk in layout.row_headers[depth]:
            by = y0 + blk.start * g.row_h
            bh = blk.span * g.row_h
            ax.add_patch(
                Rectangle(
                    (left, by),
                    lab_w[depth],
                    bh,
                    facecolor="#f7f8fa",
                    edgecolor="#ffffff",
                    linewidth=0.8,
                    zorder=1,
                )
            )
            if blk.label:
                ax.text(
                    left + 4,
                    by + bh / 2,
                    blk.label,
                    ha="left",
                    va="center",
                    fontsize=g.font_pt,
                    zorder=2,
                )

    # ---- cells --------------------------------------------------------------
    rects, fills, edges = [], [], []
    labels = []  # (x, y, text, colour) of each printed value
    for r in range(layout.stats.n_rows):
        for c in range(layout.stats.n_cols):
            cell = layout.isel(row=r, col=c)
            cx = x0 + c * cw
            cy = y0 + r * g.row_h
            if cell is None:
                ax.add_patch(
                    Rectangle(
                        (cx, cy),
                        cw,
                        g.row_h,
                        facecolor=scheme.missing,
                        edgecolor="#ffffff",
                        linewidth=0.8,
                        zorder=1,
                    )
                )
                continue
            bx = cx + g.cell_pad
            by = cy + (g.row_h - g.box_h) / 2
            for k, st in enumerate(cell.steps):
                x = bx + k * (g.box_w + g.box_gap)
                rects.append(Rectangle((x, by), g.box_w, g.box_h))
                if not st.has_data:
                    fills.append("#ffffff")
                    edges.append("#e3e6ea")
                else:
                    sw = scheme.swatch(st.family, st.level)
                    fills.append(sw.fill)
                    edges.append(sw.edge if st.significant else "#ffffff")
                    if st.text:
                        labels.append(
                            (x + g.box_w / 2, by + g.box_h / 2, st.text, sw.fg)
                        )

    pc = PatchCollection(rects, match_original=False, zorder=3)
    pc.set_facecolor(fills)
    pc.set_edgecolor(edges)
    pc.set_linewidth(0.5)
    pc.set_snap(True)
    ax.add_collection(pc)
    for x, y, text, colour in labels:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=VALUES_PT,
            family="monospace",
            color=colour,
            zorder=4,
        )

    # ---- legend: one ramp per family, plus the caveats -----------------------
    ly = y0 + g.row_h * layout.stats.n_rows + 12.0
    lx = 4.0
    sw_w, sw_h = 8.0, 10.0
    for fam in families:
        ax.text(
            lx,
            ly,
            f"{fam.negative_word}",
            fontsize=g.font_pt - 0.5,
            ha="left",
            va="center",
            color="#3b424b",
        )
        lx += len(fam.negative_word) * (g.font_pt - 0.5) * 0.58 + 5
        step = max(1, len(fam.negative) // 6)
        for swatch in list(fam.negative.swatches[::step])[::-1]:
            ax.add_patch(
                Rectangle(
                    (lx, ly - sw_h / 2),
                    sw_w,
                    sw_h,
                    facecolor=swatch.fill,
                    edgecolor=swatch.edge,
                    linewidth=0.5,
                    zorder=3,
                )
            )
            lx += sw_w + 1
        ax.add_patch(
            Rectangle(
                (lx, ly - sw_h / 2),
                sw_w,
                sw_h,
                facecolor=scheme.neutral.fill,
                edgecolor=scheme.neutral.edge,
                linewidth=0.5,
                zorder=3,
            )
        )
        lx += sw_w + 1
        for swatch in list(fam.positive.swatches[::step]):
            ax.add_patch(
                Rectangle(
                    (lx, ly - sw_h / 2),
                    sw_w,
                    sw_h,
                    facecolor=swatch.fill,
                    edgecolor=swatch.edge,
                    linewidth=0.5,
                    zorder=3,
                )
            )
            lx += sw_w + 1
        lx += 4
        ax.text(
            lx,
            ly,
            fam.positive_word,
            fontsize=g.font_pt - 0.5,
            ha="left",
            va="center",
            color="#3b424b",
        )
        lx += len(fam.positive_word) * (g.font_pt - 0.5) * 0.58 + 22

    # The border is the significance channel, so say so: without this the frames
    # are decoration as far as the reader can tell.
    if layout.stats.n_significant:
        example = scheme.swatch("error", 8)
        for label, edge in (("significant", example.edge), ("not", "#ffffff")):
            ax.add_patch(
                Rectangle(
                    (lx, ly - sw_h / 2),
                    sw_w,
                    sw_h,
                    facecolor=example.fill,
                    edgecolor=edge,
                    linewidth=0.9,
                    zorder=3,
                )
            )
            lx += sw_w + 4
            ax.text(
                lx,
                ly,
                label,
                fontsize=g.font_pt - 0.5,
                ha="left",
                va="center",
                color="#3b424b",
            )
            lx += len(label) * (g.font_pt - 0.5) * 0.58 + 10

    ax.text(4, ly + 15, foot, fontsize=g.font_pt - 1, color="#5b6470", va="center")
    ax.text(4, ly + 27, caveat, fontsize=g.font_pt - 1, color="#8a6d1f", va="center")
    return fig


def render_static(
    layout: Layout,
    path: str | Path,
    *,
    scheme: ColourScheme,
    dpi: int = 200,
    geometry: Geometry | None = None,
) -> Path:
    """Render ``layout`` to PNG, SVG or PDF, inferred from the suffix."""
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "the static backend needs matplotlib: install mlwp-scorecards[static]"
        ) from exc

    import matplotlib.pyplot as plt

    path = Path(path)
    rc = {
        "pdf.fonttype": 42,  # embed TrueType so PDF text stays selectable
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep SVG text as text, not glyph paths
        "svg.hashsalt": "mlwp-scorecards",
        "font.family": "DejaVu Sans",
        "figure.autolayout": False,
        "path.simplify": False,
    }
    with plt.rc_context(rc):
        fig = render_figure(layout, scheme=scheme, geometry=geometry)
        fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(fig)
    return path
