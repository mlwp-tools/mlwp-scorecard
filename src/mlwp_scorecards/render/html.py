"""Self-contained interactive HTML backend.

Emits one file with no external requests of any kind: no CDN, no analytics, no
webfonts. The colour ramp is applied through CSS custom properties on each box
rather than a full inline style triple, and column visibility is toggled by
injecting a single CSS rule rather than writing inline styles to every cell.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from jinja2 import Template

from ..colours import ColourScheme
from ..model import Layout

__all__ = ["render_html"]

SEP = "|"

_CSS = """
:root {{ --bw: 7px; --bh: 13px; --bg: 1px; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 18px 20px 40px;
  font: 13px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #16191d; background: #fff; }}
h1 {{ font-size: 19px; margin: 0 0 2px; }}
.sub {{ color: #5b6470; margin: 0 0 14px; font-size: 12.5px; }}
.vh {{ position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }}
fieldset {{ border: 1px solid #dde1e6; border-radius: 6px; margin: 0 0 10px;
  padding: 7px 11px 9px; }}
legend {{ font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: #5b6470; padding: 0 4px; }}
label {{ margin-right: 11px; font-size: 12px; white-space: nowrap; display: inline-block; }}
button {{ font: inherit; font-size: 11px; padding: 1px 8px; margin-left: 4px;
  border: 1px solid #c8ced6; border-radius: 4px; background: #f6f7f9; cursor: pointer; }}
button:hover {{ background: #eceff3; }}
.scroll {{ overflow: auto; max-height: 78vh; border: 1px solid #dde1e6; border-radius: 6px; }}
table.sc {{ border-collapse: separate; border-spacing: 0; font-size: 11px; }}
.sc th, .sc td {{ padding: 0; white-space: nowrap; border-right: 1px solid #fff;
  border-bottom: 1px solid #fff; }}
.sc thead th {{ position: sticky; background: #eef1f4; z-index: 3; font-weight: 600;
  padding: 3px 6px; text-align: center; }}
.sc thead tr:nth-child(1) th {{ top: 0; }}
.sc thead tr:nth-child(2) th {{ top: 23px; }}
.sc tbody th {{ position: sticky; background: #f7f8fa; z-index: 2; font-weight: 400;
  padding: 1px 7px 1px 6px; text-align: left; }}
.corner {{ left: 0; z-index: 4 !important; }}
.sc tbody th.lv {{ text-align: right; }}
td.c {{ padding: 2px 3px; line-height: 0; }}
td.c.empty {{ background: {missing}; }}
td.c > i {{ display: inline-block; width: var(--bw); height: var(--bh);
  margin-right: var(--bg); vertical-align: top; border: 1px solid #fff;
  background: var(--f, transparent); }}
td.c > i.sig {{ border-color: var(--e); }}
{ramp}
td.c > i:last-child {{ margin-right: 0; }}
i.b.nodata {{ background: repeating-linear-gradient(45deg, #fff, #fff 2px,
  #e3e6ea 2px, #e3e6ea 4px); border-color: #e3e6ea; }}
tbody tr:hover th {{ background: #eaeef3; }}
.legend {{ margin-top: 16px; font-size: 12px; color: #3b424b; max-width: 62em; }}
.legend h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
  color: #5b6470; margin: 15px 0 5px; }}
.ramp {{ display: inline-block; vertical-align: middle; margin: 0 5px; }}
.ramp i {{ display: inline-block; width: 11px; height: 13px; border: 1px solid; }}
.caveat {{ border-left: 3px solid #d8b24a; padding: 6px 0 6px 11px; margin: 7px 0;
  background: #fffdf5; }}
@media print {{
  .controls {{ display: none; }}
  .scroll {{ max-height: none; overflow: visible; border: 0; }}
  .sc thead th, .sc tbody th {{ position: static; }}
}}
"""

_JS = r"""
(function () {
  var table = document.getElementById('sc-table');
  var dyn = document.getElementById('sc-dyn').sheet;
  var groups = JSON.parse(document.getElementById('sc-groups').textContent);
  var allKeys = JSON.parse(document.getElementById('sc-colkeys').textContent);
  var SEP = JSON.parse(document.getElementById('sc-sep').textContent);
  var hidden = new Set();

  function apply() {
    while (dyn.cssRules.length) dyn.deleteRule(0);
    if (hidden.size) {
      var sel = Array.from(hidden).map(function (k) {
        return '[data-col="' + CSS.escape(k) + '"]';
      }).join(',');
      dyn.insertRule(sel + '{display:none}', 0);
    }
    var head = table.tHead.rows[0];
    for (var i = 0; i < head.cells.length; i++) {
      var th = head.cells[i], g = th.dataset.group;
      if (!g) continue;
      var vis = groups[g].filter(function (k) { return !hidden.has(k); }).length;
      th.hidden = vis === 0;
      if (vis) th.colSpan = vis;
    }
  }

  function recompute() {
    hidden.clear();
    var off = {};
    document.querySelectorAll('#sc-controls input[type=checkbox]').forEach(function (b) {
      if (!b.checked) {
        var d = b.dataset.depth;
        (off[d] = off[d] || []).push(b.dataset.key);
      }
    });
    allKeys.forEach(function (k) {
      var parts = k.split(SEP);
      for (var d in off) {
        if (off[d].indexOf(parts[d]) !== -1) { hidden.add(k); return; }
      }
    });
    apply();
  }

  var controls = document.getElementById('sc-controls');
  controls.addEventListener('change', recompute);
  controls.addEventListener('click', function (e) {
    if (e.target.tagName !== 'BUTTON') return;
    var want = e.target.dataset.setAll === '1';
    controls.querySelectorAll('input[data-depth="' + e.target.dataset.depth + '"]')
      .forEach(function (b) { b.checked = want; });
    recompute();
  });
})();
"""

_PAGE = Template(
    """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }}</title>
<style>{{ css }}</style><style id="sc-dyn"></style>
</head><body>
<h1>{{ title }}</h1>
{% if subtitle %}<p class="sub">{{ subtitle }}</p>{% endif %}

<form class="controls" id="sc-controls">
{%- for grp in control_groups %}
  <fieldset><legend>{{ grp.dim }}</legend>
  {%- for m in grp.members %}
    <label><input type="checkbox" data-depth="{{ grp.depth }}" data-key="{{ m }}" checked> {{ m }}</label>
  {%- endfor %}
    <button type="button" data-set-all="1" data-depth="{{ grp.depth }}">all</button>
    <button type="button" data-set-all="0" data-depth="{{ grp.depth }}">none</button>
  </fieldset>
{%- endfor %}
</form>

<div class="scroll">{{ table }}</div>

<div class="legend">
  <h2>How to read this</h2>
  <p>Each cell is one comparison across forecast lead time, earliest on the left:
     {{ n_lead }} boxes from {{ first_lead }} to {{ last_lead }}. Colour shows the
     difference between <b>{{ experiment }}</b> and <b>{{ control }}</b>, and its
     intensity the size of that difference relative to {{ control }}. Hover a box
     for the exact value.</p>
  {%- for fam in ramps %}
  <p><b>{{ fam.label }}</b>: {{ fam.negative_word }}
     <span class="ramp">{% for s in fam.neg %}<i style="background:{{ s.fill }};border-color:{{ s.edge }}"></i>{% endfor %}</span>
     no change
     <span class="ramp">{% for s in fam.pos %}<i style="background:{{ s.fill }};border-color:{{ s.edge }}"></i>{% endfor %}</span>
     {{ fam.positive_word }}</p>
  {%- endfor %}
  <p>A blank grey cell has no data at all, which is deliberately distinct from a cell
     whose difference happens to be zero. A hatched box is a lead time with no value
     inside a cell that otherwise has data.</p>

  <h2>What this card cannot tell you</h2>
  {%- for note in notes %}
  <div class="caveat">{{ note }}</div>
  {%- endfor %}
  <p class="sub">{{ stats_line }}</p>
</div>

<script type="application/json" id="sc-groups">{{ groups_json }}</script>
<script type="application/json" id="sc-colkeys">{{ colkeys_json }}</script>
<script type="application/json" id="sc-sep">{{ sep_json }}</script>
<script>{{ js }}</script>
</body></html>
"""
)


def _ramp_css(scheme: ColourScheme) -> str:
    """One rule per (family, direction, level) instead of a style on every box.

    A full-size card has ~20,000 boxes but only ~60 distinct colours, so carrying
    the fill inline costs roughly 800 kB for nothing.
    """
    rules = [f".c i.z {{ --f:{scheme.neutral.fill}; --e:{scheme.neutral.edge}; }}"]
    for fam_key, fam in scheme.families.items():
        for sign, ramp in (("p", fam.positive), ("n", fam.negative)):
            for i, sw in enumerate(ramp.swatches, start=1):
                rules.append(
                    f".f-{fam_key} i.{sign}{i} {{ --f:{sw.fill}; --e:{sw.edge}; }}"
                )
    return "\n".join(rules)


def _boxes(cell) -> str:
    """Emit one ``<i>`` per lead time, carrying only its level class."""
    out = []
    for st in cell.steps:
        tip = html.escape(st.tooltip, quote=True)
        if st.value is None:
            out.append(f'<i class="b nodata" title="{tip}"></i>')
            continue
        if st.level == 0:
            cls = "b z"
        else:
            cls = f"b {'p' if st.level > 0 else 'n'}{abs(st.level)}"
        if st.significant:
            cls += " sig"
        out.append(f'<i class="{cls}" title="{tip}"></i>')
    return "".join(out)


def render_html(layout: Layout, path: str | Path, *, scheme: ColourScheme) -> Path:
    """Write ``layout`` as a self-contained interactive HTML page.

    Parameters
    ----------
    layout : Layout
    path : str or Path
    scheme : ColourScheme

    Returns
    -------
    Path
        The file written.
    """
    path = Path(path)

    def esc(v: object) -> str:
        return html.escape(str(v), quote=True)

    def colkey(line) -> str:
        return SEP.join(str(k) for k in line.key)

    col_keys = [colkey(c) for c in layout.columns]
    groups: dict[str, list[str]] = {}
    for c in layout.columns:
        groups.setdefault(str(c.key[0]), []).append(colkey(c))

    head1 = [f'<th class="corner" colspan="{layout.row_depth}" rowspan="2"></th>']
    for blk in layout.column_headers[0]:
        head1.append(
            f'<th class="h0" data-group="{esc(blk.key)}" colspan="{blk.span}" '
            f'scope="colgroup">{esc(blk.label)}</th>'
        )
    head2 = []
    if layout.column_depth > 1:
        for c in layout.columns:
            head2.append(
                f'<th class="h1" data-col="{esc(colkey(c))}" scope="col">'
                f"{esc(c.headers[-1].label)}</th>"
            )
    thead = f"<tr>{''.join(head1)}</tr>" + (
        f"<tr>{''.join(head2)}</tr>" if head2 else ""
    )

    body = []
    for r, rl in enumerate(layout.rows):
        tds = []
        for d, hc in enumerate(rl.headers):
            cls = "lv" if d == layout.row_depth - 1 else ""
            # continuation rows leave the label blank rather than using rowspan,
            # so hiding a row never reflows a spanned cell into the wrong place
            text = esc(hc.label) if hc.start == r else ""
            tds.append(f'<th class="{cls}" scope="row">{text}</th>')
        for c, cl in enumerate(layout.columns):
            key = esc(colkey(cl))
            cell = layout.isel(row=r, col=c)
            if cell is None:
                tds.append(f'<td class="c empty" data-col="{key}"></td>')
            else:
                fam = cell.steps[0].family if cell.steps else "error"
                tds.append(
                    f'<td class="c f-{fam}" data-col="{key}" '
                    f'data-cell="{esc(cell.cell_id)}">{_boxes(cell)}</td>'
                )
        body.append(f"<tr>{''.join(tds)}</tr>")

    table = (
        f'<table class="sc" id="sc-table">'
        f'<caption class="vh">{esc(layout.title)}</caption>'
        f"<thead>{thead}</thead><tbody>{''.join(body)}</tbody></table>"
    )

    control_groups = []
    for depth, dim in enumerate(layout.column_dims):
        seen: list[str] = []
        for c in layout.columns:
            v = str(c.key[depth])
            if v not in seen:
                seen.append(v)
        control_groups.append({"dim": dim, "depth": depth, "members": seen})

    ramps = []
    for key, fam in scheme.families.items():
        step = max(1, len(fam.positive) // 7)
        ramps.append(
            {
                "label": "error metrics" if key == "error" else "activity metrics",
                "pos": list(fam.positive.swatches[::step]),
                "neg": list(fam.negative.swatches[::step])[::-1],
                "positive_word": fam.positive_word,
                "negative_word": fam.negative_word,
            }
        )

    s = layout.stats
    stats_line = (
        f"{s.n_rows} rows x {s.n_cols} columns; {s.n_cells_present} of "
        f"{s.n_cells_possible} crossings populated; {s.n_boxes} boxes."
    )
    if s.n_saturated:
        stats_line += f" {s.n_saturated} values sit at or beyond the top of the scale."

    page = _PAGE.render(
        title=layout.title or "Scorecard",
        subtitle=layout.subtitle,
        css=_CSS.format(missing=scheme.missing, ramp=_ramp_css(scheme)),
        js=_JS,
        table=table,
        control_groups=control_groups,
        ramps=ramps,
        notes=layout.notes,
        stats_line=stats_line,
        n_lead=len(layout.lead_times),
        first_lead=layout.lead_labels[0],
        last_lead=layout.lead_labels[-1],
        control=layout.control,
        experiment=layout.experiment,
        groups_json=json.dumps(groups),
        colkeys_json=json.dumps(col_keys),
        sep_json=json.dumps(SEP),
    )
    path.write_text(page, encoding="utf-8")
    return path
