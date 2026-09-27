"""Self-contained interactive HTML backend.

Emits one file with no external requests of any kind: no CDN, no analytics, no
webfonts. The colour ramp is applied through CSS custom properties on each box
rather than a full inline style triple, and column visibility is toggled by
injecting a single CSS rule rather than writing inline styles to every cell.

Clicking a cell opens a drill-down with two charts: the difference over lead time,
and the two sources' own values with their confidence intervals. The charts are
~80 lines of generated SVG rather than a plotting library -- the reference
implementation pulls 2.7 MB of Plotly over plain HTTP from an unpinned CDN to draw
two line charts.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from jinja2 import Template

from ..colours import ColourScheme
from ..model import Layout
from .payload import payload_for

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
td.c {{ padding: 2px 3px; line-height: 0; cursor: pointer; }}
td.c.empty {{ background: {missing}; cursor: default; }}
td.c:hover {{ outline: 2px solid #7d8894; outline-offset: -2px; }}
td.c:focus-visible {{ outline: 2px solid #06c; outline-offset: -2px; }}
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

/* ---- drill-down ---------------------------------------------------------- */
dialog#sc-detail {{ border: 1px solid #c8ced6; border-radius: 8px; padding: 0;
  max-width: 96vw; box-shadow: 0 10px 40px rgba(0,0,0,.22); }}
dialog#sc-detail::backdrop {{ background: rgba(20,24,28,.45); }}
#sc-detail .head {{ display: flex; align-items: baseline; gap: 12px;
  padding: 12px 16px 8px; border-bottom: 1px solid #eceff3; }}
#sc-detail h2 {{ font-size: 14px; margin: 0; }}
#sc-detail .meta {{ font-size: 12px; color: #5b6470; }}
#sc-detail .close {{ margin-left: auto; font: inherit; font-size: 16px;
  line-height: 1; border: 0; background: none; cursor: pointer; color: #5b6470;
  padding: 0 2px; }}
#sc-detail .charts {{ display: flex; flex-wrap: wrap; gap: 4px; padding: 8px 12px 4px; }}
#sc-detail figure {{ margin: 0; }}
#sc-detail figcaption {{ font-size: 11px; color: #5b6470; margin: 0 0 2px 46px; }}
#sc-detail .note {{ font-size: 11px; color: #5b6470; padding: 0 16px 12px;
  max-width: 62em; }}
#sc-detail .key {{ display: inline-block; width: 22px; height: 2px;
  vertical-align: middle; margin: 0 4px 0 10px; }}

@media print {{
  .controls {{ display: none; }}
  .scroll {{ max-height: none; overflow: visible; border: 0; }}
  .sc thead th, .sc tbody th {{ position: static; }}
  dialog#sc-detail {{ display: none; }}
}}
"""

#: Drawn by the JS below; kept here so both agree on the colours.
DETAIL_COLOURS = {"control": "#8a6d1f", "experiment": "#1f5c8a", "diff": "#3b424b"}

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
    Array.prototype.forEach.call(table.tHead.rows, function (head) {
      for (var i = 0; i < head.cells.length; i++) {
        var th = head.cells[i], g = th.dataset.group;
        if (!g) continue;
        var vis = groups[g].filter(function (k) { return !hidden.has(k); }).length;
        th.hidden = vis === 0;
        if (vis) th.colSpan = vis;
      }
    });
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

  /* ---- drill-down -------------------------------------------------------- */
  var dlg = document.getElementById('sc-detail');
  var dataEl = document.getElementById('sc-data');
  if (!dlg || !dataEl) return;

  var COLOURS = JSON.parse(document.getElementById('sc-colours').textContent);
  var payload = null;

  /* Inflated on first click, not at load: the page is useful without it, and on a
     full-size card this is the largest thing in the file. */
  function load() {
    if (payload) return Promise.resolve(payload);
    var bin = Uint8Array.from(atob(dataEl.textContent.trim()), function (c) {
      return c.charCodeAt(0);
    });
    if (!('DecompressionStream' in window)) {
      return Promise.reject(new Error('DecompressionStream unavailable'));
    }
    var stream = new Blob([bin]).stream()
      .pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).text().then(function (t) {
      payload = JSON.parse(t);
      return payload;
    });
  }

  var NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, kids) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    (kids || []).forEach(function (c) { n.append(c); });
    return n;
  }
  function txt(s) { return document.createTextNode(String(s)); }

  function ticks(lo, hi, n) {
    var span = hi - lo;
    if (!(span > 0)) return [lo];
    var raw = span / n;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 2.5, 5, 10].reduce(function (a, b) {
      return Math.abs(b * mag - raw) < Math.abs(a * mag - raw) ? b : a;
    }) * mag;
    var out = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + step * 1e-9; v += step) out.push(v);
    return out;
  }

  function fmt(v) {
    var a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1000 || a < 0.01) return v.toExponential(1);
    return String(Number(v.toPrecision(4)));
  }

  /* One line with an optional confidence band. `series` entries are
     {y, lo, hi, colour, name, dash}. */
  function chart(opts) {
    var x = opts.x, series = opts.series.filter(function (s) { return s.y; });
    var W = opts.w || 430, H = opts.h || 250;
    var m = { l: 58, r: 12, t: 8, b: 30 };
    var all = [];
    series.forEach(function (s) {
      s.y.forEach(function (v, i) {
        if (v == null) return;
        all.push(v);
        if (s.lo && s.lo[i] != null) all.push(s.lo[i]);
        if (s.hi && s.hi[i] != null) all.push(s.hi[i]);
      });
    });
    if (opts.zeroLine) all.push(0);
    if (!all.length) return el('svg', { width: W, height: H });
    var y0 = Math.min.apply(null, all), y1 = Math.max.apply(null, all);
    var pad = (y1 - y0) * 0.08 || Math.abs(y0) * 0.1 || 1;
    y0 -= pad; y1 += pad;

    var sx = function (i) {
      return x.length < 2 ? m.l : m.l + i * (W - m.l - m.r) / (x.length - 1);
    };
    var sy = function (v) {
      return m.t + (y1 - v) * (H - m.t - m.b) / (y1 - y0);
    };

    var g = [];
    ticks(y0, y1, 5).forEach(function (t) {
      g.push(el('line', { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t),
                          stroke: '#e8ebee' }));
      g.push(el('text', { x: m.l - 7, y: sy(t) + 3, 'text-anchor': 'end',
                          'font-size': 10, fill: '#5b6470' }, [txt(fmt(t))]));
    });
    if (opts.zeroLine) {
      g.push(el('line', { x1: m.l, x2: W - m.r, y1: sy(0), y2: sy(0),
                          stroke: '#98a2ad', 'stroke-dasharray': '4 3' }));
    }
    x.forEach(function (lab, i) {
      g.push(el('text', { x: sx(i), y: H - 10, 'text-anchor': 'middle',
                          'font-size': 9.5, fill: '#5b6470' }, [txt(lab)]));
    });

    series.forEach(function (s, si) {
      /* Error bars, offset slightly per series so two of them do not overlap
         into an unreadable smear at the same x. */
      var off = (si - (series.length - 1) / 2) * 3;
      if (s.lo && s.hi) {
        var cap = 3;
        s.y.forEach(function (v, i) {
          if (v == null || s.lo[i] == null || s.hi[i] == null) return;
          var xx = sx(i) + off, a = sy(s.lo[i]), b = sy(s.hi[i]);
          g.push(el('line', { x1: xx, x2: xx, y1: a, y2: b,
                              stroke: s.colour, 'stroke-width': 1.2 }));
          g.push(el('line', { x1: xx - cap, x2: xx + cap, y1: a, y2: a,
                              stroke: s.colour, 'stroke-width': 1.2 }));
          g.push(el('line', { x1: xx - cap, x2: xx + cap, y1: b, y2: b,
                              stroke: s.colour, 'stroke-width': 1.2 }));
        });
      }
      var pts = [];
      s.y.forEach(function (v, i) {
        if (v != null) pts.push((sx(i) + off) + ',' + sy(v));
      });
      g.push(el('polyline', { points: pts.join(' '), fill: 'none',
                              stroke: s.colour, 'stroke-width': 1.9,
                              'stroke-dasharray': s.dash || null }));
      s.y.forEach(function (v, i) {
        if (v != null) {
          g.push(el('circle', { cx: sx(i) + off, cy: sy(v), r: 2.4,
                                fill: s.colour }));
        }
      });
    });

    if (opts.ylabel) {
      g.push(el('text', { x: 12, y: H / 2, 'font-size': 10.5, fill: '#3b424b',
                          'text-anchor': 'middle',
                          transform: 'rotate(-90 12 ' + H / 2 + ')' },
                [txt(opts.ylabel)]));
    }
    return el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H,
                       role: 'img', 'aria-label': opts.ylabel || '' }, g);
  }

  function pctLabel(conf) {
    // 99.7% is 3 sigma and a real choice of level; rounding it to 100% would
    // turn a stated interval into a claim of certainty
    if (!conf) return '';
    return Number((conf * 100).toPrecision(4)) + '%';
  }

  function swatch(colour, dash) {
    var s = document.createElement('span');
    s.className = 'key';
    s.style.background = dash
      ? 'repeating-linear-gradient(90deg,' + colour + ' 0 4px,transparent 4px 7px)'
      : colour;
    return s;
  }

  function open(index) {
    load().then(function (d) {
      var c = d.cells[index];
      if (!c) return;
      dlg.querySelector('h2').textContent = c.t;
      dlg.querySelector('.meta').textContent =
        c.m + (c.u ? ' (' + c.u + ')' : '');

      var charts = dlg.querySelector('.charts');
      charts.replaceChildren();

      /* Chart 1: the paired difference in the metric's own units, with its
         interval. Plotted in units rather than percent so the error bars are on
         the same scale as the quantity -- the percentage is a ratio of two
         uncertain numbers and its interval is not simply the scaled one. */
      var haveDiffCI = c.vl && c.vu;
      var f1 = document.createElement('figure');
      var cap1 = document.createElement('figcaption');
      var src = c.s || d.experiment;
      cap1.textContent = 'Difference: ' + src + ' minus ' + d.control
        + (c.u ? ' (' + c.u + ')' : '')
        + (haveDiffCI ? ', paired ' + pctLabel(d.confidence) + ' interval' : '');
      f1.append(cap1, chart({
        x: d.labels, zeroLine: true, ylabel: c.u || 'difference',
        series: [{ y: c.v || [], lo: c.vl, hi: c.vu, colour: COLOURS.diff }]
      }));

      var f2 = document.createElement('figure');
      var cap2 = document.createElement('figcaption');
      cap2.textContent = c.m + (c.u ? ' (' + c.u + ')' : '')
        + ', each with its own ' + pctLabel(d.confidence) + ' interval';
      f2.append(cap2, chart({
        x: d.labels, ylabel: c.u || c.m,
        series: [
          { y: c.c, lo: c.cl, hi: c.cu, colour: COLOURS.control,
            name: d.control },
          { y: c.e, lo: c.el, hi: c.eu, colour: COLOURS.experiment,
            name: src, dash: '5 3' }
        ].filter(function (s) { return s.y; })
      }));
      /* A neutral cell -- the baseline's own row, or a card with no baseline --
         has a score of its own but no difference to chart. */
      if (c.v) charts.append(f1);
      charts.append(f2);

      var note = dlg.querySelector('.note');
      note.replaceChildren();
      if (c.c) note.append(swatch(COLOURS.control), txt(' ' + d.control + '   '));
      note.append(swatch(COLOURS.experiment, true), txt(' ' + src));
      var extra = document.createElement('div');
      extra.style.marginTop = '6px';
      extra.textContent = !c.v
        ? 'This source’s own score, with its own interval over forecast cases. '
          + 'It is not compared with anything, so nothing here is marked '
          + 'significant.'
        : haveDiffCI
        ? 'The difference interval is paired: it is computed per forecast case '
          + 'before averaging, so the error the two sources share cancels. It is '
          + 'therefore much tighter than the two intervals on the right, and it '
          + 'is the one that decides significance. Overlapping intervals on the '
          + 'right do not mean the difference is insignificant.'
        : 'Intervals on the right are each source’s own, over forecast cases. The '
          + 'difference has none: that needs a paired resample, which this card '
          + 'does not carry, so nothing here is marked significant and '
          + 'overlapping intervals do not mean the difference is insignificant.';
      note.append(extra);
      if (c.n) {
        var cases = document.createElement('div');
        cases.style.marginTop = '4px';
        cases.textContent = 'Cases per lead time: ' + c.n.join(', ') + '.';
        note.append(cases);
      }
      dlg.showModal();
    }).catch(function (err) {
      console.error('drill-down unavailable:', err);
    });
  }

  table.addEventListener('click', function (e) {
    var td = e.target.closest('td.c[data-i]');
    if (td) open(Number(td.dataset.i));
  });
  table.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    var td = e.target.closest('td.c[data-i]');
    if (td) { e.preventDefault(); open(Number(td.dataset.i)); }
  });
  dlg.querySelector('.close').addEventListener('click', function () { dlg.close(); });
  dlg.addEventListener('click', function (e) {
    if (e.target === dlg) dlg.close();   // click the backdrop
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
     {{ n_lead }} boxes from {{ first_lead }} to {{ last_lead }}. {% if coloured %}Colour shows the
     difference between <b>{{ experiment }}</b> and <b>{{ control }}</b>, and its
     intensity the size of that difference relative to {{ control }}.{% else %}There is no
     baseline: each box is its source's own score, and nothing is compared or marked
     significant.{% endif %}{% if show_values and coloured %} Each box prints its source's
     own score; the grey rows are <b>{{ control }}</b>'s own, which the colours are
     relative to.{% endif %} Hover a box
     for the exact value{% if has_detail %}, or click a cell for the full series{% endif %}.</p>
  {%- for fam in ramps %}
  <p><b>{{ fam.label }}</b>: {{ fam.negative_word }}
     <span class="ramp">{% for s in fam.neg %}<i style="background:{{ s.fill }};border-color:{{ s.edge }}"></i>{% endfor %}</span>
     no change
     <span class="ramp">{% for s in fam.pos %}<i style="background:{{ s.fill }};border-color:{{ s.edge }}"></i>{% endfor %}</span>
     {{ fam.positive_word }}</p>
  {%- endfor %}
  {%- if n_significant %}
  <p><b>A framed box is significant</b>
     <span class="ramp"><i style="background:{{ sig_fill }};border-color:{{ sig_edge }}"></i></span>
     — its {{ confidence_pct }} interval on the difference excludes zero. An unframed
     box
     <span class="ramp"><i style="background:{{ sig_fill }};border-color:#ffffff"></i></span>
     is not. {{ n_significant }} of {{ n_boxes }} boxes are framed.
     {%- if graded %} The frame is one bit; hover a box for the strongest level it
     reaches, up to {{ widest_pct }}.{% endif %}</p>
  {%- endif %}
  <p>A blank grey cell has no data at all, which is deliberately distinct from a cell
     whose difference happens to be zero. A hatched box is a lead time with no value
     inside a cell that otherwise has data.</p>

  <h2>What this card cannot tell you</h2>
  {%- for note in notes %}
  <div class="caveat">{{ note }}</div>
  {%- endfor %}
  <p class="sub">{{ stats_line }}</p>
</div>

{% if has_detail -%}
<dialog id="sc-detail">
  <div class="head">
    <h2></h2><span class="meta"></span>
    <button type="button" class="close" aria-label="Close">&#10005;</button>
  </div>
  <div class="charts"></div>
  <div class="note"></div>
</dialog>
<script type="application/gzip;base64" id="sc-data">{{ payload_b64 }}</script>
<script type="application/json" id="sc-colours">{{ colours_json }}</script>
{%- endif %}
<script type="application/json" id="sc-groups">{{ groups_json }}</script>
<script type="application/json" id="sc-colkeys">{{ colkeys_json }}</script>
<script type="application/json" id="sc-sep">{{ sep_json }}</script>
<script>{{ js }}</script>
</body></html>
"""
)


def _ramp_css(scheme: ColourScheme, *, values: bool = False) -> str:
    """One rule per (family, direction, level) instead of a style on every box.

    A full-size card has ~20,000 boxes but only ~60 distinct colours, so carrying
    the fill inline costs roughly 800 kB for nothing. With ``values``, each rule
    also carries the swatch's legible text colour for the number printed on it.
    """

    def rule(sel: str, sw) -> str:
        fg = f" --t:{sw.fg};" if values else ""
        return f"{sel} {{ --f:{sw.fill}; --e:{sw.edge};{fg} }}"

    rules = [rule(".c i.z", scheme.neutral)]
    for fam_key, fam in scheme.families.items():
        for sign, ramp in (("p", fam.positive), ("n", fam.negative)):
            for i, sw in enumerate(ramp.swatches, start=1):
                rules.append(rule(f".f-{fam_key} i.{sign}{i}", sw))
    return "\n".join(rules)


#: Wider boxes that hold a number each, for a card that shows values. Appended
#: only then, after the rest, so a card without values is byte-identical to before.
_VALUES_CSS = """
:root { --bw: auto; --bh: 15px; }
td.c { line-height: 15px; }
td.c > i { min-width: 30px; padding: 0 3px; font: 9.5px/13px ui-monospace, Menlo,
  Consolas, monospace; font-style: normal; text-align: center; color: var(--t, #16191d); }
tr.base th { color: #5b6470; font-style: italic; }
td.c.empty { background-clip: content-box; }
"""


def _boxes(cell) -> str:
    """Emit one ``<i>`` per lead time, carrying only its level class, and the
    printed value when the card shows values."""
    out = []
    for st in cell.steps:
        tip = html.escape(st.tooltip, quote=True)
        if not st.has_data:
            out.append(f'<i class="b nodata" title="{tip}"></i>')
            continue
        if st.level == 0:
            cls = "b z"
        else:
            cls = f"b {'p' if st.level > 0 else 'n'}{abs(st.level)}"
        if st.significant:
            cls += " sig"
        out.append(f'<i class="{cls}" title="{tip}">{html.escape(st.text)}</i>')
    return "".join(out)


def _pct(conf: float | None) -> str:
    """A confidence level as a percentage, keeping 99.7% from reading as 100%."""
    return f"{conf * 100:.4g}%" if conf else "confidence"


def render_html(
    layout: Layout,
    path: str | Path,
    *,
    scheme: ColourScheme,
    detail: bool = True,
    precision: int = 4,
) -> Path:
    """Write ``layout`` as a self-contained interactive HTML page.

    Parameters
    ----------
    layout : Layout
    path : str or Path
    scheme : ColourScheme
    detail : bool, optional
        Embed the click-through drill-down data. On a full-size card this is the
        largest thing in the file; pass False for a table-only page.
    precision : int, optional
        Significant figures kept in the drill-down data. A chart cannot show more,
        and rounding is what makes it compress.

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

    # One header row per column depth. The stylesheet makes rows 1 and 2 sticky;
    # any deeper row carries its own offset, so a card of at most two column
    # levels is written exactly as before.
    def _top(i: int) -> str:
        return f' style="top:{23 * i}px"' if i >= 2 else ""

    corner_span = max(2, layout.column_depth)
    head_rows = [
        [
            f'<th class="corner" colspan="{layout.row_depth}" '
            f'rowspan="{corner_span}"></th>'
        ]
    ]
    for blk in layout.column_headers[0]:
        head_rows[0].append(
            f'<th class="h0" data-group="{esc(blk.key)}" colspan="{blk.span}" '
            f'scope="colgroup">{esc(blk.label)}</th>'
        )
    # Middle levels: each block is a group too, keyed by its prefix, so hiding
    # columns shrinks its span exactly as it does the outermost row's.
    for depth in range(1, layout.column_depth - 1):
        row = []
        for blk in layout.column_headers[depth]:
            prefix = SEP.join(
                str(k) for k in layout.columns[blk.start].key[: depth + 1]
            )
            groups[prefix] = [colkey(c) for c in layout.columns[blk.start : blk.stop]]
            row.append(
                f'<th class="hm" data-group="{esc(prefix)}" colspan="{blk.span}" '
                f'scope="colgroup"{_top(depth)}>{esc(blk.label)}</th>'
            )
        head_rows.append(row)
    if layout.column_depth > 1:
        leaf = layout.column_depth - 1
        head_rows.append(
            [
                f'<th class="h1" data-col="{esc(colkey(c))}" scope="col"{_top(leaf)}>'
                f"{esc(c.headers[-1].label)}</th>"
                for c in layout.columns
            ]
        )
    thead = "".join(f"<tr>{''.join(row)}</tr>" for row in head_rows)

    # Cell identity for the drill-down is an integer, enumerated in the same order
    # the payload uses. Labels never participate: the reference builds ids by
    # concatenating them, which breaks on any label containing the separator.
    index_of = {
        (cell.row_key, cell.col_key): i
        for i, (_, _, cell) in enumerate(layout.iter_cells())
    }

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
                i = index_of[(cell.row_key, cell.col_key)]
                tds.append(
                    f'<td class="c f-{fam}" data-col="{key}" data-i="{i}" '
                    f'tabindex="0" role="button" '
                    f'data-cell="{esc(cell.cell_id)}">{_boxes(cell)}</td>'
                )
        is_base = any(
            cell.is_baseline
            for cell in (layout.isel(row=r, col=c) for c in range(len(layout.columns)))
            if cell is not None
        )
        body.append(f"<tr{' class=\"base\"' if is_base else ''}>{''.join(tds)}</tr>")

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

    # No baseline, nothing coloured: a colour legend would describe nothing.
    ramps = []
    for key, fam in scheme.families.items() if layout.coloured else ():
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

    payload_b64 = ""
    if detail and s.n_cells_present:
        payload_b64, raw_bytes, packed_bytes = payload_for(layout, precision=precision)
        stats_line += (
            f" Drill-down data {packed_bytes / 1024:.0f} kB "
            f"({raw_bytes / packed_bytes:.1f}x compressed)."
        )

    page = _PAGE.render(
        title=layout.title or "Scorecard",
        subtitle=layout.subtitle,
        css=_CSS.format(
            missing=scheme.missing,
            ramp=_ramp_css(scheme, values=layout.show_values),
        )
        + (_VALUES_CSS if layout.show_values else ""),
        js=_JS,
        table=table,
        control_groups=control_groups,
        ramps=ramps,
        notes=layout.notes,
        stats_line=stats_line,
        n_lead=len(layout.lead_times),
        first_lead=layout.lead_labels[0],
        last_lead=layout.lead_labels[-1],
        control=layout.baseline_source,
        experiment=layout.forecast_label,
        coloured=layout.coloured,
        show_values=layout.show_values,
        n_significant=s.n_significant,
        n_boxes=s.n_boxes,
        # The border marks a box that clears the *narrowest* level supplied, so
        # that is the number the legend has to quote; the drill-down charts draw
        # the widest, and the tooltip names whichever level each box reaches.
        confidence_pct=_pct(min(layout.confidence_levels, default=None)),
        widest_pct=_pct(layout.confidence),
        graded=len(layout.confidence_levels) > 1,
        sig_fill=scheme.swatch("error", 8).fill,
        sig_edge=scheme.swatch("error", 8).edge,
        has_detail=bool(payload_b64),
        payload_b64=payload_b64,
        colours_json=json.dumps(DETAIL_COLOURS),
        groups_json=json.dumps(groups),
        colkeys_json=json.dumps(col_keys),
        sep_json=json.dumps(SEP),
    )
    path.write_text(page, encoding="utf-8")
    return path
