# Prior work: the ECMWF scorecard page

ECMWF publishes an HTML scorecard for each IFS cycle upgrade, comparing the new cycle
(the "experiment", or e-suite) against the operational one (the "control"). This
package's layout was reverse-engineered from one of these pages:
`scorecards-47r1ENS.html`, the 47r1 ENS card. It's kept in the repo root and is
gitignored because it is 7.4 MB. The page links to sibling cards for 47r1–47r3, HRES
and ENS.

The generating software is believed to be ECMWF's **Quaver** verification system.
The page itself doesn't name its generator, and we don't have Quaver's source. So
everything below comes from the HTML and its embedded data. Where a statistical
method is described, it was **inferred from the numbers**, not read from code.

PLAN.md's appendix, *the ECMWF reference card, structurally*, covers the markup in
more detail, including the defects this package deliberately doesn't reproduce.

Notes taken 2026-09-27.

## What the card compares

An HTML comment (`__info_block_start__`) holds the run metadata:

```
dates=[2018120100,2018120200,2018120300,...,2020051412,2020051500]
steps=[24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360]
reftypes=['an', 'ob']
streams=['enfo', 'waef']
expvers=(cntrl:['0001', 'hayf', 'hbmz'], exper:['0074', 'haik', 'hail', 'hc19'])
vstreams=['oper_an', 'oper_ob', 'qrdx_an', 'qrdx_ob']
classs=['od', 'rd']
confidence=95.0
```

- **Two forecast sources.** The control (several experiment IDs stitched together)
  and the experiment. Colours show experiment minus control.
- **Two truth sources**, analyses (`an`) and observations (`ob`), both on one card.
- **Forecast cases.** About 1.5 years of twice-daily initialisations, with 15 lead
  times from T+24 to T+360.
- **Streams.** `enfo` is the ensemble and `waef` the wave ensemble, so the rows
  include `swh`.

## Layout

It's a single HTML `<table>`, with no charting library for the card itself.

- **Rows** have three nested label columns: truth source → variable (`z`, `t`, `ff`,
  `msl`, `2t`, `tp`, `swh`, …) → pressure level. Surface variables have no level.
  There are 45 rows, and each label column has its own background tint.
- **Columns** have two header levels: region (10 of them, each with a lat/lon
  tooltip, e.g. "NHem Extratropics (lat 20.0 to 90.0 …)") → metric (`rmsef`, `crps`,
  `spread`). That gives 30 leaf columns.
- **Each cell is a row of 15 small boxes**, one per lead time. 1011 of the 1350
  cells are populated, and the rest are blank grey tiles.
- **Checkboxes** show and hide regions, metrics and rows.
- **Hovering** a box shows e.g. `T+24 0.964% better (417)`: the normalised
  difference as a percentage, then the number of cases.

## Two display modes

A radio button switches between two renderings. Both are present in the markup, and
the switch only toggles `display`.

**Coloured boxes (default).**
- The fill shows the size of the normalised difference ("hue and saturation are
  proportional to normalised difference value") on a ramp of about 15 steps.
- A 1 px border in a darker shade marks a box that is significant at 95 %.
- There are four hue families. For error metrics, red means the experiment is worse
  and blue better. For `spread` and activity, purple means more active and green
  less active.

**Significance triangles.** One Unicode glyph per lead time, with seven levels:

| `siglev` | glyph | meaning |
|---|---|---|
| +3 | ▲ blue | experiment better, significant at 99.7 % |
| +2 | △ blue | better, 95 % |
| +1 | ░ blue | better, 68 % |
| 0 | █ light grey | "not really any difference" |
| −1 | ░ red | worse, 68 % |
| −2 | ▽ red | worse, 95 % |
| −3 | ▼ red | worse, 99.7 % |

The legend relates the three levels to the 1σ / 2σ / 3σ points of a normal
distribution. The two display modes carry different information. The boxes show
magnitude with a single 95 % flag. The triangles show only the confidence level.

## Click-through drill-down

Clicking a box opens two Plotly charts built from a `SubPagesData` JSON of about
3.2 MB, embedded at the bottom of the page. It has one entry per cell, keyed
`<var><level>_<region>_<metric>_<truth>`, e.g. `z500_n.hem_rmsef_an`, plus a shared
`steps` list.

| field | shape | content |
|---|---|---|
| `control_mean`, `experiment_mean` | 3 × 15 | mean score per lead time, then two error-bar arrays |
| `difference_mean` | 3 × 15 | normalised difference, then two error-bar arrays |
| `siglevs` | 15 | the −3…+3 level above |
| `popul` | 15 | number of forecast cases |
| `units` | str | e.g. `K` |

- **Top chart:** normalised difference against lead time, with error bars.
- **Bottom chart:** control (blue) and experiment (red) mean scores, with error bars.

## The statistics, as far as the numbers reveal them

Checked across all 15,165 populated (cell, lead time) entries:

- **Normalised difference.** It is signed so that **positive means the experiment
  is better**: for error metrics its sign agrees with `(ctl − exp)/ctl` in 97 % of
  entries. It is roughly that relative difference (the median ratio is 1.00, but
  the 10–90 % range is 0.88–1.11). So it is probably a mean of per-case normalised
  differences, not a ratio of the means.
- **Error bars** on the difference are symmetric, apart from float round-off. They
  are stored as *negative* numbers (Plotly draws the absolute value).
- **`siglev` is a z-threshold, exactly.** Take σ = |bar| / 1.96, so the bar is a
  95 % normal half-width, matching `confidence=95.0`. Then the value
  z = |difference| / σ splits cleanly by level:

  | `siglev` | z range observed |
  |---|---|
  | 0 | 0 – 0.992 |
  | ±1 | 0.992 – 1.960 |
  | ±2 | 1.960 – 2.976 |
  | ±3 | 2.977 – 84 |

  The boundaries are the two-sided normal quantiles for 68 %, 95 % and 99.7 %
  (0.994, 1.960, 2.968). Every one of the 6,400 non-zero `siglev`s has the same
  sign as the difference.
- **It looks like a normal-theory test on the mean difference, not a percentile
  bootstrap.** A percentile bootstrap wouldn't give exactly symmetric intervals, or
  a single σ from which all three levels follow. The card doesn't show how σ is
  estimated, e.g. whether it's inflated for serial correlation between consecutive
  cases.
- **Paired.** The difference's bar is much narrower than those on the individual
  means. For example, z500 n.hem rmsef at T+24: about ±0.24 % on the difference
  against about ±1.3 % on the control mean. That is only possible if the difference
  is taken case by case before averaging.
- **`popul` falls by 2 per 24 h of lead time**, e.g. 417 → 415 → …. Cases whose
  verification time runs past the end of the period drop out, and this is what
  would happen with twice-daily initialisations. The count also varies by variable,
  region, metric and truth source, from data availability.

## Compared with mlwp-scorecards

| | ECMWF card | mlwp-scorecards |
|---|---|---|
| Scoring | Done upstream by the generating system | Done upstream, by mxalign |
| Collapse over cases | Done by the generating system, not shown | Done here, from per-case scores |
| Significance | Normal z on the paired mean difference (inferred) | Block bootstrap on the paired per-case difference |
| Levels | 68 / 95 / 99.7 % | 68 / 95 / 99.7 % by default |
| Magnitude | Normalised (relative) difference | Relative difference, binned by `colours.py` scaling |
| Channels | Fill = magnitude, border = 95 % | Kept: fill = magnitude, border = significance |
| Polarity | `metric.substr(0,3)=="sda"` in the triangle code | Explicit `METRIC_POLARITY` table |
| Drill-down | Plotly from an unpinned CDN over `http://`, 3.2 MB payload | Self-contained, much smaller payload |
| Cell identity | Concatenated labels, duplicate `id`s | Integer index |
