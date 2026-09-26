# Prior work: Brightband Operational WeatherBench scorecard

[Operational WeatherBench](https://owb.brightband.com) (OWB) is Brightband's public
dashboard. It verifies operational and ML global models in near-real time. Its
[scorecard view](https://owb.brightband.com/scorecard?start=2026-08-09&end=2026-09-08)
compares every model against one chosen baseline model, over a date window. The
[methodology page](https://owb.brightband.com/methodology) describes the scoring.

Notes taken 2026-09-27. The page is a Next.js app that renders in the browser, so
everything below comes from its JS bundles, the methodology page and its JSON API.
The scripts in this folder re-download and re-derive all of it:

| script | does |
|---|---|
| `fetch.sh` | downloads the page, the methodology page, all JS/CSS chunks and `/api/meta` into `source/` |
| `extract.py` | splits the chunks into one pretty-printed file per webpack module (`source/modules/`) and extracts `source/methodology.txt` |
| `fetch_api.py` | saves `/api/scorecard` for both modes and all six regions (`source/api/`) |
| `analyse_api.py` | prints the API row schema and coverage, and rebuilds one card row with the page's colour rule |

```sh
./fetch.sh
uv run --no-project --with jsbeautifier --with beautifulsoup4 python extract.py
python fetch_api.py && python analyse_api.py
```

`source/` is gitignored. It holds Brightband's minified code and a snapshot of their
data, and the scripts regenerate it. The modules that matter are
`page-*/9203.js` (the scorecard page), `168-*/2168.js` (models, variables, metrics,
regions and lead times) and `310-*/6717.js` (the colour rule).

## Where the numbers come from

- **Scoring.** Scores are computed upstream with
  [WeatherBench-X](https://github.com/google-research/weatherbenchX). There is one
  stored value per model × target × variable × metric × region × **initialisation
  time** × lead time. It uses a 0.25° grid with cos(lat) area weighting and WBX
  region bounds.
- **Truth: each model against its own analysis.** Each model is verified against
  its own step-0 field. For an ensemble, that's the control member at step 0. The
  exceptions:
  - WeatherNext 2/3 and climatology are verified against the IFS analysis.
  - Precipitation is always verified against ERA5, because an analysis has no
    accumulated rainfall.

  So two models in one column were **not scored against the same truth**. The
  methodology argues this removes systematic offsets between centres' analyses.
- **Collapse over cases.** The API server does this, not the page. Per lead time,
  it pools all initialisation times in the window with a metric-appropriate rule:
  - RMSE and spread: root-mean-square of the per-cycle values.
  - CRPS and SEEPS: plain mean.
  - ACC: mean of Fisher-z.
  - Spread–skill: ratio of the window-RMS components.

  The methodology says all but ACC are "exact" poolings.
- **No common case set.** Models run different cycles:
  - Aurora: 00/12 UTC only.
  - Atlas: 12 UTC only.
  - IFS 06/18 UTC runs stop at 144 h.
  - GEFS is scored only to 10 days.

  Each model's value pools whatever cycles it has in the window. Nothing restricts
  the pooling to the cases every model has. (The home-page ranking does restrict to
  12 UTC cycles, for exactly this reason. The scorecard doesn't.)

## The API

`GET /api/scorecard?region=<r>&mode=det|prob&start=YYYY-MM-DD&end=YYYY-MM-DD` returns

```json
{"rows": [{"model": "aifs-ens-mean", "variable": "10m_wind_speed", "lead_time_h": 24, "value": 0.8615}, ...]}
```

- Each row has **one number and nothing else**: no case count, no spread, no
  interval.
- `end` is exclusive. The page sends the chosen end date + 1 day.
- The metric is implied: RMSE (SEEPS for precipitation) in `det` mode, and CRPS in
  `prob` mode.
- `/api/meta` gives the available initialisation range:
  `{"minInit":"2026-04-28T06:00Z","maxInit":"2026-09-25T18:00Z"}`.

For the 2026-08-09..09-08 global window, that was 603 rows (`det`: 13 models × 8
variables × 6 leads, less gaps) and 412 rows (`prob`: 9 ensembles).

## Layout

- **Controls:** region (global, tropics, N/S hemisphere, North America, Europe),
  baseline model (default HRES in `det`, IFS-ENS in `prob`), and date window.
  Deterministic vs probabilistic is a site-wide switch.
- **Two tables:** "Surface" (2 m T, MSLP, 10 m wind speed, 24 h precip) and "Upper
  air" (z500, T850, q700, wind vector 850, which is wind speed 850 in `prob` mode).
- **Rows are models**, with the baseline pinned first and greyed. **Columns** are
  variable → lead time, at days 1, 3, 5, 7, 10 and 15. There is one metric per
  variable, and no metric axis.
- **Each cell prints the model's own score value**, and its background colour shows
  the difference from the baseline.

## Colour

From module 6717:

```js
rel = (base - value) / Math.abs(base)       // positive = better than baseline
if (higherBetter) rel = -rel                // only ACC, which the scorecard doesn't show
pos = sign(rel) * (|100·rel| <= 1 ? 0 : min(1, log10(|100·rel|) / log10(50)))
```

- `pos` is looked up on an 11-stop RdBu ramp (`#67001f` … `#f7f7f7` … `#053061`),
  blue for better.
- The scale is **logarithmic in percent**. Anything within ±1 % is white, and it
  saturates at ±50 %. Legend ticks are at ±1, 2, 5, 10, 20, 50 %, labelled "Better ←
  % difference vs <baseline> → Worse".
- Text switches between black and white by luminance.
- Spread and spread–skill are flagged `directionless`, and the legend says "%
  difference", not "% improvement". But the scorecard page doesn't use that flag,
  and in `prob` mode it colours CRPS only.

Worked example from `analyse_api.py` (T850 RMSE, global, against HRES, day 5): IFS-ENS
mean 1.643 K against HRES 1.907 K is +13.9 %, giving ramp position +0.67, a
mid-blue.

## Significance

**There is none.** The page and API have no confidence intervals, no significance
marking and no case counts. A cell at +1.5 % gets the same pale-blue treatment
whether it rests on 30 cycles or 3.

## Compared with mlwp-scorecards

| | Brightband OWB | mlwp-scorecards |
|---|---|---|
| Comparison | N models against one baseline, one table | Two sources, one difference per cell |
| Truth | Each model's own analysis (IFS for some, ERA5 for precip) | One common truth source per row |
| Case set | Whatever cycles each model has in the window | Paired, the cases both sources share |
| Collapse over cases | Server-side, metric-aware (RMS for RMSE) | Here, a mean of per-case scores |
| Uncertainty | None | Block bootstrap, 68/95/99.7 %, case count shown |
| Cell content | The model's score as text, coloured by % difference | Colour by relative difference, significance border |
| Colour scale | Log in %, dead zone ±1 %, saturates at ±50 % | Binned relative difference (`colours.py`) |
| Metric per variable | Fixed (RMSE/SEEPS/CRPS) | Metric is a column axis |

Ideas worth considering:

- **N models against one baseline** as a layout: rows = models. Our Layout
  vocabulary could express this with `forecast_source` on rows, with one pairwise
  difference per row.
- **A log-percentage colour scale with a dead zone.** It's legible over a range of
  1–50 %, where a linear ramp would wash out.
- **Metric-aware pooling over cases.** OWB takes the RMS of per-cycle RMSEs, which
  is the pooled RMSE. A plain mean of per-case RMSEs, which is what we do, is lower.
  PLAN.md already notes that the resampling unit is still the case for non-linear
  pooling. This is a concrete reference for doing it.
- **Printing the value in the cell.** It makes the card readable without hovering.
