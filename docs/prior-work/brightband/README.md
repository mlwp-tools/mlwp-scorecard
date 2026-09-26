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
- **Truth: each model against its own analysis.** A forecast valid at *t* is
  verified against the step-0 field of the same system's run initialised at *t*.
  For an ensemble, that's the control member at step 0. The exceptions:
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
  12 UTC cycles, for exactly this reason. The scorecard doesn't.) See *Possible
  shortcomings* below for what this does to a cell.

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

## Possible shortcomings

Each item below says how it was established.

1. **The models don't share a set of forecast cases** *(from the methodology; not
   confirmed from the data, because the API returns no case counts)*. The scorecard
   pools "all initialization times in the selected date window", and models run
   different cycles. In a 30-day window at day 1, that is up to about 120 cases for
   IFS-ENS (00/06/12/18 UTC), about 60 for Aurora (00/12 UTC) and about 30 for Atlas
   (12 UTC). Beyond 144 h, IFS-ENS falls back to its 00/12 UTC runs, and GEFS stops
   at 10 days. So a cell compares a model with the baseline over different sets of
   days' weather, with different amounts of noise. The home-page ranking avoids this
   by ranking only 12 UTC cycles; the scorecard doesn't.
2. **…and so not at the same times of day** *(follows from 1)*. Lead time and cycle
   together fix the valid time. At day 1, Atlas is verified only at 12 UTC valid
   times, while IFS-ENS is averaged over all four synoptic hours. For
   diurnally-varying fields, especially 2 m temperature, part of the colour can come
   from which hours were scored, not from model skill.
3. **The models aren't scored against the same truth** *(stated in the
   methodology)*. Each model is verified against its own analysis, with the IFS
   analysis for WeatherNext and climatology, and ERA5 for precipitation. A cell
   therefore mixes model skill with differences between the verifying analyses.
   - **What the truth is.** A forecast valid at *t* is verified against the step-0
     field of the same system's run initialised at *t*, not against its own initial
     state. So OWB keeps every cycle's step-0 field as a historical series of
     analyses.
   - **Where it flatters.** That analysis blends the system's short-range forecast
     with observations. Where observations are sparse it is mostly the model's own
     forecast, so errors the model shares with its own data assimilation are partly
     in the truth and don't count against it. The effect is largest at short lead
     times.
   - **How many truths there really are.** The ML models initialised from the IFS
     analysis (GraphCast, Aurora, Atlas, and probably AIFS) have a step-0 field that
     is essentially that IFS analysis. So "own analysis" probably reduces to a few
     actual truths: IFS, GFS/GEFS, and ERA5 for precipitation. This is inferred; the
     methodology doesn't say it.

   The methodology presents own-analysis verification as the operational convention
   and a deliberate trade-off.
4. **No uncertainty and no case count** *(confirmed from the page code and the
   API)*. Nothing on the card separates a real difference from noise, so 1–3 can't be
   judged from the card either.
5. **The ±1 % dead zone hides the differences that matter for mature systems**
   *(confirmed from module 6717)*. Anything within ±1 % is white. NWP cycle upgrades
   typically move scores by about that much: the ECMWF 47r1 card's z500 n.hem
   RMSE improvement at T+24 is 0.96 %, highly significant. On this card it would be
   blank. That suits comparing very different models, but not closely matched ones.
6. **The truth grid differs for some models** *(stated in the methodology)*. GEFS
   upper air is verified against a 0.5° analysis replicated onto 0.25°, which is a
   smoother truth than everyone else's. The methodology itself warns against
   over-reading those cells.

Items 1 and 2 could be confirmed with the time-series view (chunk
`page-0b3fffd74aa12316.js`). Its API is likely to return one value per
initialisation time, which would give the actual case count per model.

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
