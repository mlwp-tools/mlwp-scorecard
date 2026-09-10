# HARMONIE-AROME DINI vs ECMWF AIFS

The synthetic tests prove the package computes what it claims on data whose answer
is known in advance. This folder does the same against **real operational output**,
where the answer is not known and the card has to earn its keep.

Three prediction sources, one common truth (the DINI analysis):

| source | what it is |
|---|---|
| `harmonie-arome` | DMI's operational limited-area physics model, ~2 km |
| `aifs` | ECMWF's global data-driven model, 0.25 deg |
| `persistence` | the analysis at initialisation, held constant — a floor, not a competitor |

All three are scored into one `verification.zarr`, so any pairwise card can be
rendered from it without re-scoring. Which source is control and which is
experiment is an argument.

## Sources

```
s3://harmonie-zarr/dini/control/<analysis-time>/single_levels.zarr
arraylake  danish-meteorological-institute/ecmwf-aifs-single-forecast-subscription
```

## Running it

```bash
uv run python tests/harmonie_dini/inspect_store.py     # what is in the bucket
uv run python tests/harmonie_dini/inspect_aifs.py      # what is in the AIFS repo
uv run python tests/harmonie_dini/check_alignment.py   # where they can be compared
uv run python tests/harmonie_dini/build_datasets.py    # truth, forecast, persistence
uv run python tests/harmonie_dini/build_aifs.py        # AIFS, regridded to DINI
uv run python tests/harmonie_dini/verify.py            # score all three
uv run python tests/harmonie_dini/make_card.py --pair all
```

Needs AWS credentials for the `dmidev-mlflow` profile (override with
`HARMONIE_AWS_PROFILE`) and `arraylake auth login`. Extraction is ~220 DINI field
reads, about 3 minutes, and lands ~80 MB locally after striding the grid by 8.

`test_harmonie.py` runs against the local datasets if they exist and skips
otherwise, so the suite stays runnable without credentials.

## Local datasets

| dataset | dims | what it is |
|---|---|---|
| `truth.zarr` | `(time, y, x)` | the DINI analysis at every analysis time |
| `forecast.zarr` | `(init_time, lead_time, y, x)` | HARMONIE-AROME |
| `aifs.zarr` | `(init_time, lead_time, y, x)` | AIFS, interpolated to the DINI points |
| `persistence.zarr` | `(init_time, lead_time, y, x)` | the analysis at init, held constant |
| `verification.zarr` | the summary the scorecard package consumes |

## Where the two models can actually be compared

They agree on nothing by default. `check_alignment.py` establishes the intersection:

- **Initialisations**: AIFS is 6-hourly, DINI 3-hourly. Five shared initialisations
  are used, all of which have a DINI analysis at every valid time.
- **Lead times**: AIFS carries 6-hourly steps, DINI hourly, so +6 h to +36 h at
  6-hourly. Lead 0 is excluded: persistence *is* the analysis there, so a relative
  difference against it is undefined.
- **Variables**: `t2m` and `pres_seasurface` map directly; `wind_speed_10m` is
  derived from the 10 m u/v components both models carry.
- **Grid**: AIFS is bilinearly interpolated from 0.25 deg onto the DINI points.

Two traps, both silent if missed:

- **AIFS reports temperature in degrees Celsius**, DINI in kelvin.
- **DINI longitudes run 316–400 deg** (a 0–360 convention that wraps past 360)
  where AIFS uses −180…180. Interpolating without folding puts every eastern DINI
  point outside the source grid and NaNs half the domain. `build_aifs.py` reports
  the finite fraction after regridding for exactly this reason.

## What the result says

`harmonie-vs-aifs`, percent better than AIFS (positive = HARMONIE better):

```
                        6h    12h    18h    24h    30h    36h
pres_seasurface rmse  26.3  -13.2  -23.8  -48.7  -63.7  -41.2
t2m             rmse  63.6   45.0   35.0   29.6   28.6   31.1
wind_speed_10m  rmse  46.9   23.8   12.3    5.7    1.9    1.0
```

Three different stories, each a coherent block rather than scattered cells:

- **Mean sea level pressure**: HARMONIE wins at +6 h, then AIFS takes over from
  +12 h and leads by 40–60% thereafter. MSLP is a smooth synoptic field, so the
  regridding penalty on AIFS is small and this is close to a fair fight.
- **2 m temperature**: HARMONIE wins throughout, but the margin halves from +6 h to
  +30 h. AIFS's error is nearly flat at ~1.1 K across all lead times, which is the
  signature of *representativeness* error rather than forecast error — a 0.25 deg
  field cannot reproduce 2 km detail, and that floor does not grow with lead time.
- **10 m wind speed**: HARMONIE wins early and converges to parity by +36 h.

## Read these results with care

Three things make this a pipeline demonstration rather than evidence about either
model, and all three are recorded in the dataset attributes and repeated on the card:

1. **The truth is HARMONIE's own analysis.** It is the state DINI was initialised
   from and is consistent with DINI's physics and orography. AIFS is being judged
   against a competitor's analysis. A fair comparison would verify both against
   observations, or against a third-party analysis.
2. **The coarse model is interpolated onto the fine grid.** This is the conventional
   direction and leaves the truth untouched, but it is not neutral: it charges AIFS
   for detail it never claimed to resolve. The flat ~1.1 K `t2m` floor is mostly
   this.
3. **Five initialisations spanning 24 hours.** That is a small and heavily
   autocorrelated sample, and the bootstrap is iid over it, so the intervals are
   optimistic. See *Two ways this goes wrong* in `PLAN.md`.

## A note on the colour scale

Against persistence, every improvement is 60–94% and lands in the top three of the
scale's fourteen steps, so that card correctly says "much better everywhere" but
cannot resolve within it. Against AIFS the differences are 1–64% and the scale
works well. The default breakpoints are spaced for comparing successive model
cycles; a model-versus-baseline card wants a coarser scale.
