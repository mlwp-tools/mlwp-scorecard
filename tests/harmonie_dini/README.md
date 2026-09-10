# HARMONIE-AROME DINI vs ECMWF AIFS

The synthetic tests prove the package computes what it claims on data whose answer
is known in advance. This folder does the same against **real operational output**,
where the answer is not known and the card has to earn its keep.

Two prediction sources:

| source | what it is |
|---|---|
| `harmonie-arome` | DMI's operational limited-area physics model, ~2 km |
| `aifs` | ECMWF's global data-driven model, 0.25 deg |

scored against **two independent truths**:

| truth | what it is | the catch |
|---|---|---|
| `dini-analysis` | gridded, whole DINI domain | it is HARMONIE's own state, so it favours HARMONIE |
| `observations` | 61 DMI stations | neutral, but only ~60 points over Denmark, and a point is not a grid mean |

Neither is "the" truth, and they do not agree. Showing both on one card is the
honest presentation, and is what the `truth_source` dimension is for -- the ECMWF
reference card stacks its `an` and `ob` blocks the same way.

## Sources

```
s3://harmonie-zarr/dini/control/<analysis-time>/single_levels.zarr
arraylake  danish-meteorological-institute/ecmwf-aifs-single-forecast-subscription
https://opendataapi.dmi.dk/v2/metObs        (open, no API key)
```

The metObs endpoint moved: `dmigw.govcloud.dk` was retired on 2026-06-30 and no
longer resolves at all. Anything still pointing at it fails at DNS rather than with
a useful error.

## Running it

```bash
uv run python tests/harmonie_dini/inspect_store.py     # what is in the bucket
uv run python tests/harmonie_dini/inspect_aifs.py      # what is in the AIFS repo
uv run python tests/harmonie_dini/check_alignment.py   # where they can be compared
uv run python tests/harmonie_dini/build_datasets.py    # truth and HARMONIE forecast
uv run python tests/harmonie_dini/build_aifs.py        # AIFS, regridded to DINI
uv run python tests/harmonie_dini/probe_metobs.py      # check the metObs API shape
uv run python tests/harmonie_dini/build_observations.py  # DMI station observations
uv run python tests/harmonie_dini/verify.py            # score both, against both truths
uv run python tests/harmonie_dini/make_card.py
```

Needs AWS credentials for the `dmidev-mlflow` profile (override with
`HARMONIE_AWS_PROFILE`) and `arraylake auth login`. Extraction is ~220 DINI field
reads, about 3 minutes, and lands ~70 MB locally after striding the grid by 8.

`test_harmonie.py` runs against the local datasets if they exist and skips
otherwise, so the suite stays runnable without credentials.

## Local datasets

| dataset | dims | what it is |
|---|---|---|
| `truth.zarr` | `(time, y, x)` | the DINI analysis at every analysis time |
| `forecast.zarr` | `(init_time, lead_time, y, x)` | HARMONIE-AROME |
| `aifs.zarr` | `(init_time, lead_time, y, x)` | AIFS, interpolated to the DINI points |
| `observations.zarr` | `(time, station)` | DMI station reports at the valid times |
| `verification.zarr` | | the summary the scorecard package consumes |

## Where the two models can actually be compared

They agree on nothing by default. `check_alignment.py` establishes the intersection
rather than assuming one:

- **Initialisations**: AIFS is 6-hourly, DINI 3-hourly. Five shared initialisations
  are used, all of which have a DINI analysis at every valid time.
- **Lead times**: AIFS carries 6-hourly steps, DINI hourly, so +6 h to +36 h at
  6-hourly. Lead 0 is excluded: it is the analysis both models start from, so it
  compares nothing.
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

**The choice of truth reverses the verdict.** RMSE, percent better than AIFS
(positive = HARMONIE better):

```
                          6h    12h    18h    24h    30h    36h
vs the DINI analysis  (HARMONIE's own state)
  pres_seasurface   26.3  -13.2  -23.8  -48.7  -63.7  -41.2
  t2m               63.6   45.0   35.0   29.6   28.6   31.1
  wind_speed_10m    46.9   23.8   12.3    5.7    1.9    1.0

vs 61 DMI stations    (neutral)
  pres_seasurface  -48.6  -23.6  -44.5    6.9  -62.9  -12.5
  t2m              -16.0  -33.9  -44.4  -49.0  -44.1  -35.8
  wind_speed_10m     0.5    2.7    2.4    3.9   -1.1    2.0
```

On 2 m temperature, HARMONIE leads by 29–64% against the analysis and *loses* by
16–49% against observations. Same models, same period, same metric; opposite
conclusion. Mean sea level pressure moves the same way. Only 10 m wind is roughly
truth-independent, and there the two models are near parity either way.

This is the home-advantage effect made quantitative. The analysis is the field
HARMONIE was initialised from and shares its physics and orography, so scoring
HARMONIE against it partly measures self-consistency rather than skill. A study
that used only the analysis would have reported the opposite result with nothing on
the page to suggest it was an artefact of the choice of truth.

It is *not* evidence that AIFS is the better model. The observation comparison has
its own biases, listed below. What it is evidence for is that a single truth source
cannot settle the question, which is why the card carries both.

Two further readings worth having:

- **AIFS's `t2m` error against the analysis is nearly flat at ~1.1 K** across all
  lead times. Forecast error grows; this does not, so most of it is
  representativeness — a 0.25 deg field cannot reproduce 2 km detail. Against
  station points that penalty largely disappears, which is part of why the sign
  flips.
- The reverse card is not the numeric negative of this one. A relative difference is
  normalised by whichever source is the control, so "+26% better than AIFS" and
  "−36% worse than HARMONIE" describe the same gap from opposite ends. Only the
  signs mirror, which is what `test_swapping_control_and_experiment_flips_the_card`
  asserts.

## Read these results with care

Four things make this a pipeline demonstration rather than evidence about either
model. All are recorded in the dataset attributes and repeated on the card:

1. **The DINI analysis is not neutral.** It is HARMONIE's own state. Its block on
   the card should be read as "how self-consistent is HARMONIE", not "how good".
2. **Neither are the observations.** 61 stations over Denmark only, against models
   covering the whole DINI domain; and a 2 m thermometer at one spot is not a 2 km
   grid mean. Models are sampled at the nearest grid point, a median 6.5 km away,
   which is itself a source of error charged to the model.
3. **The coarse model is interpolated onto the fine grid.** The conventional
   direction, and it leaves the truth untouched, but it charges AIFS for detail it
   never claimed to resolve.
4. **Five initialisations spanning 24 hours.** A small and heavily autocorrelated
   sample, bootstrapped iid, so the intervals are optimistic. The card comes out
   **92% significant**, which should be read as a warning about the resampling
   rather than as a strong result — see *Two ways this goes wrong* in `PLAN.md`,
   where a naive per-case bootstrap is measured reporting significance 44% of the
   time on data with no effect at all. A moving-block bootstrap needs more than
   five cases to be possible at all, so the honest fix here is more
   initialisations.

## What the tests actually check

Neither model is known to be right, so these assert the things that would be wrong
if the pipeline were broken rather than any expected answer:

- **Valid-time alignment** is checked directly, by feeding the truth back in as a
  perfect forecast and requiring a score of exactly zero — then requiring that a
  one-step lead offset does *not* score zero, so the check cannot pass by reading
  the same field at every lead.
- **Units** — a Celsius/kelvin slip would put `t2m` RMSE near 273 rather than 1.
- **Regridding** left no holes, which is the longitude-convention failure.
- **The card is mixed.** A uniformly one-colour card would usually mean the
  comparison had collapsed, not that one model won everything.
- **The paired difference interval is 1.0x to 3.2x tighter** than treating the two
  sources as independent, so the naive version would have hidden real results.
- **The two truths disagree**, and in the specific direction expected: HARMONIE
  scores relatively better against its own analysis than against observations, on
  every variable. That is asserted rather than left as an observation, because if
  it ever stopped holding it would mean either the observation pipeline or the
  analysis pipeline had broken.
- **metObs units** — the API reports Celsius and hectopascals, and the extraction
  converts to K and Pa.
- **Every station falls inside the model domain**, within 25 km of a grid point.
