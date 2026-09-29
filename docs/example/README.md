# Example: GraphCast vs IFS HRES, from WeatherBench 2

The scorecard at the top of the project README, end to end.

| file | what it does | needs |
|---|---|---|
| `score_weatherbench2.py` | reads GraphCast and IFS HRES forecasts and ERA5 from WeatherBench 2's public bucket, and writes per-case, area-weighted RMSE for 2020 to `tmp/wb2/wb2_graphcast_vs_hres_2020.nc` (gitignored): the package's input shape, `rmse.t2m`, `rmse.msl`, `rmse.ws10` over `forecast_source`, `spatial_region`, `lead_time`, `init_time` | network, ~0.3 GB, ~5 min |
| `make_card.py` | draws `docs/images/scorecard.png` and `scorecard.html` from those scores | the scores, and this package |

```bash
uv run --extra netcdf python docs/example/score_weatherbench2.py
uv run --extra netcdf --extra static python docs/example/make_card.py
```

Only the image and the page are committed. The scores are about 0.7 MB of noisy
floats that barely compress, and are remade from public data by the first script.

Why two scripts: `mlwp-scorecards` draws scorecards from scores it is given -- one
score per forecast case -- and never computes them itself. Normally that is done
beforehand with a verification tool such as
[mxalign](https://github.com/mlwp-tools/mxalign), on the full-resolution fields.
`score_weatherbench2.py` is a small stand-in for that step, and `make_card.py` is
the part that uses this package. What the scoring script does:

- **Data**: WeatherBench 2's 64×32 (5.625°) conservatively regridded copies of
  GraphCast (2020), IFS HRES and ERA5, read over HTTPS with no credentials.
- **Cases**: every 00 and 12 UTC initialisation of 2020 (732), lead times 1 to 10
  days every 24 h. That is enough for the default 10-day blocked bootstrap.
- **Score**: RMSE against ERA5 at the forecast's valid time, weighted by cos(latitude)
  within each region: global, the two extratropics (poleward of 20°), the tropics,
  and Europe (35–72°N, 12.5°W–42.5°E; only a few points on this grid).
- **Surface fields only** (2 m temperature, mean sea-level pressure, 10 m wind
  speed). The stores are chunked as whole global fields, with every pressure level
  in one chunk (37 for GraphCast), so a single upper-air field such as z500 would
  cost ~2 GB more; scoring a smaller region would not reduce the download at all.

Against ERA5 GraphCast has an advantage HRES does not: it was trained on ERA5, and
HRES starts from its own analysis. WeatherBench 2 reports HRES against its own
analysis for that reason; this example uses one common truth to keep the card
simple.
