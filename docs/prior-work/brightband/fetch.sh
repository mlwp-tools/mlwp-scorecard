#!/usr/bin/env bash
# Download the Brightband Operational WeatherBench scorecard page, its JS bundles,
# the methodology page and sample API responses into ./source/.
#
# The site is a Next.js app that renders in the browser, so the page HTML alone is
# empty; the logic is in the JS chunks and the numbers come from /api/scorecard.
set -euo pipefail

BASE=https://owb.brightband.com
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/source
mkdir -p "$OUT/chunks" "$OUT/api"

fetch() { curl -sSfL "$1" -o "$2"; echo "  $2 ($(wc -c <"$2" | tr -d ' ') bytes)"; }

echo "pages"
fetch "$BASE/scorecard?start=2026-08-09&end=2026-09-08" "$OUT/scorecard.html"
fetch "$BASE/methodology" "$OUT/methodology.html"

echo "js chunks"
for page in "$OUT/scorecard.html" "$OUT/methodology.html"; do
  grep -o '/_next/static/[^"?]*\.\(js\|css\)' "$page"
done | sort -u | while read -r path; do
  fetch "$BASE$path" "$OUT/chunks/$(basename "$path")"
done

echo "api"
fetch "$BASE/api/meta" "$OUT/api/meta.json"
echo "  (scorecard responses are fetched by fetch_api.py, which needs the region/mode options from the JS)"
