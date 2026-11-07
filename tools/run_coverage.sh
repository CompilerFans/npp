#!/usr/bin/env bash
# Regenerate the API inventories and coverage reports for every SDK header
# snapshot in the tree (API-11.4 / API-12.2 / API-12.8) plus the unified
# contract used for the build (API-unified -> api_catalog.csv).
#
# Usage:
#   tools/run_coverage.sh [path-to-libmpp.so]
#
# The optional library is read with `nm -D -C` so macro-generated wrappers
# (NPPI_COPY_WRAPPER(...) and friends) count as implemented; without it only
# the source scan is used.  Per-version reports land in
# api_analysis/coverage_<version>.{csv,md}; the unified contract refreshes the
# committed api_analysis/coverage.{csv,md}.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

lib=${1:-$root/build/src/libmpp.so}
symbols=$(mktemp)
trap 'rm -f "$symbols"' EXIT

echo "== analyze_api.py (11.4 / 12.2 / 12.8)"
python3 tools/analyze_api.py --api-dir "$root" --versions 11.4 12.2 12.8 13.4 --output api_analysis

if [[ -f "$lib" ]]; then
  echo "== symbols from $lib"
  nm -D -C --defined-only "$lib" > "$symbols"
else
  echo "== no library at $lib; coverage uses the source scan only"
  : > "$symbols"
fi

echo "== union of 11.4 / 12.2 / 12.8 / 13.4"
python3 - <<'PY'
import csv
from pathlib import Path

rows = {}
for version in ("11.4", "12.2", "12.8", "13.4"):
    with open(f"api_analysis/api_functions_{version}.csv") as f:
        for r in csv.DictReader(f):
            rows[r["function_name"]] = r  # later versions win on collisions
out = Path("api_analysis/api_functions_union.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["function_name", "module", "return_type", "params", "signature"])
    w.writeheader()
    for name in sorted(rows):
        w.writerow(rows[name])
with open("api_analysis/api_catalog.csv") as f:
    catalog = {r["function_name"] for r in csv.DictReader(f)}
union = set(rows)
print(f"- Union functions: {len(union)}")
print(f"- Union == API-unified contract: {union == catalog}")
missing, extra = sorted(catalog - union), sorted(union - catalog)
if missing:
    print(f"- In unified but not in any version header: {len(missing)}")
if extra:
    print(f"- In version headers but not in unified: {len(extra)}")
PY

for version in 11.4 12.2 12.8 13.4 union; do
  echo "== coverage $version"
  python3 tools/check_coverage.py \
    --api-csv "api_analysis/api_functions_$version.csv" \
    --src-dir src --test-dir test --symbols "$symbols" \
    --output "api_analysis/coverage_$version.csv" \
    --summary "api_analysis/coverage_$version.md" > /dev/null
  grep -E "^- Total API Functions|^- Families with at least one" "api_analysis/coverage_$version.md"
done

echo "== coverage unified (API-unified contract)"
python3 tools/check_coverage.py \
  --api-csv api_analysis/api_catalog.csv \
  --src-dir src --test-dir test --symbols "$symbols" \
  --output api_analysis/coverage.csv \
  --summary api_analysis/coverage_summary.md > /dev/null
grep -E "^- Total API Functions|^- Families with at least one" api_analysis/coverage_summary.md
