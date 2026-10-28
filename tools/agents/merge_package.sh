#!/usr/bin/env bash
# Merge a reviewed work-package branch into the integration branch and validate.
#
#   tools/agents/merge_package.sh <package-id> [--allow-partial]
#
# Refuses to merge if the package still has remaining functions unless
# --allow-partial is given.  Runs the local build, full unit suite and the
# differential verification, then refreshes coverage and the package list.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTEGRATION_BRANCH="${MCPP_INTEGRATION_BRANCH:-feat/npp-api-expansion}"
WORKTREE_ROOT="${MCPP_WORKTREE_ROOT:-$HOME/worktrees}"
GOLDEN_DIR="${NPP_GOLDEN_DIR:-$ROOT/artifacts/5090/npp_records}"

pkg="${1:-}"
allow_partial=false
if [[ "${2:-}" == "--allow-partial" ]]; then allow_partial=true; fi
if [[ -z "$pkg" ]]; then
  echo "usage: $0 <package-id> [--allow-partial]" >&2
  exit 2
fi
worktree="$WORKTREE_ROOT/mcMPP-$pkg"
branch="pkg/$pkg"
if [[ ! -d "$worktree" ]]; then
  echo "error: no worktree at $worktree" >&2
  exit 2
fi
if [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
  echo "error: worktree has uncommitted changes; commit or discard them first" >&2
  git -C "$worktree" status --short | head -30
  exit 2
fi

remaining=$(python3 - "$worktree/api_analysis/api_catalog.csv" "$ROOT/api_analysis/packages/$pkg.csv" <<'PY'
import csv, sys
catalog = {r["function_name"]: r for r in csv.DictReader(open(sys.argv[1]))}
rows = list(csv.DictReader(open(sys.argv[2])))
print(sum(1 for r in rows if catalog.get(r["function_name"], {}).get("status") != "real"))
PY
)
echo ">> package $pkg remaining functions: $remaining"
if [[ "$remaining" != "0" && "$allow_partial" != true ]]; then
  echo "error: package is incomplete; pass --allow-partial to merge anyway" >&2
  exit 2
fi

current_branch=$(git -C "$ROOT" branch --show-current)
if [[ "$current_branch" != "$INTEGRATION_BRANCH" ]]; then
  echo ">> switching $ROOT to $INTEGRATION_BRANCH"
  git -C "$ROOT" checkout "$INTEGRATION_BRANCH"
fi

echo ">> merging $branch into $INTEGRATION_BRANCH"
git -C "$ROOT" merge --no-ff "$branch" -m "Merge $branch: $pkg"

echo ">> building"
(cd "$ROOT" && ./build.sh >/tmp/mcpp_merge_build.log 2>&1) || { tail -40 /tmp/mcpp_merge_build.log; exit 1; }

echo ">> full unit suite"
(cd "$ROOT" && LD_LIBRARY_PATH="$ROOT/build/src" ./build/unit_tests) | tail -5

echo ">> differential verification"
if [[ -d "$GOLDEN_DIR" ]]; then
  (cd "$ROOT" && NPP_VERIFY_DIR="$GOLDEN_DIR" LD_LIBRARY_PATH="$ROOT/build/src" ./build/diff_tests) | tail -8
else
  echo "(no goldens at $GOLDEN_DIR; skipping diff verification)"
fi

echo ">> coverage + package list"
(cd "$ROOT" && ./update_coverage.sh >/tmp/mcpp_merge_cov.log 2>&1)
(cd "$ROOT" && python3 tools/make_package_list.py --package "$pkg") || true

echo ">> merge complete; review docs/npp_api_progress.md and commit the ledger update"
