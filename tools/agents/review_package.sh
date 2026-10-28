#!/usr/bin/env bash
# Review a completed work-package branch before merging.
#
#   tools/agents/review_package.sh <package-id>
#
# Prints commits, diffstat, unexpected files and the remaining-function count
# from the worktree's catalog.  It never modifies the worktree or merges.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTEGRATION_BRANCH="${MCPP_INTEGRATION_BRANCH:-feat/npp-api-expansion}"
WORKTREE_ROOT="${MCPP_WORKTREE_ROOT:-$HOME/worktrees}"

pkg="${1:-}"
if [[ -z "$pkg" ]]; then
  echo "usage: $0 <package-id>" >&2
  exit 2
fi
worktree="$WORKTREE_ROOT/mcMPP-$pkg"
branch="pkg/$pkg"
if [[ ! -d "$worktree" ]]; then
  echo "error: no worktree at $worktree" >&2
  exit 2
fi

echo "== branch =="
git -C "$worktree" log --oneline "$INTEGRATION_BRANCH".."$branch" || true
echo
echo "== diffstat =="
git -C "$worktree" diff --stat "$INTEGRATION_BRANCH"..."$branch" || true
echo
echo "== worktree status =="
git -C "$worktree" status --short | head -40 || true
echo
echo "== files outside expected areas =="
unexpected=$(git -C "$worktree" diff --name-only "$INTEGRATION_BRANCH"..."$branch" | \
  grep -vE '^(src/|test/|mpp_scripts/families/|docs/|api_analysis/)' || true)
if [[ -n "$unexpected" ]]; then
  echo "$unexpected"
else
  echo "(none)"
fi
echo
echo "== package coverage (from the worktree catalog) =="
if [[ -f "$worktree/api_analysis/api_catalog.csv" ]]; then
  python3 - "$worktree/api_analysis/api_catalog.csv" "$ROOT/api_analysis/packages/$pkg.csv" <<'PY'
import csv, sys
catalog = {r["function_name"]: r for r in csv.DictReader(open(sys.argv[1]))}
rows = list(csv.DictReader(open(sys.argv[2])))
from collections import Counter
c = Counter()
for r in rows:
    status = catalog.get(r["function_name"], {}).get("status", "unknown")
    c[status] += 1
print(f"total={len(rows)} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))
PY
else
  echo "(no api_catalog.csv in worktree; agent did not refresh coverage)"
fi
