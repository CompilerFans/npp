#!/usr/bin/env bash
# Verify a package branch in the main checkout (build + unit + diff).
#
#   tools/agents/verify_package.sh <package-id> [--keep]
#
# Used with the coding-parallel / verify-serial workflow: package agents run in
# worktrees with --no-build, then this script builds and tests each pkg/<id>
# branch serially in the main checkout (the only place with a working cu-bridge
# build cache).
#
# Refuses to run while another package/ref agent owns the main checkout.
# Logs: /tmp/verify_<pkg>_<stamp>.*.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_ENV="${MCPP_BUILD_ENV:-$HOME/.config/mcpp-build.env}"
LOCK="/tmp/mcpp_verify.lock"

pkg="${1:-}"
shift || true
keep=0
for arg in "$@"; do
  case "$arg" in
    --keep) keep=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
if [[ -z "$pkg" ]]; then
  echo "usage: $0 <package-id> [--keep]" >&2
  exit 2
fi

branch="pkg/$pkg"
git -C "$ROOT" show-ref --verify --quiet "refs/heads/$branch" || {
  echo "error: branch $branch not found" >&2; exit 2
}

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "error: another verification is running ($LOCK)" >&2
  exit 2
fi

if pgrep -f 'run_package\.sh' >/dev/null 2>&1; then
  echo "error: a package agent owns the main checkout; wait until it exits" >&2
  pgrep -af 'run_package\.sh' | head -5 >&2
  exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
  echo "error: main checkout is dirty; commit or stash first" >&2
  git -C "$ROOT" status --short | head -20 >&2
  exit 2
fi

orig="$(git -C "$ROOT" branch --show-current)"
stamp="$(date +%Y%m%d-%H%M%S)"
blog="/tmp/verify_${pkg}_${stamp}.build.log"
ulog="/tmp/verify_${pkg}_${stamp}.unit.log"
dlog="/tmp/verify_${pkg}_${stamp}.diff.log"

restore() {
  if [[ "$keep" != "1" ]]; then
    git -C "$ROOT" checkout -q "$orig" 2>/dev/null || true
  fi
}
trap restore EXIT

echo ">> verify $branch (from $orig)"
git -C "$ROOT" checkout -q "$branch"

if [[ -f "$BUILD_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$BUILD_ENV"
fi

echo ">> build"
set +e
( cd "$ROOT" && ./build.sh ) >"$blog" 2>&1
build_rc=$?
set -e
if grep -qE 'ninja: error|CMake Error|(^| )error:' "$blog"; then
  build_rc=1
fi
if [[ $build_rc -ne 0 ]]; then
  echo "FAIL build (log: $blog)"
  grep -E 'ninja: error|CMake Error|(^| )error:' "$blog" | head -10
  exit 1
fi
echo "OK build"

echo ">> unit tests"
set +e
( cd "$ROOT" && LD_LIBRARY_PATH=build/src ./build/unit_tests ) >"$ulog" 2>&1
set -e
unit_summary="$(grep -E '\[  PASSED  \]|\[  FAILED  \]' "$ulog" | tail -2 | tr '\n' ' ')"
if grep -q '\[  FAILED  \]' "$ulog"; then
  echo "FAIL unit: $unit_summary (log: $ulog)"
  exit 1
fi
echo "OK unit: $unit_summary"

echo ">> differential tests"
set +e
( cd "$ROOT" && NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src ./build/diff_tests ) >"$dlog" 2>&1
set -e
diff_summary="$(grep -E '\[  PASSED  \]|\[  FAILED  \]' "$dlog" | tail -2 | tr '\n' ' ')"
if grep -q '\[  FAILED  \]' "$dlog"; then
  echo "FAIL diff: $diff_summary (log: $dlog)"
  exit 1
fi
echo "OK diff: $diff_summary"

echo "PASS $branch"
