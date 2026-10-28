#!/usr/bin/env bash
# One-command validation helpers for API work packages.
#
#   tools/validate.sh local  [target] [filter]  build locally and run a target
#   tools/validate.sh full   [filter]           run the full local unit suite
#   tools/validate.sh record [filter]           record goldens on the 5090
#   tools/validate.sh verify [dir] [filter]     verify locally against goldens
#   tools/validate.sh diff   [filter]           record + pull + verify
#   tools/validate.sh remote [target] [filter]  run a target on the 5090
#
# Defaults: target=diff_tests, golden dir=artifacts/5090/npp_records.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOLDEN_DIR="${NPP_GOLDEN_DIR:-$ROOT/artifacts/5090/npp_records}"

local_build() {
  (cd "$ROOT" && ./build.sh --lib-only >/dev/null)
  (cd "$ROOT" && cmake --build build -j"$(nproc)")
}

local_test() {
  local target="${1:-diff_tests}"
  local filter="${2:-}"
  local bin="$ROOT/build/$target"
  if [[ ! -x "$bin" ]]; then
    echo "error: $bin not built" >&2
    exit 2
  fi
  # The shell's global LD_LIBRARY_PATH points at the installed MACA SDK, which
  # may be older than the tree under test.  Prefer the freshly built library.
  export LD_LIBRARY_PATH="$ROOT/build/src:${LD_LIBRARY_PATH:-}"
  if [[ -n "$filter" ]]; then
    "$bin" --gtest_filter="$filter"
  else
    "$bin"
  fi
}

case "${1:-}" in
  local)
    shift
    local_build
    local_test "${1:-diff_tests}" "${2:-}"
    ;;
  full)
    shift
    local_build
    local_test unit_tests "${1:-}"
    ;;
  record)
    shift
    "$ROOT/tools/remote5090.sh" record "${1:-}"
    "$ROOT/tools/remote5090.sh" pull
    ;;
  verify)
    shift
    dir="${1:-$GOLDEN_DIR}"
    filter="${2:-}"
    NPP_VERIFY_DIR="$dir" local_test diff_tests "$filter"
    ;;
  diff)
    shift
    filter="${1:-}"
    "$ROOT/tools/remote5090.sh" record "$filter"
    "$ROOT/tools/remote5090.sh" pull
    NPP_VERIFY_DIR="$GOLDEN_DIR" local_test diff_tests "$filter"
    ;;
  remote)
    shift
    "$ROOT/tools/remote5090.sh" test "${1:-diff_tests}" "${2:-}"
    ;;
  *)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
