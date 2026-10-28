#!/usr/bin/env bash
# Launch a work-package subagent.
#
#   tools/agents/run_package.sh <package-id> [--dry-run] [--in-place]
#
# The package must have docs/agent_packages/<id>.md committed on the
# integration branch.  The agent runs with the gateway environment from
# ~/.config/mcpp-agent.env and commits on branch pkg/<id>; the lead reviews
# and merges afterwards.  Pushes are forbidden.
#
# --in-place  work in the main checkout instead of a fresh worktree.  Fresh
#             worktrees cannot configure the cu-bridge CUDA toolchain with
#             CMake 4.4, so this is the default for this environment (also
#             selectable with MCPP_AGENT_IN_PLACE=1).  The main checkout must
#             be clean; the runner checks out pkg/<id> from the integration
#             branch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTEGRATION_BRANCH="${MCPP_INTEGRATION_BRANCH:-feat/npp-api-expansion}"
WORKTREE_ROOT="${MCPP_WORKTREE_ROOT:-$HOME/worktrees}"
AGENT_ENV="${MCPP_AGENT_ENV:-$HOME/.config/mcpp-agent.env}"
LOG_DIR="$ROOT/artifacts/agents"
TIMEOUT_SECS="${MCPP_AGENT_TIMEOUT:-21600}" # 6h

pkg="${1:-}"
dry_run=false
in_place="${MCPP_AGENT_IN_PLACE:-0}"
shift || true
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    --in-place) in_place=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
if [[ -z "$pkg" ]]; then
  echo "usage: $0 <package-id> [--dry-run] [--in-place]" >&2
  exit 2
fi

pkg_doc="$ROOT/docs/agent_packages/$pkg.md"
pkg_csv="$ROOT/api_analysis/packages/$pkg.csv"
if [[ ! -f "$pkg_doc" ]]; then
  echo "error: missing $pkg_doc" >&2
  exit 2
fi
if [[ ! -f "$pkg_csv" ]]; then
  echo "error: missing $pkg_csv (run tools/make_package_list.py --package $pkg)" >&2
  exit 2
fi
if ! git -C "$ROOT" show-ref --verify --quiet "refs/heads/$INTEGRATION_BRANCH"; then
  echo "error: integration branch $INTEGRATION_BRANCH not found" >&2
  exit 2
fi
if ! git -C "$ROOT" ls-tree -r --name-only "$INTEGRATION_BRANCH" -- "docs/agent_packages/$pkg.md" | grep -q .; then
  echo "error: $pkg_doc is not committed on $INTEGRATION_BRANCH; commit it first" >&2
  exit 2
fi

branch="pkg/$pkg"
if [[ "$in_place" == "1" ]]; then
  workdir="$ROOT"
  if ! $dry_run; then
    if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
      echo "error: main checkout is dirty; commit or stash before an in-place run" >&2
      git -C "$ROOT" status --short | head -30
      exit 2
    fi
    current=$(git -C "$ROOT" branch --show-current)
    if [[ "$current" != "$branch" ]]; then
      if git -C "$ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
        echo "error: branch $branch exists but is not checked out; merge or delete it first" >&2
        exit 2
      fi
      git -C "$ROOT" checkout -b "$branch" "$INTEGRATION_BRANCH"
    fi
  fi
else
  workdir="$WORKTREE_ROOT/mcMPP-$pkg"
  if [[ -d "$workdir/.git" || -f "$workdir/.git" ]]; then
    echo ">> reusing worktree $workdir"
  else
    mkdir -p "$WORKTREE_ROOT"
    git -C "$ROOT" worktree add -b "$branch" "$workdir" "$INTEGRATION_BRANCH"
  fi
fi

if [[ ! -f "$AGENT_ENV" ]]; then
  echo "error: missing $AGENT_ENV (gateway credentials)" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$AGENT_ENV"

# Fresh worktrees have no CMake cache; point FindCUDAToolkit at the cu-bridge
# toolkit so ./build.sh can configure without a pre-existing build directory.
BUILD_ENV="${MCPP_BUILD_ENV:-$HOME/.config/mcpp-build.env}"
if [[ -f "$BUILD_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$BUILD_ENV"
fi

mkdir -p "$LOG_DIR"
stamp="$(date +%Y%m%d-%H%M%S)"
log="$LOG_DIR/$pkg-$stamp.log"

system_prompt='You are an autonomous NPP API implementation subagent for the mcMPP repository.
Follow docs/agent_packages/<package>.md exactly. Work only in the current checkout.
Never push to Gerrit or any remote, never force-push, never rewrite the integration branch.
Commit your work on the current branch. If a requirement is impossible or ambiguous, stop and
write the question into your final report instead of guessing. Prefer the existing executor,
policy-template and differential-test patterns over new frameworks.
NEVER run cmake_maca or make_maca: they recreate and delete ~/cu-bridge, breaking the local
toolchain for everyone. Use the cached ./build.sh in the main checkout; do not try to "repair" the toolkit.'

prompt="Read docs/agent_packages/$pkg.md and execute it end to end. The package function list is api_analysis/packages/$pkg.csv. Work in the current checkout, commit on the current branch ($branch), and finish with the report described in the document. Do not push."

echo ">> package : $pkg"
echo ">> workdir : $workdir"
echo ">> branch  : $branch"
echo ">> log     : $log"
echo ">> model   : ${ANTHROPIC_MODEL:-<unset>}"

cmd=(claude -p --bare --permission-mode bypassPermissions
     --allowedTools "Bash Read Write Edit Glob Grep"
     --append-system-prompt "$system_prompt"
     "$prompt")

if $dry_run; then
  printf '>> cd %q && ' "$workdir"
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

cd "$workdir"
set +e
timeout "$TIMEOUT_SECS" "${cmd[@]}" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e

echo ">> agent exit code: $rc"
echo ">> last commits:"
git -C "$workdir" log --oneline -5 || true
echo ">> checkout status:"
git -C "$workdir" status --short | head -30 || true
exit "$rc"
