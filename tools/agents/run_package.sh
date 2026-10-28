#!/usr/bin/env bash
# Launch a work-package subagent in an isolated git worktree.
#
#   tools/agents/run_package.sh <package-id> [--dry-run]
#
# The package must have docs/agent_packages/<id>.md committed on the
# integration branch.  The agent runs with the gateway environment from
# ~/.config/mcpp-agent.env, works in ~/worktrees/mcMPP-<id> on branch
# pkg/<id>, and commits there.  The lead reviews and merges afterwards; the
# agent is explicitly forbidden from pushing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTEGRATION_BRANCH="${MCPP_INTEGRATION_BRANCH:-feat/npp-api-expansion}"
WORKTREE_ROOT="${MCPP_WORKTREE_ROOT:-$HOME/worktrees}"
AGENT_ENV="${MCPP_AGENT_ENV:-$HOME/.config/mcpp-agent.env}"
LOG_DIR="$ROOT/artifacts/agents"
TIMEOUT_SECS="${MCPP_AGENT_TIMEOUT:-21600}" # 6h

pkg="${1:-}"
dry_run=false
if [[ "${2:-}" == "--dry-run" ]]; then dry_run=true; fi
if [[ -z "$pkg" ]]; then
  echo "usage: $0 <package-id> [--dry-run]" >&2
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

worktree="$WORKTREE_ROOT/mcMPP-$pkg"
branch="pkg/$pkg"
if [[ -d "$worktree/.git" || -f "$worktree/.git" ]]; then
  echo ">> reusing worktree $worktree"
else
  mkdir -p "$WORKTREE_ROOT"
  git -C "$ROOT" worktree add -b "$branch" "$worktree" "$INTEGRATION_BRANCH"
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
Follow docs/agent_packages/<package>.md exactly. Work only in the current git worktree.
Never push to Gerrit or any remote, never force-push, never rewrite the integration branch.
Commit your work on the current branch. If a requirement is impossible or ambiguous, stop and
write the question into your final report instead of guessing. Prefer the existing executor,
policy-template and differential-test patterns over new frameworks.'

prompt="Read docs/agent_packages/$pkg.md and execute it end to end. The package function list is api_analysis/packages/$pkg.csv. Work in the current worktree, commit on the current branch ($branch), and finish with the report described in the document. Do not push."

echo ">> package : $pkg"
echo ">> worktree: $worktree"
echo ">> branch  : $branch"
echo ">> log     : $log"
echo ">> model   : ${ANTHROPIC_MODEL:-<unset>}"

cmd=(claude -p --bare --permission-mode bypassPermissions
     --append-system-prompt "$system_prompt"
     "$prompt")

if $dry_run; then
  printf '>> cd %q && ' "$worktree"
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

cd "$worktree"
set +e
timeout "$TIMEOUT_SECS" "${cmd[@]}" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e

echo ">> agent exit code: $rc"
echo ">> last commits:"
git -C "$worktree" log --oneline -5 || true
echo ">> worktree status:"
git -C "$worktree" status --short | head -30 || true
exit "$rc"
