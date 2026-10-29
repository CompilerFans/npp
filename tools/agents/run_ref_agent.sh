#!/usr/bin/env bash
# Launch a reference-test development subagent in an isolated worktree.
#
#   tools/agents/run_ref_agent.sh <id> <module.h> [<module.h> ...]
#
# Each agent gets:
#   * branch ref/<id> and worktree ~/worktrees/mcMPP-ref-<id>
#   * remote 5090 directory /home/metax/mcMPP-ref-<id> (MCPP_5090_DIR)
#   * the user-scope mcpp-5090 MCP server for sync/build/test/record/pull
# Agents commit on ref/<id>; they never push.  The lead reviews and merges.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTEGRATION_BRANCH="${MCPP_INTEGRATION_BRANCH:-feat/npp-api-expansion}"
WORKTREE_ROOT="${MCPP_WORKTREE_ROOT:-$HOME/worktrees}"
AGENT_ENV="${MCPP_AGENT_ENV:-$HOME/.config/mcpp-agent.env}"
MCP_CFG="${MCPP_5090_MCP_CONFIG:-$HOME/.config/mcpp-5090-mcp.json}"
LOG_DIR="$ROOT/artifacts/agents"
TIMEOUT_SECS="${MCPP_AGENT_TIMEOUT:-21600}"
REF_DOC="${MCPP_REF_DOC:-REFERENCE_TESTS.md}"

id="${1:-}"
shift || true
modules="$*"
if [[ -z "$id" || -z "$modules" ]]; then
  echo "usage: $0 <id> <module.h> [<module.h> ...]" >&2
  exit 2
fi
if [[ "$modules" == "ALL" ]]; then
  modules_desc="all modules (scope defined by docs/agent_packages/$REF_DOC)"
else
  modules_desc="$modules"
fi

branch="ref/$id"
worktree="$WORKTREE_ROOT/mcMPP-ref-$id"
if [[ -d "$worktree/.git" || -f "$worktree/.git" ]]; then
  echo ">> reusing worktree $worktree"
else
  mkdir -p "$WORKTREE_ROOT"
  git -C "$ROOT" worktree add -b "$branch" "$worktree" "$INTEGRATION_BRANCH"
fi

# Reference builds need googletest; worktrees start with uninitialized submodules.
git -C "$worktree" submodule update --init --recursive >/dev/null 2>&1 || true

if [[ ! -f "$AGENT_ENV" ]]; then
  echo "error: missing $AGENT_ENV" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$AGENT_ENV"

# Isolated remote directory for this agent.
export MCPP_5090_DIR="/home/metax/mcMPP-ref-$id"

mkdir -p "$LOG_DIR"
stamp="$(date +%Y%m%d-%H%M%S)"
log="$LOG_DIR/ref-$id-$stamp.log"

allowed="Bash Read Write Edit Glob Grep mcp__mcpp-5090__sync mcp__mcpp-5090__build mcp__mcpp-5090__test mcp__mcpp-5090__record mcp__mcpp-5090__pull mcp__mcpp-5090__status mcp__mcpp-5090__run"

system_prompt="You are a reference-test development subagent for mcMPP.
Read docs/agent_packages/$REF_DOC and execute it end to end.
Only work in the current worktree and the isolated remote directory in MCPP_5090_DIR.
Do not modify the integration branch, do not push, never run cmake_maca/make_maca.
Commit generated tests, skip manifests and module overrides on ref/<id>.
Keep changes to the shared generator small and general; report them."

prompt="Read docs/agent_packages/$REF_DOC. Your assigned modules: $modules_desc. Generate, build, and record reference tests for these modules on the 5090 until record is fully green (allowed skips documented). Commit on the current branch ($branch) and finish with the report described in the document. Do not push."

echo ">> agent   : $id"
echo ">> doc     : $REF_DOC"
echo ">> modules : $modules_desc"
echo ">> worktree: $worktree"
echo ">> remote  : $MCPP_5090_DIR"
echo ">> log     : $log"
echo ">> model   : ${ANTHROPIC_MODEL:-<unset>}"

cmd=(claude -p --bare --permission-mode bypassPermissions
     --mcp-config "$MCP_CFG" --strict-mcp-config
     --allowedTools "$allowed"
     --append-system-prompt "$system_prompt"
     "$prompt")

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
