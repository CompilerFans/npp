#!/usr/bin/env bash
# Drive the RTX 5090 / CUDA 13.4 reference machine.
#
# Usage:
#   tools/remote5090.sh sync                    rsync this tree to the remote host
#   tools/remote5090.sh build [dir]             configure + build NVIDIA NPP targets
#   tools/remote5090.sh test [target] [filter]  run diff_tests (default) or a target
#   tools/remote5090.sh record [filter]         record goldens with NPP_RECORD_DIR
#   tools/remote5090.sh pull [src] [dest]       copy artifacts back
#   tools/remote5090.sh run '<command>'         run an arbitrary remote command
#   tools/remote5090.sh shell                   interactive shell
#
# Credentials and host configuration live outside the repository in
# ~/.config/mcpp-5090.env (chmod 600), e.g.:
#   export SSHPASS='...'
#   export MCPP_5090_HOST='metax@10.2.120.6'
#   export MCPP_5090_DIR='~/mcMPP'
# They may also be provided through the environment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${MCPP_5090_ENV:-$HOME/.config/mcpp-5090.env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

HOST="${MCPP_5090_HOST:-metax@10.2.120.6}"
REMOTE_DIR="${MCPP_5090_DIR:-~/mcMPP}"
BUILD_DIR="${MCPP_5090_BUILD_DIR:-build-nvidia}"
RECORD_DIR_REMOTE="${MCPP_5090_RECORD_DIR:-$REMOTE_DIR/$BUILD_DIR/npp_records}"
RECORD_DIR_LOCAL="${MCPP_5090_RECORD_LOCAL:-$ROOT/artifacts/5090/npp_records}"
REMOTE_PATH='export PATH=$HOME/tools/cmake/bin:$HOME/tools:/usr/local/cuda/bin:$PATH'
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

if [[ -z "${SSHPASS:-}" ]]; then
  echo "error: SSHPASS is not set (expected in $ENV_FILE)" >&2
  exit 2
fi

ssh_run() {
  sshpass -e ssh "${SSH_OPTS[@]}" "$HOST" "$@"
}

sync_tree() {
  echo ">> rsync $ROOT/ -> $HOST:$REMOTE_DIR/"
  sshpass -e rsync -az --delete \
    --exclude=.git \
    --exclude=build \
    --exclude=build-* \
    --exclude=build_maca* \
    --exclude=.claude \
    --exclude=tools/__pycache__ \
    --exclude='*.pyc' \
    --exclude=.npp_backup_full \
    -e "ssh ${SSH_OPTS[*]}" \
    "$ROOT/" "$HOST:$REMOTE_DIR/"
}

remote_build() {
  local dir="${1:-$BUILD_DIR}"
  local jobs="${MCPP_5090_JOBS:-20}"
  echo ">> remote build ($dir, -j$jobs)"
  ssh_run "$REMOTE_PATH; cd $REMOTE_DIR && \
    (cmake -S . -B $dir -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DBUILD_TESTS=ON -DBUILD_BENCHMARK=OFF \
       -DNPP_WARNINGS_AS_ERRORS=OFF -DUSE_NVIDIA_NPP=ON >/tmp/mcpp_cfg.log 2>&1) && \
    cmake --build $dir -j$jobs"
}

remote_test() {
  local target="${1:-diff_tests}"
  local filter="${2:-}"
  local dir="${MCPP_5090_BUILD_DIR:-build-nvidia}"
  local cmd="cd $REMOTE_DIR/$dir && ./$target"
  if [[ -n "$filter" ]]; then
    cmd="$cmd --gtest_filter='$filter'"
  fi
  ssh_run "$cmd"
}

remote_record() {
  local filter="${1:-}"
  local dir="${MCPP_5090_BUILD_DIR:-build-nvidia}"
  local out="$RECORD_DIR_REMOTE"
  local cmd="mkdir -p $out && cd $REMOTE_DIR/$dir && NPP_RECORD_DIR=$out ./diff_tests"
  if [[ -n "$filter" ]]; then
    cmd="$cmd --gtest_filter='$filter'"
  fi
  ssh_run "$cmd"
}

case "${1:-}" in
  sync)
    sync_tree
    ;;
  build)
    shift
    sync_tree
    remote_build "${1:-$BUILD_DIR}"
    ;;
  test)
    shift
    remote_test "${1:-diff_tests}" "${2:-}"
    ;;
  record)
    shift
    sync_tree
    remote_record "${1:-}"
    ;;
  pull)
    shift
    src="${1:-$RECORD_DIR_REMOTE}"
    dest="${2:-$RECORD_DIR_LOCAL}"
    mkdir -p "$dest"
    echo ">> rsync $HOST:$src/ -> $dest/"
    sshpass -e rsync -az -e "ssh ${SSH_OPTS[*]}" "$HOST:$src/" "$dest/"
    ;;
  run)
    shift
    ssh_run "$REMOTE_PATH; $*"
    ;;
  shell)
    ssh_run
    ;;
  *)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
