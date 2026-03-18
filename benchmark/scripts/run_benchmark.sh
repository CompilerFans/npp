#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$BENCHMARK_DIR")"
RESULTS_DIR="$BENCHMARK_DIR/results"

USE_NVIDIA_NPP="OFF"
FILTER=""
OUTPUT=""
LIST_ONLY=false
FORCE_BUILD=false
NO_BUILD=false
JSON_OUTPUT=false
VERBOSE=false
EXTRA_ARGS=()

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}  NPP/MPP Performance Benchmark Runner${NC}"
  echo -e "${BLUE}========================================${NC}"
}

print_info() {
  echo -e "${CYAN}[INFO]${NC} $1"
}

print_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
  cat <<'EOF'
Usage:
  ./benchmark/scripts/run_benchmark.sh [options]

Options:
  --use-nvidia-npp      Run benchmark linked to NVIDIA NPP
  -f, --filter PATTERN  Run only benchmarks whose names contain PATTERN
  -o, --output FILE     Output prefix or file path
  -l, --list            List available benchmark entries
  --build               Force rebuild before running
  --no-build            Do not auto-build if executable is missing
  --json                Output JSON
  --verbose, -v         Print command details
  --help, -h            Show this help
EOF
  exit 0
}

require_value() {
  local opt="$1"
  local value="${2:-}"
  if [ -z "$value" ] || [[ "$value" == -* ]]; then
    print_error "Option $opt requires a value"
    exit 1
  fi
}

default_output_prefix() {
  date +"$RESULTS_DIR/benchmark_results_%Y%m%d_%H%M%S"
}

normalize_output_file() {
  local output_prefix="$1"
  local ext="$2"
  if [[ "$output_prefix" == *"$ext" ]]; then
    output_prefix="${output_prefix%$ext}"
  fi
  echo "${output_prefix}${ext}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-nvidia-npp)
      USE_NVIDIA_NPP="ON"
      shift
      ;;
    -f|--filter)
      require_value "$1" "${2:-}"
      FILTER="$2"
      shift 2
      ;;
    -o|--output)
      require_value "$1" "${2:-}"
      OUTPUT="$2"
      shift 2
      ;;
    -l|--list)
      LIST_ONLY=true
      shift
      ;;
    --build)
      FORCE_BUILD=true
      shift
      ;;
    --no-build)
      NO_BUILD=true
      shift
      ;;
    --json)
      JSON_OUTPUT=true
      EXTRA_ARGS+=("--json")
      shift
      ;;
    --verbose|-v)
      VERBOSE=true
      EXTRA_ARGS+=("--verbose")
      shift
      ;;
    --help|-h)
      show_help
      ;;
    *)
      print_error "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ "$LIST_ONLY" = false ] && [ "$JSON_OUTPUT" = false ] && [ -z "$OUTPUT" ]; then
  OUTPUT="$(default_output_prefix)"
fi

get_benchmark_exe() {
  if [ "$1" = "ON" ]; then
    echo "$PROJECT_DIR/build-nvidia/benchmark/npp_benchmark"
  else
    echo "$PROJECT_DIR/build/benchmark/npp_benchmark"
  fi
}

ensure_benchmark_built() {
  local exe
  exe="$(get_benchmark_exe "$1")"

  if [ "$NO_BUILD" = true ]; then
    if [ ! -x "$exe" ]; then
      print_error "Benchmark executable not found: $exe"
      exit 1
    fi
    return
  fi

  if [ "$FORCE_BUILD" = true ] || [ ! -x "$exe" ]; then
    if [ "$1" = "ON" ]; then
      "$SCRIPT_DIR/build_benchmark.sh" --use-nvidia-npp
    else
      "$SCRIPT_DIR/build_benchmark.sh"
    fi
  fi
}

run_benchmark() {
  local exe
  exe="$(get_benchmark_exe "$1")"

  if [ "$1" = "ON" ]; then
    print_info "Running NVIDIA NPP benchmark..."
  else
    print_info "Running benchmark against the in-tree project NPP..."
  fi
  print_info "Using benchmark executable: $exe"

  local -a cmd=("$exe")
  if [ -n "$FILTER" ]; then
    cmd+=("-f" "$FILTER")
  fi

  if [ -n "$OUTPUT" ]; then
    local out_ext=".csv"
    if [ "$JSON_OUTPUT" = true ]; then
      out_ext=".json"
    fi
    local out_file
    out_file="$(normalize_output_file "$OUTPUT" "$out_ext")"
    mkdir -p "$(dirname "$out_file")"
    cmd+=("-o" "$out_file")
    print_info "Results will be saved to: $out_file"
  fi

  if [ "$LIST_ONLY" = true ]; then
    cmd=("$exe" "-l")
  fi

  if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  if [ "$VERBOSE" = true ]; then
    print_info "Command: ${cmd[*]}"
  fi

  echo ""
  "${cmd[@]}"
}

mkdir -p "$RESULTS_DIR"
print_header
ensure_benchmark_built "$USE_NVIDIA_NPP"
run_benchmark "$USE_NVIDIA_NPP"

echo ""
print_success "Benchmark completed!"
