#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$BENCHMARK_DIR")"

BUILD_TYPE="Release"
USE_NVIDIA_NPP="OFF"
CUDA_ARCH=""
JOBS="$(nproc 2>/dev/null || echo 4)"
CLEAN_BUILD=false
FORCE_REBUILD=false
WARNINGS_AS_ERRORS="OFF"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
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
  ./benchmark/scripts/build_benchmark.sh [options]

Options:
  --use-nvidia-npp  Build benchmark in build-nvidia/ and link NVIDIA NPP
  --debug, -d       Debug build
  --release         Release build (default)
  --clean           Remove the target build directory before configuring
  --rebuild         Force full reconfigure
  --arch N          Set CUDA architecture explicitly
  --no-werror       Disable warnings-as-errors
  -j N              Parallel jobs for build
  --help, -h        Show this help
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

detect_cuda_arch() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local compute_cap
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    if [[ "$compute_cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
      echo "${compute_cap//./}"
      return
    fi
  fi
  echo "75"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-nvidia-npp)
      USE_NVIDIA_NPP="ON"
      shift
      ;;
    --debug|-d)
      BUILD_TYPE="Debug"
      shift
      ;;
    --release)
      BUILD_TYPE="Release"
      shift
      ;;
    --clean)
      CLEAN_BUILD=true
      shift
      ;;
    --rebuild)
      FORCE_REBUILD=true
      shift
      ;;
    --arch)
      require_value "$1" "${2:-}"
      CUDA_ARCH="$2"
      shift 2
      ;;
    --no-werror)
      WARNINGS_AS_ERRORS="OFF"
      shift
      ;;
    -j)
      require_value "$1" "${2:-}"
      JOBS="$2"
      shift 2
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

if ! command -v cmake >/dev/null 2>&1; then
  print_error "cmake not found"
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  print_error "nvcc not found; benchmark build requires CUDA toolchain"
  exit 1
fi

if [ -z "$CUDA_ARCH" ]; then
  CUDA_ARCH="$(detect_cuda_arch)"
fi

if [ "$USE_NVIDIA_NPP" = "ON" ]; then
  BUILD_DIR="$PROJECT_DIR/build-nvidia"
  LIB_NAME="NVIDIA NPP"
else
  BUILD_DIR="$PROJECT_DIR/build"
  LIB_NAME="MPP (in-tree)"
fi

if [ "$CLEAN_BUILD" = true ]; then
  print_info "Cleaning $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

if [ "$FORCE_REBUILD" = true ] && [ -d "$BUILD_DIR" ]; then
  print_info "Removing stale CMake cache from $BUILD_DIR"
  rm -rf "$BUILD_DIR/CMakeCache.txt" "$BUILD_DIR/CMakeFiles"
fi

mkdir -p "$BUILD_DIR"

print_info "Configuring benchmark against $LIB_NAME"
print_info "Build directory: $BUILD_DIR"
print_info "Build type: $BUILD_TYPE"
print_info "CUDA arch: $CUDA_ARCH"

cmake \
  -S "$PROJECT_DIR" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARK=ON \
  -DNPP_WARNINGS_AS_ERRORS="$WARNINGS_AS_ERRORS" \
  -DUSE_NVIDIA_NPP="$USE_NVIDIA_NPP" \
  -DWITHOUT_CU_BRIDGE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

print_info "Building npp_benchmark with -j $JOBS"
cmake --build "$BUILD_DIR" --target npp_benchmark -j "$JOBS"

if [ ! -x "$BUILD_DIR/benchmark/npp_benchmark" ]; then
  print_error "Benchmark executable not found: $BUILD_DIR/benchmark/npp_benchmark"
  exit 1
fi

print_success "Benchmark build completed"
echo "Executable: $BUILD_DIR/benchmark/npp_benchmark"
