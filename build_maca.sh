#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-maca}"
BUILD_TYPE="Release"
JOBS="$(nproc)"
BUILD_TESTS="ON"
BUILD_EXAMPLES="ON"
BUILD_BENCHMARK="OFF"
BUILD_SHARED_LIBS="ON"
WARNINGS_AS_ERRORS="OFF"
CUCC_TARGET=""

usage() {
  cat <<'EOF'
Usage: ./build_maca.sh [options] [clean]

Options:
  -d, --debug          Build with debug symbols
  -j N                 Use N parallel build jobs
  --build-dir DIR      Override the build directory
  --target ARCH        Set CUCC_TARGETS (for example, xcore1000)
  --no-tests           Do not build unit tests
  --no-examples        Do not build examples
  --benchmark          Build benchmarks
  --lib-only           Build only the MPP library
  --shared             Build shared libraries (default)
  --static             Build static libraries
  --werror             Treat warnings as errors
  clean                Remove the selected build directory
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  -d | --debug)
    BUILD_TYPE="Debug"
    shift
    ;;
  -j)
    JOBS="$2"
    shift 2
    ;;
  --build-dir)
    BUILD_DIR="$2"
    shift 2
    ;;
  --target)
    CUCC_TARGET="$2"
    shift 2
    ;;
  --no-tests)
    BUILD_TESTS="OFF"
    shift
    ;;
  --no-examples)
    BUILD_EXAMPLES="OFF"
    shift
    ;;
  --benchmark)
    BUILD_BENCHMARK="ON"
    shift
    ;;
  --lib-only)
    BUILD_TESTS="OFF"
    BUILD_EXAMPLES="OFF"
    BUILD_BENCHMARK="OFF"
    shift
    ;;
  --shared)
    BUILD_SHARED_LIBS="ON"
    shift
    ;;
  --static)
    BUILD_SHARED_LIBS="OFF"
    shift
    ;;
  --werror)
    WARNINGS_AS_ERRORS="ON"
    shift
    ;;
  clean)
    rm -rf "${BUILD_DIR}"
    echo "Removed ${BUILD_DIR}"
    exit 0
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
  esac
done

if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid job count: ${JOBS}" >&2
  exit 2
fi

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export CUCC_PATH="${CUCC_PATH:-${MACA_PATH}/tools/cu-bridge}"
export CUDA_PATH="${CUDA_PATH:-${CUCC_PATH}}"
export CUCC_CMAKE_ENTRY="${CUCC_CMAKE_ENTRY:-2}"
export WCUDA_HOME="${WCUDA_HOME:-${TMPDIR:-/tmp}/npp-cu-bridge-${USER:-user}}"
export PATH="${CUCC_PATH}/tools:${CUCC_PATH}/bin:${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LIBRARY_PATH="${CUCC_PATH}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"

if [[ -n "${CUCC_TARGET}" ]]; then
  export CUCC_TARGETS="${CUCC_TARGET}"
fi

for tool in cucc cmake_maca make_maca; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Required cu-bridge tool not found: ${tool}" >&2
    exit 1
  fi
done

mkdir -p "${BUILD_DIR}"
BUILD_DIR="$(cd "${BUILD_DIR}" && pwd)"

echo "Configuring MACA build in ${BUILD_DIR}"
echo "MACA_PATH=${MACA_PATH} CUCC_CMAKE_ENTRY=${CUCC_CMAKE_ENTRY} BUILD_TYPE=${BUILD_TYPE}"

# cmake_maca Entry2 defaults to Unix Makefiles. Passing -G explicitly is
# avoided because some cu-bridge releases preserve generator quotes.
cmake_maca -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DMACA_ROOT="${MACA_PATH}" \
  -DBUILD_TESTS="${BUILD_TESTS}" \
  -DBUILD_EXAMPLES="${BUILD_EXAMPLES}" \
  -DBUILD_BENCHMARK="${BUILD_BENCHMARK}" \
  -DBUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
  -DNPP_WARNINGS_AS_ERRORS="${WARNINGS_AS_ERRORS}" \
  -DUSE_NVIDIA_NPP=OFF

echo "Building with make_maca -j${JOBS}"
(
  cd "${BUILD_DIR}"
  make_maca -j"${JOBS}"
)

echo "MACA build completed: ${BUILD_DIR}"
